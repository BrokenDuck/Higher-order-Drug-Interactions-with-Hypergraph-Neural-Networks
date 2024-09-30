import itertools
import argparse
import os.path as osp
import time
import random

import pandas as pd
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import kaiming_uniform, zeros
from torch_geometric.nn.aggr import MLPAggregation
from torch_geometric.utils import scatter, softmax
from torch_geometric.logging import log

parser = argparse.ArgumentParser()
parser.add_argument(
    "--enc_hgraph",
    type=str,
    default="enc.csv",
    help="The encoder hypergraph that decomposes the molecules.",
    required=True,
)
parser.add_argument(
    "--dec_hgraph",
    type=str,
    default="dec.csv",
    help="The decoder hypergraph that represents node interactions.",
    required=True,
)
parser.add_argument(
    "--dec_hedge_lab",
    type=str,
    default="lab.csv",
    help="The labels for the hyperedges of the decoder.",
    required=True,
)
parser.add_argument(
    "--num_node_features",
    type=int,
    default=128,
    help="The number of features for the node embedings obtained with hyperedge attention in the encoder.",
)
parser.add_argument(
    "--num_query_features",
    type=int,
    default=64,
    help="The number of features for the computation of the attention coefficients in the encoder.",
)
parser.add_argument(
    "--num_hedge_features",
    type=int,
    default=128,
    help="The number of features of latent representation of the molecules during and after the encoder.",
)
parser.add_argument(
    "--lr", type=float, default=0.005, help="The learning rate of the Adam optimizer."
)
parser.add_argument(
    "--test_ratio",
    type=float,
    default=0.2,
    help="The ratio of the decoder labels used for test in the train-val-test split.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=500,
    help="The number of epochs to train the model for.",
)
parser.add_argument(
    "--stop_early",
    type=int,
    default=200,
    help="The number of epochs with no improvement until training is stopped.",
)
parser.add_argument(
    "--seed",
    type=int,
    help="The seed for the randomness in case reproducability is needed.",
)
args = parser.parse_args()

device = torch_geometric.device("auto")

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

#### Generating Graphs ####

# Generate the hypergraph for the encoder
df = pd.read_csv(args.enc_hgraph)
hedge_index = torch.tensor(df.values, dtype=torch.long).t().squeeze().contiguous()
assert (
    hedge_index.size(0) == 2
), "The file defining the encoder hypergraph needs two columns to define the hypergraph in bipartite form."
num_nodes = int(hedge_index[0].max()) + 1
num_hedges = int(hedge_index[1].max()) + 1
p = torch.ones(num_nodes, args.num_node_features)
q = torch.eye(num_hedges)
encoder_hgraph = Data(
    num_nodes=num_nodes,
    num_hedges=num_hedges,
    p=p,
    q=q,
    hedge_index=hedge_index,
).to(device)

# Generate the hypergraph for the decoder
df = pd.read_csv(args.dec_hgraph)
hedge_index = torch.tensor(df.values, dtype=torch.long).t().squeeze().contiguous()
assert (
    hedge_index.size(0) == 2
), "The file defining the decoder hypergraph needs two columns to define the hypergraph in bipartite form."
num_nodes = int(hedge_index[0].max()) + 1
num_hedges = int(hedge_index[1].max()) + 1
df = pd.read_csv(args.dec_hedge_lab)
labels = torch.tensor(df.values, dtype=torch.float).t().squeeze().contiguous()
assert labels.size(0) == num_hedges, "There needs to be one label per hyperedge."
num_test_hedges = round(args.test_ratio * num_hedges)
num_train_hedges = num_hedges - num_test_hedges
train_mask, test_mask = torch.randperm(num_hedges).split(
    [num_train_hedges, num_test_hedges]
)
decoder_hgraph = Data(
    num_nodes=num_nodes,
    num_node_features=args.num_hedge_features,
    num_hedges=num_hedges,
    num_hedge_features=1,
    hedge_index=hedge_index,
    train_mask=train_mask,
    num_train_hedges=num_train_hedges,
    test_mask=test_mask,
    num_test_hedges=num_test_hedges,
    labels=labels,
).to(device)

#### Generate HGNN ####


class AttentionConv(MessagePassing):
    r"""The hypergraph self-attention based convolutional operator from the `"HyGNN: Drug-Drug Interaction Prediction via Hypergraph Neural Network" <https://arxiv.org/abs/2206.12747>`_ paper.

    It consists of two layers:
    1. hyperedge-level attention
    2. node level attention

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        node_channels (int): Size of the node features.
        query_channels (int): Size of the intermediate vectors used in attention computation.
        heads (int, attention): Number of heads to use for attention. Defaults to 1.
        negative_slope (float, optional): Slope to use for the leaky ReLU function in the attention computation. Defaults to 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        node_channels: int,
        query_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_channels = node_channels
        self.query_channels = query_channels
        self.negative_slope = negative_slope

        self.w1 = Linear(
            in_channels, node_channels, bias=False, weight_initializer="glorot"
        )
        self.w4 = Linear(
            node_channels, out_channels, bias=False, weight_initializer="glorot"
        )

        self.w2 = Parameter(torch.empty(in_channels, query_channels))
        self.w3 = Parameter(torch.empty(node_channels, query_channels))
        self.w5 = Parameter(torch.empty(node_channels, query_channels))
        self.w6 = Parameter(torch.empty(in_channels, query_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.w1.reset_parameters()
        self.w4.reset_parameters()
        kaiming_uniform(self.w2, fan=self.in_channels, a=self.negative_slope)
        kaiming_uniform(self.w3, fan=self.node_channels, a=self.negative_slope)
        kaiming_uniform(self.w5, fan=self.node_channels, a=self.negative_slope)
        kaiming_uniform(self.w6, fan=self.in_channels, a=self.negative_slope)

    def forward(
        self,
        p: Tensor,
        q: Tensor,
        hedge_index: Tensor,
    ) -> Tensor:
        """Runs the forward pass of the module.

        Args:
            p (Tensor): Node feature matrix
            q (Tensor): Edge feature matrix
            hedge_index (Tensor): The hypergraph connection matrix

        Returns:
            Tensor: _description_
        """
        # p has shape [num_nodes, node_features]
        # q has shape [num_hedges, in_channels]
        # hedge_index has shape [2, n]

        num_nodes = p.size(0)
        num_hedges = q.size(0)
        n = hedge_index[0].size(0)

        ## Hyperedge Level-Attention ##

        # q_p has shape [num_edges, node_channels]
        q_p = self.w1(q)
        # q_pp has shape [num_edges, query_channels]
        q_pp = q @ self.w2
        # q_ppp has shape [num_nodes, query_channels]
        q_ppp = p @ self.w3

        # p_i has shape [n, query_channels]
        p_i = q_ppp[hedge_index[0]]
        # q_i has shape [n, query_channels]
        q_j = q_pp[hedge_index[1]]

        # e has shape [n]
        e = F.leaky_relu((p_i * q_j).sum(-1), self.negative_slope)
        e = softmax(e, hedge_index[0], num_nodes=num_nodes)

        # We propagate from the hedge to the node
        p_out = self.propagate(
            hedge_index.flip([0]), x=q_p, alpha=e, size=(num_hedges, num_nodes)
        )

        ## Node level attention ##

        # p_p has shape [num_nodes, out_channels]
        p_p = self.w4(p_out)
        # p_pp has shape [num_nodes, query_channels]
        p_pp = p_out @ self.w5
        # p_ppp has shape [num_edges, query_channels]
        p_ppp = q @ self.w6

        # p_i has shape [n, query_channels]
        p_i = p_pp[hedge_index[0]]
        # q_i has shape [n, query_channels]
        q_j = p_ppp[hedge_index[1]]

        # v has shape [n]
        v = F.leaky_relu((p_i * q_j).sum(-1), self.negative_slope)
        v = softmax(v, hedge_index[1], num_nodes=num_hedges)

        # We propagate from the node to the hedge
        q_out = self.propagate(
            hedge_index, x=p_p, alpha=v, size=(num_nodes, num_hedges)
        )

        return q_out

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        # x_j has shape [X, node_channels/out_channels]
        # e has shape [n]

        out = 1 / np.sqrt(self.query_channels) * alpha.view(-1, 1) * x_j
        return out


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        num_edges_features_in: int,
        num_edges_features_out: int,
        num_node_features: int,
        query_channels: int,
    ):
        super().__init__()
        self.ac = AttentionConv(
            in_channels=num_edges_features_in,
            out_channels=num_edges_features_out,
            query_channels=query_channels,
            node_channels=num_node_features,
        )

    def forward(self, p: Tensor, q: Tensor, hedge_index: Tensor) -> Tensor:
        q = self.ac(p, q, hedge_index)
        return q


class DotDecoder(MessagePassing):
    def __init__(self, in_channels: int):
        super().__init__(flow="source_to_target", node_dim=0, aggr="mul")

        self.in_channels = in_channels

    def forward(self, p: Tensor, hedge_index: Tensor) -> Tensor:
        num_nodes = p.size(0)
        num_hedges = int(hedge_index[1].max()) + 1

        out = self.propagate(hedge_index, x=p, size=(num_nodes, num_hedges))
        return out

    def update(self, out: Tensor) -> Tensor:
        return out.sum(-1)


class MLPDecoder(MessagePassing):
    def __init__(self, in_channels: int, max_hedge_degree: int):
        aggr = MLPAggregation(
            in_channels=in_channels,
            hidden_channels=in_channels / 2,
            out_channels=1,
            max_num_elements=max_hedge_degree,
        )
        super().__init__(flow="source_to_target", node_dim=0, aggr=aggr)

    def forward(self, p: Tensor, hedge_index: Tensor) -> Tensor:
        num_nodes = p.size(0)
        num_hedges = int(hedge_index[1].max()) + 1

        out = self.propagate(hedge_index, x=p, size=(num_nodes, num_hedges))
        return out


encoder = AttentionEncoder(
    num_edges_features_in=encoder_hgraph.num_hedges,
    num_edges_features_out=args.num_hedge_features,
    num_node_features=args.num_node_features,
    query_channels=args.num_query_features,
).to(device)

decoder = DotDecoder(in_channels=args.num_hedge_features).to(device)

# _, pos_counts = torch.unique(decoder_pos_hgraph.hedge_index[1], return_counts=True)
# _, neg_counts = torch.unique(decoder_neg_hgraph.hedge_index[1], return_counts=True)
# decoder = MLPDecoder(
#     in_channels=args.num_hedge_features,
#     max_hedge_degree=max(pos_counts.max(), neg_counts.max())
# )

#### Loss and Optimizer ####
optimizer = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr
)


def compute_loss(preds: Tensor, lables: Tensor) -> float:
    preds = torch.cat([pos_preds, neg_preds])
    labels = torch.cat([torch.ones(pos_num_hedges), torch.zeros(neg_num_hedges)])
    return F.binary_cross_entropy_with_logits(preds, labels)


def train():
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    # Encode the molecules
    embedings = encoder(
        p=encoder_hgraph.p,
        q=encoder_hgraph.q,
        hedge_index=encoder_hgraph.hedge_index,
    )

    # Decode the embedings to obtain the predictions
    preds = decoder(
        p=embedings,
        hedge_index=decoder_hgraph.hedge_index,
    )

    # Compute the loss and do gradient descent
    loss = F.binary_cross_entropy_with_logits(
        preds[decoder_hgraph.train_mask],
        decoder_hgraph.labels[decoder_hgraph.train_mask],
    )
    loss.backward()
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    encoder.eval()
    decoder.eval()

    # Encode the molecules
    embedings = encoder(
        p=encoder_hgraph.p,
        q=encoder_hgraph.q,
        hedge_index=encoder_hgraph.hedge_index,
    )

    # Decode the embedings to obtain the predictions
    preds = decoder(
        p=embedings,
        hedge_index=decoder_hgraph.hedge_index,
    )
    preds = torch.sigmoid(preds).round()

    accs = []
    for mask in [decoder_hgraph.train_mask, decoder_hgraph.test_mask]:
        accs.append(
            int((preds[mask] == decoder_hgraph.labels[mask]).sum()) / int(mask.sum())
        )
    return accs


best_test_acc = 0
patience = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, test_acc = test()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience = 0
    else:
        patience += 1
    if patience > 200:
        break
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
