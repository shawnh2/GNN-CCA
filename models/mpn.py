import dgl
import dgl.function as df
import torch
import torch.nn as nn


class MPN(nn.Module):
    """Message Passing Neural Network."""

    def __init__(self, device, ckpt=None):
        super(MPN, self).__init__()
        # Learnable MLP encoders, original design:
        # => Linear(38, 32) + ReLU()
        self.node_msg_encoder = nn.Sequential(
            nn.Linear(38, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # => Linear(70, 6) + ReLU()
        self.edge_msg_encoder = nn.Sequential(
            nn.Linear(70, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )
        self.to(device)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def apply_adjacent_edges(self, edges):
        edge_msg = self.edge_msg_encoder(
            torch.cat((edges.dst['x'], edges.src['x'], edges.data['x']), 1)
        )
        return {'em': edge_msg}

    def apply_adjacent_nodes(self, edges):
        node_msg = self.node_msg_encoder(
            torch.cat((edges.dst['x'], edges.data['em']), 1)
        )
        return {'msg': node_msg}

    def forward(self, graph, x_node, x_edge):
        with graph.local_scope():
            graph.ndata['x'] = x_node
            graph.edata['x'] = x_edge
            for n in range(graph.num_nodes()):
                adj_edges = dgl.in_subgraph(graph, [n]).edges()
                # Message passing for edges.
                graph.apply_edges(self.apply_adjacent_edges, adj_edges)
                # Message passing for nodes.
                # Obtain message from its adjacent nodes.
                graph.send_and_recv(adj_edges, self.apply_adjacent_nodes, df.sum('msg', 'nm'))
            return graph.ndata['nm'], graph.edata['em']
