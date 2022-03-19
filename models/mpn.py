import dgl
import dgl.function as dfn
import torch
import torch.nn as nn


class MPN(nn.Module):
    """Message Passing Neural Network."""

    def __init__(self, device, ckpt=None):
        super(MPN, self).__init__()
        self.node_msg_encoder = nn.Sequential(
            nn.Linear(38, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.edge_msg_encoder = nn.Sequential(
            nn.Linear(70, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )
        self.to(device)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def apply_forward_edges(self, edges):
        edge_msg = self.edge_msg_encoder(
            torch.cat((edges.src['x'], edges.dst['x'], edges.data['x']), 1)
        )
        return {'em': edge_msg}

    def apply_backward_edges(self, edges):
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
            # Message passing for edges:
            # All two directions edges should obtain the same message.
            graph.apply_edges(
                self.apply_forward_edges,
                edges=graph.filter_edges(lambda edges: edges.data['b_mask'] == 0)
            )
            graph.apply_edges(
                self.apply_backward_edges,
                edges=graph.filter_edges(lambda edges: edges.data['b_mask'] == 1)
            )
            # Message passing for nodes:
            # Obtain message from its adjacent nodes (only consider one direction here).
            for n in range(graph.num_nodes()):
                graph.send_and_recv(
                    edges=dgl.in_subgraph(graph, [n]).edges(),
                    message_func=self.apply_adjacent_nodes,
                    reduce_func=dfn.sum('msg', 'nm')
                )
            return graph.ndata['nm'], graph.edata['em']
