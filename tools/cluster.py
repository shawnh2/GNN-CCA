import cv2
import numpy as np
import networkx as nx
from sklearn import metrics

from tools.utils import get_color


class ClusterDetections:

    def __init__(self, inputs, targets, graph):
        self.inputs = inputs.squeeze(1).cpu()
        self.targets = targets.squeeze(1).int().cpu()
        self.graph = graph

        self._cam_info = graph.ndata['cam'].cpu().numpy()
        self._box_info = graph.ndata['box'].cpu().numpy()
        self._act_edges = np.zeros(inputs.shape[0], dtype=np.int32)
        self._n_cams = np.unique(self._cam_info).size
        self._n_node = graph.num_nodes()
        self._G = nx.Graph()  # draw graph from active edges
        self._G.add_nodes_from([n for n in range(self._n_node)])
        self._G.add_weighted_edges_from(self._filter_edges())

    def _filter_edges(self):
        # Only keep the active edges, filter out the non-active ones.
        weight_active_edges = []
        edges_id = np.where(self.inputs >= 0.5)[0]
        for edge_id, u, v in zip(edges_id, *self.graph.find_edges(edges_id)):
            weight_active_edges.append(
                (int(u), int(v), float(self.inputs[edge_id]))
            )
        self._act_edges[edges_id] = 1
        return weight_active_edges

    def scores(self):
        return (
            metrics.adjusted_rand_score(self.targets, self._act_edges),  # ARI
            metrics.adjusted_mutual_info_score(self.targets, self._act_edges),  # AMI
            metrics.accuracy_score(self.targets, self._act_edges),  # ACC
            metrics.homogeneity_score(self.targets, self._act_edges),  # H
            metrics.completeness_score(self.targets, self._act_edges),  # C
            metrics.v_measure_score(self.targets, self._act_edges)  # V-m
        )

    def pruning_and_splitting(self):
        """Postprocessing."""
        self.pruning()
        self.splitting()
        # return nx.connected_components(self._G)

    def pruning(self):
        """
        Flow constraint: meaning that each node can be connected to,
        at most, other M - 1 nodes, being M the number of total cameras.
        """
        for cc in nx.connected_components(self._G):
            sub_graph = self._G.subgraph(cc)
            for node_id in cc:
                flow = sub_graph.degree(node_id)
                if flow > self._n_cams - 1:
                    prun_edges = sub_graph.edges(node_id, data=True)
                    min_edge = min(prun_edges, key=lambda x: x[-1]['weight'])
                    u, v = min_edge[:2]
                    self._G.remove_edge(u, v)
                    # Deactivate the deleted edge.
                    self._act_edges[self.graph.edge_ids(u, v)] = 0

    def splitting(self):
        """
        Cardinality constraint: namely the maximum cardinality of each identity set
        should be, at most, the total number of cameras.
        """
        for cc in nx.connected_components(self._G):
            sub_graph = self._G.subgraph(cc)
            for node_id in cc:
                edges = sub_graph.edges(node_id)
                if len(edges) == 0:
                    continue
                edge_uv = np.array([e for e in edges], dtype=np.int32)
                dst_cam = self._cam_info[edge_uv[:, 1]]
                error_cam = np.where(np.bincount(dst_cam) > 1)[0]
                for err_cam_id in error_cam:
                    split_edges = edge_uv[dst_cam == err_cam_id]
                    edge_weight = [sub_graph.get_edge_data(*e.tolist())['weight']
                                   for e in split_edges]
                    u, v = split_edges[np.argmin(edge_weight)]
                    self._G.remove_edge(u, v)
                    # Deactivate the deleted edge.
                    self._act_edges[self.graph.edge_ids(u, v)] = 0

    def visualize(self,
                  images_path: list,
                  output_path: str,
                  line_thickness=2,
                  font_thickness=1,
                  font_scale=1.0):
        gids = np.zeros(self._n_node, dtype=np.int32)  # global id
        for gid, cc in enumerate(nx.connected_components(self._G)):
            gids[list(cc)] = gid

        imgs = []
        nid = 0  # node id, also local id
        for cam_id, img_path in enumerate(images_path):
            boxes = self._box_info[self._cam_info == cam_id]
            img = cv2.imread(img_path)
            for box in boxes:
                gid = int(gids[nid])
                x, y, w, h = box
                color = get_color(gid)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, line_thickness)
                cv2.putText(img, str(gid), (x, y),
                            cv2.FONT_HERSHEY_PLAIN, font_scale, color, font_thickness)
                nid += 1
            imgs.append(img)
        cv2.imwrite(output_path, np.vstack(imgs))
