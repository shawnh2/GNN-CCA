import random

import cv2
import dgl
import torch
import torch.nn.functional as F
from torchvision import transforms

from tools.sequence import Sequence


class DataLoader:

    def __init__(self,
                 feature_extractor,
                 device,
                 batch_size,
                 is_training=True,
                 shuffle=True):
        self.device = device
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle

        self._extractor = feature_extractor
        self._transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.4),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomErasing(0.3)
        ])

    def flow(self, *sequence: Sequence):
        """Sample a batch of graphs and features for training."""
        frames = [(i, frame_id)
                  for i, seq in enumerate(sequence)
                  for frame_id in seq.avail_frames()]
        if self.shuffle:
            random.shuffle(frames)

        batch = len(frames) // self.batch_size
        for i in range(batch):
            data = []
            for (seq_id, frame_id) in frames[i * self.batch_size: (i + 1) * self.batch_size]:
                seq = sequence[seq_id]
                args = self.construct_graph(
                    torch.tensor(seq[frame_id], dtype=torch.int32),
                    self.load_frames_from_paths(seq.get_frame_images(frame_id)),
                    seq.H
                )
                if args is not None:
                    data.append(args)
            yield data

    def load_frames_from_paths(self, img_path):
        """Load frames from a list of image-path using opencv.
        :return: a list of images with shape (C, H, W)
        """
        imgs = []
        for i, img_p in enumerate(img_path):
            img = cv2.imread(img_p)
            imgs.append(self.process_img(img))
        return imgs

    @staticmethod
    def process_img(img):
        """Convert a numpy.ndarray image to torch.Tensor image."""
        res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = torch.from_numpy(res)
        res = torch.permute(res, (2, 0, 1))  # (H, W, C) => (C, H, W)
        return res.float()

    def construct_graph(self, frame_meta_info, frame_images, frame_h_mats):
        """
        Construct a graph from one frame across all cameras.

        Args:
            frame_meta_info: Detections across all the cameras with same frame id,
                             data format [tlwh, global_person_id, camera_id]
            frame_images: A list that contains the current frame images
                          across all cameras with shape (C, H, W).
            frame_h_mats: A tensor that contains the homography matrix for all cameras.
        Returns:
            A Graph with its features or None.
        """
        # Graph without any nodes.
        node_nbr = frame_meta_info.shape[0]
        if node_nbr == 0:
            return None

        # Indexing graph.
        u, v = [], []
        dets = []  # bounding box detections
        lbls = []  # labels of edges
        projs = torch.zeros(node_nbr, 3, dtype=torch.float32)  # projected coordinates
        for n1 in range(node_nbr):
            src_pid, src_cid = frame_meta_info[n1, -2:]
            for n2 in range(n1, node_nbr):
                # Constraint: skip the connection between two identical cameras
                dst_pid, dst_cid = frame_meta_info[n2, -2:]
                if dst_cid == src_cid:
                    continue
                u.append(n1)
                v.append(n2)
                # 1 if two detections have the same identity, otherwise 0
                lbls.append(1 if dst_pid == src_pid else 0)
            x, y, w, h = frame_meta_info[n1, :4]
            # Obtain camera projection by projecting the foot point of bounding box.
            proj = torch.matmul(frame_h_mats[int(src_cid)],
                                torch.tensor([x + w / 2, y, 1], dtype=torch.float32))
            projs[n1] = proj / proj[-1]
            # Obtain bounding box image.
            det = frame_images[int(src_cid)][:, y: y + h, x: x + w]
            det = transforms.Resize((256, 128))(det)
            if self.is_training:
                det = self._transforms(det)
            dets.append(det.unsqueeze(0))

        # Graph without any edges.
        edge_nbr = len(u)
        if edge_nbr == 0:
            return None

        graph = dgl.graph((u + v, v + u), device=self.device)  # undirected graph
        y_true = torch.tensor(lbls + lbls, dtype=torch.float32, device=self.device).unsqueeze(1)  # (E, 1)
        # Backward edges mask.
        graph.edata['b_mask'] = torch.tensor([0] * edge_nbr + [1] * edge_nbr, dtype=torch.int32, device=self.device)
        # Obtain the initial node appearance feature.
        node_feature = self._extractor(torch.cat(dets))  # (N, 3, 256, 128) -> (N, 512)
        # Obtain the initial edge appearance feature and spatial information.
        embedding = torch.vstack((
            torch.pairwise_distance(node_feature[u], node_feature[v]),  # euclid distance
            torch.cosine_similarity(node_feature[u], node_feature[v]),  # cosine distance
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),  # l1 distance
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device)   # l2 distance
        ))  # (4, E/2)
        embedding = F.normalize(embedding, p=2, dim=1).T  # (E/2, 4)
        edge_feature = torch.cat((embedding, embedding))  # (E, 4)

        return graph, node_feature, edge_feature, y_true
