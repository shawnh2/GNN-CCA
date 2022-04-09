import random
from typing import Any
from itertools import cycle

import cv2
import dgl
import torch
from tqdm import trange
from torchvision import transforms


class DataLoader:

    def __init__(self,
                 sequences: list,
                 feature_extractor,
                 device,
                 batch_size: int,
                 val_batch_size: int,
                 is_training=True,
                 shuffle=True):
        self.sequences = sequences
        self.device = device
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.is_training = is_training
        self.shuffle = shuffle

        self._extractor = feature_extractor
        self._transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomErasing()
        ])

    def train_set(self):
        frames = [(i, frame_id)
                  for i, seq in enumerate(self.sequences)
                  for frame_id in seq.train_frames]
        if self.shuffle:
            random.shuffle(frames)
        return self._flow(frames, self.batch_size, False)

    def val_set(self):
        frames = [(i, frame_id)
                  for i, seq in enumerate(self.sequences)
                  for frame_id in seq.val_frames]
        return self._flow(frames, self.val_batch_size, True)

    def _flow(self, frames: list, batch_size: int, status_bar: bool):
        """Sample a batch of graphs and features for training or validating."""
        iteration = len(frames) // batch_size
        ranger = trange(iteration) if status_bar else range(iteration)
        for i in ranger:
            batch = [Any] * batch_size
            nbr = 0
            for (seq_id, frame_id) in cycle(frames[i * batch_size: (i + 1) * batch_size]):
                seq = self.sequences[seq_id]
                data = self.construct_graph(
                    torch.tensor(seq[frame_id], dtype=torch.int32),
                    self.load_frames_from_paths(seq.get_frame_images(frame_id)),
                    seq.H
                )
                if data is not None:
                    batch[nbr] = data
                    nbr += 1
                    if nbr == batch_size:
                        break
            yield batch

    def load_frames_from_paths(self, img_path):
        """Load frames from a list of image-path using opencv.

        Returns:
            A list of images with shape (C, H, W)
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

    def construct_graph(self, frame_meta_info, frame_images, cam_h_mats):
        """
        Construct a graph from one frame across all cameras.

        Args:
            frame_meta_info: Detections across all the cameras with same frame id,
                             data format [tlwh, global_person_id, camera_id].
            frame_images: A list that contains the current frame images
                          across all cameras with shape (C, H, W).
            cam_h_mats: A tensor that contains the homography matrix for all cameras.
        Returns:
            A Graph with its features or None.
        """
        # Graph without any nodes.
        node_nbr = frame_meta_info.shape[0]
        if node_nbr == 0:
            return None

        # Indexing graph.
        u, v, lbls = [], [], []
        for n1 in range(node_nbr):
            src_pid, src_cid = frame_meta_info[n1, -2:]
            for n2 in range(n1 + 1, node_nbr):
                # Constraint: skip the connection between two identical cameras
                dst_pid, dst_cid = frame_meta_info[n2, -2:]
                if dst_cid == src_cid:
                    continue
                u.append(n1)
                v.append(n2)
                # 1 if two detections have the same identity, otherwise 0
                lbls.append(1 if dst_pid == src_pid else 0)
        # Graph without any edges.
        edge_nbr = len(u)
        if edge_nbr == 0:
            return None

        # Obtaining features.
        bbdts = torch.zeros(node_nbr, 3, 256, 128, dtype=torch.float32)  # bounding box detections
        projs = torch.zeros(node_nbr, 3, dtype=torch.float32)  # projected coordinates
        for i in range(node_nbr):
            x, y, w, h, src_cid = frame_meta_info[i, [0, 1, 2, 3, -1]]
            # Obtain camera projection by projecting the foot point of bounding box.
            proj = torch.matmul(cam_h_mats[int(src_cid)],
                                torch.tensor([x + w / 2, y + h, 1], dtype=torch.float32))
            projs[i] = proj / proj[-1]
            # Obtain bounding box image.
            det = frame_images[int(src_cid)][:, y: y + h, x: x + w]
            det = transforms.Resize((256, 128))(det)
            if self.is_training:
                det = self._transform(det)
            bbdts[i] = det

        # Constructing graph
        graph = dgl.graph((u + v, v + u), device=self.device)  # undirected graph
        graph.ndata['cam'] = frame_meta_info[:, -1].to(self.device)

        y_true = torch.tensor(lbls + lbls, dtype=torch.float32, device=self.device).unsqueeze(1)  # (E, 1)
        # Obtain the initial node appearance feature.
        node_feature = self._extractor(bbdts)  # (N, 512)
        # Obtain the initial edge appearance feature and spatial information.
        embedding = torch.vstack((
            torch.pairwise_distance(node_feature[u], node_feature[v]),  # euclid distance
            torch.cosine_similarity(node_feature[u], node_feature[v]),  # cosine distance
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),  # l1 distance
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device)   # l2 distance
        )).T  # (E/2, 4)
        edge_feature = torch.cat((embedding, embedding))  # (E, 4)

        return graph, node_feature, edge_feature, y_true
