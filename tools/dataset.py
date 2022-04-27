import json
import os.path as osp

import cv2
import dgl
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset


__all__ = ["EPFLDataset"]


class BaseGraphDataset(Dataset):
    """Base class for Graph Dataset."""

    def __init__(self, seq_names: list, mode: str, feature_extractor, dataset_dir: str):
        assert mode in ("train", "eval", "test")
        assert len(seq_names) != 0

        self.seq_names = seq_names
        self.mode = mode
        self.device = feature_extractor.device
        self.feature_extractor = feature_extractor
        self.dataset_dir = dataset_dir

        self._transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.3),
            T.RandomErasing()
        ])
        # ====== These values can be loaded via self.load_dataset() ======
        self._H = []  # homography matrices, H[seq_id][cam_id] => torch.Tensor(3*3)
        self._P = []  # images name pattern, F[seq_id][cam_id] => image path pattern
        self._S = []  # frames in sequences, S[seq_id] => frame based dict (key type: str)
        self._SFI = None  # a (N*2) size tensor, store <seq_id, frame_id>
        self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def load_images(self, seq_id: int, frame_id: int, tensor=True):
        imgs = []
        for img_path in self._P[seq_id]:
            img = cv2.imread(img_path.format(frame_id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if tensor:
                img = T.ToTensor()(img)  # (C, H, W), float
            else:
                img = torch.from_numpy(img)
                img = torch.permute(img, (2, 0, 1))  # (C, H, W), uint8
            imgs.append(img)
        return imgs

    def __len__(self):
        return self._SFI.shape[0]

    def __getitem__(self, index):
        sid, fid = tuple(map(int, self._SFI[index]))
        frames = torch.tensor(self._S[sid][str(fid)], dtype=torch.int32)
        n_node = frames.shape[0]
        # images of current frame
        frame_images = self.load_images(sid, fid)
        # projected coordinates
        projs = torch.zeros(n_node, 3, dtype=torch.float32)
        # bounding box detections
        bdets = torch.zeros(n_node, 3, 256, 128, dtype=torch.float32)

        u, v, lbls = [], [], []
        for n1 in range(n_node):
            src_tid, src_cid = frames[n1, -2:]
            for n2 in range(n1 + 1, n_node):
                # Constraint: skip the connection between two identical cameras.
                dst_tid, dst_cid = frames[n2, -2:]
                if dst_cid != src_cid:
                    u.append(n1)
                    v.append(n2)
                    # True edges: if two tracks have the same identity.
                    lbls.append(1 if dst_tid == src_tid else 0)
            x, y, w, h = frames[n1, :4]
            # Obtain camera projection by projecting the foot point of bounding box.
            proj = torch.matmul(self._H[sid][src_cid],
                                torch.tensor([x + w / 2, y + h, 1], dtype=torch.float32))
            projs[n1] = proj / proj[-1]
            # Obtain bounding box detections image.
            det = frame_images[int(src_cid)][:, y: y + h, x: x + w]
            det = T.Resize((256, 128))(det)
            if self.mode == 'train':
                det = self._transforms(det)
            bdets[n1] = det

        graph = dgl.graph((u + v, v + u), idtype=torch.int32, device=self.device)  # undirected graph
        graph.ndata['cam'] = frames[:, -1].to(self.device)  # for validation purpose
        graph.ndata['box'] = frames[:, :4].to(self.device)  # for visualization purpose
        other = {'seq_id': sid, 'frame_id': fid}

        y_true = torch.tensor(lbls + lbls, dtype=torch.float32, device=self.device).unsqueeze(1)  # (E, 1)
        # Obtain the initial node appearance feature.
        node_feature = self.feature_extractor(bdets)  # (N, 512)
        # Obtain the initial edge appearance feature and spatial information.
        embedding = torch.vstack((
            torch.pairwise_distance(node_feature[u], node_feature[v]),  # euclid distance
            torch.cosine_similarity(node_feature[u], node_feature[v]),  # cosine distance
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1).to(self.device),  # l1 distance
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2).to(self.device)   # l2 distance
        )).T  # (E / 2, 4)
        edge_feature = torch.cat((embedding, embedding))  # (E, 4)

        return graph, node_feature, edge_feature, y_true, other


class EPFLDataset(BaseGraphDataset):

    def load_dataset(self):
        with open(osp.join(self.dataset_dir, 'metainfo.json')) as fp:
            meta_info = json.load(fp)

        if len(self.seq_names) == 1 and self.seq_names[0] == 'all':
            self.seq_names = list(meta_info.keys())

        SFI = []
        for seq_id, name in enumerate(self.seq_names):
            output_path = osp.join(self.dataset_dir, name, 'output', f'{self.mode}.json')
            with open(output_path, 'r') as fp:
                frames = json.load(fp)
            frames_id = list(map(int, frames.keys()))
            f_idx = torch.tensor(frames_id, dtype=torch.int32).unsqueeze(1)
            s_idx = torch.full_like(f_idx, seq_id)
            SFI.append(torch.hstack([s_idx, f_idx]))
            self._S.append(frames)
            self._H.append(torch.tensor(meta_info[name]['homography']))
            self._P.append([f'{self.dataset_dir}/{name}/output/frames/{{}}_{i}.jpg'
                            for i in range(meta_info[name]['cam_nbr'])])
        self._SFI = torch.vstack(SFI)
