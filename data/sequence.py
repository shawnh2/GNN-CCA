# Generate a frame-based sequence from one dataset

import csv
import os.path as osp


class Sequence:
    def __init__(self,
                 dataset_path: str,
                 dataset_meta_info: dict):
        self._frames = {}
        self.cam_nbr = dataset_meta_info['cam_nbr']
        self.annots = [osp.join(dataset_path, dataset_meta_info['annot_fn_pattern'].format(c))
                       for c in range(self.cam_nbr)]
        self.videos = [osp.join(dataset_path, dataset_meta_info['video_fn_pattern'].format(c))
                       for c in range(self.cam_nbr)]
        self.H = dataset_meta_info['homography']
        self._load()

    def _load(self):
        for cam_id, annotation in enumerate(self.annots):
            fp = open(annotation, 'r')
            rd = csv.reader(fp, delimiter=' ')
            # row format: track_id, x_min, y_min, x_max, y_max, frame_number, lost, occluded, generated, label
            for row in rd:
                # filter out the lost one
                if row[6] == '1':
                    continue
                pid, x_min, y_min, x_max, y_max, fid = tuple(map(int, row[:6]))
                self._frames.setdefault(fid, []).append((
                    x_min, y_min, x_max - x_min, y_max - y_min,  # tlwh
                    pid,      # global person id
                    cam_id))  # camera id
            fp.close()

    def frames(self) -> list:
        return list(self._frames.keys())

    def __getitem__(self, fid: int):
        return self._frames.get(fid)

    def __len__(self):
        return len(self._frames)
