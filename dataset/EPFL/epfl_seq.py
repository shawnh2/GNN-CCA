import csv
import os.path as osp

import torch

from tools.sequence import Sequence


class EPFLSequence(Sequence):
    """
    Load EPFL dataset meta information into Sequence.
    Its annotations row format:
    (track_id, x_min, y_min, x_max, y_max, frame_number, lost, occluded, generated, label).
    """

    def __init__(self, dataset_path: str, dataset_meta_info: dict):
        self.frames_range = dataset_meta_info['valid_frames_range']
        self.cam_nbr = dataset_meta_info['cam_nbr']
        self.annots = [osp.join(dataset_path, dataset_meta_info['annot_fn_pattern'].format(c))
                       for c in range(self.cam_nbr)]
        self.videos = [osp.join(dataset_path, dataset_meta_info['video_fn_pattern'].format(c))
                       for c in range(self.cam_nbr)]
        self.shape = dataset_meta_info['video_hwc']
        self.img_nm = dataset_path + '/frames/{}_{}.jpg'

        self.H = torch.tensor(dataset_meta_info['homography'])
        super(EPFLSequence, self).__init__()

    def _load(self):
        for cam_id, annotation in enumerate(self.annots):
            fp = open(annotation, 'r')
            rd = csv.reader(fp, delimiter=' ')
            for row in rd:
                # Filter out the lost one.
                if row[6] == '1':
                    continue
                pid, x_min, y_min, x_max, y_max, fid = tuple(map(int, row[:6]))
                # Filter out the frame that outside the valid frames range.
                if fid > self.frames_range[1] or fid < self.frames_range[0]:
                    continue
                # Update frame meta information.
                self._frames.setdefault(fid, []).append(
                    (x_min, y_min, x_max - x_min, y_max - y_min, pid, cam_id)
                )
            fp.close()

    def get_frame_images(self, frame_id):
        return [self.img_nm.format(frame_id, camera_id) for camera_id in range(self.cam_nbr)]
