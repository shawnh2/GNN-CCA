import os
import csv
import json

from base import Preprocess


class EPFLPreprocess(Preprocess):

    def load_annotations(self):
        for cam_id, annot_name in enumerate(self.annots_name):
            fp = open(os.path.join(self.dataset_path, annot_name))
            rd = csv.reader(fp, delimiter=' ')
            # Within annotations format:
            # (track_id, x_min, y_min, x_max, y_max, frame_number, lost, occluded, generated, label)
            for row in rd:
                # Filter out the lost one.
                if row[6] == '1':
                    continue
                track_id, x_min, y_min, x_max, y_max, frame_id = tuple(map(int, row[:6]))
                # Filter out the frame that outside the valid frames range.
                if frame_id > self.valid_frames_range[1] or \
                   frame_id < self.valid_frames_range[0]:
                    continue
                self.frames.setdefault(frame_id, []).append(
                    (x_min, y_min, x_max - x_min, y_max - y_min, track_id, cam_id)
                )
            fp.close()


def preprocess_epfl(dataset_dir):
    with open(os.path.join(dataset_dir, 'metainfo.json'), 'r') as fp:
        meta_info = json.load(fp)

    for dataset_name in meta_info.keys():
        dmi = meta_info[dataset_name]
        epfl = EPFLPreprocess(
            dataset_name,
            dataset_dir,
            [dmi['annot_fn_pattern'].format(cam_id) for cam_id in range(dmi['cam_nbr'])],
            [dmi['video_fn_pattern'].format(cam_id) for cam_id in range(dmi['cam_nbr'])],
            dmi['eval_ratio'],
            dmi['test_ratio'],
            dmi['valid_frames_range']
        )
        epfl.process()


if __name__ == '__main__':
    preprocess_epfl("./dataset/EPFL")
