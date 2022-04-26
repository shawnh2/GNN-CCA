import os
import json
import random

import cv2
import tqdm


class Preprocess:

    def __init__(self,
                 dataset_name: str,
                 base_dir: str,
                 annots_name: list,
                 videos_name: list,
                 eval_ratio: float,
                 test_ratio: float,
                 valid_frames_range=None,
                 image_format='jpg',
                 random_seed=202204):
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(base_dir, dataset_name)
        self.annots_name = annots_name  # must correspond to cameras id
        self.videos_name = videos_name  # also correspond to annotation files
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.valid_frames_range = valid_frames_range
        self.image_format = image_format
        self.output_path = os.path.join(self.dataset_path, 'output')
        self.frames_output_path = os.path.join(self.output_path, 'frames')
        self.frames = {}

        random.seed(random_seed)

    def process(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self.frames_output_path):
            os.mkdir(self.frames_output_path)

        self.load_annotations()
        self.filter_frames()
        self.load_videos()
        self.train_test_split()

    def load_annotations(self):
        """
        Parse annotation files that across all cameras to obtain frames(self).
        Within data format:
            [frame_id]: (top_left_x, top_left_y, width, height, track_id, camera_id)
        """
        raise NotImplementedError

    def filter_frames(self):
        # Filter out frames that the person only appear in one camera view.
        invalid_frames_id = []
        for frame_id, frame in self.frames.items():
            cams = set([sample[-1] for sample in frame])
            if len(cams) <= 1:
                invalid_frames_id.append(frame_id)

        for frame_id in invalid_frames_id:
            self.frames.pop(frame_id)

    def select_frames(self, frames_id: list):
        ret = {}
        for frame_id in frames_id:
            ret[frame_id] = self.frames[frame_id]
        return ret

    def load_videos(self):
        frames_id = list(self.frames.keys())
        caps = [cv2.VideoCapture(os.path.join(self.dataset_path, vn))
                for vn in self.videos_name]
        avail_frames = sorted(frames_id)
        cur_frame_id = 0
        for frame_id in tqdm.trange(0, avail_frames[-1] + 1, desc=self.dataset_name):
            # Skip the invalid frame.
            if frame_id != avail_frames[cur_frame_id]:
                for cap in caps:
                    cap.read()
                continue
            # Capture one frame across all cameras.
            for cam_id, cap in enumerate(caps):
                ret, img = cap.read()
                if ret:
                    cv2.imwrite(os.path.join(self.frames_output_path,
                                             f'{frame_id}_{cam_id}.{self.image_format}'), img)
                else:
                    raise ValueError(f"Cannot load frame {frame_id} from "
                                     f"video {self.dataset_name} at cam {cam_id}.")
            cur_frame_id += 1

    def train_test_split(self):
        frames_id = list(self.frames.keys())
        random.shuffle(frames_id)
        n = len(frames_id)
        n_test = int(self.test_ratio * n)
        n_eval = int(self.eval_ratio * (n - n_test))
        test_frames_id = frames_id[-n_test:]
        rest_frames_id = frames_id[:-n_test]
        eval_frames_id = rest_frames_id[-n_eval:]
        train_frames_id = rest_frames_id[:-n_eval]

        train_frames = self.select_frames(train_frames_id)
        eval_frames = self.select_frames(eval_frames_id)
        test_frames = self.select_frames(test_frames_id)
        self.save_json(train_frames, 'train')
        self.save_json(eval_frames, 'eval')
        self.save_json(test_frames, 'test')

    def save_json(self, obj, name):
        with open(os.path.join(self.output_path, f'{name}.json'), 'w') as fp:
            json.dump(obj, fp)
