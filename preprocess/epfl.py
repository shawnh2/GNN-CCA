import os

import cv2
import tqdm

from sequence import EPFLSequence
from meta_info import EPFL


def preprocess_epfl():
    # Save video frames to local disk.
    for dataset_nm, dataset_info in EPFL.items():
        dataset_dir = f'./dataset/EPFL/{dataset_nm}'
        outputs_dir = os.path.join(dataset_dir, 'frames')
        if not os.path.exists(outputs_dir):
            os.mkdir(outputs_dir)

        seq = EPFLSequence(dataset_dir, EPFL[dataset_nm])
        avail_frames = sorted(seq.avail_frames)
        cur_avail_frame = 0
        caps = [cv2.VideoCapture(vp) for vp in seq.videos]
        for frame_id in tqdm.trange(seq.frames_range[0], seq.frames_range[1] + 1, desc=dataset_nm):
            # Skip the invalid frame.
            if frame_id != avail_frames[cur_avail_frame]:
                for cap in caps:
                    cap.read()
                continue
            # Capture one frame across all cameras.
            for i, cap in enumerate(caps):
                ret, img = cap.read()
                if ret:
                    cv2.imwrite(os.path.join(outputs_dir, f'{frame_id}_{i}.png'), img)
                else:
                    raise ValueError(f'Cannot load frame {frame_id} from video {dataset_nm} at cam {i}.')
            cur_avail_frame += 1


if __name__ == '__main__':
    preprocess_epfl()
