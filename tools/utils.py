import os


def udf_collate_fn(batch):
    return batch


def get_color(idx):
    idx = int(idx) * 3
    return (37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
