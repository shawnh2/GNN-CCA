def udf_collate_fn(batch):
    return batch


def get_color(idx):
    idx = idx * 3
    return (37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255
