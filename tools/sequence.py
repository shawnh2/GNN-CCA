
class Sequence:
    """
    Generate a frame-based sequence from one dataset.

    Within frames structure:
        {frame_id [int]: (top-left-x [int],
                          top-left-y [int],
                          width [int],
                          height [int],
                          global person id [int],
                          camera id [int])}

    Every subclass should have these attributes:
        - H: ground plane homography

    And these methods:
        - get_frame_images(self, frame_id): return the image file path list at a certain frame.
    """

    def __init__(self):
        self._frames = {}  # only store the available frames
        self._load()

    def _load(self):
        raise NotImplementedError()

    def get_frame_images(self, frame_id):
        raise NotImplementedError()

    def avail_frames(self) -> list:
        """Return all the available frame id in this sequence."""
        return list(self._frames.keys())

    def __getitem__(self, frame_id: int):
        return self._frames.get(frame_id)

    def __len__(self):
        return len(self._frames)
