import numpy as np
import torch
from cv2 import VideoCapture,CAP_PROP_POS_MSEC
# from moviepy.editor import VideoFileClip

from .base import Loader


class MoviePy(Loader):

    def __init__(self, video_path, parent_dir=''):
        super().__init__(video_path, parent_dir)
        # self.video = VideoFileClip(self.path, audio=False)
        self.video = VideoCapture(self.path)   
        # self.fps = self.video.fps
        self.fps = self.video.get(5)   # 帧速率

    def read_iter(self, batch_size=1, limit=None, stride=1, start=0):
        images, image_ids = [], []
        # positions = np.arange(
        #     start, self.video.duration * self.fps, stride / self.fps)
        positions = np.arange(start, self.video.get(7), stride / self.fps)
        for image_id, pos in enumerate(positions[:limit]):
            # image = self.video.get_frame(pos)

            cap.set(CAP_PROP_POS_MSEC, pos * 1000)
            success, image = cap.read()  # BGR
            image = image[..., ::-1]  # RGB

            image = np.ascontiguousarray(image[:, :, ::-1])
            image = torch.as_tensor(image)
            images.append(image)
            image_ids.append(image_id)
            if len(images) == batch_size:
                yield images, image_ids
                images, image_ids = [], []
        if len(images) > 0:
            yield images, image_ids
