import torch
import numpy as np
import cv2
from urllib.request import urlopen
import os
import platform
from ultralytics import YOLO
# from ultralytics.yolo.utils.plotting import Annotator


class YoloV8ImageObjectDetection:
    """
    Class implements Yolo8 model to make inferences on a youtube video using Opencv2.
    """
    
    PATH        = os.environ.get("YOLO_WEIGHTS_PATH", "yolov8n.pt")
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.10"))
    IOU         = float(os.environ.get("YOLO_IOU_THRESHOLD", "0.50"))

    def __init__(self, url: str = None, chunked: bytes = None):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self._URL = url
        self._bytes = chunked
        self.model = self._load_model()
        self.device = self._get_device()
        self.classes = self.model.names

    def _get_device(self):
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_image_from_chunked(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        arr = np.asarray(bytearray(self._bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'
        return img

    def _get_image_from_url(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        req = urlopen(self._URL)
        self._bytes = req.read()
        return self._get_image_from_chunked()

    def _load_model(self):
        """
        Loads Yolo8 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = YOLO(YoloV8ImageObjectDetection.PATH)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame, conf=YoloV8ImageObjectDetection.CONF_THRESH, save_conf=True, iou=YoloV8ImageObjectDetection.IOU)
        return results

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        for r in results:
            boxes = r.boxes
            labels = []
            for box in boxes:
                b = box.xyxy[
                    0
                ]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                l = self.model.names[int(c)]
                labels.append(l)
        frame = results[0].plot()
        return frame, labels

    async def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """

        if self._URL:
            frame = self._get_image_from_url()
        else:
            frame = self._get_image_from_chunked()

        results = self.score_frame(frame)
        frame, labels = self.plot_boxes(results, frame)
        return frame, set(labels)
