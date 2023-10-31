# For machine learning
import torch
# For array computations
import numpy as np
# For image decoding / editing
import cv2
# For environment variables
import os
# For detecting which ML Devices we can use
import platform
# For actually using the YOLO models
from ultralytics import YOLO

class YoloV8ImageObjectDetection:
    PATH        = os.environ.get("YOLO_WEIGHTS_PATH", "yolov8n.pt")    # Path to a model. yolov8n.pt means download from PyTorch Hub
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.70")) # Confidence threshold

    def __init__(self, chunked: bytes = None):
        """Initializes a yolov8 detector with a binary image
        
        Arguments:
            chunked (bytes): A binary image representation
        """
        self._bytes = chunked
        self.model = self._load_model()
        self.device = self._get_device()
        self.classes = self.model.names

    def _get_device(self):
        """Gets best device for your system

        Returns:
            device (str): The device to use for YOLO for your system
        """
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self):
        """Loads Yolo8 model from pytorch hub or a path on disk

        Returns:
            model (Model) - Trained Pytorch model
        """
        model = YOLO(YoloV8ImageObjectDetection.PATH)
        return model

    async def __call__(self):
        """This function is called when class is executed.
        It analyzes a single image passed to its constructor
        and returns the annotated image and its labels
        
        Returns:
            frame (numpy.ndarray): Frame with bounding boxes and labels ploted on it.
            labels (list(str)): The corresponding labels that were found
        """
        frame = self._get_image_from_chunked()
        results = self.score_frame(frame)
        frame, labels = self.plot_boxes(results, frame)
        return frame, set(labels)
    
    def _get_image_from_chunked(self):
        """Loads an openCV image from the raw image bytes passed by
        the API.

        Returns: 
            img (numpy.ndarray): opencv2 image object from the raw binary
        """
        arr = np.asarray(bytearray(self._bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'
        return img
    
    def score_frame(self, frame):
        """Scores a single image with a YoloV8 model

        Arguments:
            frame (numpy.ndarray): input frame in numpy/list/tuple format.

        Returns:
            results list(ultralytics.engine.results.Results)): Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(
            frame, 
            conf=YoloV8ImageObjectDetection.CONF_THRESH, 
            save_conf=True
        )
        return results

    def class_to_label(self, x):
        """For a given label value, return corresponding string label.
        Arguments:
            x (int): numeric label

        Returns:   
            class (str): corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """Takes a frame and its results as input, 
        and plots the bounding boxes and label on to the frame.

        Arguments:
            results (list(ultralytics.engine.results.Results)): contains labels and coordinates predicted by model on the given frame.
            frame (numpy.ndarray): Frame which has been scored.
        
        Returns:
            frame (numpy.ndarray): Frame with bounding boxes and labels ploted on it.
            labels (list(str)): The corresponding labels that were found
        """
        for r in results:
            boxes = r.boxes
            labels = []
            for box in boxes:
                c = box.cls
                l = self.model.names[int(c)]
                labels.append(l)
        frame = results[0].plot()
        return frame, labels
