#!python

import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import time
import cv2
import sys
import pytube


confidence = float(sys.argv[1])
nms = float(sys.argv[2])


def download_youtube(url):
    youtube = pytube.YouTube(url)
    video = youtube.streams.first()
    video = video.download()
    return video


# video_to_predict = download_youtube(video_url)


def load_yolo(config_path, weights_path):

    """This function will load in class names and yolov4"""
    class_names = []
    with open("yolo/cars.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))
    net = cv2.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 256)
    return model, colors, class_names


video = cv2.VideoCapture("yolo/chicagotest.mp4")
model, colors, class_names = load_yolo(
    "yolo/yolov4-tiny-custom_best.weights", "yolo/yolov4-tiny-custom.cfg"
)
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, size
)

while True:
    ret, frame = video.read()

    if ret == True:
        start = time.time()
        classes, scores, boxes = model.detect(frame, confidence, nms)
        end = time.time()
        result.write(frame)
        start = time.time()
        classes, scores, boxes = model.detect(frame, confidence, nms)
        end = time.time()

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = colors[int(classid) % len(colors)]
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(
                frame,
                label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        end_drawing = time.time()

        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (
            1 / (end - start),
            (end_drawing - start_drawing) * 1000,
        )
        cv2.putText(
            frame, fps_label, (0, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2
        )
        cv2.imshow("detections", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
video.release()
result.release()

cv2.destroyAllWindows()