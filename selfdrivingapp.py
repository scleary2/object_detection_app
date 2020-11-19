import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import pytube
import time


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def download_youtube(url):
    youtube = pytube.YouTube(url)
    video = youtube.streams.first()
    video = video.download("./")
    return video


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


def image_preds(image, confidence, nms):
    img = cv2.imread(image)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model, colors, class_names = load_yolo(
        "yolo/yolov4-tiny-custom_best.weights", "yolo/yolov4-tiny-custom.cfg"
    )
    classes, scores, boxes = model.detect(img, confidence, nms)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = colors[int(classid) % len(colors)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(img, box, color, 2)
        cv2.putText(
            img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # cv2.imshow("test_img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    st.image(img, use_column_width=True)


def video_preds(video):
    vc = cv2.VideoCapture(video)
    frame_width, frame_height = int(video.get(3)), int(video.get(4))
    out = cv2.VideoWriter(
        "output.avi", CV_FOURCC("M", "J", "P", "G"), 40, (frame_width, frame_height)
    )
    model, colors, class_names = load_yolo(
        "yolo/yolov4-tiny-custom_best.weights", "yolo/yolov4-tiny-custom.cfg"
    )
    while True:
        (grabbed, frame) = vc.read()
        if not grabbed:
            exit()

        start = time.time()
        classes, scores, boxes = model.detect(frame, 0.3, 0.4)
        end = time.time()
        out = out.write(frame)
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
            frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        # cv2.imshow("detections", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        st.video(frame, format="video/mp4", start_time=0)
    vc.release()
    out.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def main():
    """ Object Detection App"""

    st.title("Object Detection App")
    st.text("Build with Streamlit and OpenCV")

    activities = ["Image", "Video", "Webcam"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Image":
        st.subheader("Object Detection")

        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if image_file is not None:
            our_image = Image.open(image_file)
            our_image.save("test_image.jpg")
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image, height=608, width=608)

        else:
            st.image(our_image, width=300)

        if st.button("Process"):
            yolo_out = image_preds("test_image.jpg", 0.3, 0.4)

        # Object Detection
        # st.sidebar.title("What to do")
        # feature_choice = st.sidebar.selectbox(
        #     "Choose the app mode", ["Run the app", "Select"]
        # )
        # if feature_choice == "Select":
        #     pass
        # elif feature_choice == "Run the app":
        #     st.button("process")
        #     yolo_out = image_preds("test_image.jpg", 0.4, 0.2)

    elif choice == "Video":
        st.subheader("Detections on Video")
        # user_input = st.text_input("What is the url for youtube?")

        video_file = st.file_uploader("Upload Video", type=["mp4"])

        if video_file is not None:

            video = st.video(video_file, format="video/mp4", start_time=0)
        #     user_input = st.text_input("What is the url for youtube?")
        #     video = download_youtube(f"{user_input}")

        if st.button("Process"):
            # video = download_youtube(f"{user_input}")
            video_out = video_preds(video)


if __name__ == "__main__":
    main()