#### Streamlit App Hosted with Heroku
[Yolo Self Driving Car App](https://yolo-self-driving.herokuapp.com/)

#### Problem Statement
Is it possible, to utilize transfer learning, to create a real-time object detection model for self-driving vehicles?

#### Image Collection
The dataset used for this app is the Udacity Self Driving Car Dataset #2. I downloaded this dataset using [RoboFlow](https://public.roboflow.com/object-detection/self-driving-car). It consists of 15,000 images and annotation files.

#### Darknet Framework
All training is done through the Darknet framework. The repo can be found [here](https://github.com/AlexeyAB/darknet). Also, you will need to download the YOLOv4-tiny.weights file from [here](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights). These weights are the pre-trained weights for YOLOv4-tiny on the Coco dataset and will be used at the start of training.

        cd darknet
        ./darknet.exe detector train data/cars.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.weights -clear -gpus 0 -map

#### Requirements
I recommend creating a new conda environment named Yolo.
Then you need to install the requirements.txt file.

        pip install -r requirements.txt

#### My Custom Weights
Unfortunately, due to the large file size of my custom trained yolo weights I had to add them to my .gitignore file. This will prevent being able to run detections if you clone the repo. Instead, you will need to follow instructions in Darknet Framework on how to compile Darknet and where to download the weights. Once this is done, you will be able to utilize the darknet framework to train your own custom model and from there all the scripts will work.

#### Next Steps
- Implement my video detection script in Streamlit.
- Create counter for each instance of an object.
- Experiment with DeepSort to see if I can create an object tracker