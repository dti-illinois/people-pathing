# people-pathing
Detects and tracks paths of individual people in a video

## Usage

To use, ensure you have `PyTorch` and `OpenCV` installed on your environment.
The YOLO model requires the `config` directory in your project directory have the following files:

```
config
+-- coco.data
+-- coco.names
+-- yolov3.cfg
+-- yolov3.weights
```

Run `python app.py` to try it out on a video file.
