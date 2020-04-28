'''
People Pathing

Author: Rishi Masand 2020

Adapted from the following resource
Title: Object Detection and Tracking
Author: Chris Fotache
Link: https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98)
'''

import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from lib.sort import Sort
from lib.models import Darknet
from lib.utils import load_classes, non_max_suppression

warnings.filterwarnings('ignore')


class PeoplePathing():
    '''
    People Pathing

    Derives paths of people moving in a video
    '''
    config_path = 'config/yolov3.cfg'
    weights_path = 'config/yolov3.weights'
    class_path = 'config/coco.names'
    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4

    def __init__(self):
        self.model = Darknet(self.config_path, img_size=self.img_size)
        self.model.load_weights(self.weights_path)
        self.model.eval()
        self.classes = load_classes(self.class_path)
        self.tensor = torch.FloatTensor

    def _get_image_objects(self, img):
        '''Detects objects in image'''

        # Scale and pad image
        ratio = min(self.img_size/img.size[0], self.img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([
            transforms.Resize((imh, imw)),
            transforms.Pad((
                max(int((imh-imw)/2), 0),
                max(int((imw-imh)/2), 0),
                max(int((imh-imw)/2), 0),
                max(int((imw-imh)/2), 0)
                ), (128, 128, 128)),
            transforms.ToTensor()])

        # Convert image to tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(self.tensor))

        # Get detected objects from image
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(
                detections, 80,
                self.conf_thres, self.nms_thres)
        return detections[0]

    def get_paths(self, video_file_name, num_frames,
                  show_detection=False, silent=False):
        '''Gets paths of objects in video'''

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        vid = cv2.VideoCapture(video_file_name)
        if num_frames:
            num_frames = min(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)), num_frames)
        else:
            num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        mot_tracker = Sort()

        object_paths = {}

        for frame_idx in range(num_frames):
            if not silent:
                print('Processing frame: ' + str(frame_idx))

            _, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pilimg = Image.fromarray(frame)
            objects = self._get_image_objects(pilimg)

            img = np.array(pilimg)
            pad_x = max(img.shape[0] - img.shape[1], 0) * \
                (self.img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * \
                (self.img_size / max(img.shape))
            unpad_h = self.img_size - pad_y
            unpad_w = self.img_size - pad_x
            if objects is not None:
                tracked_objects = mot_tracker.update(objects.cpu())

                for x_1, y_1, x_2, y_2, obj_id, cls_pred in tracked_objects:
                    if obj_id in object_paths:
                        object_paths[obj_id].append(
                            (abs((x_2 + x_1) / 2), abs((y_2 + y_1) / 2)))
                    else:
                        object_paths[obj_id] = [(abs((x_2 + x_1) / 2),
                                                 abs((y_2 + y_1) / 2))]

                    if show_detection:
                        box_h = int(((y_2 - y_1) / unpad_h) * img.shape[0])
                        box_w = int(((x_2 - x_1) / unpad_w) * img.shape[1])
                        y_1 = int(((y_1 - pad_y // 2) / unpad_h) *
                                  img.shape[0])
                        x_1 = int(((x_1 - pad_x // 2) / unpad_w) *
                                  img.shape[1])

                        color = colors[int(obj_id) % len(colors)]
                        color = [i * 255 for i in color]
                        cls = self.classes[int(cls_pred)]
                        cv2.rectangle(frame,
                                      (x_1, y_1),
                                      (x_1 + box_w, y_1 + box_h),
                                      color, 4)
                        cv2.rectangle(frame,
                                      (x_1, y_1 - 35),
                                      (x_1 + len(cls) * 19 + 60, y_1),
                                      color, -1)
                        cv2.putText(frame,
                                    cls + '-' + str(int(obj_id)),
                                    (x_1, y_1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 3)
            if show_detection:
                plt.figure(figsize=(12, 8))
                plt.title('Video Stream')
                plt.imshow(frame)
                plt.show(block=False)
                plt.pause(0.25)
                plt.close()
        return object_paths

    def plot_object_paths(self, object_paths):
        '''Plots object paths'''

        for object_id in object_paths:
            object_path = object_paths[object_id]
            x_path, y_path = [list(t) for t in zip(*object_path)]
            plt.plot(x_path, y_path)
        plt.show()

    def plot_object_paths_on_image(self, object_paths, image_file_name):
        '''Plots object paths on image'''

        _, a_x = plt.subplots()
        img = plt.imread(image_file_name)
        a_x.imshow(img, extent=[0, 610, 0, 340])
        for object_id in object_paths:
            object_path = object_paths[object_id]
            x_path, y_path = [list(t) for t in zip(*object_path)]
            a_x.plot(x_path, y_path)
        plt.show()
