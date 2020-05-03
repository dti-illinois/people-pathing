'''
People Pathing

Author: Rishi Masand 2020

Detection and tracking adapted from the following resource
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

    Determines paths of multiple people moving in a video
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

    def _detect_image_objects(self, img):
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

        # get video
        video = cv2.VideoCapture(video_file_name)

        # determine number of frames to process
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames:
            frame_count = min(frame_count, num_frames)

        # get set of colors for bounding boxes
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # keep track of people between frames and their paths
        tracker = Sort()
        measured_paths = {}

        if not silent:
            print('Processing ' + str(frame_count) + ' frames.')

        # process frames one at a time
        for frame_idx in range(frame_count):
            if not silent:
                print('Processing frame: ' + str(frame_idx))

            # get frame image
            _, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(frame)

            # get objects detected in image
            objects = self._detect_image_objects(pilimg)

            # calculate image paddings used by object detector
            img = np.array(pilimg)
            pad_x = max(img.shape[0] - img.shape[1], 0) * \
                (self.img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * \
                (self.img_size / max(img.shape))
            unpad_h = self.img_size - pad_y
            unpad_w = self.img_size - pad_x

            # process next frame if no objects detected
            if objects is None:
                # show frame if necessary
                if show_detection:
                    plt.figure(figsize=(12, 8))
                    plt.title('Frame')
                    plt.imshow(frame)
                    plt.show(block=False)
                    plt.pause(0.1)
                    plt.close()
                continue

            # update object tracker
            tracked_objects = tracker.update(objects.cpu())

            for x_1, y_1, x_2, y_2, obj_id, pred_class in tracked_objects:
                # verify that object is person
                if self.classes[int(pred_class)] != 'person':
                    continue

                # get bounding box frame
                box_h = int(((y_2 - y_1) / unpad_h) * img.shape[0])
                box_w = int(((x_2 - x_1) / unpad_w) * img.shape[1])
                y_1 = int(((y_1 - pad_y // 2) / unpad_h) *
                          img.shape[0])
                x_1 = int(((x_1 - pad_x // 2) / unpad_w) *
                          img.shape[1])

                # update measured path (tracking at mid-bottom of box)
                mid_x = x_1 + box_w / 2
                bottom_y = y_1 + box_h
                if obj_id in measured_paths:
                    measured_paths[obj_id].append((mid_x, bottom_y))
                else:
                    measured_paths[obj_id] = [(mid_x, bottom_y)]

                # only draw on frame if being shown
                if not show_detection:
                    continue
                # get color for object
                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]

                # get class name
                class_name = self.classes[int(pred_class)]

                # draw bounding box
                cv2.rectangle(frame,
                              (x_1, y_1),
                              (x_1 + box_w, y_1 + box_h),
                              color, 4)

                # draw object label
                cv2.rectangle(frame,
                              (x_1, y_1 - 35),
                              (x_1 + len(class_name) * 19 + 60, y_1),
                              color, -1)
                cv2.putText(frame,
                            class_name + '-' + str(int(obj_id)),
                            (x_1, y_1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 3)

            # show frame if necessary
            if show_detection:
                plt.figure(figsize=(12, 8))
                plt.title('Video Stream')
                plt.imshow(frame)
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()

        image_size = (img.shape[1], img.shape[0])
        return (image_size, measured_paths)

    def plot_paths(self, image_size, paths):
        '''Plots paths'''

        # configure plot as per image size
        plt.figure(figsize=(12, 8))
        plt.xlim(0, image_size[0])
        plt.ylim(image_size[1], 0)

        for object_id in paths:
            path = paths[object_id]
            x_path, y_path = [list(t) for t in zip(*path)]
            plt.plot(x_path, y_path)
        plt.show()

    def plot_paths_on_image(self, image_size, paths, image_file_name):
        '''Plots object paths on image'''

        _, a_x = plt.subplots()
        img = plt.imread(image_file_name)
        plt.xlim(0, image_size[0])
        plt.ylim(image_size[1], 0)
        a_x.imshow(img, extent=[0, 610, 340, 0])
        for object_id in paths:
            path = paths[object_id]
            x_path, y_path = [list(t) for t in zip(*path)]
            a_x.plot(x_path, y_path)
        plt.show()
