# -*- coding: utf-8 -*-

!wget https://raw.githubusercontent.com/MaxMLgh/YOLO_tutorial/master/net/coco.txt -O coco.txt
!wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3.cfg
!wget https://pjreddie.com/media/files/yolov3-tiny.weights -O yolov3-tiny.weights
!wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg -O yolov3-tiny.cfg
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -O yolov4.weights
!wget https://raw.githubusercontent.com/MaxMLgh/YOLO_tutorial/master/net/yolov4.cfg -O yolov4.cfg
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights -O yolov4-tiny.weights
!wget https://raw.githubusercontent.com/MaxMLgh/YOLO_tutorial/master/net/yolov4-tiny.cfg -O yolov4-tiny.cfg

!pip install opencv-python==4.5.1.48 numpy==1.19.2

import cv2
import numpy as np
np.set_printoptions(suppress=True)
from urllib.request import Request, urlopen
import urllib
import time
import os
import traceback
from google.colab.patches import cv2_imshow

class Detection:
    font = cv2.FONT_HERSHEY_PLAIN
    colors = ((255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (128,0,0))

    def __init__(self, model_name = 'yolov3', input_height=416, path_yolo_classes='coco.txt',
                 output_height=800, is_scale_output = True, MIN_confidence=0.5, IOU_threshold=0.6,
                 is_blob_aspect_ratio=True, anchor_box_show=False, grid_show=False, show_text_on_box=True,
                 is_recording=False, show_text_left=False):
        self.model_name = model_name
        self.net = None
        self.anchors = None

        self.grids_per_height = round(input_height/32)
        self.grids_per_width = self.grids_per_height
        self.input_height = self.grids_per_height * 32
        self.input_width = self.input_height
        if input_height%32:
            print('''Значение input_height={} неделимо на 32, вместо него будет использоваться input_height={}.
Выберите значение input_height, которое является целым числом, кратным 32 (например, 320,416,620)).'''.format(input_height,
                                                                                self.input_height))

        with open(path_yolo_classes, 'r') as f:
            self.classes = f.read().splitlines()

        self.anchor_box_show = anchor_box_show
        self.grid_show = grid_show
        self.show_text_on_box = show_text_on_box
        self.show_text_left = show_text_left
        self.is_recording = is_recording
        self.is_any_frame_recorded = False

        self.MIN_confidence = MIN_confidence
        self.IOU_threshold = IOU_threshold
        self.FPS = 0.0

        self.img = None
        self.img_name = None
        self.img_with_drawings = None
        self.img_height, self.img_width = None, None
        self.boxes = None
        self.confidences = None
        self.best_class_ids = None
        self.grid_cells = None
        self.anchor_boxes = None
        self.bounding_box_centers = None
        self.detection_outputs = None

        self.is_blob_aspect_ratio = is_blob_aspect_ratio

        self.is_scale_output = is_scale_output
        self.output_height = output_height

        if cv2.cuda.getCudaEnabledDeviceCount():
            self.is_cuda = True
        else:
            self.is_cuda = False



    def configure_net(self, model_name=None):
          if model_name!=None:
            self.model_name = model_name

          path_weights = '{}.weights'.format(self.model_name)
          path_cfg = '{}.cfg'.format(self.model_name)
          self.net = cv2.dnn.readNet(path_weights, path_cfg)

          if self.is_cuda:
              self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
              self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


          with open('{}.cfg'.format(self.model_name), 'r') as f:
              cfg = f.read().splitlines()
              anchors_line = [line for line in cfg if 'anchors' in line][0].replace('anchors', '').replace('=', '')
              anchors = np.array([int(num) for num in anchors_line.split(',')])
              anchors = anchors.reshape(int(len(anchors)/6), 3, 2)[::-1]
              self.anchors = anchors


    def detect(self, img):
        if self.is_blob_aspect_ratio:
            ratio_width2height = img.shape[1]/img.shape[0]
            self.grids_per_width = round((self.input_height * ratio_width2height)/32)
            self.input_width = self.grids_per_width*32
        else:
            self.input_width = self.input_height
            self.grids_per_width = self.grids_per_height


        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.input_width, self.input_height),
                                 (0, 0, 0), swapRB=True, crop=False)

        self.net.setInput(blob)
        output_layers_names = self.net.getUnconnectedOutLayersNames()
        self.layerOutputs = self.net.forward(output_layers_names)


        if self.is_scale_output:
            img = image_resize(img, height = self.output_height)
        self.img = img
        self.img_height, self.img_width, _ = self.img.shape

        boxes = []
        confidences = []
        best_class_ids = []
        grid_cells = []
        anchor_boxes = []
        bounding_box_centers = []
        detection_outputs = []


        for i, output in enumerate(self.layerOutputs):

            if self.model_name == 'yolov4':
                if i==0:
                    i=2
                elif i==2:
                    i=0


            for j, detection in enumerate(output):

                scores = detection[5:]
                best_class_id = np.argmax(scores)
                confidence = detection[4] * scores[best_class_id]

                if confidence > 0.001:
                    anchor_box = self.anchors[i][j % 3]
                    grid_cell = [int(j / 3) % (self.grids_per_width * 2 ** i),
                                 int(j / (self.grids_per_width * 3 * 2 ** i))]

                    center_x = round(detection[0] * self.img_width)
                    center_y = round(detection[1] * self.img_height)
                    w = round(detection[2] * self.img_width)
                    h = round(detection[3] * self.img_height)
                    x = round(center_x - w / 2)
                    y = round(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    best_class_ids.append(best_class_id)
                    grid_cells.append(grid_cell)
                    anchor_boxes.append(anchor_box)
                    bounding_box_centers.append((center_x, center_y))
                    detection_outputs.append(i)

        self.boxes = boxes
        self.confidences = confidences
        self.best_class_ids = best_class_ids
        self.grid_cells = grid_cells
        self.anchor_boxes = anchor_boxes
        self.bounding_box_centers = bounding_box_centers
        self.detection_outputs = detection_outputs


    def draw_img(self):
        img = self.img.copy()
        if self.show_text_left:
            cv2.putText(img, "IOU:  {0:.0%}".format(self.IOU_threshold), (20, 40), self.font, 3, (0, 0, 255), 3)
            cv2.putText(img, "CONF: {0:.0%}".format(self.MIN_confidence), (20, 80), self.font, 3, (255, 0, 0), 3)


        indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.MIN_confidence, self.IOU_threshold)
        if len(indexes) > 0:
            for c, i in enumerate(indexes.flatten()):
                x, y, w, h = self.boxes[i]
                label = str(self.classes[self.best_class_ids[i]])
                confidence = self.confidences[i]
                color = self.colors[c%len(self.colors)]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                if self.show_text_on_box:
                    cv2.putText(img, '{}.{} {:.0%}'.format(c+1, label, confidence), (x+2, y-5),
                                self.font, 2, color, 3)
                elif self.show_text_left:
                    cv2.putText(img, '{}'.format(c+1), (x+2, y-5), self.font, 2, color, 3)

                num_of_grids_w = (self.grids_per_width * 2 ** self.detection_outputs[i])
                num_of_grids_h = (self.grids_per_height * 2 ** self.detection_outputs[i])
                grid_corner_x= int(round(self.grid_cells[i][0] * self.img_width / num_of_grids_w))
                grid_corner_y = int(round(self.grid_cells[i][1] * self.img_height / num_of_grids_h))
                grid_w = round(self.img_width / num_of_grids_w)
                grid_h = round(self.img_height / num_of_grids_h)
                if self.grid_show:
                    cv2.rectangle(img, (grid_corner_x, grid_corner_y), (grid_corner_x+ grid_w, grid_corner_y + grid_h),
                                  color, int(4 / 2 ** self.detection_outputs[i]))
                    cv2.circle(img, (self.bounding_box_centers[i]), 3, color, 4)

                ab_center_x = round(grid_corner_x+ grid_w * 0.5)
                ab_center_y = round(grid_corner_y + grid_h * 0.5)
                ab_width = self.anchor_boxes[i][0] * self.img_width / self.input_width
                ab_height = self.anchor_boxes[i][1] * self.img_height / self.input_height
                if self.anchor_box_show:
                    cv2.rectangle(img, (round(ab_center_x - 0.5 * ab_width),
                                        round(ab_center_y - 0.5 * ab_height)),
                                       (round(ab_center_x + 0.5 * ab_width),
                                        round(ab_center_y + 0.5*ab_height)),
                                       color, int(4 / 2 ** self.detection_outputs[i]))
                    cv2.rectangle(img, (round(ab_center_x - 0.5 * ab_width),
                                        round(ab_center_y - 0.5 * ab_height)),
                                       (round(ab_center_x + 0.5 * ab_width),
                                        round(ab_center_y + 0.5*ab_height)),
                                       (255,255,255), 1)
                    text = '{}:{} {:.0%} {}({})'.format(c+1, label, confidence, self.anchor_boxes[i],
                                    self.detection_outputs[i])
                else:
                    text = '{}:{} {:.0%}'.format(c+1, label, confidence)
                if self.show_text_left:
                    cv2.putText(img, text, (20, 210 + 30 * c),self.font, 2, color, 3)
        if self.show_text_left:
            cv2.putText(img, '{mn} {iw}x{ih}'.format(mn=self.model_name, iw=self.input_width,
                        ih=self.input_height), (20, 110),self.font, 2, (255,0,255), 3)
            cv2.putText(img, 'FPS: {:.2f}'.format(self.FPS), (20, 140),
                    self.font, 2, (255,0,255), 3)
            cv2.putText(img, 'REC:{}'.format('ON' if self.is_recording else "OFF"), (20, 170),
            self.font, 2, ((0, 255, 0) if self.is_recording else (0, 0, 255)), 3)

        self.img_with_drawings = img
        cv2_imshow(img)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

Det = Detection(model_name = 'yolov4', input_height=416, path_yolo_classes='coco.txt',
                 output_height=800, is_scale_output = True, MIN_confidence=0.5, IOU_threshold=0.8,
                 is_blob_aspect_ratio=True, anchor_box_show=False, grid_show=False, show_text_on_box=True,
                 is_recording=False, show_text_left=True)
Det.configure_net()
url_to_img = '''

https://www.majorcars.ru/files/upload/images/202077_Volvo_Cars.jpg

'''

try:
    req = Request(url_to_img, headers={'User-Agent': 'Mozilla/5.0'})
    req = urlopen(req).read()
    arr = np.asarray(bytearray(req), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    print('Размер изображения {}.'.format(img.shape[:2]))
    if not int(img.shape[0]):
        print("Файл не найден, попробуйте другой")

    start_time = time.time()
    Det.detect(img)
    Det.FPS = 1/(time.time() - start_time)
    Det.draw_img()
except urllib.error.HTTPError:
    print("Файл не найден, попробуйте другой")
    print(traceback.format_exc())
