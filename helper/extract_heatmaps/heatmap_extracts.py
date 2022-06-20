import argparse
import csv
import sys
import os
import glob
import shutil
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time

from HRNet.lib import models
from HRNet.lib.config import cfg
from HRNet.lib.config import update_config
from HRNet.lib.core.function import get_final_preds
from HRNet.lib.utils.transforms import get_affine_transform


class HRNetHeatmapExtractor:

    COCO_KEYPOINT_INDEXES = {
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
    }

    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    SKELETON = [
        [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
    ]

    CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    NUM_KPTS = 17

    VIDEO_EXT = ['mov', 'mp4']

    CTX = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def __init__(self, cfg):
        """
        cfg     : configuration file
        opts    : Modify config options using the command-line
        """
        self.cfg = cfg

        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.box_model.to(self.CTX)
        self.box_model.eval()

        self.pose_model = eval('models.'+cfg['MODEL']['NAME']+'.get_pose_net')(
            cfg, is_train=False
        )

        if cfg['TEST']['MODEL_FILE']:
            print('=> loading model from {}'.format(cfg['TEST']['MODEL_FILE']))
            self.pose_model.load_state_dict(torch.load(cfg['TEST']['MODEL_FILE'], map_location=self.CTX), strict=False)
        else:
            print('expected model defined in config at TEST.MODEL_FILE')

        self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=cfg['GPUS'])
        self.pose_model.to(self.CTX)
        self.pose_model.eval()

    def extract_features(self, video, getHeatmapImage=False, return_kpts=False):
        cudnn.benchmark = self.cfg['CUDNN']['BENCHMARK']
        torch.backends.cudnn.deterministic = self.cfg['CUDNN']['DETERMINISTIC']
        torch.backends.cudnn.enabled = self.cfg['CUDNN']['ENABLED']

        assert video.split('.')[-1].lower() in self.VIDEO_EXT
        torch.cuda.empty_cache()

        vidcap = cv2.VideoCapture(video)

        heatmap_list = []
        kpts_list = []
        if getHeatmapImage:
            kpts_accept = [5,6,7,8,9,10,11,12,13,14]

            fourcc = cv2.VideoWriter_fourcc(*'MJEG')
            hm_full_save_path = os.path.join(self.cfg['OUTPUT_DIR'], 'heatmap_full.avi')
            out_hm_full = cv2.VideoWriter(hm_full_save_path, fourcc, 30.0, (self.cfg['TEST']['HEATMAP_OUT'][0], self.cfg['TEST']['HEATMAP_OUT'][1]))

            bodypart_save_paths = []
            out_bodyparts = []

            for idx, idx_kpt in enumerate(kpts_accept):
                bodypart_save_paths.append(os.path.join(self.cfg['OUTPUT_DIR'], f'heatmap_bodypart_{idx_kpt}.avi'))
                out_bodyparts.append(cv2.VideoWriter(bodypart_save_paths[-1], fourcc, 30.0, (self.cfg['TEST']['HEATMAP_OUT'][0], self.cfg['TEST']['HEATMAP_OUT'][1])))
        count = 0
        while True:
            ret, image_bgr = vidcap.read()
            if ret:
                count += 1
                if count % 100 == 0:
                    print(f"Extracted {count} heatmaps features")
                # if count > 80:
                #     break
                img_shape = [image_bgr.shape[1], image_bgr.shape[0]]
                heatmap_out_shape = self.cfg['TEST']['HEATMAP_OUT']
                image = image_bgr[:, :, [2, 1, 0]]

                input = []
                img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                heatmap_ori = np.zeros((heatmap_out_shape[1], heatmap_out_shape[0], self.NUM_KPTS))
                img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(self.CTX)
                input.append(img_tensor)

                # object detection box
                pred_boxes = self.get_person_detection_boxes(self.box_model, input, threshold=0.9)
                assert len(pred_boxes) <= 1, 'currently only support 1 person'

                # pose estimation
                kpts_box_list = []
                if len(pred_boxes) >= 1:
                    for box_ in pred_boxes:
                        box = np.array(box_)
                        box_out = self.get_box_out(box, img_shape, heatmap_out_shape)
                        box_out = np.ceil(box_out).astype(np.uint8)
                        center, scale = self.box_to_center_scale(box, cfg['MODEL']['IMAGE_SIZE'][0], cfg['MODEL']['IMAGE_SIZE'][1])
                        image_pose = image.copy() if cfg['DATASET']['COLOR_RGB'] else image_bgr.copy()
                        pose_preds, heatmap = self.get_pose_estimation_prediction(self.pose_model, image_pose, center, scale)
                        kpts_box_list.append(pose_preds)
                        if len(pose_preds)>=1:
                            for kpt in pose_preds:
                                self.draw_pose(kpt,image_bgr) # draw the poses
                        heatmap_person_out = cv2.resize(heatmap, (box_out[1,0]-box_out[0,0],box_out[1,1]-box_out[0,1]))
                        heatmap_ori[box_out[0,1]:box_out[1,1], box_out[0,0]:box_out[1,0]] = heatmap_person_out

                if return_kpts:
                    kpts_list.append(kpts_box_list)
                heatmap_list.append(heatmap_ori)
                

                if getHeatmapImage:
                    heatmap_img = heatmap_list[-1] * 255.0
                    heatmap_img = heatmap_img.astype(np.uint8).clip(0, 255)
                    for idx, idx_kpt in enumerate(kpts_accept):
                        heatmap_part = heatmap_img[..., idx_kpt]
                        # print(heatmap_part.shape)
                        # exit(0)
                        out_bodyparts[idx].write(np.stack([heatmap_part]*3, axis=-1))
                    heatmap_img = np.sum(heatmap_img.astype(np.float32), axis=-1).clip(0, 255).astype(np.uint8)
                    out_hm_full.write(np.stack([heatmap_img]*3, axis=-1))
    
            else:
                print("Heatmaps Extraction: Done!")

        cv2.destroyAllWindows()
        vidcap.release()

        if getHeatmapImage:
            out_hm_full.release()
            for idx in range(len(kpts_accept)):
                out_bodyparts[idx].release()

        if return_kpts:
            return heatmap_list, kpts_list
        return heatmap_list

    def draw_pose(self, keypoints,img):
        """draw the keypoints and the skeletons.
        :params keypoints: the shape should be equal to [17,2]
        :params img:
        """
        assert keypoints.shape == (self.NUM_KPTS,2)
        for i in range(len(self.SKELETON)):
            kpt_a, kpt_b = self.SKELETON[i][0], self.SKELETON[i][1]
            x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
            cv2.circle(img, (int(x_a), int(y_a)), 6, self.CocoColors[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 6, self.CocoColors[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), self.CocoColors[i], 2)

    def draw_bbox(self, box,img):
        """draw the detected bounding box on the image.
        :param img:
        """
        cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)


    def get_person_detection_boxes(self, model, img, threshold=0.5):
        pred = model(img)
        pred_classes = [self.COCO_INSTANCE_CATEGORY_NAMES[i]
                        for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                    for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        if not pred_score or max(pred_score)<threshold:
            return []
        # Get list of index with score greater than threshold
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_score = pred_score[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_classes = pred_classes[:pred_t+1]

        person_boxes = []
        for idx, box in enumerate(pred_boxes):
            if pred_classes[idx] == 'person':
                person_boxes.append(box)

        #TODO: get person with highest area
        #get person with highest score
        return person_boxes[:1]


    def get_pose_estimation_prediction(self, pose_model, image, center, scale):
        rotation = 0

        # pose estimation transformation
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        # pose estimation inference
        model_input = transform(model_input).unsqueeze(0)
        # switch to evaluate mode
        pose_model.eval()
        with torch.no_grad():
            # compute output heatmap
            output = pose_model(model_input)
            preds, _ = get_final_preds(
                cfg,
                output.clone().cpu().numpy(),
                np.asarray([center]),
                np.asarray([scale]))

            return preds, output.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)


    def box_to_center_scale(self, box, model_image_width, model_image_height):
        """convert a box to center,scale information required for pose transformation
        Parameters
        ----------
        box : list of tuple
            list of length 2 with two tuples of floats representing
            bottom left and top right corner of a box
        model_image_width : int
        model_image_height : int

        Returns
        -------
        (numpy array, numpy array)
            Two numpy arrays, coordinates for the center of the box and the scale of the box
        """
        center = np.zeros((2), dtype=np.float32)

        bottom_left_corner = box[0]
        top_right_corner = box[1]
        box_width = top_right_corner[0]-bottom_left_corner[0]
        box_height = top_right_corner[1]-bottom_left_corner[1]
        bottom_left_x = bottom_left_corner[0]
        bottom_left_y = bottom_left_corner[1]
        center[0] = bottom_left_x + box_width * 0.5
        center[1] = bottom_left_y + box_height * 0.5

        aspect_ratio = model_image_width * 1.0 / model_image_height
        pixel_std = 200

        if box_width > aspect_ratio * box_height:
            box_height = box_width * 1.0 / aspect_ratio
        elif box_width < aspect_ratio * box_height:
            box_width = box_height * aspect_ratio
        scale = np.array(
            [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale


    def get_box_out(self, box, ori_shape, des_shape):
        box_out = box.copy()
        ratio = (des_shape[0] / ori_shape[0], des_shape[1] / ori_shape[1])
        box_out[:, 0] = box[:, 0] * ratio[0]
        box_out[:, 1] = box[:, 1] * ratio[1]

        return box_out
