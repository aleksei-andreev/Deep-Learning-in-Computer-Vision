#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import numpy as np

import cv2

import torch

from YOLOX_main.yolox.data.data_augment import ValTransform
from YOLOX_main.yolox.data.datasets import COCO_CLASSES
from YOLOX_main.yolox.exp import get_exp
from YOLOX_main.yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device="gpu" if torch.cuda.is_available() else "cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        # img_info = {"id": 0}
        # if isinstance(img, str):
        #     img_info["file_name"] = os.path.basename(img)
        #     img = cv2.imread(img)
        # else:
        #     img_info["file_name"] = None

        # height, width = img.shape[:2]
        # img_info["height"] = height
        # img_info["width"] = width
        # img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        # img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            # t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        # return outputs, img_info

    # def visual(self, output, img_info, cls_conf=0.35):
        # ratio = img_info["ratio"]
        # img = img_info["raw_img"]
        # if output is None:
        #    return img
        output = outputs[0].cpu().detach().numpy()
        output = output[output[:, 6] == 0]

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]

        # cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        h, _ = bboxes.shape
        output = np.c_[-np.ones((h, 2)), bboxes, scores, -np.ones((h, 3))]

        # vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return output
      
