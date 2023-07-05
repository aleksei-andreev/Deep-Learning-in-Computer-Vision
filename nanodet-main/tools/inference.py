# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import time

import cv2
import torch

from nanodet-main.nanodet.data.transform import Pipeline
from nanodet-main.nanodet.model.arch import build_model
from nanodet-main.nanodet.util import load_model_weight
from nanodet-main.nanodet.util import cfg
from nanodet-main.nanodet.data.collate import naive_collate
from nanodet-main.nanodet.data.batch_process import stack_batch_img


class Predictor(object):
    # def __init__(self, cfg, model_path, logger, device="cuda:0"):
    def __init__(self, cfg, model_path, confidence_minimum):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        # load_model_weight(model, ckpt, logger)
        load_model_weight(model, ckpt)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.confidence_minimum = confidence_minimum

    def inference(self, img):
        # img_info = {}
        img_info = {"id":0}
        img_info["file_name"] = None
        # if isinstance(img, str):
        #     img_info["file_name"] = os.path.basename(img)
        #     img = cv2.imread(img)
        # else:
        #     img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        # meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        meta = naiva_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        results = [results[0][lbl] for lbl in results[0] if cfg.class_names[lbl] == "person"][0]
        results = np.array([pers for pers in results if pers[4] > self.confidence_minimum])
        results[:, 2] -= results[:, 0]
        results[:, 3] -= results[:, 1]
        h, _ = results.shape
        results = np.c_[-np.ones((h, 2)), results, -np.ones((h, 3))]
        # return meta, results
        return results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        self.model.head.show_result(
            meta["raw_img"], dets, class_names, score_thres=score_thres, show=True
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
