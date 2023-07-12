import os
import torch
import numpy as np
from torchreid.utils import FeatureExtractor
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from nanodet_main.nanodet.util import cfg, load_config
from nanodet_main.tools.inference import Predictor
from YOLOX_main.yolox.exp.build import get_exp_by_file
from YOLOX_main.tools.inference import Predictor

det_choice = {
  0: {"model":"None", "config":"None", "descr":"MOT benchmarks", "short":"MOT"},
  1: {"model":"model_weights/nanodet-plus-m_416.pth", "config":"nanodet-main/config/nanodet-plus-m_416.yml", "descr":"NanoDet-Plus-M", "short":"nanodet_plus_m"},
  2: {"model":"model_weights/nanodet-plus-m-1.5x_416.pth", "config":"nanodet-main/config/nanodet-plus-m-1.5x_416.yml", "descr":"NanoDet-Plus-M-1.5x", "short":"nanodet_plus_m_1.5x"},
  3: {"model":"model_weights/yolox_tiny.pth", "config":"YOLOX-main/exps/default/yolox_tiny.py", "descr":"YOLOX-Tiny", "short":"yolox_tiny"},
  4: {"model":"model_weights/yolox_l.pth", "config":"YOLOX-main/exps/default/yolox_l.py", "descr":"YOLOX-Large", "short":"yolox_l"},
  5: {"model":"model_weights/mask_rcnn.pk1", "config":"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", "descr":"Mask R-CNN (R50-FPN)", "short":"mask_rcnn"},
  6: {"model":"model_weights/cascade_mask_rcnn.pk1", "config":"Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml", "descr":"Cascade Mask R-CNN (R50-FPN)", "short":"cascade_mask_rcnn"}
}

ext_choice = {
  0: {"path":"model_weights/mars-small128.pb", "name":"None", "descr":"Default DeepSORT Feature Extractor", "short":"default"},
  1: {"path":"model_weights/shufflenet.pth.tar", "name":"shufflenet", "descr":"ShuffleNet", "short":"shufflenet"},
  2: {"path":"model_weights/mlfn.pth.tar", "name":"mlfn", "descr":"Multilayer Feedforward Neural Network", "short":"mlfn"},
  3: {"path":"model_weights/mobilenetv2.pth", "name":"mobilenetv2_x1_0", "descr":"MobileNetV2", "short":"mobilenetv2"},
  4: {"path":"model_weights/osnet.pth", "name":"osnet_x1_0", "descr":"OSNet", "short":"osnet"},
  5: {"path":"model_weights/osnet_ain.pth", "name":"osnet_ain_x1_0", "descr":"OSNet-AIN", "short":"osnet_ain"},
  6: {"path":"model_weights/osnet_ibn.pth", "name":"osnet_ibn_x1_0", "descr":"OSNet-IBN", "short":"osnet_ibn"},
}

def detector(mode, choice, dir, thresh):
  config = choice[mode]["cfg"]
  wghts = choice[mode]["model"]
  if not mode:
    class Predictor():
      def __init__(self, dir, thresh):
        self.dets = np.loadtxt(os.path.join(dir, "det/det.txt"), delimiter=",")
        self.frames = self.dets[:, 0].astype(int)
        self.frame = 1
        self.thresh = thresh

      def inference(self, img):
        mask = self.frames == self.frame
        res = self.dets[mask]
        res = res[res[:, 4] > self.thresh]
        self.frame += 1
        return res

    det = Predictor(dir, thresh)
    return det

  elif mode in (1, 2):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    load_config(cfg, config)
    det = Predictor(cfg, wghts, thresh)
    return det

  elif mode in (3, 4):
    exp = get_exp_by_file(config)
    exp.test_size = (640, 640)
    exp.test_conf = thresh
    model = exp.get_model()
    ckpt = torch.load(wghts, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if torch.cuda.is_available():
      model.cuda()
    model.eval()
    det = Predictor(model, exp)
    return det

  elif mode in (5, 6):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = wghts

    class Det_Predictor():
      def __init__(self, cfg, thresh):
        self.predictor = DefaultPredictor(cfg)
        self.thresh = thresh

      def inference(self, img):
        output = self.predictor(img)
        output["instances"] = output["instances"][output["instances"].pred_classes == 0]
        output["instances"] = output["instances"][output["instances"].scores > self.thresh]
        bbxs = output["instances"].pred_boxes.tensor.cpu().numpy()
        bbxs[:, 2] -= bbxs[:, 0]
        bbxs[:, 3] -= bbxs[:, 1]
        scrs = output["instances"].scores.cpu().numpy()
        h, _ = bbxs.shape
        output = np.c_[-np.ones((h, 2)), bbxs, scrs, -np.ones((h, 3))]
        return output

    det = Det_Predictor(cfg, thresh)
    return det

class FeatureExtractorCustom():
  def __init__(self, name, path):
    self.extractor = FeatureExtractor(model_name=name, model_path=path, device="cuda" if torch.cuda.is_available() else "cpu")

  def __call__(self, img, bbxs):
    bbxs = np.array(bbxs)
    bbxs[:, 2] += bbxs[:, 0]
    bbxs[:, 3] += bbxs[:, 1]
    bbxs = bbxs.astype(int)
    bbxs[:, :2] = np.maximum(0, bbxs[:, :2])
    bbxs[:, 2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbxs[:, 2:])
    img_ptchs = [img[bbx[1]:bbx[3], bbx[0]:bbx[2], :] for bbx in bbxs]
    return self.extractor(img_ptchs).cpu().numpy()

def create_feat_ext(name, path):
  ext = FeatureExtractorCustom(name, path)

  def encoder(img, bbxs):
    return extractor(img, bbxs)
  return encoder
