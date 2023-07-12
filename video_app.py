# vim: expandtab:ts=4:sw=4
# from __future__ import division, print_function, absolute_import

import argparse
import os
from time import time
import cv2
import numpy as np
from deep_sort_master.application_util import preprocessing
from deep_sort_master.application_util import visualization
from deep_sort_master.deep_sort import nn_matching
from deep_sort_master.deep_sort.detection import Detection
from deep_sort_master.deep_sort.tracker import Tracker
import deep_sort_master.tools.generate_detections as gd
from inference import FeatureExtractorCustom as fec
import inference as infr


def gen_frame_feats(enc, rows, img):
  feats = enc(img, rows[:, 2:6].copy())
  dets_out = [np.r_[(row, feat)] for row, feat in zip(rows, feats)]
  return dets_out


# def gather_sequence_info(sequence_dir, detection_file):
def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    vid = os.path.join(sequence_dir, "vid.mp4")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    # detections = None
    # if detection_file is not None:
    #    detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    # if len(image_filenames) > 0:
        # min_frame_idx = min(image_filenames.keys())
        # max_frame_idx = max(image_filenames.keys())
    # else:
        # min_frame_idx = int(detections[:, 0].min())
        # max_frame_idx = int(detections[:, 0].max())
    min_frame_idx = min(image_filenames.keys())
    max_frame_idx = max(image_filenames.keys())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    # feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "update_ms": update_ms,
        "vid": vid,
        "rate": int(info_dict["frameRate"])
    }
    return seq_info


def create_detections(detections, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    # frame_indices = detection_mat[:, 0].astype(np.int)
    # mask = frame_indices == frame_idx

    detection_list = []
    # for row in detection_mat[mask]:
    for row in detections:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


# def run(sequence_dir, detection_file, output_file, min_confidence,
#         nms_max_overlap, min_detection_height, max_cosine_distance,
#         nn_budget, display):
def run(mode, enc, sequence_dir, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    # seq_info = gather_sequence_info(sequence_dir, detection_file)
    seq_info = gather_sequence_info(sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    COLAB = colab()
    det = infr.detector(mode, infr.det_choice, sequence_dir, min_confidence)
    global end_t
    end_t = time()
    if COLAB:
      global cap
      cap = cv2.VideoCapture(seq_info["vid"])
      if cap.isOpened() == False:
        print("Error")

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)
        if COLAB:
          ret, img = cap.read()
          if not ret:
            return
        else:
          img = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        rows = det.inference(img)
        dets = gen_frame_feats(enc, rows, img)

        # Load image and generate detections.
        # detections = create_detections(
        #     seq_info["detections"], frame_idx, min_detection_height)
        # detections = [d for d in detections if d.confidence >= min_confidence]
        dets = create_detections(dets, min_detection_height)

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        # Update visualization.
        if display:
            # image = cv2.imread(
            #     seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(img.copy())
            # vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            global end_t
            vis.draw_frames_per_sec(time() - end_t)
            end_t = time()

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)
    if COLAB:
      cap.release()

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def colab():
  return True


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")


def setup():
  det_msg = ("\nMake your choice among detection and segmentation models:\n" + "".join([f"{k}. {infr.det_choice[k]['descr']}.\n" for k in infr.det_choice]))
  det_mode = int(input(det_msg))
  if det_mode not in range(len(infr.det_choice)):
    print("Error. Not in range")
  ext_msg = ("\n Make your choice among ReID models:\n" + "".join([f"{k}. {infr.ext_choice[k]['descr']}.\n" for k in infr.ext_choice]))
  ext_mode = int(input(ext_msg))
  if ext_mode not in range(len(infr.ext_choice)):
    print("Error. Not in range")
  if ext_mode == 0:
    enc = gd.create_box_encoder(infr.ext_choice["ext_mode"]["path"], batch_size=32)
  elif ext_mode in range(1, len(infr.ext_choice)):
    enc = fec.create_feat_ext(infr.ext_choice[ext_mode]["name"], infr.ext_choice[ext_mode]["path"])
  else:
    print("Error")
  output_dir = f"result/{infr.det_choice[det_mode]['short']}_{infr.ext_choice[ext_mode]['short']}/data"
  return det_mode, enc, output_dir

# def parse_args():
def parse_args(args=None):
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    # parser.add_argument(
    #     "--detection_file", help="Path to custom detections.", default=None,
    #     required=True)
    # parser.add_argument(
    #     "--output_file", help="Path to the tracking output file. This file will"
    #     " contain the tracking results on completion.",
    #     default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.4, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    if not args:
      return parser.parse_args()
    return parser.parse_args(args)

def main(args):
  mode, enc, output_dir = setup()
  output_file = os.path.join(output_dir, f"{os.path.basename(args.sequence_dir)}.txt")
  run(mode, enc, args.sequence_dir, output_file, args.min_confidence, args.nms_max_overlap, args.min_detection_height, args.max_cosine_distance, args.nn_budget, args.display)

def run_vid(vid, display="True"):
  main(parse_args([f"--sequence_dir=./MOT/{vid}", f"--display={display}"]))

if __name__ == "__main__":
    # args = parse_args()
    # run(
    #     args.sequence_dir, args.detection_file, args.output_file,
    #     args.min_confidence, args.nms_max_overlap, args.min_detection_height,
    #     args.max_cosine_distance, args.nn_budget, args.display)
    main(parse_args())
