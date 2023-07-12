# vim: expandtab:ts=4:sw=4
import argparse
import os
import video_app


def parse_args(args=None):
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    # parser.add_argument(
    #     "--detection_dir", help="Path to detections.", default="detections",
    #     required=True)
    parser.add_argument(
        "--output_dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="base")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.35, type=float)
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
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Results",
        default=True, type=vid_app.bool_string)
    if not args:
      return parser.parse_args()
    # return parser.parse_args()
    return parser.parse_args(args)


def main(args):
    mode, enc, output_dir = vid_app.setup()
    # os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)
    for sequence in sequences:
        # print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(args.mot_dir, sequence)
        # detection_file = os.path.join(args.detection_dir, "%s.npy" % sequence)
        output_file = os.path.join(args.output_dir, "%s.txt" % sequence)
        if not os.path.isdir(sequence_dir):
          continue
        print("Running sequence %s" % sequence)
        vid_app.run(
            mode, enc, sequence_dir, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, args.display)


def run_fold(path="./MOT", display="True"):
  main(parse_args([f"--mot_dir={path}",f"--display={display}"]))


if __name__ == "__main__":
    main(parse_args())
