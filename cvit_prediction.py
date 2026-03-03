import sys, os
import argparse
import torch
import json
from time import perf_counter
from datetime import datetime

from helpers.loader import load_data
from model.pred_func import *


# ---------------- BASIC VIDEO MODE ----------------
def vids(cvit_weight, root_dir="sample_prediction_data",
         dataset=None, num_frames=15, net=None, fp16=False):

    result = set_result()
    model = load_cvit(cvit_weight, net, fp16)

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        if is_video(curr_vid):
            result, _, _, _ = predict(
                curr_vid, model, fp16,
                result, num_frames, net, "uncategorized"
            )

    return result


# ---------------- FACEFORENSICS ----------------
def faceforensics(cvit_weight,
                  root_dir="FaceForensics\\data",
                  dataset=None,
                  num_frames=15,
                  net=None,
                  fp16=False):

    result = set_result()
    model = load_cvit(cvit_weight, net, fp16)
    return result


# ---------------- TIMIT ----------------
def timit(cvit_weight,
          root_dir="DeepfakeTIMIT",
          dataset=None,
          num_frames=15,
          net=None,
          fp16=False):

    result = set_result()
    model = load_cvit(cvit_weight, net, fp16)
    return result


# ---------------- DFDC ----------------
def dfdc(cvit_weight,
         root_dir="deepfake-detection-challenge\\train_sample_videos",
         dataset=None,
         num_frames=15,
         net=None,
         fp16=False):

    result = set_result()
    model = load_cvit(cvit_weight, net, fp16)
    return result


# ---------------- CELEB ----------------
def celeb(cvit_weight,
          root_dir="Celeb-DF-v2",
          dataset=None,
          num_frames=15,
          net=None,
          fp16=False):

    result = set_result()
    model = load_cvit(cvit_weight, net, fp16)
    return result


# ---------------- PREDICT CORE ----------------
def predict(
    vid,
    model,
    fp16,
    result,
    num_frames,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
):

    df = df_face(vid, num_frames)

    if fp16:
        df = df.half()

    y, y_val = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )

    result = store_result(
        result,
        os.path.basename(vid),
        y,
        y_val,
        klass,
        correct_label,
        compression,
    )

    return result, accuracy, count, [y, y_val]


# ---------------- STREAMLIT FUNCTION ----------------
def predict_video(uploaded_file):

    # save uploaded file temporarily
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    cvit_weight = "cvit2_deepfake_detection_ep_50.pth"
    net = "cvit2"
    fp16 = False
    num_frames = 15

    model = load_cvit(cvit_weight, net, fp16)

    result, _, _, pred = predict(
        temp_path,
        model,
        fp16,
        set_result(),
        num_frames,
        net,
        "streamlit"
    )

    return real_or_fake(pred[0])


# ---------------- CLI MODE ----------------
def main():
    print("CLI mode not used in Streamlit")


if __name__ == "__main__":
    main()
