import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label
import operator


def calc_avg_classif(results_dict: dict, thresholds, output_file):
    avg = pd.DataFrame(columns=["Thr", "TP", "FP", "TN", "FN", "Accuracy", "Conf"])
    for threshold in thresholds:
        # TP, FP, FN, TN, RT
        results = [0, 0, 0, 0]
        mean_conf = 0
        for vid, res_dict in results_dict.items():  # for each video
            results = [res + new for res, new in zip(results, res_dict[threshold][:4])]

            mean_conf += res_dict[threshold][-1]

        # switched values fn <-> tn, as requested by J.B.
        tp, fp, tn, fn = results[0], results[1], results[2], results[3]

        try:
            acc = (tp + tn) / (tp + fp + fn + tn)
        except:
            acc = -1
        try:
            mean_conf /= len(results_dict.items())
        except:
            mean_conf = 0

        # switched values fn <-> tn, as requested by J.B.
        row = [threshold, tp, fp, tn, fn, acc, mean_conf]
        avg.loc[-1] = row

        avg.index += 1
        avg.sort_index()
        avg.reset_index(inplace=True, drop=True)

    print(avg)

    avg.to_csv(output_file)


def calc_avg(results_dict: dict, thresholds, output_file):
    avg = pd.DataFrame(columns=["Thr", "TP", "FP", "TN", "FN", 'Accuracy', "Precision", "Recall", "Specificity", "F1", "F2", "Mean RT"])
    for threshold in thresholds:
        # TP, FP, FN, TN, RT
        results = [0, 0, 0, 0]
        sums = [0, 0, 0, 0]
        srt = 0
        drt = 1e-7
        for vid, res_dict in results_dict.items():  # for each video
            results = [res + new for res, new in zip(results, res_dict[threshold][:-1])]
            #sums = [val + new for val, new in zip(sums, results)]
            #print(res_dict[threshold][:-1])
            #print(sum)
            #print(results)
            srt = srt + res_dict[threshold][-1] if res_dict[threshold][-1] != -1 else srt
            drt = drt + 1 if res_dict[threshold][-1] != -1 else drt

        # switched values fn <-> tn, as requested by J.B.
        tp, fp, tn, fn = results[0], results[1], results[2], results[3]

        try:
            acc = (tp + tn) / (tp + fp + fn + tn)
        except:
            acc = -1
        try:
            pre = tp / (tp + fp)
        except:
            pre = -1
        try:
            rec = tp / (tp + fn)
        except:
            rec = -1
        try:
            spec = tn / (fp + tn)
        except:
            spec = -1
        try:
            mean_rt = srt / drt
        except:
            mean_rt = -1

        f1 = (2*pre*rec) / (pre+rec)
        f2 = (5*pre*rec) / ( (4*pre) + rec)

        # switched values fn <-> tn, as requested by J.B.
        row = [threshold, tp, fp, tn, fn, acc, pre, rec, spec, f1, f2, mean_rt]
        avg.loc[-1] = row

        avg.index += 1
        avg.sort_index()
        avg.reset_index(inplace=True, drop=True)

    print(avg)

    avg.to_csv(output_file)


def save_detection_plot(output_folder, threshold, vid_folder, video_gt, video_pred):
    title = "Video: {} - threshold: {}".format(vid_folder.split("/")[-1], threshold)
    plt.title(title)
    plt.plot(video_gt, color='blue')
    plt.plot(video_pred, color='gold')
    plt.savefig(os.path.join(output_folder, "detect_plot-{}-{}.png".format(vid_folder.split("/")[-1], threshold)))
    plt.clf()


def process_videos(videos, thrs, vid_folder):
    video_len = len(os.listdir(vid_folder)) + 1
    video_gt = np.zeros((video_len, 1))
    video_pred = np.zeros((video_len, 1))
    
    det_df = videos[0]
    loc_df = videos[1]
    
    # HISTOLOGIAS DE VIDEOS DE TEST (eventually should be loaded from file)
    no_adenomas = [2, 16]

    vid_id = int(vid_folder.split('/')[-1])
    histologia_real = 0 if (vid_id in no_adenomas) else 1

    # 8-connected
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    first_polyp_detection = dict()
    first_polyp_localization = dict()
    results_detection = dict()
    results_localization = dict()
    results_classif = dict()

    for thr in thrs:
        first_polyp_detection[thr] = -1
        first_polyp_localization[thr] = -1
        results_detection[thr]=[0,0,0,0,0]
        results_localization[thr]=[0,0,0,0,0]
        results_classif[thr]=[0,0,0,0,0]

    # For RT calc
    first_polyp_apparition = -1

    for frame in sorted(os.listdir(vid_folder)):

        frame_id = int(frame.split("_")[0].split("-")[1])
        im_frame = Image.open(os.path.join(vid_folder, frame))
        im_frame_np = np.asarray(im_frame, dtype=int)
        is_polyp = im_frame_np.sum() > 0
        video_gt[frame_id] = 1.1 if is_polyp else 0

        labeled_frame, max_polyp = label(im_frame, structure=kernel)

        if is_polyp and first_polyp_apparition == -1:
            first_polyp_apparition = frame_id

        for thr in thrs:
            det_tp, det_fp, det_tn, det_fn = 0, 0, 0, 0
            loc_tp, loc_fp, loc_tn, loc_fn = 0, 0, 0, 0
            histo_tp, histo_fp, histo_tn, histo_fn = 0, 0, 0, 0
            
            det = loc_df.loc[loc_df[3] >= thr]
            loc = loc_df.loc[loc_df[3] >= thr]
            frame_output_det = det.loc[det[0] == frame_id]
            frame_output_loc = loc.loc[loc[0] == frame_id]

            if frame_output_det.empty:
                if is_polyp:
                    det_fn += 1
                else:
                    det_tn += 1
            else:
                pred_out = frame_output_det[1].tolist()[0]
                video_pred[frame_id] = 0.9
                if pred_out:
                    if is_polyp:
                        det_tp += 1
                        if first_polyp_detection[thr] == -1:
                            first_polyp_detection[thr] = frame_id
                    else:
                        det_fp += 1
                else:
                    if is_polyp:
                        det_fn += 1
                    else:
                        det_tn += 1


            if frame_output_loc.empty: # Si no hay deteccion
                if is_polyp: # Pero hay polipo -> FN
                    loc_fn += max_polyp
                else:
                    loc_tn += 1
            else:
                already_located = []

                for loc_row in frame_output_loc.iterrows():
                    frame_pred = True
                    loc_row = loc_row[1]
                    centroid_x = int(loc_row[1])
                    centroid_y = int(loc_row[2])

                    if frame_pred:
                        if is_polyp:
                            if im_frame_np[centroid_y, centroid_x] != 0:
                                if labeled_frame[centroid_y, centroid_x] not in already_located:
                                    loc_tp += 1
                                    already_located += [labeled_frame[centroid_y, centroid_x]]

                                    if first_polyp_localization[thr] == -1:
                                        first_polyp_localization[thr] = frame_id

                                    # HISTOLOGIAS:
                                    histologia_red = int(loc_row[4])

                                    if (histologia_red == 0) and (histologia_real == 0):
                                        histo_tn += 1
                                    elif (histologia_red == 0) and (histologia_real == 1):
                                        histo_fn += 1
                                    elif (histologia_red == 1) and (histologia_real == 0):
                                        histo_fp += 1
                                    elif (histologia_red == 1) and (histologia_real == 1):
                                        histo_tp += 1
                            else:
                                loc_fp += 1
                        else:
                            loc_fp += 1
                    else:
                        if not is_polyp:
                            loc_tn += 1

                detected_in_frame = len(set(already_located))
                loc_fn += (max_polyp - detected_in_frame)

            results_detection[thr][:4] = list(map(operator.add, results_detection[thr][:4], [loc_tp, loc_fp, loc_tn, loc_fn]))
            results_localization[thr][:4] = list(map(operator.add, results_localization[thr][:4], [loc_tp, loc_fp, loc_tn, loc_fn]))
            results_classif[thr][:4] = list(map(operator.add, results_classif[thr][:4], [histo_tp, histo_fp, histo_tn, histo_fn]))

    for thr in thrs:
        det_rt = first_polyp_detection[thr] - first_polyp_apparition if first_polyp_detection[thr] != -1 else -1
        loc_rt = first_polyp_localization[thr] - first_polyp_apparition if first_polyp_localization[thr] != -1 else -1

        results_detection[thr][4] = det_rt
        results_localization[thr][4] = loc_rt

        positives = results_classif[thr][0] + results_classif[thr][1] # tp + fp
        negatives = results_classif[thr][2] + results_classif[thr][3] # tn + fn

        pred_histo = 1 if positives >= negatives else 0

        if(positives+negatives) == 0:
            conf = 0
            acc = 0
        else:
            conf = positives/(positives+negatives) if positives >= negatives else negatives/(positives+negatives)
            acc = (histo_tp + histo_tn) / (positives + negatives)

        results_classif[thr][4:]=[acc, histologia_real, pred_histo, conf]

    return results_detection, results_localization, results_classif, video_gt, video_pred


def do_giana_eval(path_to_input_det,
                  path_to_input_loc,
                  path_to_gt,
                  path_to_output_det,
                  path_to_output_loc,
                  path_to_output_classif,
                  thr=0, series=False):

    # DEBUGGING !!!!!
    nvids = 18  # should be 18

    if series:
        thrs = [x / 10 for x in range(1, 10)]
    elif thr!=0:
        thrs = [thr]
    else:
        thrs = [0]

    files_detection = sorted(os.listdir(path_to_input_det))[0:nvids]
    files_localization = sorted(os.listdir(path_to_input_loc))[0:nvids]

    results_detection = {}
    results_localization = {}
    results_classif = {}

    # for each video:
    for detection, localization in zip(files_detection, files_localization):

        detection_csv = os.path.join(path_to_input_det, detection)
        detection_df = pd.read_csv(detection_csv, header=None)

        localization_csv = os.path.join(path_to_input_loc, localization)
        localization_df = pd.read_csv(localization_csv, header=None)

        # both named the same
        vid_name = localization_csv.split("/")[-1].split(".")[0]
        gt_vid_folder = os.path.join(path_to_gt, str(int(vid_name)))
        print('Processing video', vid_name, "...")

        #res_detection, res_localization, res_classif = generate_results_per_video((detection_df, localization_df), thrs, gt_vid_folder)
        res_detection, res_localization, res_classif, video_gt, video_pred = process_videos((detection_df, localization_df), thrs, gt_vid_folder)

        pd.DataFrame.from_dict(res_detection, columns=["TP", "FP", "TN", "FN", "RT"], orient='index').to_csv(
            os.path.join(path_to_output_det, "d{}.csv".format(vid_name)))
        results_detection[vid_name] = res_detection

        pd.DataFrame.from_dict(res_localization, columns=["TP", "FP", "TN", "FN", "RT"], orient='index').to_csv(
            os.path.join(path_to_output_loc, "l{}.csv".format(vid_name)))
        results_localization[vid_name] = res_localization

        pd.DataFrame.from_dict(res_classif, columns=["TP", "FP", "TN", "FN", "Acc", "Histo-real", "Histo-pred", "Conf"], orient='index').to_csv(
            os.path.join(path_to_output_classif, "l{}.csv".format(vid_name)))
        results_classif[vid_name] = res_classif

    # Save to csv
    avg_det_out_file = os.path.join(path_to_output_det, "average.csv")
    avg_loc_out_file = os.path.join(path_to_output_loc, "average.csv")
    avg_classif_out_file = os.path.join(path_to_output_classif, "average.csv")
    
    calc_avg(results_detection, thrs, avg_det_out_file)
    calc_avg(results_localization, thrs, avg_loc_out_file)
    calc_avg_classif(results_classif, thrs, avg_classif_out_file)


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("--res", "--results_root", type=str, default='results')
    ap.add_argument("--thr", "--threshold", type=float, default=0)
    ap.add_argument("--team", "--team", type=str, required=True)
    ap.add_argument("--out", "--output_folder", type=str, default=None)
    ap.add_argument("--list", action='store_true', help="threshold series")

    args = ap.parse_args()
    return args


def main():
    params = parse_args()
    
    team = params.team.split('.bbox.json')[0].split('json/')[-1]

    out = params.out
    if out is None:
        out = os.path.join(params.res, "results_giana")

    path_to_input_det = os.path.join(params.res, "Detection", team)
    path_to_input_loc = os.path.join(params.res, "Localization", team)

    path_to_gt = "/home/marina/Downloads/DATASETS/cvcvideoclinicdbtest/masks/"

    path_to_output_det = os.path.join(out, "Detection", team)
    path_to_output_loc = os.path.join(out, "Localization", team)
    path_to_output_classif = os.path.join(out, "Classif", team)
    
    if not os.path.exists(path_to_output_det):
        os.makedirs(path_to_output_det)
    if not os.path.exists(path_to_output_loc):
        os.makedirs(path_to_output_loc)
    if not os.path.exists(path_to_output_classif):
        os.makedirs(path_to_output_classif)

    do_giana_eval(path_to_input_det,
                  path_to_input_loc,
                  path_to_gt,
                  path_to_output_det,
                  path_to_output_loc,
                  path_to_output_classif,
                  params.thr,
                  params.list)


if __name__ == '__main__':
    main()