import os

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label


def calculate_average_classif_results(results_dict: dict, thresholds, output_file):
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
        acc = (tp + tn) / (tp + fp + fn + tn)
        mean_conf /= len(results_dict.items())


        # switched values fn <-> tn, as requested by J.B.
        row = [threshold, tp, fp, tn, fn, acc, mean_conf]
        avg.loc[-1] = row

        avg.index += 1
        avg.sort_index()
        avg.reset_index(inplace=True, drop=True)

    print(avg)

    avg.to_csv(output_file)


def calculate_average_results(results_dict: dict, thresholds, output_file):
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
        acc = (tp + tn) / (tp + fp + fn + tn)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (fp + tn)
        mean_rt = srt / drt
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


def process_video_for_detection(file, has_confidence, thresh, vid_folder):
    video_len = len(os.listdir(vid_folder)) + 1
    video_gt = np.zeros((video_len, 1))
    video_pred = np.zeros((video_len, 1))

    first_polyp = -1
    first_detected_polyp = -1

    tp, fp, fn, tn = 0, 0, 0, 0
    for frame in sorted(os.listdir(vid_folder)):

        polyp_n = int(frame.split("_")[0].split("-")[1])
        im_frame = Image.open(os.path.join(vid_folder, frame))
        is_polyp = np.asarray(im_frame).sum() > 0
        video_gt[polyp_n] = 1.1 if is_polyp else 0

        if is_polyp and first_polyp == -1:
            first_polyp = polyp_n

        frame_output = file.loc[file[0] == polyp_n]
        if has_confidence:
            frame_output = frame_output.loc[frame_output[2] >= thresh]

        if frame_output.empty:
            if is_polyp:
                fn += 1
            else:
                tn += 1
        else:
            pred_out = frame_output[1].tolist()[0]
            if pred_out:
                if is_polyp:
                    tp += 1
                    if first_detected_polyp == -1:
                        first_detected_polyp = polyp_n
                else:
                    fp += 1
            else:
                if is_polyp:
                    fn += 1
                else:
                    tn += 1

            video_pred[polyp_n] = 0.9

    rt = first_detected_polyp - first_polyp if first_detected_polyp != -1 else -1

    # switched values fn <-> tn, as requested by J.B.
    return [tp, fp, tn, fn, rt], video_gt, video_pred


def process_video_for_localization(file, has_confidence, threshold, vid_folder):
    tp, fp, tn, fn = 0, 0, 0, 0
    histo_tp, histo_fp, histo_tn, histo_fn = 0, 0, 0, 0

    # HISTOLOGIAS DE VIDEOS DE TEST (eventually should be loaded from file)
    no_adenomas = [2, 16]

    first_polyp = -1
    first_detected_polyp = -1
    i = 0

    vid_n = int(vid_folder.split('/')[-1])
    histologia_real = 0 if (vid_n in no_adenomas) else 1

    for frame in sorted(os.listdir(vid_folder)):
        i+=1

        #print("frame", i)
        polyp_n = int(frame.split("_")[0].split("-")[1])
        im_frame = Image.open(os.path.join(vid_folder, frame))
        im_frame_np = np.asarray(im_frame, dtype=int)
        is_polyp = im_frame_np.sum() > 0

        # 8-connected
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        labeled_frame, max_polyp = label(im_frame, structure=kernel)

        if is_polyp and first_polyp == -1:
            first_polyp = polyp_n
        frame_output = file.loc[file[0] == polyp_n]
        if has_confidence:
            frame_output = frame_output.loc[frame_output[3] >= threshold]

        #if i>35:
        #  break
        #print(frame)

        if frame_output.empty:
            if is_polyp:
                fn += max_polyp
            else:
                tn += 1
        else:
            already_detected = []

            for detection_row in frame_output.iterrows():
                detection = detection_row[1]
                frame_pred = True
                centroid_x = int(detection[1])
                centroid_y = int(detection[2])

                #print("Detection:",centroid_x, centroid_y)
                #print(im_frame_np[centroid_y-5:centroid_y+5, centroid_x-5:centroid_x+5])

                if frame_pred:
                    if is_polyp:
                        if im_frame_np[centroid_y, centroid_x] != 0:
                            if labeled_frame[centroid_y, centroid_x] not in already_detected:
                                tp += 1
                                already_detected += [labeled_frame[centroid_y, centroid_x]]

                                if first_detected_polyp == -1:
                                    first_detected_polyp = polyp_n

                                # HISTOLOGIAS:
                                histologia_red = int(detection[4])

                                if (histologia_red == 0) and (histologia_real == 0):
                                    histo_tn += 1
                                elif (histologia_red == 0) and (histologia_real == 1):
                                    histo_fn += 1
                                elif (histologia_red == 1) and (histologia_real == 0):
                                    histo_fp += 1
                                elif (histologia_red == 1) and (histologia_real == 1):
                                    histo_tp += 1
                        else:
                            fp += 1
                    else:
                        fp += 1
                else:
                    if not is_polyp:
                        tn += 1

            detected_in_frame = len(set(already_detected))
            fn += (max_polyp - detected_in_frame)

    rt = first_detected_polyp - first_polyp if first_detected_polyp != -1 else -1

    positives = histo_fp + histo_tp
    negatives = histo_fn + histo_tn
    pred_histo = 1 if positives >= negatives else 0
    if(positives+negatives) == 0:
        conf = 0
        acc = 0
    else:
        conf = positives/(positives+negatives) if positives >= negatives else negatives/(positives+negatives)
        acc = (histo_tp + histo_tn) / (positives + negatives)

    # switched values fn <-> tn, as requested by J.B.
    return [tp, fp, tn, fn, rt], [histo_tp, histo_fp, histo_tn, histo_fn, acc, histologia_real, pred_histo, conf]


def generate_results_per_video(videos, confidences, thresholds, gt):
    detect_dict = {}
    local_dict = {}
    classif_dict = {}
    for threshold in thresholds:
        # TODO change plots
        res_detection, _, _ = process_video_for_detection(videos[0], confidences[0], threshold, gt)
        res_localization, res_classif = process_video_for_localization(videos[1], confidences[1], threshold, gt)
        print("  -thr",threshold, "done...")

        detect_dict[threshold] = res_detection
        local_dict[threshold] = res_localization
        classif_dict[threshold] = res_classif
    return detect_dict, local_dict, classif_dict


def do_giana_eval(folder_detection, folder_localization, folder_gt, root_folder_output, team, thr=0, series=False):

    # DEBUGGING !!!!!
    nvids = 18  # should be 18

    folder_output_detection = os.path.join(root_folder_output, "Detection/"+team)
    folder_output_localization = os.path.join(root_folder_output, "Localization/"+team)
    folder_output_classif = os.path.join(root_folder_output, "Classif/"+team)
    average_detection_output_file = os.path.join(folder_output_detection, "average.csv")
    average_localization_output_file = os.path.join(folder_output_localization, "average.csv")
    average_classif_output_file = os.path.join(folder_output_classif, "average.csv")

    if series:
        thresholds = [x / 10 for x in range(1, 10)]
    elif thr!=0:
        thresholds = [thr]
    else:
        thresholds = [0]

    if not os.path.exists(folder_output_detection):
        os.makedirs(folder_output_detection)
    if not os.path.exists(folder_output_localization):
        os.makedirs(folder_output_localization)
    if not os.path.exists(folder_output_classif):
        os.makedirs(folder_output_classif)

    files_detection = sorted(os.listdir(folder_detection))[0:nvids]
    files_localization = sorted(os.listdir(folder_localization))[0:nvids]

    results_detection = {}
    results_localization = {}
    results_classif = {}

    # for each video:
    for detection, localization in zip(files_detection, files_localization):

        detection_csv = os.path.join(folder_detection, detection)
        detection_df = pd.read_csv(detection_csv, header=None)
        detection_confidence = detection_df.shape[1] > 2

        localization_csv = os.path.join(folder_localization, localization)
        localization_df = pd.read_csv(localization_csv, header=None)
        localization_confidence = localization_df.shape[1] > 3

        # both named the same
        vid_name = localization_csv.split("/")[-1].split(".")[0]
        gt_vid_folder = os.path.join(folder_gt, str(int(vid_name)))
        print('Processing video', vid_name, "...")
        res_detection, res_localization, res_classif = generate_results_per_video((detection_df, localization_df),
                                                                     (detection_confidence, localization_confidence),
                                                                     thresholds, gt_vid_folder)

        pd.DataFrame.from_dict(res_detection, columns=["TP", "FP", "TN", "FN", "RT"], orient='index').to_csv(
            os.path.join(folder_output_detection, "d{}.csv".format(vid_name)))
        results_detection[vid_name] = res_detection

        pd.DataFrame.from_dict(res_localization, columns=["TP", "FP", "TN", "FN", "RT"], orient='index').to_csv(
            os.path.join(folder_output_localization, "l{}.csv".format(vid_name)))
        results_localization[vid_name] = res_localization

        pd.DataFrame.from_dict(res_classif, columns=["TP", "FP", "TN", "FN", "Acc", "Histo-real", "Histo-pred", "Conf"], orient='index').to_csv(
            os.path.join(folder_output_classif, "l{}.csv".format(vid_name)))
        results_classif[vid_name] = res_classif

    calculate_average_results(results_detection, thresholds, average_detection_output_file)
    calculate_average_results(results_localization, thresholds, average_localization_output_file)
    calculate_average_classif_results(results_classif, thresholds, average_classif_output_file)

    #nvids = len(results_detection)

    global_detection_list = np.zeros([nvids*len(thresholds), 7])
    global_localization_list = np.zeros([nvids*len(thresholds), 7])
    global_classif_list = np.zeros([nvids*len(thresholds), 10])

    i=0;
    j=0;
    k=0;
    for vidname in sorted(results_detection.keys()):

        vid = int(vidname)

        for key, vals in results_detection[vidname].items():

            global_detection_list[i, :] = ([vid] + [key] + vals)
            i += 1

        #print(np.around(global_detection_list, decimals=4))

        for key, vals in results_localization[vidname].items():
            global_localization_list[j, :] = ([vid] + [key] + vals)
            j += 1

        for key, vals in results_classif[vidname].items():
            global_classif_list[k, :] = ([vid] + [key] + vals)
            k += 1


    #print("")

    columns = ["Video", "Thr", "TP", "FP", "TN", "FN", "RT"]
    detframe = pd.DataFrame(global_detection_list, columns=columns)
    locframe = pd.DataFrame(global_localization_list, columns=columns)
    classifframe = pd.DataFrame(global_classif_list, columns=["Video", "Thr", "TP", "FP", "TN", "FN", "Acc", "Histo-real", "Histo-pred", "Conf"])

    print("")

    detframe.to_csv(os.path.join(folder_output_detection, "detection.csv"))
    locframe.to_csv(os.path.join(folder_output_localization, "localization.csv"))
    classifframe.to_csv(os.path.join(folder_output_classif, "classification.csv"))


if __name__ == '__main__':
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument("--res", "--results_root", type=str, default='results')
    ap.add_argument("--thr", "--threshold", type=float, default=0)
    ap.add_argument("--team", "--team", type=str, required=True)
    ap.add_argument("--out", "--output_folder", type=str, default=None)
    ap.add_argument("--list", action='store_true', help="threshold series")

    params = ap.parse_args()
    team = params.team.split('.bbox.json')[0].split('json/')[-1]

    folder_detection = os.path.join(params.res, "Detection")
    folder_detection = os.path.join(folder_detection, team)
    folder_localization = os.path.join(params.res, "Localization")
    folder_localization = os.path.join(folder_localization, team)
    output_folder = params.out

    if output_folder is None:
        output_folder = os.path.join(params.res, "results_giana")
    folder_gt = "/home/marina/Downloads/DATASETS/cvcvideoclinicdbtest/masks/"

    do_giana_eval(folder_detection, folder_localization, folder_gt, output_folder, team, params.thr, params.list)
