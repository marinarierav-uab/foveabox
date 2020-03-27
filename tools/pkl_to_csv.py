import argparse
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_csv(team, output, original):

    if output=="":
        output = team.split('.pkl')[0].split('pkl/')[-1]

    writelines_detection = dict()
    writelines_localization = dict()

    for vid in range(1, 19):
        writelines_detection[format(int(vid), '03d')] = []
        writelines_localization[format(int(vid), '03d')] = []

    with open(original) as json_file:
        data = json.load(json_file)
        images = data['images']

    with open(team, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    if not os.path.exists('results/Detection/' + output):
        os.makedirs('results/Detection/' + output)  # make new output folder
    if not os.path.exists('results/Localization/' + output):
        os.makedirs('results/Localization/' + output)  # make new output folder

    already_seen_vids = []
    i=0
    for image_id, image in enumerate(data):

        det = image[0]
        if len(det)>0:
            image_name = images[image_id]['file_name']
            nvid = image_name.split('-')[0]
            if nvid not in already_seen_vids:
                i=0
                already_seen_vids.append(nvid)
            det = det[0]
            x1 = int(det[0])
            y1 = int(det[1])
            x2 = int(det[2])
            y2 = int(det[3])

            cx = int(x1 + (x2-x1)/2)
            cy = int(y1 + (y2-y1)/2)

            conf = det[4]
            clase = 1

            # DETECT
            writelines_detection[nvid].append([i, 1, conf])
            # LOCAL
            writelines_localization[nvid].append([i, cx, cy, conf, clase])
            i+=1

    for vid in range(1, 19):
        nvid = format(int(vid), '03d')

        if writelines_detection[nvid]:
            with open('results/Detection/' + output + '/' + format(int(nvid), '02d') + '.csv', 'w') as file:
                for line in writelines_detection[nvid]:
                    file.write(str(line[0]) + ',' + str(line[1]) + ',' + str(line[2]) + '\n')  # separated by ,

        if writelines_localization[nvid]:
            with open('results/Localization/' + output + '/' + format(int(nvid), '02d') + '.csv', 'w') as file:
                for line in writelines_localization[nvid]:
                    file.write(
                        str(line[0]) + ',' + str(line[1]) + ',' + str(line[2]) + ',' + str(line[3]) + ',' + str(
                            line[4]) + '\n')  # separated by ,


    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, help='name of pkl with detected bbox annotations file')
    parser.add_argument('--original', type=str, help='name of json original annotation file')
    parser.add_argument("--out", "--output_folder", type=str, default=None)

    opt = parser.parse_args()
    print(opt)

    show_csv(opt.pkl, opt.out, opt.original)
