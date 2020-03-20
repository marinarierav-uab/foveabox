import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_csv(team, output, original):

    if output=="":
        output = team.split('.bbox.json')[0].split('json/')[-1]

    writelines_detection = dict()
    writelines_localization = dict()

    for vid in range(1, 19):
        writelines_detection[format(int(vid), '03d')] = []
        writelines_localization[format(int(vid), '03d')] = []

    with open(original) as json_file:
        data = json.load(json_file)
        images = data['images']

    with open(team) as json_file:
        data = json.load(json_file)

    if not os.path.exists('results/Detection/' + output):
        os.makedirs('results/Detection/' + output)  # make new output folder
    if not os.path.exists('results/Localization/' + output):
        os.makedirs('results/Localization/' + output)  # make new output folder

    already_seen_vids = []

    acc_id = 0
    for i, image in enumerate(data):

        image_id = image['image_id']
        image_name = images[image_id-1]['file_name']

        nvid = image_name.split('-')[0]

        if nvid not in already_seen_vids:
            acc_id = image_id-1
            already_seen_vids.append(nvid)

        det = image['bbox']
        x1 = int(det[0])
        y1 = int(det[1])
        w = int(det[2])
        h = int(det[3])

        cx = int(x1 + w/2)
        cy = int(y1 + h/2)

        conf = image['score']
        clase = image['category_id']

        """
        --------->>> IMAGE_ID <<<------------
        """

        # DETECT
        writelines_detection[nvid].append([image_id-acc_id-1, 1, conf])
        # LOCAL
        writelines_localization[nvid].append([image_id-acc_id-1, cx, cy, conf, clase])

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
    parser.add_argument('--json', type=str, help='name of json detected bbox annotations file')
    parser.add_argument('--original', type=str, help='name of json original annotation file')
    parser.add_argument("--out", "--output_folder", type=str, default=None)

    opt = parser.parse_args()
    print(opt)

    show_csv(opt.json, opt.out, opt.original)
