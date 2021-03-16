#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import shutil

from skimage import feature
from skimage import color
from skimage import io


def clear_directory(dir_path: str):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    else:
        print("Directory '%s' already exists" % dir_path)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e_rr:
                print("Failed to delete ", file_path, "Reason: ", e_rr)
                exit(-1)
        print("Directory '%s' was cleaned" % dir_path)


def read_color_codes(file_path):
    colors_list = list()
    with open(file_path) as file_with_colors:
        _ = file_with_colors.readline()
        for row in file_with_colors:
            row = row.split(",")
            colors = [int(item) for item in row[2:5]]
            colors_list.append(colors)

    return colors_list


def zhou_operator(target_mask):
    img_edges = feature.canny(target_mask)

    # Проходим по строкам
    line_1_points = list()
    line_1_xs = list()
    line_1_ys = list()
    epsilon = 2
    for row in range(img_edges.shape[0]):
        row_edges = np.nonzero(img_edges[row])
        row_edges = row_edges[0]
        if len(row_edges):
            avr_in_row = (row_edges[0] + row_edges[-1])/2
            if row <= 0 + epsilon or row >= img_edges.shape[0] - epsilon:
                print("BAD ROW FOR CODED TARGET")
                line_1_points.clear()
                line_1_xs.clear()
                line_1_ys.clear()
                break
            line_1_points.append([avr_in_row, row])
            line_1_xs.append(avr_in_row)
            line_1_ys.append(row)

    # Проходим по столбцам
    line_2_points = list()
    line_2_xs = list()
    line_2_ys = list()
    for col in range(img_edges.shape[1]):
        col_edges = np.nonzero(img_edges[:, col])
        col_edges = col_edges[0]
        if len(col_edges):
            avr_in_col = (col_edges[0] + col_edges[-1])/2
            if col <= 0 + epsilon or col >= img_edges.shape[1] - epsilon:
                print("BAD COL FOR CODED TARGET")
                line_2_points.clear()
                line_2_xs.clear()
                line_2_ys.clear()
                break
            line_2_points.append([col, avr_in_col])
            line_2_xs.append(col)
            line_2_ys.append(avr_in_col)

    # находим пересечение двух прямых
    xs = None
    ys = None
    if len(line_1_xs)*len(line_1_ys) and len(line_2_xs)*len(line_2_ys):
        m1, b1 = np.polyfit(line_1_xs, line_1_ys, 1)
        m2, b2 = np.polyfit(line_2_xs, line_2_ys, 1)

        A = np.array([[m1, -1], [m2, -1]])
        B = np.array([[-b1], [-b2]])

        xs, ys = np.linalg.solve(A, B)
        return [float(xs), float(ys)]
    return [xs, ys]


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--input_dir', default="C:\\Data\\zhou_operator\\LABEL")
    argument_parser.add_argument('-o', '--output_dir', default="C:\\Data\\zhou_operator\\Centers")
    argument_parser.add_argument('-c', '--classes_file', default="classes.csv")

    args = argument_parser.parse_args()

    list_of_colors = read_color_codes(args.classes_file)
    image_names_list = os.listdir(args.input_dir)

    clear_directory(args.output_dir)

    for image_name in image_names_list:
        img = io.imread(os.path.join(args.input_dir, image_name))

        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        elif img.shape[2] < 3 or img.shape[2] > 5:
            print("IMG %s is non RGB or RGBA. We don't know, what to do..." % (os.path.join(args.input_dir, image_name)))
            continue

        out_name, _ = image_name.split(".")
        out_name += ".csv"

        with open(os.path.join(args.output_dir, out_name), "w") as result_file:
            for target_index in range(len(list_of_colors)):
                mask_coded_target = np.all(img == list_of_colors[target_index], axis=-1)
                print("CODED TARGET --> ", target_index+1)
                p = zhou_operator(mask_coded_target)
                print("====")
                result_file.write("%d,%s,%s\n" % (target_index+1, p[0], p[1]))

        print("FILE %s with centers is READY" % os.path.join(args.output_dir, out_name))


if __name__ == "__main__":
    main()
