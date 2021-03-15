#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

from skimage import feature
from skimage import color
from skimage import io


def read_color_codes(file_path):
    colors_list = list()
    with open(file_path) as file_with_colors:
        _ = file_with_colors.readline()
        for row in file_with_colors:
            row = row.split(",")
            colors = [int(item) for item in row[2:5]]
            colors_list.append(colors)

    return colors_list


in_dir_images = "D:\\Tasks\\python\\CodedTargets\\Data\\LABEL"

def main():
    list_of_colors = read_color_codes("classes.csv")
    image_names_list = os.listdir(in_dir_images)

    image_index = 35  # 18
    target_index = 47
    img = io.imread(os.path.join(in_dir_images, image_names_list[image_index]))
    print("IMG --> ", image_names_list[image_index])
    print("CODED TARGET --> ", target_index+1)

    if img.shape[2] == 4:
        img = color.rgba2rgb(img)
    elif img.shape[2] < 3 or img.shape[2] > 5:
        print("Our img non RGB or RGBA. We don't know, what to do...")
        return 1

    mask_coded_target = np.all(img == list_of_colors[target_index], axis=-1)

    img_edges = feature.canny(mask_coded_target)

    # Проходим по строкам
    line_1_points = list()
    line_1_xs = list()
    line_1_ys = list()
    for row in range(img_edges.shape[0]):
        row_edges = np.nonzero(img_edges[row])
        row_edges = row_edges[0]
        if len(row_edges):
            avr_in_row = (row_edges[0] + row_edges[-1])/2
            if row_edges[0] == row_edges[-1]:
                print("bad row")
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
            if col_edges[0] == col_edges[-1]:
                print("bad col")
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

    print("xs = ", xs, "ys = ", ys)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True)
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Image")

    ax2.imshow(img_edges)
    # Отображаем точки первой линии
    for point in line_1_points:
        plt.scatter(point[0], point[1], s=1, facecolor='red')
    # Отображаем точки второй линии
    for point in line_2_points:
        plt.scatter(point[0], point[1], s=1, facecolor='red')

    if xs is not None and ys is not None:
        plt.scatter(xs, ys, s=2, facecolor='green')

    ax2.axis("off")
    ax2.set_title("Edges")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()