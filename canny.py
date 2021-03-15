#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from skimage import feature
from skimage import color
from skimage import io

in_image = "D:\\Tasks\\python\\CodedTargets\\Data\\ellipse.png"


def main():
    img = io.imread(in_image)
    if img.shape[2] == 4:
        img_gray = color.rgb2gray(color.rgba2rgb(img))
    elif img.shape[2] == 3:
        img_gray = color.rgb2gray(img)
    else:
        print("Our img non RGB or RGBA. We don't know, what to do...")
        return 1
    edges = feature.canny(img_gray)
    print(edges.shape)
    print(edges)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True)
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Image")

    ax2.imshow(edges)
    ax2.axis("off")
    ax2.set_title("Edges")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
