---
layout: post
title: Cartoonizing an image
subtitle: Using bilateral filter and some edge detection
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/banner.jpeg
tags: [computer vision, python]
author: Nhi Nguyen
comments: true
---

**Bilateral filter:** reduce the color palette or the numbers of colors that are used in the image. This mimics a cartoon drawing, wherein a cartoonist typically has few colors to work with

**Edge detection:** to generate bold silhouettes. Note that before doing edge detection, we need to blur the image to suppress small-scale noise and details

**Pseudocode:**

1. First, apply a bilateral filter to reduce the color palette of the image.
2. Then, convert the original color image into grayscale.
3. After that, apply a median blur to reduce image noise.
4. Use adaptive thresholding to detect and emphasize the edges in an edge mask.
5. Finally, combine the color image from step 1 with the edge mask from step 4.

```python
import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

def cartoonize(rgb_image, num_pyr_downs=2, num_bilaterals=7):
    # STEP 1 -- Apply a bilateral filter to reduce the color palette of
    # the image.
    downsampled_img = rgb_image
    # Downsample the image using multiple pyrDown calls
    for _ in range(num_pyr_downs):
        downsampled_img = cv.pyrDown(downsampled_img)
    # apply multiple bilateral filters
    for _ in range(num_bilaterals):
        filterd_small_img = cv.bilateralFilter(downsampled_img, 9, 9, 7)
    # upsample it to the original size
    filtered_normal_img = filterd_small_img
    for _ in range(num_pyr_downs):
        filtered_normal_img = cv.pyrUp(filtered_normal_img)

    # check the result of filtered_normal_img
    cv2_imshow(cv.resize(filtered_normal_img, (0, 0), fx=0.5, fy=0.5))

    # make sure resulting image has the same dims as original
    if filtered_normal_img.shape != rgb_image.shape:
        filtered_normal_img = cv.resize(
            filtered_normal_img, (rgb_image.shape[1], rgb_image.shape[0]))

    # STEP 2 -- Convert the original color image into grayscale.
    img_gray = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
    # STEP 3 -- Apply median blur to reduce image noise.
    img_blur = cv.medianBlur(img_gray, 7)

    # STEP 4 -- Use adaptive thresholding to detect and emphasize the edges
    # in an edge mask.
    gray_edges = cv.adaptiveThreshold(img_blur, 255,
                                       cv.ADAPTIVE_THRESH_MEAN_C,
                                       cv.THRESH_BINARY, 9, 2)
    # STEP 5 -- Combine the color image from step 1 with the edge mask
    # from step 4.
    rgb_edges = cv.cvtColor(gray_edges, cv.COLOR_GRAY2RGB)
    rgb_edges_resized = cv.resize(rgb_edges, (filtered_normal_img.shape[1], filtered_normal_img.shape[0]))
    return cv.bitwise_and(filtered_normal_img, rgb_edges_resized)

# change this to your image's relative path
path = '/content/small-dog-owners-1.jpeg'
rgb_img = cv.imread(path, 1)
cv2_imshow(cv.resize(rgb_img, (0, 0), fx=0.5, fy=0.5))
img = cartoonize(rgb_img, num_pyr_downs=2, num_bilaterals=7)
cv2_imshow(cv.resize(img, (0, 0), fx=0.5, fy=0.5))
```
