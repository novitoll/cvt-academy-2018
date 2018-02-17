import glob
import cv2
import numpy as np

import cvt_utils.tools as tl

exts = ['JPG']
jpgs_paths = [fs for ext in exts for fs in glob.glob(r'code/Data/*.%s' % ext)]
scale = 0.45

RED = (0, 0, 255)


def read_inverted_wb_img():
    """load 1 inverted to gray image in memory"""
    for jpg_path in jpgs_paths:
        origin_img = cv2.imread(jpg_path)
        gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        yield origin_img, gray


def main():
    for origin, gray in read_inverted_wb_img():
        vertical_projection = np.sum(gray, axis=1) / 255

        vertical_projected_image = tl.getDrawProjectionVer(origin, vertical_projection)

        cv2.imshow("Window", tl.concat_hor((origin, vertical_projected_image)))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
