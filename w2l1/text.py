import glob
import cv2
import numpy as np

import cvt_utils.tools as tl

# exts = ['jpg', 'JPG']
exts = ['JPG']
jpgs_paths = [fs for ext in exts for fs in glob.glob(r'code/Data/*.%s' % ext)]
mask_blur_size = (21, 1)
mask_morph_kernel = np.ones((1, 15), dtype=np.uint8)
scale = 0.45

RED = (0, 0, 255)


def read_inverted_wb_img():
    """load 1 inverted to gray image in memory"""
    for jpg_path in jpgs_paths:
        origin_img = cv2.imread(jpg_path)
        gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        yield origin_img, gray


def get_height_contour(c):
    x, y, w, h = cv2.boundingRect(c)
    return h


def main():
    for origin, gray in read_inverted_wb_img():

        # blur
        blur = cv2.blur(gray, mask_blur_size)

        # Otsu thresholds
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morph opening (erosion + dilate)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, mask_morph_kernel, iterations=1)

        # find contours
        img, contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # invert back from gray to BGR to see RED contour
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # get mean of contour height
        h_mean = np.mean(map(get_height_contour, contours))

        # draw contour as rectangle
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > h_mean:
                text_regions.append((x, y, x + w, y + h))  # bottom-left-coordinate, top-left, bottom-right, top-right
                cv2.rectangle(vis, (x, y), (x + w, y + h), RED, 2)

        # crop origin image by given coordinates of rectangle contours and concat them vertically in one image
        text_img_lines = [origin[tr[1]:tr[3], tr[0]:tr[2]] for tr in text_regions]
        text_img = tl.concat_ver(text_img_lines)

        row1 = tl.concat_hor((origin, gray, blur))
        row2 = tl.concat_hor((opening, vis, text_img))

        final_img = tl.concat_ver((row1, row2))
        cv2.imshow("window", cv2.resize(final_img, (0, 0), fx=scale, fy=scale))
        cv2.imwrite("output.png", final_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
