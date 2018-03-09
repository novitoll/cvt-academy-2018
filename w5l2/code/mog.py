import cv2
import numpy as np

width = 1280 * 2
height = 720 * 2
scaleMOG2 = 0.3

backsub_params = dict(history=300,
                      varThreshold=10,
                      detectShadows=False)

backsub = cv2.createBackgroundSubtractorMOG2(**backsub_params)
learning_rate = 0.01
area = width * height * scaleMOG2 * scaleMOG2
mask_thr = area * 0.005

videoPath = "./video.mp4"
cap = cv2.VideoCapture(videoPath)
kernel = np.ones((3, 3), dtype=np.uint8)

while True:
    ret, frame_data = cap.read()

    if ret:
        frame_data = cv2.resize(frame_data, (0, 0), fx=0.5, fy=0.5)
        motion_mask = backsub.apply(frame_data, None, learning_rate)

        # reduce mask noise with Morph Opening (erosion + dilation)
        opening = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        cv2.imshow("motion mask", motion_mask)
        cv2.imshow("motion mask with opening", opening)
        cv2.imshow("frame_data", frame_data)

        k = cv2.waitKey(15)
        if k == 27:
            break
    else:
        break

cap.release()
