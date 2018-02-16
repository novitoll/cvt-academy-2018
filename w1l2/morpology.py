import cv2
import numpy as np
import code.tools as tl

img = cv2.imread("code/morphology1.png", 0)

_, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), dtype=np.uint8)

erosion = cv2.erode(th, kernel, iterations=1)
final_img = tl.concat_ver((img, erosion))
cv2.imwrite("code/erosion.png", final_img)
cv2.imshow("Erosion", final_img)
cv2.waitKey(0)

dilate = cv2.dilate(th, kernel, iterations=1)
final_img = tl.concat_ver((img, dilate))
cv2.imwrite("code/dilate.png", final_img)
cv2.imshow("Dilate", final_img)
cv2.waitKey(0)

# opening
img = cv2.imread("code/morphology2.png", 0)
_, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
final_img = tl.concat_ver((img, opening))
cv2.imwrite("code/opening.png", final_img)
cv2.imshow("opening", final_img)
cv2.waitKey(0)

# closing
img = cv2.imread("code/morphology3.png", 0)
_, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
final_img = tl.concat_ver((img, closing))
cv2.imwrite("code/closing.png", final_img)
cv2.imshow("opening", final_img)
cv2.waitKey(0)
