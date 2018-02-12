import cv2
import numpy as np

img = cv2.imread("small.png")

cv2.imshow("Window", img)
cv2.waitKey(0)

# practice 1.3
# show white 50x50 1-channel image
img_50_50_1 = np.zeros([50, 50, 1], dtype=np.uint8)
img_50_50_1.fill(255)

# practice 1.4
# fill 10x10 with black in the centre
img_50_50_1[20:30, 20:30] = 0
cv2.imshow("Window", img_50_50_1)
cv2.waitKey(5000)

car = cv2.imread("car.jpg")
gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)

cv2.imshow("Window-origin", gray)
cv2.waitKey(0)

# practice: 2D filter
kernel = np.ones((3, 3), dtype=np.float64) / 9
filtered = cv2.filter2D(gray, -1, kernel=kernel)
cv2.imshow("Window-conv-filter", filtered)
cv2.waitKey(0)

# practice: Gaussian filter
filtered = cv2.GaussianBlur(gray, (25, 25), 1)
cv2.imshow("Window-conv-filter", filtered)
cv2.waitKey(0)
