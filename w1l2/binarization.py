import cv2
import cvt_utils.tools as tl

# take as white-black == 0 (no BGR)
img = cv2.imread("code/sudoku.png", 0)

assert len(img.shape) == 2  # only X, Y, without BGR channels

min_th = 127
max_th = 255

# basic
get_basic_th = lambda coef: list(cv2.threshold(img, min_th, max_th, coef))[1]

basic_binary_th_img1 = tl.concat_hor((img, get_basic_th(cv2.THRESH_BINARY)))
basic_binary_th_img2 = tl.concat_hor((img, get_basic_th(cv2.THRESH_BINARY_INV)))
basic_binary_th_img = tl.concat_ver((basic_binary_th_img1, basic_binary_th_img2))

cv2.imwrite("code/sudoku-basic-binary-th.png", basic_binary_th_img)
cv2.imshow("Sudoku", basic_binary_th_img)
cv2.waitKey(0)

# adaptive
block_size = 11  # 11 x 11
th_adaptive_mean = cv2.adaptiveThreshold(img, max_th, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2)
th_adaptive_gaus = cv2.adaptiveThreshold(img, max_th, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

row1 = tl.concat_hor((img, th_adaptive_mean))
row2 = tl.concat_hor((img, th_adaptive_gaus))

final_img = tl.concat_ver((row1, row2))

cv2.imwrite("code/sudoku-adaptive-th.png", final_img)
cv2.imshow("Sudoku-2", final_img)
cv2.waitKey(0)

# otsu

# Otsu's thresholding
_, th2 = cv2.threshold(img, 0, max_th, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 1)
_, th3 = cv2.threshold(blur, 0, max_th, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

row1 = tl.concat_hor((img, th2))
row2 = tl.concat_hor((img, th3))
final_img = tl.concat_ver((row1, row2))

cv2.imwrite("code/sudoku-otsu-th.png", final_img)
cv2.imshow("Sudoku-3", final_img)
cv2.waitKey(0)

ret_o, th_o = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print "Otsu threshold %d" % ret_o
