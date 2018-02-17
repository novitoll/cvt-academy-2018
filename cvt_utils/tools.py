import cv2
import numpy as np


def concat_hor(imgs, color=(0, 255, 0)):
    m = 0
    s = 0
    bs = 1
    for img in imgs:
        m = max(m, img.shape[0])
        s += img.shape[1] + 2 * bs

    image = np.zeros((m + 2 * bs, s, 3))

    x = 0
    for img in imgs:
        if len(img.shape) == 3:
            imgg = cv2.copyMakeBorder(img.copy(), bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=color)
            image[0:imgg.shape[0], x:x + imgg.shape[1], :] = imgg
        else:
            imgg = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), bs, bs, bs, bs, cv2.BORDER_CONSTANT,
                                      value=color)
            image[0:imgg.shape[0], x:x + imgg.shape[1], :] = imgg
        x += img.shape[1] + 2 * bs

    return np.asarray(image, dtype=np.uint8)


def concat_ver(imgs):
    m = 0
    s = 0
    bs = 1
    for img in imgs:
        m = max(m, img.shape[1])
        s += img.shape[0] + 2 * bs

    image = np.zeros((s, m + 2 * bs, 3))

    y = 0
    for img in imgs:
        if len(img.shape) == 3:
            imgg = cv2.copyMakeBorder(img.copy(), bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=(0, 255, 0))
            image[y:y + imgg.shape[0], 0:imgg.shape[1], :] = imgg
        else:
            imgg = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), bs, bs, bs, bs, cv2.BORDER_CONSTANT,
                                      value=(0, 255, 0))
            image[y:y + imgg.shape[0], 0:imgg.shape[1], :] = imgg
        y += img.shape[0] + 2 * bs

    return np.asarray(image, dtype=np.uint8)
