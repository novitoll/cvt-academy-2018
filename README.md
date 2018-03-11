### Concepts

Since lectures are 30% - CV, 70% - ML (from basic up to DL according to syllabus), I'd like to focus more on CV-related info,
that's why I mix in this repo also some additional non-lectures related stuff in [_cv directory](https://github.com/Novitoll/cvt-academy-2018/tree/master/_cv).

* W1L1 - Introduction to CV
    * openCV
    * image read and work with numpy matrix
    * 2D conv filter
    * Gaussian filter

---

* W1L2 - Binarization
    * basic
    * adaptive
    * otsu

    ![Example](https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w1l2/code/sudoku-adaptive-th.png)

---

* W2L1 - Practice & Gradients
    * Find texts from image

    Example of the perfect input without noise with just contouring:

    <img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w2l1/output.png" width="50%" height="50%">

    Example of the input with the noise and how gradients of vertical projection + Sobel smoothing works

    <img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w2l1/output-gradient.png" width="50%" height="50%">

---

* W2L2 - Projection & HOG
    * Find peaks of projection
    * Sobel X, Y combined gradient magnitude + angel
    * HOG (Histogram of oriented gradients)

    <img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w2l2/code/hog.png" width="50%" height="50%">

---

* W3L1 - Line detection
    * Hough space
    * PClines
    * Canny algorithm (edge detector)

    <img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w3l1/code/hough_space_lines.png" width="50%" height="50%">
    <img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w3l1/code/canny0.png" width="50%" height="50%">

---

* W3L2 - Feedforward NN & RNN (Guest lecture)
    * Will come back to this later

---

* W4L1 - Perspective transformation
    * perspective transformation
    * affine transformation

---

* W5L1 - Optical flow
    * Taylor formula
    * Optical flow in general
    * Lucas-Kanade optical flow (1981)
        Usage in:
        * Structure from Motion
        * Video Compression
        * Video Stabilization
    * Eigenvalue and eigenvector

    #### TODO: re-factor car_counter app (HOG features + SVM)

    <img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w5l1/pics/optflow-LK.png" width="50%" height="50%">
---

* W5L2 - Background subtraction
    * Detect traffic light switching (single Gaussian)
    * MOG - Gaussian Mixture-based Background/Foreground Segmentation Algorithm

    MOG | Morph. opening + MOG
    
    <img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w5l2/pics/mog2.gif">
---

### NB! I already passed Andrew Ng's CS229 courses few years ago, so will briefly recap only math part

---

* W6L1 - ML intro: Linear regression & Gradient descent
    * Linear Regression (generalized)
    * BGD, SGD

    <img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/w6l1/pics/sgd2.png">

---

* W7L1 - Regularization (L1, L2)

---

* W8L1 - Logistic regression

---

* W8L2 - Naive Bayes

