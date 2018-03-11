### CV related notes

#### Metrics
* Dice (Sørensen–Dice index)

Comparing the similarity of two samples.

* Sørensen–Dice coefficient: `DSC = 2TP / (2TP + FP + FN)`
* Dice coefficient : `DC = |A∩B| /|A|+|B∖A|`

* Jaccard (Intersection of union)

<img src="https://raw.githubusercontent.com/Novitoll/cvt-academy-2018/master/_cv/pics/IoU.png" width="50%" height="50%">
* `J = S / (2 - S), where S - Sørensen–Dice coefficient, J - Jaccard index`

#### Augmentation

* flip horizontally
* random crops/scales
* color jittering
* rotation

Test-time augmentation (TTA) - profit during test

#### Segmentation

* OpenCV:
  * [Watershed transformation](http://cmm.ensmp.fr/~beucher/wtshed.html)
* Neural nets:
  * [U-net](https://arxiv.org/pdf/1505.04597.pdf)

#### Refs:
* http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html

