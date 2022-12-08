import numpy as np
import pydensecrf.densecrf as dcrf

import numpy as np
import pydensecrf.densecrf as dcrf
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

def CRFs(coarse,label):
    img=np.array(coarse)
    labels=np.array(label)
    n_labels=3

    use_2d = False
    if use_2d:
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=1, zero_unsure=None)
        #U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，功能只是位置而已
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(5, ), srgb=(5, ), rgbim=img,compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        U = unary_from_labels(img, n_labels, gt_prob=0.5, zero_unsure=False)

        #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        feats = create_pairwise_gaussian(sdims=(1.25,1.25), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=4.25,kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        feats = create_pairwise_bilateral(sdims=(25,25), schan=(0.1,),
                                          img=labels, chdim=-1)
        d.addPairwiseEnergy(feats, compat=3.25,
                            kernel=dcrf.CONST_KERNEL,
                            normalization=dcrf.NORMALIZE_BEFORE)

    Q = d.inference(5)
    MAP = np.argmax(Q, axis=0)

    refine=MAP.reshape(img.shape)
    plt.imshow(refine)
    plt.show()
    return refine



