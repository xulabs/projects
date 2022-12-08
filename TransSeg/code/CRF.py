from densecrf import  CRFs
import numpy as np
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
from metrics_3 import *

data=np.load('/Users/zixintang/Documents/single_cell_segmentation/img/hk2/unet/ori_dic.npy')
dir = '/Users/zixintang/Documents/single_cell_segmentation/img/hk2/unet'

def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [68, 1, 84],
        1: [32, 144, 140],  # cornsilk
        2: [253, 231, 36],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

"""
for i in range (data.shape[0]):
    B = data[i, : ,:].astype('uint8')
    plt.imshow(B)
    plt.show()
    #Image.fromarray(B).save(dir + "/" + str(i)+ ".tif")

    C = create_visual_anno(B)
    Image.fromarray(C).save(dir + "/" + str(i) + ".tif")
    #cv2.imwrite(dir+'/'+str(i)+'.png',C)
"""


labeldir='/Users/zixintang/Desktop/BIB_'
#A = data[0, : ,:].astype('uint8')
A = data[0,].astype('uint8')
gt_A= cv2.imread(labeldir+'/label000.png')
gt_A = cv2.cvtColor(gt_A, cv2.COLOR_BGR2GRAY)

#B = data[1, : ,:]
B = data[1, ]
gt_B= cv2.imread(labeldir+'/label001.png')
gt_B = cv2.cvtColor(gt_B, cv2.COLOR_BGR2GRAY)
#plt.imshow(B)
#plt.show()

#C = data[2, : ,:]
C = data[2, ]
gt_C= cv2.imread(labeldir+'/label002.png')
gt_C = cv2.cvtColor(gt_C, cv2.COLOR_BGR2GRAY)
#plt.imshow(gt_B)


#plt.imshow(C)
#plt.show()

#refine = dense_crf(np.array(img).astype(np.uint8), coarse)
pred=CRFs(C,gt_C)


pic0 = create_visual_anno(A)
Image.fromarray(pic0).save(dir + "/" + str(0) + ".tif")

pic1 = create_visual_anno(B)
Image.fromarray(pic1).save(dir + "/" + str(1_1) + ".tif")

pic2 = create_visual_anno(C)
Image.fromarray(pic2).save(dir + "/" + str(2_2) + ".tif")


pixel_acc = pixel_accuracy(pred, gt_C)
iou = mean_IU(pred, gt_C)
boundary_iou = boundary_iou(gt_C, pred)
quality = computeQualityMeasures(pred, gt_C)
