import numpy as np
import cv2
import SimpleITK as sitk
def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    sum_n_ii = 0
    sum_t_i = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i
    print("pixel accuracy = %f" %pixel_accuracy_)
    return pixel_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''
    check_size(eval_segm, gt_segm)
    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    IU = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        IU[i] = n_ii / (t_i + n_ij - n_ii)
    mean_IU_ = np.sum(IU) / n_cl_gt
    print("miou = %f" % mean_IU_)
    return mean_IU_

def mean_IU_boundary(gt_segm, eval_segm):
    gt = convert_binary(gt_segm)
    pred = convert_binary(eval_segm)
    check_size(pred, gt)
    cl, n_cl = union_classes(pred, gt)
    _, n_cl_gt = extract_classes(gt)
    eval_mask, gt_mask = extract_both_masks(pred, gt, cl, n_cl)
    IU = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        IU[i] = n_ii / (t_i + n_ij - n_ii)
    # boundary_iou = np.sum(IU) / n_cl_gt

    if len(IU) == 2:
        print("boundary_iou_new = %f" % IU[1])
        return IU[1]
    else:
        print("boundary_iou_new = 0.0")
        return 0.0

def convert_binary(mask):
    h, w = mask.shape
    mask_b = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            if mask[row, col] != 2:
                mask_b[row, col] = 0
            else:
                mask_b[row, col] = 2
    return mask_b

def convert_binary_inner(mask):
    h, w = mask.shape
    mask_b = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            if mask[row, col] != 0:
                mask_b[row, col] = 256
            else:
                mask_b[row, col] = 0
    return mask_b

def get_disk_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))

def compute_boundary_acc(gt, seg):

    gt = convert_binary(gt)
    seg = convert_binary(seg)

    gt = gt > 128
    seg = seg > 128

    gt = gt.astype(np.uint8)
    seg = seg.astype(np.uint8)

    h, w = gt.shape

    min_radius = 1
    max_radius = (w+h)/300
    num_steps = 5

    seg_acc = [None] * num_steps

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)

        kernel = get_disk_kernel(curr_radius)
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0


        gt_in_bound = gt[boundary_region]
        seg_in_bound = seg[boundary_region]


        num_edge_pixels = (boundary_region).sum()
        num_seg_gd_pix = ((gt_in_bound) * (seg_in_bound) + (1-gt_in_bound) * (1-seg_in_bound)).sum()



        seg_acc[i] = num_seg_gd_pix / num_edge_pixels

    acc = sum(seg_acc)/num_steps
    print('boundary_acc=%f' % acc)
    return acc
# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt,  dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_binary = convert_binary(gt)
    dt_binary = convert_binary(dt)
    #gt_boundary = mask_to_boundary(gt_binary, dilation_ratio)
    #dt_boundary = mask_to_boundary(dt_binary, dilation_ratio)
    intersection = ((gt_binary * dt_binary) > 0).sum()
    union = ((gt_binary + dt_binary) > 0).sum()
    boundary_iou = intersection / union
    print('boundary_iou=%f' % boundary_iou)
    return boundary_iou


def computeQualityMeasures(pred, gt):
    quality = dict()
    lP=convert_binary(pred)
    lT=convert_binary(gt)
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["dice"] = dicecomputer.GetDiceCoefficient()
    print('avgHausdorff=%f,Hausdorff=%f,dice=%f'%(quality["avgHausdorff"],quality["Hausdorff"],quality["dice"]))
    return quality



'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)
    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm) #cl=[0 1 2]
    n_cl = len(cl) #n_cl=3
    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)
    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))
    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c
    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise
    return height, width

def check_size(eval_segm, gt_segm):

    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)
    #print(h_e, w_e)
    #print(h_g, w_g)
    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")
'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
