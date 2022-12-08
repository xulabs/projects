import torch
import numpy as np
import random
import PIL.Image as Image
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
from torchvision import transforms
from torch import Tensor
from functools import partial
from operator import itemgetter
from utils.losses import class2one_hot,one_hot2dist
import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
import utils.helpers as helpers

palette = [[0], [1], [2]]
num_classes = 3
D = Union[Image.Image, np.ndarray, Tensor]

def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0)  # Then pop the element to go back to img shape
    ])

def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        gt_transform(resolution, K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

def transform(img):
    img=(img-np.amin(img))*1.0/(np.amax(img)-np.amin(img))#img*1.0 transform array to double
    img=img*1.0/np.median(img)
    img_h=img.shape[0]
    img_w=img.shape[1]
    return np.reshape(img,(1,img_h,img_w))

class Dataset_folder(torch.utils.data.Dataset):
    def __init__(self, dic, phase, labels):
        self.dic = dic
        self.phase = phase
        self.labels = labels.astype(int)
        # self.labels = sorted(os.listdir(labels_dir))
        self.disttransform = dist_map_transform([1, 1], 3)

    def __len__(self):
        return len(self.dic)

    def __getitem__(self, i):
        dic = self.dic[i]
        phase = self.phase[i]
        label = self.labels[i]
        # label = np.array(io.imread(self.labels_dir + self.labels[i]))

        seed=np.random.randint(0,2**32) # make a seed with numpy generator 
        torch.manual_seed(seed)
        
        dic_trans = transform(dic).astype(np.float32)
        phase_trans = transform(phase).astype(np.float32)

        label_h = label.shape[0]
        label_w = label.shape[1]
        label_map = np.zeros([3, label_h, label_w])    
        for r in range(label_h):
            for c in range(label_w):
                label_map[label[r][c], r, c] = 1
        
        # #get mask 1 is inside
        # mask = np.zeros([img_h, img_w])    
        # for r in range(label_h):
        #     for c in range(label_w):
        #         if label[r][c] == 1:
        #             mask[r, c] = 1
        #         else:
        #             mask[r, c] = 0

        # #get boundary (2)
        # boundary = np.zeros([img_h, img_w])    
        # for r in range(label_h):
        #     for c in range(label_w):
        #         if label[r][c] == 2:
        #             boundary[r, c] = 1
        #         else:
        #             boundary[r, c] = 0

        # #get boundary map
        # boundary_map = np.zeros([2, img_h, img_w])    
        # for r in range(label_h):
        #     for c in range(label_w):
        #         if label[r][c] == 2:
        #             boundary_map[1, r, c] = 1
        #         else:
        #             boundary_map[0, r, c] = 1

        # apply this seed to target/label tranfsorms  
        label = torch.tensor(label,dtype=torch.float)
        label_map = torch.tensor(label_map,dtype=torch.float)
        dist_map= self.disttransform(label)
        # boundary_map = torch.tensor(boundary_map,dtype=torch.float)
        # boundary = boundary.astype(np.float32)
        # boundary = torch.tensor(boundary,dtype=torch.float).unsqueeze(dim=0)
        # mask = mask.astype(np.float32)
        # mask = torch.tensor(mask,dtype=torch.float).unsqueeze(dim=0)
        

        return dic_trans, phase_trans, label_map, dist_map, label

class UnetDataset(torch.utils.data.Dataset):
    def __init__(self, dic, labels):
        self.dic = dic
        self.labels = labels.astype(int)

    def __len__(self):
        return len(self.dic)

    def __getitem__(self, i):
        dic = self.dic[i]
        label = self.labels[i]
        # label = np.array(io.imread(self.labels_dir + self.labels[i]))

        seed=np.random.randint(0,2**32) # make a seed with numpy generator 
        torch.manual_seed(seed)
        
        dic_trans = transform(dic).astype(np.float32)
       
        label_h = label.shape[0]
        label_w = label.shape[1]
        label_map = np.zeros([3, label_h, label_w])    
        for r in range(label_h):
            for c in range(label_w):
                label_map[label[r][c], r, c] = 1
        
        
        label = torch.tensor(label,dtype=torch.float)
        label_map = torch.tensor(label_map,dtype=torch.float)

        return dic_trans, label_map,label


palette = [[0], [1], [2]]
num_classes = 3


class IsbiCellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'F:\isbi'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\train\images\*')
        self.mask_paths = glob(self.root + r'\train\label\*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        # self.test_img_paths = glob(self.root + r'\test\test_images\*')
        # self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class NMuMgDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = '/shared/home/v_zixin_tang/dataset/NmuMg/'
        #self.root = r'F:\single_cell_segmentation-master\NMuMg_phase_contrast'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform =transform
        self.target_transform =target_transform

    def getDataPath(self):

        self.img_paths = glob(self.root + 'train/Img/*')
        self.mask_paths = glob(self.root + 'train/BIB/*')
        self.test_img_paths = glob(self.root + 'test/enh_data/*')
        self.test_mask_paths = glob(self.root + 'test/BIB/*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(sorted(self.img_paths), sorted(self.mask_paths), test_size=0.2, random_state=42)
        # self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return sorted(self.train_img_paths),sorted(self.train_mask_paths)
        if self.state == 'val':
            return sorted(self.val_img_paths),sorted(self.val_mask_paths)
        if self.state == 'test':
            return sorted(self.test_img_paths),sorted(self.test_mask_paths)

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        '''
        #pic = Image.open(pic_path)
        #pic = np.array(pic)
        #pic = np.expand_dims(pic, axis=2)
        pic= cv2.imread(pic_path)
        pic = pic.astype('float32')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        '''
        pic = Image.open(pic_path)
        pic = Image.fromarray(np.uint8(pic))
        pic = pic.convert('L')
        pic = np.array(pic, dtype='float32')
        pic = np.expand_dims(pic, axis=2)
        mask = Image.open(mask_path)
        mask = Image.fromarray(np.uint8(mask))
        mask = np.asarray(mask)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        pic = pic.astype('float32')  # / 255
        mask = mask.astype('float32')  # / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class T47DDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'/shared/home/v_zixin_tang/dataset/T47D_fluorescence'
        #self.root = r'F:\dataset\T47D_fluorescence'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
    def getDataPath(self):
        self.img_paths = glob(self.root + r'/train/data/*')
        self.mask_paths = glob(self.root + r'/train/label/*')
        self.test_img_paths = glob(self.root + r'/test/data/*')
        self.test_mask_paths = glob(self.root + r'/test/label/*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split((self.img_paths), (self.mask_paths), test_size=0.2, random_state=42)
        # self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return (self.train_img_paths), (self.train_mask_paths)
        if self.state == 'val':
            return (self.val_img_paths), (self.val_mask_paths)
        if self.state == 'test':
            return (self.test_img_paths), (self.test_mask_paths)

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        '''
        #pic = Image.open(pic_path)
        #pic = np.array(pic)
        #pic = np.expand_dims(pic, axis=2)
        pic = cv2.imread(pic_path)
        pic = pic.astype('float32')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        '''
        pic = Image.open(pic_path)
        pic = Image.fromarray(np.uint8(pic))
        pic = pic.convert('L')
        pic = np.array(pic, dtype='float32')
        pic = np.expand_dims(pic, axis=2)
        mask = Image.open(mask_path)
        mask = Image.fromarray(np.uint8(mask))
        mask = np.asarray(mask)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        pic = pic.astype('float32')  # / 255
        mask = mask.astype('float32')  # / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path
    def __len__(self):
        return len(self.pics)

class HK2_DICDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = '/shared/home/v_zixin_tang/dataset/HK2_DIC/'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
    def getDataPath(self):
        self.img_paths = glob(self.root + 'train/transfer/*')
        self.mask_paths = glob(self.root + 'train/enh_BIB/*')
        self.test_img_paths = glob(self.root + 'test/transfer/*')
        self.test_mask_paths = glob(self.root + 'test/BIB/*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(sorted(self.img_paths), sorted(self.mask_paths), test_size=0.2, random_state=42)
        # self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return sorted(self.train_img_paths), sorted(self.train_mask_paths)
        if self.state == 'val':
            return sorted(self.val_img_paths), sorted(self.val_mask_paths)
        if self.state == 'test':
            return sorted(self.test_img_paths), sorted(self.test_mask_paths)

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        """
        pic = Image.open(pic_path)
        pic = Image.fromarray(np.uint8(pic))
        pic = pic.convert('L')
        pic = np.array(pic, dtype='float32')
        pic = np.expand_dims(pic, axis=2)
        mask = Image.open(mask_path)
        mask = Image.fromarray(np.uint8(mask))
        mask = np.asarray(mask)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, palette)
        pic = pic.astype('float32')  # / 255
        mask = mask.astype('float32')  # / 255
        """
        img = Image.open(pic_path)
        img = np.array(img)
        img = img.astype('float32')
        #img = np.array(img, dtype='float32')
        mask = Image.open(mask_path)
        mask = Image.fromarray(np.uint8(mask))
        mask = np.array(mask)
        seed=np.random.randint(0,2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))
        img = img * 1.0 / np.median(img)
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_x = np.reshape(img, (1,img_h, img_w))
        mask_h = mask.shape[0]
        mask_w = mask.shape[1]
        label_map=np.zeros([mask_h,mask_w,3])
        for r in range(mask_h):
            for c in range(mask_w):
                label_map[r, c, mask[r][c]] = 1
        random.seed(seed)
        torch.manual_seed(seed)
        img_y=np.transpose(label_map,(2,0,1))
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        '''
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        '''
        return img_x, img_y,pic_path,mask_path
    def __len__(self):
        return len(self.pics)


