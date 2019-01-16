# Subtomogram classification network model compression
This project developed model compression techniques to compress deep subtomogram classification networks into compact models achieving 2-3 times speedup without loss in accuracy.

Please refer to our paper for more details:

Guo, Jialiang, Bo Zhou, Xiangrui Zeng, Zachary Freyberg, and Min Xu. "Model compression for faster structural separation of macromolecules captured by Cellular Electron Cryo-Tomography." In International Conference Image Analysis and Recognition, pp. 144-152. Springer, Cham, 2018. 

https://link.springer.com/chapter/10.1007/978-3-319-93000-8_17

https://arxiv.org/abs/1801.10597

## Knowledge distillation flowchart
<img src="https://user-images.githubusercontent.com/31047726/51223193-e7c20700-190e-11e9-9eb2-eaaf58aa5078.png" width="700">

## Architectures of the teacher network and student networks
<img src="https://user-images.githubusercontent.com/31047726/51223192-e7c20700-190e-11e9-8384-ed7db0d3973c.png" width="700">


## Key prerequisites
* [EMAN2](http://blake.bcm.edu/emanwiki/EMAN2/Install)
* [keras](https://keras.io/#installation)
* [tensorflow-gpu](https://www.tensorflow.org/install/)
* numpy
```
pip install numpy
```

* scipy
```
pip install scipy
```
* pypng
```
pip install pypng
```



## Installation 
```
git clone https://github.com/xulabs/projects.git
```


## Example dataset
http://xulab-gpu0.pc.cc.cmu.edu/d38122b6-cca3-11e7-9e38-a32bc9f19922/autoencoder/example.tgz

Please download this dataset and extract it into your working directory. 

File subvolumes_example_1.pickle and tomogram.rec are realistically simulated CECT data files as described in the supplementary material of our paper.

File subvolumes_example_2.pickle is some pose normalized representative small subvolumes from the COS-7 tomogram 1 as descrbed in our paper.

## Protocal
### Step 1. Prepare input dataset
There are four inputs， CECT small subvolumes can be extracted from a tomogram using particle picking methods such as Difference of Gaussian.

1. A python pickle data file of CECT small subvolumes, this data file should be prepared as follows:

  + d is the small subvolume data file.

  + d is a dictionary consists 'v_siz' and 'vs'.

  + d['v_siz'] is an numpy.ndarray specifying the shape of the small subvolume. For example, d['v_siz'] = array([32,32,32]).

  + d['vs'] is a dictionary with keys of uuids specifying each small subvolume.

  + d['vs'][an example uuid] is a dictionary consists 'center', 'id', and 'v'.

  + d['vs'][an example uuid]['center'] is the center of the small subvolume in the tomogram. For example, d['vs'][an example  uuid]['center'] = [110,407,200].

  + d['vs'][an example uuid]['id'] is the specific uuid.

  + d['vs'][an example uuid]['v'] are voxel values of the small subvolume, which is an numpy.ndarray of shape d['v_siz']. 

2. A tomogram file in .rec format, which is only required when performing pose normalization.

3. Whether the optional pose normalization step should be applied. Input should be True or False.

4. The number of clusters. This should be an positive integer such as 100.

### Step 2. Autoencoder3D training
This step trains the Autoencoder3D network. We recommend users to use default network training parameters including learning rate, number of epochs and etc. However, they can be modified in autoencoder.py.

First, change to the working directory where the input data file is stored. 
```
cd autoencoder
```
Second, train the Autoencoder3D network with the four inputs in correct order.

```
python autoencoder.py example/subvolumes_example_1.pickle example/tomogram.rec True 100
```

When not using pose normalization, the tomogram file input can be None.

```
python autoencoder.py example/subvolumes_example_2.pickle None False 4
```

### Step 3. Manual selection of small subvolume clusters
Autoencoder3D training step will have two output folders.

1. A 'model' directory for the trained autoencoder models

2. A 'clus-center' directory for the resulting clusters. There should be two pickle files in 'clus-center'. 'kmeans.pickle' stores the uuids for each cluster. 'ccents.pickle' stores the decoded cluster centers. The 'fig' folder under 'clus-center' directory contains the 2D slices of decoded cluster center. User can use the figures as guide for manual selection.

Manual selection clues are provided in the folder 'fig' under 'clus-center'. Each picture is a 2D slices presentation of a decoded small subvolume cluster center. The picture name such as '035--47.png' refers to cluster 35 which consists 47 small subvolumes.

### Step 4. Optional Encoder-decoder Semantic Segmentation 3D network training.
Based on the manual selection results, Encoder-decoder Semantic Segmentation 3D (EDSS3D) network can be trained and applied for another tomogram dataset. 

The EDSS3D training is similar to Autoencoder3D training. Please refer to seg_src.py for source code.
