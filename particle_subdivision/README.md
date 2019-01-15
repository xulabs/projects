# Inception3D and DSRF3D network for subtomogram particle subdivision
The two networks perform deep learning based classification of Cellular Electron Cryo Tomography subtomogram data

Please refer to our paper for more details:

Xu, Min, Xiaoqi Chai, Hariank Muthakana, Xiaodan Liang, Ge Yang, Tzviya Zeev-Ben-Mordehai, and Eric P. Xing. "Deep learning-based subdivision approach for large scale macromolecules structure recovery from electron cryo tomograms." Bioinformatics 33, no. 14 (2017): i13-i22.https://doi.org/10.1016/j.jsb.2017.12.015 

https://doi.org/10.1093/bioinformatics/btx230

https://arxiv.org/abs/1701.08404

## Network architecture
<img src="https://user-images.githubusercontent.com/31047726/51214221-437a9900-18eb-11e9-9f42-27fd51543f6f.jpg" width="1000">


## Key prerequisites
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

## Example structure recovery results
![alt text](https://user-images.githubusercontent.com/31047726/51214220-437a9900-18eb-11e9-8084-23b726f0ef96.jpg)
