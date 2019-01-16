# Subtomogram classification network model compression
This project developed model compression techniques to compress deep subtomogram classification networks into compact models achieving 2-3 times speedup without loss in accuracy.

Please refer to our paper for more details:

Guo, Jialiang, Bo Zhou, Xiangrui Zeng, Zachary Freyberg, and Min Xu. "Model compression for faster structural separation of macromolecules captured by Cellular Electron Cryo-Tomography." In International Conference Image Analysis and Recognition, pp. 144-152. Springer, Cham, 2018. 

https://link.springer.com/chapter/10.1007/978-3-319-93000-8_17

https://arxiv.org/abs/1801.10597

## Knowledge distillation flowchart
<img src="https://user-images.githubusercontent.com/31047726/51223193-e7c20700-190e-11e9-9eb2-eaaf58aa5078.png" width="700">

## Architectures of the teacher network and student networks

models.py contains the code using the keras platform to construct the teacher network and studnet networks.

<img src="https://user-images.githubusercontent.com/31047726/51223192-e7c20700-190e-11e9-8384-ed7db0d3973c.png" width="700">


## Key prerequisites
* [keras](https://keras.io/#installation)
* [tensorflow-gpu](https://www.tensorflow.org/install/)
* numpy
```
pip install numpy
```





## Installation 
```
git clone https://github.com/xulabs/projects.git
```


