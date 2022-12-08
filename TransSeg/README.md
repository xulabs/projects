# single_cell_segmentation
## Abstract 
Motivation: Single cell segmentation is a fundamental but challenging task in the field of cell biology.
While existing deep learning-based segmentation methods have achieved acceptable performance, there
is still room for improving the performance on segmenting cell boundaries (or cytoderms), specifically
in terms of separation, integrity, and continuity of cells and cell boundaries. We performed comparative
study on a number of live-cell image datasets using existing deep learning-based biomedical image
segmentation methods and identified the main reasons that cause the inferior performance are different
image styles and class unbalanced issue.
Results: Based on the pre-study results, we propose a novel TransSeg method for single cell segmentation
with a focus on improving the segmentation performance on cytoderm. The pipeline includes a style-
transfer aware modality adaptation module, an unbalanced boundary enhancing loss, and a dense
Conditional Random Fields-based post-processing module to improve the performance throughout
the whole segmentation process from the perspectives of pre-processing, network training, and
post-processing. TransSeg significantly improves the performance of the baseline model through
comprehensive experiments on fluorescence, phase contrast, and differential inference contrast (DIC)
image
## Method
* style-transfer aware modality adaptation module
* unbalanced boundary enhancing loss
* dense Conditional Random Fields-based post-processing module

## Tutorial 
### Requirement 

* PyTorch>=1.10.2
* NumPy>=1.21.6
* torchvision>=0.11.3
* matplotlib>=3.5.1
* scipy>=1.8.0
* pillow>=9.0.1
* opencv>=4.5.5

### Data
The data is available at https://github.com/opnumten/single_cell_segmentation. Please refer to the paper 
doi:10.1016/j.compbiomed.2019.04.006 Wang et al, Learn to Segment Single Cells with Deep Distance Estimator 
and Deep Cell Detector for more information.


### Training
train CycleGAN model for data generation
```
data_gen.ipynb
```
train U_Net segmentation model
```
python main.py
```
applying densecrf for post-processing
```
python CRF.py
```
