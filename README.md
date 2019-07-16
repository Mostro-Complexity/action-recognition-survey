# Action Recognition With Depth Sequences
Matlab and Python code for graduation design of Qiu Siyu.

[Research on Action and Behavior Recognition Technology Based on Kinect Somatosensory Information]

By Nanjing Normal University Qiu Siyu.
************************************************************************************
## Experimental Environment
* Matlab 2018b  
* Anaconda 3.6

## Experimental setting:

Cross-subject - half of the subjects used for training and the remaining half used for testing.
Results are averaged over 10 different training and test subject combinations.


## Datasets

We provide pre-computed skeleton sequences for all the datasets supported:
* [MSR Action 3D](http://research.microsoft.com/en-us/um/people/zliu/ActionRecoRsrc)

## Run

### For hand-craft features
The matlab file "run.m" runs the experiments for UTKinect-Action, Florence3D-Action and MSRAction3D datasets using 4 
different skeletal representations: 'absolute joint positions', 'relative joint positions', 'eigen joint', 'histograms of joint'.

The file "skeletal_action_classification.m" contains the code for entire pipeline:
Step 1: Skeletal representation ('absolute joint positions' or 'relative joint positions' or 'eigen joint' or 'histograms of joint')
Step 2: Temporal modeling (DTW and Fourier Temporal Pyramid)
Step 3: Classification: One-vs-All linear SVM (implemented as kernel SVM with linear kernel)

### For high-level learned features
Use the file "get_cnn_advanced_features.py" and you will get high-level learned features as "*.mat" files.

```python
python get_cnn_advanced_features.py
```
## Results

Due to problems in the implementation of some models, the experimental results are not ideal. The values in the results are for reference only.

