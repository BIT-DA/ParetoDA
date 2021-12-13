## ParetoDA

This repo provides a demo for the NIPS 2021 paper "Pareto Domain Adaptation" on the VisDA-2017 dataset.
[[Paper]](https://openreview.net/forum?id=frgb7FsKWs3)

### Requirements

* `Python 3.6`
* `Pytorch 1.1.0`

### Training from scratch 
Please first download the VisDA-2017 dataset from https://github.com/VisionLearningGroup/taskcv-2017-public. Then update the train and validation files with suffix ".txt" following styles below:
```
data/visda2017/train/aeroplane/aeroplane_src_001.jpg 0
...
```
```
data/visda2017/validation/aeroplane/aeroplane_001.jpg 0
...
```


Then train on VisDA2017 with ResNet101:

```
python DANN+ParetoDA.py --gpu_id 0 --arch resnet101 --train_path xxx --val_path xxx
```




### Acknowledgements
Some codes are adapted from [EPOSearch](https://github.com/dbmptr/EPOSearch). We thank them for their excellent projects.

### Contact

If you have any problem about our code, feel free to contact
fangruilv@bit.edu.cn
or describe your problem in Issues.