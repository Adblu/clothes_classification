# Clothes Classification 

## Instalation
* install requirements
* install mmclassification: [link](https://github.com/open-mmlab/mmclassification)

## Train model:
```
python train.py

```
## Evaluation
* By default, mmclassification provides evaluation metrices during training. To see run:
```
python display_stats.py
```
 you should see:

![](src/figures/experiment_summary.png?raw=true)

## Visualization:
* Run:
```
python model_inference.py
```
![](src/figures/your_file_21.jpeg?raw=true)

## Notes

* To activate transfer learning specify in train_config.py:
```
TRANSFER_LEARNING = True
```
* pth file for experiment "clothes_experiment" is stored [here](https://ufile.io/i1uyquk4)
* pth file for experiment "clothes_experiment_tf" is stored [here](https://ufile.io/xiokecie)
* transfer learning checkpoint download [link](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth),
after download place to src/checkopints
* images dataset [link](https://ufile.io/q6pefkmn)  