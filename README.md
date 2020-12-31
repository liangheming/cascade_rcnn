# Cascade RCNN
This is an unofficial pytorch implementation of CascadeRCNN object detection as described in [Cascade R-CNN Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726) by Zhaowei Cai and Nuno Vasconcelos

## requirement
```text
tqdm
pyyaml
numpy
opencv-python
pycocotools
torch >= 1.5
torchvision >=0.6.0
```
## result
we trained this repo on 4 GPUs with batch size 32(8 image per node).the total epoch is 24(about 180k iter),Adam with cosine lr decay is used for optimizing.
finally, this repo achieves 39.4 mAp at 736px(max thresh) resolution with resnet50 backbone.(about 21.94)
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.412
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.605
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.446
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.232
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.449
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.568
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.518
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.547
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.336
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.589
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.718
```

## training
for now we only support coco detection data.
### COCO
* modify main.py (modify config file path)
```python
from solver.ddp_mix_solver import DDPMixSolver
if __name__ == '__main__':
    processor = DDPMixSolver(cfg_path="your own config path") 
    processor.run()
```
* custom some parameters in *config.yaml*
```yaml
model_name: cascade_rcnn
data:
  train_annotation_path: data/coco/annotations/instances_train2017.json
#  train_annotation_path: data/coco/annotations/instances_val2017.json
  val_annotation_path: data/coco/annotations/instances_val2017.json
  train_img_root: data/coco/train2017
#  train_img_root: data/coco/val2017
  val_img_root: data/coco/val2017
  max_thresh: 768
  use_crowd: False
  batch_size: 8
  num_workers: 4
  debug: False
  remove_blank: Ture

model:
  num_cls: 80
  backbone: resnet50
  pretrained: True
  reduction: False
  fpn_channel: 256
  fpn_bias: True
  anchor_sizes: [32.0, 64.0, 128.0, 256.0, 512.0]
  anchor_scales: [1.0, ]
  anchor_ratios: [0.5, 1.0, 2.0]
  strides: [4.0, 8.0, 16.0, 32.0, 64.0]
  box_score_thresh: 0.05
  box_nms_thresh: 0.5
  box_detections_per_img: 100

optim:
  optimizer: Adam
  lr: 0.0001
  milestones: [24,]
  warm_up_epoch: 0
  weight_decay: 0.0001
  epochs: 24
  sync_bn: True
  amp: True
val:
  interval: 1
  weight_path: weights


gpus: 0,1,2,3
```
**detailed settings reference to nets.cascade_rcnn.default_cfg**
* run train scripts
```shell script
nohup python -m torch.distributed.launch --nproc_per_node=4 main.py >>train.log 2>&1 &
```

## TODO
- [x] Color Jitter
- [x] Perspective Transform
- [x] Mosaic Augment
- [x] MixUp Augment
- [x] IOU GIOU DIOU CIOU
- [x] Warming UP
- [x] Cosine Lr Decay
- [x] EMA(Exponential Moving Average)
- [x] Mixed Precision Training (supported by apex)
- [x] Sync Batch Normalize
- [ ] PANet(neck)
- [ ] BiFPN(EfficientDet neck)
- [ ] VOC data train\test scripts
- [ ] custom data train\test scripts
- [ ] MobileNet Backbone support

## Reference
```text
@inproceedings{cai18cascadercnn,
  author = {Zhaowei Cai and Nuno Vasconcelos},
  Title = {Cascade R-CNN: Delving into High Quality Object Detection},
  booktitle = {CVPR},
  Year  = {2018}
}
```