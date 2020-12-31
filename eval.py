import os
import time
import torch
import yaml
import json

import cv2 as cv
import numpy as np
from tqdm import tqdm
from nets.cascade_rcnn import CascadeRCNN
from datasets.coco import coco_ids, rgb_mean, rgb_std
from utils.augmentations import RandScaleMinMax
from utils.model_utils import AverageLogger


def coco_eavl(anno_path="/home/huffman/data/annotations/instances_val2017.json", pred_path="predicts.json"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoGt = COCO(anno_path)  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(pred_path)  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


@torch.no_grad()
def eval_model(weight_path="weights/faster_rcnn_resnet50_last.pth", device="cuda:0"):
    from pycocotools.coco import COCO
    device = torch.device(device)
    with open("config/cascade.yaml", 'r') as rf:
        cfg = yaml.safe_load(rf)
    net = CascadeRCNN(**{**cfg['model'], 'pretrained': False, 'box_nms_thresh': 0.6})
    net.load_state_dict(torch.load(weight_path, map_location="cpu")['ema'])
    net.to(device)
    net.eval()
    data_cfg = cfg['data']
    basic_transform = RandScaleMinMax(min_threshes=[640], max_thresh=data_cfg['max_thresh'])
    coco = COCO(data_cfg['val_annotation_path'])
    coco_predict_list = list()
    time_logger = AverageLogger()
    pbar = tqdm(coco.imgs.keys())
    for img_id in pbar:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(data_cfg['val_img_root'], file_name)
        img = cv.imread(img_path)
        h, w, _ = img.shape
        img, ratio = basic_transform.scale_img(img,
                                               min_thresh=640)
        h_, w_ = img.shape[:2]
        padding_size = max(h_, w_)
        img_inp = np.ones((padding_size, padding_size, 3)) * np.array((103, 116, 123))
        img_inp[:h_, :w_, :] = img
        img_inp = (img_inp[:, :, ::-1] / 255.0 - np.array(rgb_mean)) / np.array(rgb_std)
        img_inp = torch.from_numpy(img_inp).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float().to(device)
        tic = time.time()
        predict = net(img_inp, valid_size=[(padding_size, padding_size)])[0]
        duration = time.time() - tic
        time_logger.update(duration)
        pbar.set_description("fps:{:4.2f}".format(1 / time_logger.avg()))
        if predict is None:
            continue
        predict[:, [0, 2]] = (predict[:, [0, 2]] / ratio).clamp(min=0, max=w)
        predict[:, [1, 3]] = (predict[:, [1, 3]] / ratio).clamp(min=0, max=h)
        box = predict.cpu().numpy()
        coco_box = box[:, :4]
        coco_box[:, 2:] = coco_box[:, 2:] - coco_box[:, :2]
        for p, b in zip(box.tolist(), coco_box.tolist()):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': coco_ids[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})
    with open("predicts.json", 'w') as file:
        json.dump(coco_predict_list, file)
    coco_eavl(anno_path=data_cfg['val_annotation_path'], pred_path="predicts.json")


if __name__ == '__main__':
    eval_model()
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.410
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.609
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.439
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.446
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.329
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.509
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.324
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.572
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.703
