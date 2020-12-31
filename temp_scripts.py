import torch
from nets.cascade_rcnn import CascadeRCNN
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    dataset = COCODataSets(img_root="/home/huffman/data/coco/val2017",
                           annotation_path="/home/huffman/data/coco/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           remove_blank=True,
                           max_thresh=768,
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=dataset.collect_fn)
    net = CascadeRCNN()
    for img_input, valid_size, targets, batch_len in dataloader:
        out = net(img_input, valid_size=valid_size, targets={"target": targets, "batch_len": batch_len})
        box_loss = sum([v for k, v in out.items() if "box" in k])
        cls_loss = sum([v for k, v in out.items() if "cls" in k])
        loss = sum([v for k, v in out.items()])
        print(box_loss, cls_loss, loss)
        break
