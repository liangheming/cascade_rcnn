import torch
import torch.nn.functional as F
from torch import nn
from nets import resnet
from nets.roi_pooling import MultiScaleRoIAlign
from nets.common import FrozenBatchNorm2d
from losses.commons import IOULoss
from utils.boxs_utils import box_iou
from torchvision.ops.boxes import batched_nms


class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_reg[..., :2] = anchors_xy + scale_reg[..., :2] * anchors_wh
        scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchors_wh
        scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
        scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]

        return scale_reg


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
                matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    @staticmethod
    def set_low_quality_matches_(matches, all_matches, match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None], as_tuple=False
        )
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class AnchorGenerator(object):
    def __init__(self, anchor_sizes, anchor_scales, anchor_ratios, strides):
        self.anchor_sizes = anchor_sizes
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.strides = strides
        self.anchor_per_grid = len(self.anchor_ratios) * len(self.anchor_scales)
        assert len(anchor_sizes) == len(strides)

    @staticmethod
    def __get_anchor_delta(anchor_size, anchor_scales, anchor_ratios):
        """
        :param anchor_size:
        :param anchor_scales: list
        :param anchor_ratios: list
        :return: [len(anchor_scales) * len(anchor_ratio),4]
        """
        scales = torch.tensor(anchor_scales).float()
        ratio = torch.tensor(anchor_ratios).float()
        scale_size = (scales * anchor_size)
        w = (scale_size[:, None] * ratio[None, :].sqrt()).view(-1) / 2
        h = (scale_size[:, None] / ratio[None, :].sqrt()).view(-1) / 2
        delta = torch.stack([-w, -h, w, h], dim=1)
        return delta

    def build_anchors(self, feature_maps):
        """
        :param feature_maps:
        :return: list(anchor) anchor:[all,4] (x1,y1,x2,y2)
        """
        assert len(self.anchor_sizes) == len(feature_maps)
        anchors = list()
        for stride, size, feature_map in zip(self.strides, self.anchor_sizes, feature_maps):
            # 9*4
            anchor_delta = self.__get_anchor_delta(size, self.anchor_scales, self.anchor_ratios)
            _, _, ny, nx = feature_map.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            # h,w,4
            grid = torch.stack([xv, yv, xv, yv], 2).float()
            anchor = (grid[:, :, None, :] + 0.5) * stride + anchor_delta[None, None, :, :]
            anchor = anchor.view(-1, 4)
            anchors.append(anchor.to(feature_map.device))
        return anchors

    def __call__(self, feature_maps):
        anchors = self.build_anchors(feature_maps)
        return anchors


class BalancedSampler(object):
    def __init__(self,
                 positive_thresh=0,
                 negative_val=-1,
                 sample_num=512,
                 positive_fraction=0.5):
        super(BalancedSampler, self).__init__()
        self.positive_fraction = positive_fraction
        self.positive_thresh = positive_thresh
        self.negative_val = negative_val
        self.sample_num = sample_num

    def __call__(self, match_idx):
        positive = torch.nonzero(match_idx >= self.positive_thresh, as_tuple=False).squeeze(1)
        negative = torch.nonzero(match_idx == self.negative_val, as_tuple=False).squeeze(1)
        num_pos = int(self.sample_num * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.sample_num - num_pos
        num_neg = min(negative.numel(), num_neg)
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
        pos_idx_per_image = positive[perm1]
        neg_idx_per_image = negative[perm2]
        return pos_idx_per_image, neg_idx_per_image


class FPN(nn.Module):
    def __init__(self, in_channels, out_channel, bias=True):
        super(FPN, self).__init__()
        self.latent_layers = list()
        self.out_layers = list()
        for channels in in_channels:
            self.latent_layers.append(nn.Conv2d(channels, out_channel, 1, 1, bias=bias))
            self.out_layers.append(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=bias))
        self.latent_layers = nn.ModuleList(self.latent_layers)
        self.out_layers = nn.ModuleList(self.out_layers)
        self.max_pooling = nn.MaxPool2d(1, 2)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xs):
        num_layers = len(xs)
        for i in range(num_layers):
            xs[i] = self.latent_layers[i](xs[i])
        for i in range(num_layers):
            layer_idx = num_layers - i - 1
            if i == 0:
                xs[layer_idx] = self.out_layers[layer_idx](xs[layer_idx])
            else:
                d_l = nn.UpsamplingBilinear2d(size=xs[layer_idx].shape[-2:])(xs[layer_idx + 1])
                xs[layer_idx] = self.out_layers[layer_idx](d_l + xs[layer_idx])
        xs.append(self.max_pooling(xs[-1]))
        return xs


class RPNHead(nn.Module):
    def __init__(self, in_channel, anchor_per_grid):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.cls = nn.Conv2d(in_channel, anchor_per_grid, 1, 1)
        self.box = nn.Conv2d(in_channel, anchor_per_grid * 4, 1, 1)
        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        cls = []
        box = []
        bs = x[0].shape[0]
        for feature in x:
            t = F.relu(self.conv(feature))
            cls.append(self.cls(t).permute(0, 2, 3, 1).contiguous().view(bs, -1, 1))
            box.append(self.box(t).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4))
        return cls, box


default_cfg = {
    "num_cls": 80,
    "backbone": "resnet18",
    "pretrained": True,
    "reduction": False,
    "norm_layer": None,
    "fpn_channel": 256,
    "fpn_bias": True,
    "anchor_sizes": [32., 64., 128., 256., 512.],
    "anchor_scales": [2 ** 0, ],
    "anchor_ratios": [0.5, 1.0, 2.0],
    "strides": [4., 8., 16., 32., 64.],
    "rpn": {
        "pre_nms_top_n_train": 2000,
        "post_nms_top_n_train": 2000,
        "pre_nms_top_n_test": 1000,
        "post_nms_top_n_test": 1000,
        "pos_iou_thr": 0.7,
        "neg_iou_thr": 0.3,
        "allow_low_quality_matches": True,
        "nms_thresh": 0.7,
        "sample_size": 256,
        "positive_fraction": 0.5,
        "bbox_weights": [1.0, 1.0, 1.0, 1.0],
        "iou_type": "giou"
    },
    "roi": {
        "featmap_names": ['0', '1', '2', '3'],
        "output_size": 7,
        "sampling_ratio": 2
    },
    "cascade_head": {
        "num_stages": 3,
        # "num_stages": 1,
        "stage_loss_weights": [1, 0.5, 0.25],
        # "stage_loss_weights": [1.0, ],
        "fc_out_channels": 1024,
        "bbox_head": [
            {
                "pos_iou_thr": 0.5,
                "neg_iou_thr": 0.5,
                "allow_low_quality_matches": False,
                "add_gt_as_proposals": True,
                "sample_size": 512,
                "positive_fraction": 0.25,
                "bbox_weights": [0.1, 0.1, 0.2, 0.2],
                "iou_type": "giou"
            },
            {
                "pos_iou_thr": 0.6,
                "neg_iou_thr": 0.6,
                "allow_low_quality_matches": False,
                "add_gt_as_proposals": True,
                "sample_size": 512,
                "positive_fraction": 0.25,
                "bbox_weights": [0.05, 0.05, 0.1, 0.1],
                "iou_type": "giou"
            },
            {
                "pos_iou_thr": 0.7,
                "neg_iou_thr": 0.7,
                "allow_low_quality_matches": False,
                "add_gt_as_proposals": True,
                "sample_size": 512,
                "positive_fraction": 0.25,
                "bbox_weights": [0.033, 0.033, 0.067, 0.067],
                "iou_type": "giou"
            },
        ]
    },
    "box_score_thresh": 0.05,
    "box_nms_thresh": 0.5,
    "box_detections_per_img": 100

}


class RPN(nn.Module):
    def __init__(self,
                 rpn_head,
                 pre_nms_top_n,
                 post_nms_top_n,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 allow_low_quality_matches=True,
                 nms_thresh=0.7,
                 sample_size=256,
                 positive_fraction=0.5,
                 iou_type="giou",
                 bbox_weights=None,
                 ):
        super(RPN, self).__init__()
        self.rpn_head = rpn_head
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.proposal_matcher = Matcher(
            pos_iou_thr,
            neg_iou_thr,
            allow_low_quality_matches=allow_low_quality_matches
        )
        self.box_coder = BoxCoder(bbox_weights)
        self.bce = nn.BCEWithLogitsLoss()
        self.box_loss = IOULoss(iou_type=iou_type)
        self.nms_thresh = nms_thresh
        self.sampler = BalancedSampler(positive_thresh=0,
                                       negative_val=Matcher.BELOW_LOW_THRESHOLD,
                                       sample_num=sample_size,
                                       positive_fraction=positive_fraction)

    def get_pre_nms_top_n(self):
        return self.pre_nms_top_n['train'] if self.training else self.pre_nms_top_n['test']

    def get_post_nms_top_n(self):
        return self.post_nms_top_n['train'] if self.training else self.post_nms_top_n['test']

    def filter_proposals(self, proposals, objectness, anchor_nums_per_level, valid_size):
        """
        :param proposals:[bs,anchor_nums,4]
        :param objectness:[bs,anchor_nums,1]
        :param anchor_nums_per_level:list()
        :param valid_size:[bs,2](w,h)
        :return:
        """
        bs = proposals.shape[0]
        device = proposals.device
        objectness = objectness.squeeze(-1)
        levels = torch.cat([torch.full((n,), idx, dtype=torch.int64, device=device)
                            for idx, n in enumerate(anchor_nums_per_level)], dim=0)[None, :].repeat(bs, 1)
        anchor_idx = list()
        offset = 0
        for ob in objectness.split(anchor_nums_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.get_pre_nms_top_n(), num_anchors)
            _, top_k = ob.topk(pre_nms_top_n, dim=1)
            anchor_idx.append(top_k + offset)
            offset += num_anchors
        anchor_idx = torch.cat(anchor_idx, dim=1)
        batch_idx = torch.arange(bs, device=device)[:, None]
        objectness = objectness[batch_idx, anchor_idx]
        levels = levels[batch_idx, anchor_idx]
        proposals = proposals[batch_idx, anchor_idx]

        final_boxes = list()
        final_scores = list()
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, valid_size):
            width, height = img_shape
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(min=0, max=width)
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(min=0, max=height)
            keep = ((boxes[..., 2] - boxes[..., 0]) > 1e-3) & ((boxes[..., 3] - boxes[..., 1]) > 1e-3)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[:self.get_post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, proposal, anchors_all, targets):
        gt_boxes = targets['target'].split(targets['batch_len'])
        batch_idx = list()
        anchor_idx = list()
        gt_idx = list()
        for idx, gt in enumerate(gt_boxes):
            if len(gt) == 0:
                match_idx = torch.full_like(anchors_all[:, 0], fill_value=Matcher.BELOW_LOW_THRESHOLD).long()
            else:
                gt_anchor_iou = box_iou(gt[:, 1:], anchors_all)
                match_idx = self.proposal_matcher(gt_anchor_iou)
            positive_idx, negative_idx = self.sampler(match_idx)
            batch_idx.append(([idx] * len(positive_idx), [idx] * len(negative_idx)))
            gt_idx.append(match_idx[positive_idx].long())
            anchor_idx.append((positive_idx, negative_idx))
        all_batch_idx = sum([sum(item, []) for item in batch_idx], [])
        all_anchor_idx = torch.cat([torch.cat(item) for item in anchor_idx])
        all_cls_target = torch.tensor(sum([[1] * len(item[0]) + [0] * len(item[1])
                                           for item in anchor_idx], []),
                                      device=objectness.device, dtype=objectness.dtype)
        all_cls_predicts = objectness[all_batch_idx, all_anchor_idx]

        cls_loss = self.bce(all_cls_predicts, all_cls_target[:, None])
        all_positive_batch = sum([item[0] for item in batch_idx], [])
        all_positive_anchor = torch.cat([item[0] for item in anchor_idx])
        all_predict_box = proposal[all_positive_batch, all_positive_anchor]
        all_gt_box = torch.cat([i[j][:, 1:] for i, j in zip(gt_boxes, gt_idx)], dim=0)
        box_loss = self.box_loss(all_predict_box, all_gt_box).sum() / len(all_gt_box)
        return cls_loss, box_loss

    def forward(self, xs, anchors, valid_size, targets=None):
        objectness, pred_bbox_delta = self.rpn_head(xs)
        anchors_num_per_layer = [len(anchor) for anchor in anchors]
        anchors_all = torch.cat([anchor for anchor in anchors], dim=0)
        objectness = torch.cat([obj for obj in objectness], dim=1)
        pred_bbox_delta = torch.cat([delta for delta in pred_bbox_delta], dim=1)
        proposals = self.box_coder.decoder(pred_bbox_delta, anchors_all)
        boxes, scores = self.filter_proposals(proposals.detach(),
                                              objectness.detach(),
                                              anchors_num_per_layer,
                                              valid_size)
        losses = dict()

        if self.training:
            assert targets is not None
            cls_loss, box_loss = self.compute_loss(objectness, proposals, anchors_all, targets)
            losses['rpn_cls_loss'] = cls_loss
            losses['rpn_box_loss'] = box_loss
        return boxes, losses


class FasterRCNNSimpleBoxHead(nn.Module):
    def __init__(self, in_channels, inner_channels, num_cls=80):
        super(FasterRCNNSimpleBoxHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, inner_channels)
        self.fc7 = nn.Linear(inner_channels, inner_channels)
        self.cls = nn.Linear(inner_channels, num_cls + 1)
        self.bbox = nn.Linear(inner_channels, 4)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls.weight, 0, 0.01)
        nn.init.constant_(self.cls.bias, 0)
        nn.init.normal_(self.bbox.weight, 0, 0.001)
        nn.init.constant_(self.bbox.bias, 0)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        scores = self.cls(x)
        bbox_deltas = self.bbox(x)
        return scores, bbox_deltas


class ROIHead(nn.Module):
    def __init__(self,
                 roi_pooling,
                 box_head,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5,
                 allow_low_quality_matches=False,
                 add_gt_as_proposals=True,
                 sample_size=512,
                 positive_fraction=0.25,
                 bbox_weights=None,
                 iou_type="giou"):
        super(ROIHead, self).__init__()
        self.box_head = box_head
        self.roi_pooling = roi_pooling
        self.matcher = Matcher(high_threshold=pos_iou_thr,
                               low_threshold=neg_iou_thr,
                               allow_low_quality_matches=allow_low_quality_matches)
        self.add_gt_as_proposals = add_gt_as_proposals
        self.sampler = BalancedSampler(positive_thresh=0,
                                       negative_val=Matcher.BELOW_LOW_THRESHOLD,
                                       sample_num=sample_size,
                                       positive_fraction=positive_fraction)
        self.box_coder = BoxCoder(bbox_weights)
        self.iou_loss = IOULoss(iou_type=iou_type)
        self.ce = nn.CrossEntropyLoss()

    def compute_loss(self, proposals, cls_predicts, box_predicts, targets):
        gt_boxes = targets['target'].split(targets['batch_len'])
        loss_cls_predicts = list()
        loss_box_predicts = list()
        loss_cls_targets = list()
        loss_box_targets = list()
        for p, c, b, g in zip(proposals, cls_predicts, box_predicts, gt_boxes):
            if len(g) == 0:
                match_idx = torch.full_like(p[:, 0], fill_value=Matcher.BELOW_LOW_THRESHOLD).long()
            else:
                gt_anchor_iou = box_iou(g[:, 1:], p)
                match_idx = self.matcher(gt_anchor_iou).long()
            positive_idx, negative_idx = self.sampler(match_idx)
            gt_idx = match_idx[positive_idx]
            loss_cls_predicts.append(c[positive_idx])
            loss_box_predicts.append(b[positive_idx])
            loss_cls_targets.append(g[gt_idx][:, 0].long())
            loss_box_targets.append(g[gt_idx][:, 1:])

            loss_cls_predicts.append(c[negative_idx])
            loss_cls_targets.append(torch.full((len(negative_idx),), -1, device=c.device, dtype=torch.long))
        loss_cls_predicts = torch.cat(loss_cls_predicts)
        loss_cls_targets = torch.cat(loss_cls_targets) + 1
        loss_box_predicts = torch.cat(loss_box_predicts)
        loss_box_targets = torch.cat(loss_box_targets)
        cls_loss = self.ce(loss_cls_predicts, loss_cls_targets)
        box_loss = self.iou_loss(loss_box_predicts, loss_box_targets).sum() / len(loss_box_targets)
        return cls_loss, box_loss

    def forward(self, xs, proposals, valid_size, targets=None):
        """
        :param xs: feature dict
        :param proposals:
        :param valid_size:
        :param targets:
        :return:
        """
        hw_size = [(s[1], s[0]) for s in valid_size]
        ori_nums_per_batch = [len(p) for p in proposals]
        nums_per_batch = ori_nums_per_batch
        if self.training and self.add_gt_as_proposals:
            gt_boxes = targets['target'].split(targets['batch_len'])
            proposals = [torch.cat([p, g[:, 1:]]) for p, g in zip(proposals, gt_boxes)]
            nums_per_batch = [len(p) for p in proposals]
        box_features = self.roi_pooling(xs, proposals, hw_size)
        cls_predicts, box_predicts = self.box_head(box_features)
        box_predicts = self.box_coder.decoder(box_predicts, torch.cat(proposals))
        cls_predicts = cls_predicts.split(nums_per_batch)
        box_predicts = box_predicts.split(nums_per_batch)
        loss = dict()
        if self.training:
            cls_loss, box_loss = self.compute_loss(proposals, cls_predicts, box_predicts, targets)
            loss['roi_cls_loss'] = cls_loss
            loss['roi_box_loss'] = box_loss
        boxes = [box[:l].detach() for l, box in zip(ori_nums_per_batch, box_predicts)]
        cls = [c[:l].detach() for l, c in zip(ori_nums_per_batch, cls_predicts)]
        return boxes, cls, loss


class CascadeHead(nn.Module):
    def __init__(self, num_cls, pooling_layer, stage_loss_weights, num_stages, fc_out_channels, detail_cfg):
        super(CascadeHead, self).__init__()
        self.stage_loss_weights = stage_loss_weights
        self.num_stages = num_stages
        self.fc_out_channels = fc_out_channels
        self.detail_cfg = detail_cfg
        feature_map_channel_size = pooling_layer.channel_size
        roi_resolution = pooling_layer.output_size[0]
        head_in_channel = feature_map_channel_size * roi_resolution ** 2
        assert len(self.stage_loss_weights) == num_stages == len(detail_cfg)
        self.roi_heads = list()
        for i in range(num_stages):
            box_head = FasterRCNNSimpleBoxHead(head_in_channel, self.fc_out_channels, num_cls)
            roi_head = ROIHead(pooling_layer, box_head, **detail_cfg[i])
            self.roi_heads.append(roi_head)
        self.roi_heads = nn.ModuleList(self.roi_heads)

    def forward(self, feature_dict, boxes, valid_size, targets):
        loss_sum = dict()
        all_cls = list()
        num_per_batch = [len(b) for b in boxes]
        for i in range(self.num_stages):
            boxes, cls, loss = self.roi_heads[i](feature_dict, boxes, valid_size, targets)
            if self.training:
                loss_sum['{:d}_cls_loss'.format(i)] = loss['roi_cls_loss'] * self.stage_loss_weights[i]
                loss_sum['{:d}_box_loss'.format(i)] = loss['roi_box_loss'] * self.stage_loss_weights[i]
            else:
                all_cls.append(torch.cat(cls, dim=0))
        if not self.training:
            all_cls = torch.stack(all_cls, dim=-1)
            if all_cls.dtype == torch.float16:
                all_cls = all_cls.float()
            all_cls = all_cls.softmax(dim=-2).mean(-1)
            all_cls = all_cls.split(num_per_batch)
        return boxes, all_cls, loss_sum


class CascadeRCNN(nn.Module):
    def __init__(self, **kwargs):
        super(CascadeRCNN, self).__init__()
        self.cfg = {**default_cfg, **kwargs}
        self.backbone = getattr(resnet, self.cfg['backbone'])(
            pretrained=self.cfg['pretrained'],
            reduction=self.cfg['reduction']
        )
        self.fpn = FPN(in_channels=self.backbone.inner_channels,
                       out_channel=self.cfg['fpn_channel'],
                       bias=self.cfg['fpn_bias'])
        self.anchor_generator = AnchorGenerator(
            anchor_sizes=self.cfg['anchor_sizes'],
            anchor_scales=self.cfg['anchor_scales'],
            anchor_ratios=self.cfg['anchor_ratios'],
            strides=self.cfg['strides']
        )
        self.anchors = None
        rpn_head = RPNHead(self.cfg['fpn_channel'], anchor_per_grid=self.anchor_generator.anchor_per_grid)
        rpn_pre_nms_top_n = {"train": self.cfg['rpn']['pre_nms_top_n_train'],
                             "test": self.cfg['rpn']['pre_nms_top_n_test']}
        rpn_post_nms_top_n = {"train": self.cfg['rpn']['post_nms_top_n_train'],
                              "test": self.cfg['rpn']['post_nms_top_n_test']}
        self.feature_keys = ['0', '1', '2', '3', 'pool']

        self.rpn = RPN(
            rpn_head=rpn_head,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
            pos_iou_thr=self.cfg['rpn']['pos_iou_thr'],
            neg_iou_thr=self.cfg['rpn']['neg_iou_thr'],
            allow_low_quality_matches=self.cfg['rpn']['allow_low_quality_matches'],
            nms_thresh=self.cfg['rpn']['nms_thresh'],
            sample_size=self.cfg['rpn']['sample_size'],
            positive_fraction=self.cfg['rpn']['positive_fraction'],
            bbox_weights=self.cfg['rpn']['bbox_weights'],
            iou_type=self.cfg['rpn']['iou_type']
        )

        roi = MultiScaleRoIAlign(
            featmap_names=self.cfg['roi']['featmap_names'],
            output_size=self.cfg['roi']['output_size'],
            sampling_ratio=self.cfg['roi']['sampling_ratio']
        )
        roi.channel_size = self.cfg['fpn_channel']
        self.cascade_head = CascadeHead(self.cfg['num_cls'],
                                        roi,
                                        self.cfg['cascade_head']['stage_loss_weights'],
                                        num_stages=self.cfg['cascade_head']['num_stages'],
                                        fc_out_channels=self.cfg['cascade_head']['fc_out_channels'],
                                        detail_cfg=self.cfg['cascade_head']['bbox_head'])

    def forward(self, x, valid_size, targets=None):
        if self.anchors:
            anchor_num = sum([a.shape[0] for a in self.anchors]) / self.anchor_generator.anchor_per_grid
        else:
            anchor_num = -1
        xs = self.backbone(x)
        xs = self.fpn(xs)
        xs_resolution = sum([i.shape[-2] * i.shape[-1] for i in xs])
        if xs_resolution != anchor_num:
            self.anchors = self.anchor_generator(xs)
        boxes, rpn_losses = self.rpn(xs, self.anchors, valid_size, targets)
        feature_dict = dict()
        for k, v in zip(self.feature_keys, xs):
            feature_dict[k] = v
        box_predicts, cls_predicts, roi_losses = self.cascade_head(feature_dict, boxes, valid_size, targets)
        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        if self.training:
            return losses
        else:
            return self.post_process(cls_predicts, box_predicts, valid_size)

    def post_process(self, cls_predicts, box_predicts, valid_size):
        predicts = list()
        for cls, box, wh in zip(cls_predicts, box_predicts, valid_size):
            box[..., [0, 2]] = box[..., [0, 2]].clamp(min=0, max=wh[0])
            box[..., [1, 3]] = box[..., [1, 3]].clamp(min=0, max=wh[1])
            scores = cls[:, 1:]
            labels = torch.arange(scores.shape[-1], device=cls.device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = box.unsqueeze(1).repeat(1, scores.shape[-1], 1).reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            inds = torch.nonzero(scores > self.cfg['box_score_thresh'], as_tuple=False).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            keep = ((boxes[..., 2] - boxes[..., 0]) > 1e-2) & ((boxes[..., 3] - boxes[..., 1]) > 1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = batched_nms(boxes, scores, labels, self.cfg['box_nms_thresh'])
            keep = keep[:self.cfg['box_detections_per_img']]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            pred = torch.cat([boxes, scores[:, None], labels[:, None]], dim=-1)
            predicts.append(pred)
        return predicts


if __name__ == '__main__':
    CascadeRCNN()
