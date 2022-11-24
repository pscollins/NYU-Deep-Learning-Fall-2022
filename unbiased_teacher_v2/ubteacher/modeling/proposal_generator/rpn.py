# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from detectron2.layers import cat
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import move_device_like, batched_nms


def masked_binary_cross_entropy_with_logits(pred_objectness_logits,
                                           gt_labels,
                                           gt_confids,
                                           valid_mask):
    if gt_labels.device.type == 'xla':
        default_weights = gt_confids if gt_confids else torch.ones_like(gt_labels)
        masked_weights = torch.where(valid_mask, default_weights, torch.zeros_like(default_weights))
        return F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1),
                gt_labels.to(torch.float32),
                weight=masked_weights,
                reduction="sum",
            )
    else:
        masked_weights = gt_confids[valid_mask] if gt_confids is not None else None
        return F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            weight=masked_gt_confids,
            reduction="sum",
        )



@PROPOSAL_GENERATOR_REGISTRY.register()
class PseudoLabRPN(RPN):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
        compute_loss: bool = True,
        compute_val_loss: bool = False,
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if (self.training and compute_loss) or compute_val_loss:

            if gt_instances[0].has("scores"):  # has confidence; then weight loss
                gt_labels, gt_boxes, gt_confids = self.label_and_sample_anchors_pseudo(
                    anchors, gt_instances
                )
            else:  # no confidence of each proposal
                gt_labels, gt_boxes = self.label_and_sample_anchors(
                    anchors, gt_instances
                )
                gt_confids = None

            losses = self.losses(
                anchors,
                pred_objectness_logits,
                gt_labels,
                pred_anchor_deltas,
                gt_boxes,
                gt_confids,
            )
            losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        else:  # inference
            losses = {}

        if anchors[0].device.type == 'xla':
            proposals = self.predict_proposal_static_shape(anchors,
                                               pred_objectness_logits,
                                               pred_anchor_deltas,
                                               images.image_sizes)

        else:
            proposals = self.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )

        return proposals, losses


    def predict_proposal_static_shape(self, anchors, pred_objectness_logits, pred_anchor_deltas,
                                      image_sizes):
        # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/proposal_generator/rpn.py#L501-L512
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals_static_shape(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )


    def label_and_sample_anchors_pseudo(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
            list[Tensor]:
                i-th element is a R tensor. The values are the matched gt scores for each
                anchor. Values are undefined for those anchors not labeled as 1.

        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        scores = [x.scores for x in gt_instances]

        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        gt_confids = []

        for image_size_i, gt_boxes_i, scores_i in zip(image_sizes, gt_boxes, scores):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(
                match_quality_matrix
            )
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(
                    image_size_i, self.anchor_boundary_thresh
                )
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)
            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_confidence = torch.zeros_like(
                    matched_idxs
                )  # no boxes in the label --> no loss
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
                gt_confidence = scores_i[matched_idxs]

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
            gt_confids.append(gt_confidence)

        return gt_labels, matched_gt_boxes, gt_confids

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_confids: List[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        if gt_labels.device.type != 'xla':
            num_pos_anchors = pos_mask.sum().item()
            num_neg_anchors = (gt_labels == 0).sum().item()
            storage = get_event_storage()
            storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
            storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        # localization loss is not weighted
        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0
        if gt_confids:  # weights
            gt_confids = torch.stack(gt_confids)  # (N, sum(Hi*Wi*Ai))
            objectness_loss = masked_binary_cross_entropy_with_logits(
                pred_objectness_logits, gt_labels, gt_confids, valid_mask)
            # objectness_loss = F.binary_cross_entropy_with_logits(
            #     cat(pred_objectness_logits, dim=1)[valid_mask],
            #     gt_labels[valid_mask].to(torch.float32),
            #     weight=gt_confids[valid_mask],
            #     reduction="sum",
            # )
        else:  # no weights
            # objectness_loss = F.binary_cross_entropy_with_logits(
            #     cat(pred_objectness_logits, dim=1)[valid_mask],
            #     gt_labels[valid_mask].to(torch.float32),
            #     reduction="sum",
            # )
            objectness_loss = masked_binary_cross_entropy_with_logits(
                pred_objectness_logits, gt_labels, None, valid_mask)
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses


# def find_top_rpn_proposals_static_shape(
def find_top_rpn_proposals_static_shape(
    proposals,
    pred_objectness_logits,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_size,
    training,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.
    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.
    Returns:
        list[Instances]: list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
    """
    num_images = len(image_sizes)
    device = (
        proposals[0].device
        if torch.jit.is_scripting()
        else ("cpu" if torch.jit.is_tracing() else proposals[0].device)
    )
    print('find proposals RPN')

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = move_device_like(torch.arange(num_images, device=device), proposals[0])
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            move_device_like(
                torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device),
                proposals[0],
            )
        )

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        # boxes = torch.where(
        #     keep,
        #     boxes,
        #     torch.zeros_like(boxes))
        scores_per_img = torch.where(keep, scores_per_img, torch.zeros_like(scores_per_img))
        # lvl = torch.where(keep, lvl, 0)
        # if _is_tracing() or keep.sum().item() != len(boxes):
        #     boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        # res.proposal_boxes = boxes[keep]
        # res.objectness_logits = scores_per_img[keep]
        res.proposal_boxes = boxes
        res.objectness_logits = scores_per_img
        results.append(res)
    return results
