import torch

_EPSILON = 1e-6

def _bboxes_side_lengths(bboxes):
    # [B, 4] -> 2 x [B]
    dx = bboxes[:, 0] - bboxes[:, 2]
    dy = bboxes[:, 1] - bboxes[:, 3]
    return dx, dy


def bbox_area(bboxes):
    # [B, 4]
    #  where the last dimension has the form
    #   [x0, y0, x1, y1]
    assert bboxes.dim() == 2
    assert bboxes.shape[-1] == 4
    dx, dy = _bboxes_side_lengths(bboxes)
    signed_area = dx * dy
    # don't assume any particular ordering
    return torch.abs(signed_area)


def sort_corners(bboxes):
    # [B, 4]
    #  where the last dimension has the form
    #   [x0, y0, x1, y1]
    assert bboxes.dim() == 2
    assert bboxes.shape[-1] == 4
    # solve for:
    #  x0_hat = min(x0, x1)
    #  x1_hat = max(x0, x1)
    #  etc
    # and return
    #  [x0_hat, y0_hat, ...]

    # Collect into [B, [x0, x1]]
    xs = torch.hstack((bboxes[:, 0].unsqueeze(-1), bboxes[:, 2].unsqueeze(-1)))
    # [B, [y0, y1]]
    ys = torch.hstack((bboxes[:, 1].unsqueeze(-1), bboxes[:, 3].unsqueeze(-1)))

    x_lo, x_hi = torch.aminmax(xs, dim=-1)
    y_lo, y_hi = torch.aminmax(ys, dim=-1)

    return torch.stack([x_lo, y_lo, x_hi, y_hi], dim=-1)


def bbox_intersection(left_bboxes, right_bboxes):
    # both tensors should have shape
    # [B, 4]
    #  where the last dimension has the form
    #   [x0, y0, x1, y1]
    # with x0 <= x1, y0 <= y1
    assert left_bboxes.dim() == right_bboxes.dim() == 2
    assert left_bboxes.shape[-1] == right_bboxes.shape[-1] == 4
    assert torch.equal(left_bboxes, sort_corners(left_bboxes))
    assert torch.equal(left_bboxes, sort_corners(left_bboxes))

    # [B, 4, 2]
    joined_bboxes = torch.stack((left_bboxes, right_bboxes), dim=-1)
    # 2 x [B, 4]
    lo_bboxes, hi_bboxes = torch.aminmax(joined_bboxes.squeeze(dim=-1), dim=-1)

    # 4 x [B]
    x_lo, x_hi = hi_bboxes[:, 0], lo_bboxes[:, 2]
    y_lo, y_hi = hi_bboxes[:, 0], lo_bboxes[:, 3]

    # [B, 4]
    intersection = torch.stack([x_lo, y_lo, x_hi, y_hi], dim=-1)
    # 2 x [B]
    dx, dy = _bboxes_side_lengths(intersection)
    # [B, 1]
    signed_area = (dx * dy).unsqueeze(-1)

    intersection = torch.where(signed_area > _EPSILON,
                               intersection,
                               torch.zeros_like(intersection))
    # TODO(pscollins): There will probably be some duplicated computation here,
    # since the caller also wants the area.
    return intersection

def generalized_intersection_over_union(predicted_bboxes, ground_truth_bboxes):
    # https://giou.stanford.edu/
    #
    # inputs are [B, 4], where the last dimension has the format
    #   [x0, y0, x1, y1]
    pass
