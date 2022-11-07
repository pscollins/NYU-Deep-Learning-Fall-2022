import loss

import unittest

import torch

def mktensor(xs):
    return torch.tensor(xs, dtype=torch.float32)

class LossTest(unittest.TestCase):

    def test_bbox_area(self):
        bboxes = [
            [0, 0, 1, 1],
            [0, 0, 2, 2],
            [1, 1, 0, 0],
            [2, 2, 0, 0],
            [1, 1, 2, 2],
            [0, 0, 1, 2],
        ]
        expected = [
            1,
            4,
            1,
            4,
            1,
            2
        ]
        area = loss.bbox_area(torch.tensor(bboxes, dtype=torch.float32))
        torch.testing.assert_close(area, torch.tensor(expected,
                                                      dtype=torch.float32))

    def test_sort_corners(self):
        bboxes = [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [1, 1, 0, 0],
            [1, 2, 0, 1],
        ]
        expected = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [0, 0, 1, 1],
            [0, 1, 1, 2]
        ]
        actual = loss.sort_corners(mktensor(bboxes))
        torch.testing.assert_close(actual, mktensor(expected))

    def test_intersection(self):
        bbox_pairs = [
            ([0, 0, 0, 0], [0, 0, 0, 0]),
            ([0, 0, 2, 2], [0, 0, 2, 2]),
            ([0, 0, 2, 2], [1, 1, 2, 2]),
            ([0, 0, 2, 2], [-1, -1, 3, 3]),
            ([0, 0, 1, 1], [0, 0, 1, -1]),
        ]
        expected = [
            [0, 0, 0, 0],
            [0, 0, 2, 2],
            [1, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0]
        ]

        left_bboxes, right_bboxes = zip(*bbox_pairs)

        actual = loss.bbox_intersection(mktensor(left_bboxes), mktensor(right_bboxes))
        torch.testing.assert_close(actual, mktensor(expected))


if __name__ == '__main__':
    unittest.main()
