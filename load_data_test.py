import load_data

import unittest

import torch
import torchvision

Label = load_data.Label

class DataLoaderTest(unittest.TestCase):
    def test_load_paths(self):
        roots_to_expected = {
            load_data.TRAINING_DATA_ROOT: 30000,
            load_data.VALIDATION_DATA_ROOT: 20000,
        }
        for (root, expected) in roots_to_expected.items():
            images = load_data.get_labeled_image_paths(root)
            labels = load_data.get_labeled_label_paths(root)

            self.assertEqual(len(images), len(labels))
            self.assertEqual(len(images), expected)
            self.assertEqual(len(labels), expected)

            self.assertIn(root, images[0])
            self.assertIn(root, labels[0])

    def test_collect_paths(self):
        paths = [
            'labeled_data/training/labels/17103.yml',
            'labeled_data/training/labels/1432.yml',
        ]
        expected = {
            17103: 'labeled_data/training/labels/17103.yml',
            1432: 'labeled_data/training/labels/1432.yml',
        }

        self.assertEqual(load_data.collect_paths_by_index(paths), expected)

    def test_collect_examples_by_index(self):
        label_paths = [
            'labeled_data/training/labels/17103.yml',
            'labeled_data/training/labels/1432.yml',
        ]
        image_paths = [
            'labeled_data/training/images/17103.JPEG',
            'labeled_data/training/images/1432.JPEG',
        ]

        expected = {
            17103: (
                'labeled_data/training/images/17103.JPEG',
                'labeled_data/training/labels/17103.yml',
                ),
            1432:  (
                'labeled_data/training/images/1432.JPEG',
                'labeled_data/training/labels/1432.yml',
                )
            }
        self.assertEqual(
            load_data.collect_examples_by_index(image_paths, label_paths),
            expected)

    def test_load_examples(self):
        label_yaml = """bboxes:
- - 277
  - 98
  - 439
  - 291
- - 163
  - 106
  - 306
  - 286
- - 84
  - 136
  - 217
  - 331
image_size:
- 500
- 375
labels:
- antelope
- antelope
- antelope
        """
        expected = [
            Label([277, 98, 439, 291], 'antelope'),
            Label([163, 106, 306, 286], 'antelope'),
            Label([84, 136, 217, 331], 'antelope')
        ]
        self.assertEqual(load_data.parse_labels(label_yaml),
                         expected)

    def test_load_class_index(self):
        class_index = load_data.load_class_index()
        self.assertEqual(class_index['fig'], 68)

    def test_load_inverted_class_index(self):
        inverted_class_index = load_data.load_inverted_class_index()
        self.assertEqual(inverted_class_index[68], 'fig')

    def test_load_dataset(self):
        ds = load_data.LabeledDataset(root_dir=load_data.VALIDATION_DATA_ROOT)

        self.assertEqual(20000, len(ds))

        image, bboxes, classes = ds[100]

        self.assertTrue(isinstance(image, torch.Tensor))
        # [C, H, W]
        self.assertEqual(3, image.dim())

        # [idx, 4]
        self.assertEqual(2, bboxes.dim())
        self.assertEqual(4, bboxes.shape[1])

        # [idx]
        self.assertEqual(1, classes.dim())
        self.assertEqual(bboxes.shape[0], classes.shape[0])

    @unittest.skip('slow test')
    def test_labeled_dataset_consistency(self):
        roots = [
            load_data.TRAINING_DATA_ROOT, load_data.VALIDATION_DATA_ROOT
        ]

        for root in roots:
            with self.subTest(root=root):
                ds = load_data.LabeledDataset(root)
                for img, bboxes, classes in ds:
                    # mostly just test that things load
                    self.assertEqual(bboxes.shape[0], classes.shape[0])


    def test_unlabeled_dataset_transforms(self):
        def transform(_):
            return torch.ones((4, 4))
        def augment(tensor):
            return tensor - 1

        ds = load_data.UnlabeledDataset(transform=transform, augment=augment)

        augmented, img = ds[10]

        torch.testing.assert_close(img, torch.ones((4, 4)))
        torch.testing.assert_close(augmented, torch.zeros((4, 4)))


    def test_crop_tensor_to_bbox(self):
        img = torch.ones(1, 150, 100)
        bboxes_and_expecteds = [
            ([0, 0, 100, 100], (100, 100)),
            ([0, 100, 100, 150], (50, 100)),
            ([0, 0, 1, 1], (1, 1)),
            ([1, 1, 2, 2], (1, 1)),
            (torch.tensor([0, 0, 1, 1]), (1, 1)),
        ]

        for bbox, expected in bboxes_and_expecteds:
            cropped = load_data.crop_tensor_to_bbox(img, bbox)
            self.assertEqual(cropped.shape, (1,) + expected)


    def test_classifier_dataset_fake(self):
        class FakeDataset:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                img = torch.ones(1, 100, 100)
                bboxes = torch.tensor([
                    [0, 0, 1, 1],
                    [0, 0, 2, 2]
                    ])
                classes = torch.tensor([12, 155])

                return (img, bboxes, classes)

        ds = load_data.ClassifierDataset(root_dir='', dataset_factory=FakeDataset)

        MAX_ATTEMPTS = 100

        seen_classes = set()
        for _ in range(MAX_ATTEMPTS):
            img, class_idx = ds[0]
            seen_classes.add(class_idx)
            expected_shape = {
                12: (1, 1, 1),
                155: (1, 2, 2),
            }[class_idx.item()]
            self.assertEqual(img.shape, expected_shape)

            if len(seen_classes) == 2:
                # success, we saw both classes
                return

        # fail the test
        self.assertTrue(False)

    def test_classifier_dataset_basic(self):
        # just verify that things look reasonable
        idx_to_class = load_data.load_inverted_class_index()

        COUNT = 5
        ds = load_data.ClassifierDataset(root_dir=load_data.TRAINING_DATA_ROOT)
        for i in range(5):
            img, class_idx = ds[i]
            self.assertIn(class_idx.item(), idx_to_class)
            # 3 channels
            self.assertEqual(3, img.shape[0])

    def test_classifier_dataset_resize(self):
        # just verify that things look reasonable
        idx_to_class = load_data.load_inverted_class_index()

        COUNT = 5
        resize = torchvision.transforms.Resize(224)
        ds = load_data.ClassifierDataset(root_dir=load_data.TRAINING_DATA_ROOT,
                                         outer_transform=resize)
        for i in range(5):
            img, class_idx = ds[i]
            self.assertIn(class_idx.item(), idx_to_class)
            # 3 channels
            self.assertEqual(3, img.shape[0])
            # resize increases the shorter side
            self.assertLessEqual(224, min(img.shape[1:]))


    def test_detr_coco_wrapper(self):
        fake_ds = {
            0: (
                # image
                torch.ones((3, 10, 10)),
                # bboxes
                torch.ones((4, 10)),
                # classes
                torch.ones(4),
            ),
            2: (
                # image
                torch.zeros((3, 10, 11)),
                # bboxes
                torch.zeros((4, 10)),
                # classes
                torch.zeros(4),
            )
            }

        ds = load_data.DetrCocoWrapper(fake_ds)


        img_0, target_0 = ds[0]
        self.assertEqual(target_0['image_id'], 0)
        torch.testing.assert_close(img_0, fake_ds[0][0])
        torch.testing.assert_close(target_0['boxes'], fake_ds[0][1])
        torch.testing.assert_close(target_0['labels'], fake_ds[0][2])
        torch.testing.assert_close(target_0['orig_size'], torch.tensor([10, 10]))


        img_2, target_2 = ds[2]
        self.assertEqual(target_2['image_id'], 2)
        torch.testing.assert_close(img_2, fake_ds[2][0])
        torch.testing.assert_close(target_2['boxes'], fake_ds[2][1])
        torch.testing.assert_close(target_2['labels'], fake_ds[2][2])
        torch.testing.assert_close(target_2['orig_size'], torch.tensor([10, 11]))
        # TODO(pscollins): test for PIL case


    def test_detr_coco_wrapper_transform(self):
        fake_ds = {
            0: (
                # image
                torch.ones((3, 10, 10)),
                # bboxes
                torch.ones((4, 10)),
                # classes
                torch.ones(4),
            ),
        }

        def transform(img, target):
            torch.testing.assert_close(img, fake_ds[0][0])
            target['boxes'] = target['boxes'] * 2
            return img, target

        ds = load_data.DetrCocoWrapper(fake_ds, transform=transform)

        img_0, target_0 = ds[0]
        self.assertEqual(target_0['image_id'], 0)
        torch.testing.assert_close(img_0, fake_ds[0][0])
        torch.testing.assert_close(target_0['boxes'], torch.ones((4, 10)) * 2)




if __name__ == '__main__':
    unittest.main()
