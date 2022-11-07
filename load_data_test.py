import unittest

import load_data
import torch

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
        self.assertEqual(class_index['fig'], 38)

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






if __name__ == '__main__':
    unittest.main()
