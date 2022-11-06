import unittest
import load_data

Label = load_data.Label


class DataLoaderTest(unittest.TestCase):
    def test_load_paths(self):
        roots_to_expected = {
            load_data.TRAINING_DATA_ROOT: 30000,
            load_data.VALIDATION_DATA_ROOT: 20000,
        }
        for (root, expected) in roots_to_expected.items():
            num_images = len(load_data.get_labeled_image_paths(root))
            num_labels = len(load_data.get_labeled_label_paths(root))

            self.assertEqual(num_images, num_labels)
            self.assertEqual(num_images, expected)
            self.assertEqual(num_labels, expected)

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





if __name__ == '__main__':
    unittest.main()
