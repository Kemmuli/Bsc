import numpy as np
from os import path
import glob
import csv


def load_dataset():

    feature_types = ['GCC', 'magspec', 'mel', 'ilds', 'phase_diffs_cossine', 'phase_diff', 'mel_gcc_phat']
    feature_ids = {}
    input_shapes = {}
    for feature in feature_types:
        dataset = {'train': [], 'validation': [], 'test': []}
        data_fp = './direction-dataset/audio/*_' + feature + '.npy'
        data_fnames = glob.glob(data_fp)

        for name in data_fnames:
            basename = path.splitext(path.basename(name))
            suffix = '_' + feature
            id = basename[0].replace(suffix, '')
            if 'split3' in name:
                dataset['validation'].append(id)
            if 'split4' in name:
                dataset['test'].append(id)
            if 'split1' in name or 'split2' in name:
                dataset['train'].append(id)

        feature_ids[feature] = dataset
        test_fp = './direction-dataset/audio/' + dataset['train'][0] + '_' + feature + '.npy'
        shape = np.load(test_fp).shape
        input_shapes[feature] = shape

    np.save('./input_shapes', input_shapes, allow_pickle=True)
    np.save('./data_id_dict', feature_ids, allow_pickle=True)


def store_true_values():
    classtypes = ['front', 'right', 'back', 'left']
    class_fp = './direction-dataset/metadata/split*'
    class_fnames = glob.glob(class_fp)
    classes = {}
    true_class = None

    for i, name in enumerate(class_fnames):
        with open(name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            line = 0
            for row in reader:
                if line == 0:
                    line += 1
                else:
                    true_class = row[-1]
        csv_file.close()

        for direction in range(len(classtypes)):  # Check which class is the value.
            if true_class == classtypes[direction]:
                basename = path.splitext(path.basename(name))
                id = basename[0]
                classes[id] = int(direction)

    np.save('./class_labels', classes, allow_pickle=True)


if __name__ == "__main__":
    # Save the dataset
    store_true_values()
    load_dataset()


