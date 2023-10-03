import os
import tensorflow as tf
import numpy as np
import utils
from training_bsc import create_generators
# Get current working directory
wd = os.getcwd()

# Load list of feature types and class labels
feature_types = ['GCC', 'mel_gcc_phat', 'mel', 'phase_diffs_cossine', 'magspec', 'ilds', 'phase_diff']
classtypes = ['front', 'right', 'back', 'left']
input_shapes = np.load('input_shapes.npy', allow_pickle=True)
feature_ids = np.load('data_id_dict.npy', allow_pickle=True)
labels = np.load('class_labels.npy', allow_pickle=True).item()

best_acc = 0
best_feature = ''

for feature in feature_types:
    _, _, test_gen = create_generators(feature=feature,
                                       feature_ids=feature_ids,
                                       input_shapes=input_shapes,
                                       labels=labels)

    model_path = os.path.join(wd, f'{feature}_model')
    best_model_path = os.path.join(model_path, 'best')
    acc = np.load(os.path.join(best_model_path, 'accuracy.npy'))
    if acc > best_acc:
        best_acc = acc
        best_feature = feature
    mp = os.path.join(best_model_path, feature)
    model = tf.keras.models.load_model(mp)
    print(f'Feature: {feature}\nAccuracy: {acc}')
    model.summary()
    cm = utils.get_confusion_matrix(model, test_gen)
    per_class_acc = utils.calculate_per_class_accuracy(cm)
    utils.plot_confusion_matrix(cm, classtypes, feature)
    utils.plot_per_class_accuracy(per_class_acc, general_acc=acc, classtypes=classtypes, feature=feature)


print(f'Best feature is: {best_feature}\nWith an accuracy of: {best_acc}')