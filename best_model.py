import os
import tensorflow as tf

# Get current working directory
import numpy as np

wd = os.getcwd()

feature_types = ['GCC', 'mel_gcc_phat', 'mel', 'phase_diffs_cossine', 'magspec', 'ilds', 'phase_diff']
best_acc = 0
best_feature = ''

for feature in feature_types:
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


print(f'Best feature is: {best_feature}\nWith an accuracy of: {best_acc}')