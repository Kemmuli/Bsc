from typing import Tuple, Optional
import os
import numpy as np
import tensorflow as tf

import logging
import utils
from training_bsc import create_generators
from keras.utils import plot_model


# Initialize logging
logging.basicConfig(level=logging.INFO)


def load_and_evaluate_model(
    feature: str,
    feature_ids: np.ndarray,
    input_shapes: np.ndarray,
    labels: dict,
    wd: str,
    classtypes: list
) -> Tuple[Optional[float], Optional[str]]:
    _, _, test_gen = create_generators(
        feature=feature,
        feature_ids=feature_ids,
        input_shapes=input_shapes,
        labels=labels
    )

    model_path = os.path.join(wd, f'{feature}_model')
    best_model_path = os.path.join(model_path, 'best')
    logging.info(f'Current working directory: {wd} '
                 f'\nCurrent model path: {model_path} '
                 f'\nCurrent best_model_path: {best_model_path}')

    # Check if accuracy file exists
    acc_file_path = os.path.join(best_model_path, 'accuracy.npy')
    if not os.path.exists(acc_file_path):
        logging.warning(f"Accuracy file does not exist for feature {feature}. Skipping.")
        return None, None

    acc = np.load(acc_file_path)
    mp = os.path.join(best_model_path, feature)

    # Check if model exists
    if not os.path.exists(mp):
        logging.warning(f"Model does not exist for feature {feature}. Skipping.")
        return None, None

    model = tf.keras.models.load_model(mp)
    logging.info(f'Feature: {feature}\nAccuracy: {acc}')
    model.summary()
    for layer in model.layers:
        layer_config = layer.get_config()
        print(layer_config)

    cm = utils.get_confusion_matrix(model, test_gen)
    per_class_acc = utils.calculate_per_class_accuracy(cm)
    """
    # Generate the plot
    plot_model(
        model,
        to_file=f"{feature}.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=200,
        show_layer_activations=True,
        show_trainable=False,
    )
    """
    utils.plot_confusion_matrix(cm, classtypes, feature)
    utils.plot_per_class_accuracy(per_class_acc, general_acc=acc, classtypes=classtypes, feature=feature)

    return acc, feature


def main() -> None:
    wd: str = os.getcwd()
    feature_types: list = ['GCC', 'mel_gcc_phat', 'mel', 'phase_diffs_cossine',
                           'magspec', 'ilds', 'phase_diff', 'cossine_gcc']
    classtypes: list = ['front', 'right', 'back', 'left']
    input_shapes: np.ndarray = np.load('input_shapes.npy', allow_pickle=True)
    feature_ids: np.ndarray = np.load('data_id_dict.npy', allow_pickle=True)
    labels: dict = np.load('class_labels.npy', allow_pickle=True).item()  # type: ignore

    best_acc: float = 0
    best_feature: str = ''
    feature_accuracies: dict = {}  # To store the accuracies of each feature

    for feature in feature_types:
        acc, feature = load_and_evaluate_model(feature, feature_ids, input_shapes, labels, wd, classtypes)
        if acc:
            feature_accuracies[feature] = acc  # Store the accuracy
            if acc > best_acc:
                best_acc = acc
                best_feature = feature

    logging.info(f'Best feature is: {best_feature}\nWith an accuracy of: {best_acc}')

    utils.plot_accuracies(feature_accuracies, f'./accuracies')


if __name__ == "__main__":
    main()
