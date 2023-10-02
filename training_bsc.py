import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import utils

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from datagen_bsc import DataGen


def create_generators(feature, feature_ids, input_shapes, labels):
    dataset = feature_ids.item().get(feature)
    input_shape = input_shapes.item().get(feature)
    params = {'batch_size': 16, 'dim': input_shape, 'n_classes': 4,
              'feature': feature, 'shuffle': True}
    training_gen = DataGen(dataset['train'], labels, **params)
    validation_gen = DataGen(dataset['validation'], labels, **params)
    test_gen = DataGen(dataset['test'], labels, **params)

    return training_gen, validation_gen, test_gen


def define_model(input_shape):
    model = Sequential()
    if input_shape[0] == 128:
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same',
                         input_shape=input_shape))
        model.add(MaxPooling2D((4, 4)))
        model.add(Dropout(0.0))

        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((4, 4)))
        model.add(Dropout(0.0))
        """
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((4, 4)))
        model.add(Dropout(0.0))

        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 4)))
        model.add(Dropout(0.0))"""

        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='sigmoid'))

        optimize = Adam(0.001, 0.9)
        model.compile(optimizer=optimize, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())
    else:
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same',
                         input_shape=input_shape))
        model.add(MaxPooling2D((8, 4)))
        model.add(Dropout(0.0))

        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((4, 4)))
        model.add(Dropout(0.0))

        """model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((4, 4)))
        model.add(Dropout(0.1))

        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 4)))
        model.add(Dropout(0.1))"""

        model.add(Flatten())

        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='sigmoid'))

        optimize = Adam(0.001, 0.9)
        model.compile(optimizer=optimize, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())

    tensorboard_callback = TensorBoard(log_dir='./logs')

    return model, tensorboard_callback


if __name__ == "__main__":

    # Get current working directory
    wd = os.getcwd()

    # Load list of feature types and class labels
    feature_types = ['GCC', 'mel_gcc_phat', 'mel', 'phase_diffs_cossine', 'magspec', 'ilds', 'phase_diff']
    classtypes = ['front', 'right', 'back', 'left']
    input_shapes = np.load('input_shapes.npy', allow_pickle=True)
    feature_ids = np.load('data_id_dict.npy', allow_pickle=True)
    labels = np.load('class_labels.npy', allow_pickle=True).item()

    # Initialize list to store model accuracy for each feature type
    model_accs = []

    # Iterate through each feature type
    for feature in feature_types:
        # Load the dataset for the current feature type

        training_gen, validation_gen, test_gen = create_generators(feature=feature,
                                                                   feature_ids=feature_ids,
                                                                   input_shapes=input_shapes,
                                                                   labels=labels)

        # Define early stopping callback
        callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Define model and tensorboard callback
        model, tensorboard_callback = define_model(input_shape)

        # Train the model
        history = model.fit_generator(generator=training_gen, epochs=100,
                                      use_multiprocessing=False,
                                      validation_data=validation_gen,
                                      verbose=1, callbacks=[callback, tensorboard_callback])

        # Evaluate the model on the test set
        loss, acc = model.evaluate(x=test_gen, verbose=0)

        # Save the model and its accuracy
        utils.save_model(model, feature, acc, wd)

        print(f"Accuracy: {acc:.3g}")
        model_accs.append((feature, acc))

        # Plot learning curve
        utils.plot_results(history, feature)
        # Plot confusion matrix
        cm = utils.get_confusion_matrix(model, test_gen)
        utils.plot_confusion_matrix(cm, feature=feature, classes=classtypes)
        plt.close()
        per_class_acc = utils.calculate_per_class_accuracy(cm)


    # Print the accuracy of each model
    for model in model_accs:
        feature, acc = model
        print(f'{feature} with accuracy: {acc:.3g}')


