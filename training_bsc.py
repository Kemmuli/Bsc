import numpy as np
import matplotlib.pyplot as plt
import glob
import json

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from datagen_bsc import DataGen


def define_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (2, 4), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (2, 4), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(4, activation='sigmoid'))

    optimize = Adam(0.001, 0.9)
    model.compile(optimizer=optimize, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    return model


def plot_results(history, feature, accuracy):

    plt.suptitle(feature)

    plt.subplot(211)
    plt.title('Loss, Green = validation, Blue = training')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='green', label='test')

    plt.subplot(212)
    plt.title('Accuracy, Green = validation, Blue = training')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='green', label='test')

    plt.show()


if __name__ == "__main__":

    feature_types = ['magspec', 'GCC', 'mel', 'ilds', 'phase_diffs_cossine', 'phase_diff']
    classtypes = ['front', 'right', 'back', 'left']
    input_shapes = np.load('input_shapes.npy', allow_pickle=True)
    feature_ids = np.load('data_id_dict.npy', allow_pickle=True)
    labels = np.load('class_labels.npy', allow_pickle=True).item()

    for feature in feature_types:
        dataset = feature_ids.item().get(feature)
        input_shape = input_shapes.item().get(feature)
        params = {'batch_size': 16, 'dim': input_shape, 'n_classes': 4,
                  'feature': feature, 'shuffle': True}
        training_gen = DataGen(dataset['train'], labels, **params)
        validation_gen = DataGen(dataset['validation'], labels, **params)
        test_gen = DataGen(dataset['test'], labels, **params)

        callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model = define_model(input_shape)
        history = model.fit_generator(generator=training_gen, epochs=100,
                                        use_multiprocessing=False,
                                        validation_data=validation_gen,
                                        verbose=1, callbacks=[callback])
        loss, acc = model.evaluate(x=test_gen, verbose=0)
        savename = './' + feature + '_model'
        model.save(savename)
        print(f"Accuracy: {acc:.3g}")
        # graph of learning
        plot_results(history, feature, acc)