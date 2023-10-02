import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from datagen_bsc import DataGen


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


def plot_results(history, feature):

    plt.figure()
    plt.suptitle(feature)

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], color='blue', label='Train')
    plt.plot(history.history['val_loss'], color='green', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], color='blue', label='Train')
    plt.plot(history.history['val_accuracy'], color='green', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the figure to a file
    save_dir = 'figures'
    # Get the current time as a string
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{feature}_{time_str}.png')
    plt.savefig(save_path)
    #plt.show()
    plt.close()


def save_model(model, feature, acc, wd):
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join(wd, f'{feature}_model')
    model_name = os.path.join(model_path, f'{feature}_{time_str}')
    model.save(model_name)
    best_model_path = os.path.join(model_path, 'best')
    if os.path.exists(best_model_path):
        prev_acc = np.load(os.path.join(best_model_path, 'accuracy.npy'))
        if acc > prev_acc:
            model.save(os.path.join(best_model_path, feature))
            np.save(os.path.join(best_model_path, 'accuracy.npy'), acc)
    else:
        model.save(os.path.join(best_model_path, feature))
        np.save(os.path.join(best_model_path, 'accuracy.npy'), acc)

    return


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
        dataset = feature_ids.item().get(feature)
        input_shape = input_shapes.item().get(feature)
        params = {'batch_size': 16, 'dim': input_shape, 'n_classes': 4,
                  'feature': feature, 'shuffle': True}
        training_gen = DataGen(dataset['train'], labels, **params)
        validation_gen = DataGen(dataset['validation'], labels, **params)
        test_gen = DataGen(dataset['test'], labels, **params)

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
        save_model(model, feature, acc, wd)

        print(f"Accuracy: {acc:.3g}")
        model_accs.append((feature, acc))

        # Plot learning curve
        plot_results(history, feature)

        target_names = []
        for key in training_gen.indexes:
            target_names.append(key)

        # Plot Confusion matrix
        Y_true = np.argmax()

        Y_pred = model.predict_generator(test_gen)
        y_pred = np.argmax(Y_pred, axis=1)

        print(f'Confusion matrix')
        # TODO: Make sure n_classes is what is wanted in 'testing_generator.classes'
        cm = confusion_matrix(test_gen.n_classes, y_pred)
        plot_confusion_matrix(cm, target_names, title='Confusion Matrix')


    # Print the accuracy of each model
    for model in model_accs:
        feature, acc = model
        print(f'{feature} with accuracy: {acc:.3g}')


