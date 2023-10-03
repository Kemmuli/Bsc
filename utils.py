import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import itertools
import os
import datetime

from sklearn.metrics import confusion_matrix


classtypes = ['front', 'right', 'back', 'left']


def plot_accuracies(accuracy_dict: dict, save_dir: str) -> None:
    df = pd.DataFrame(list(accuracy_dict.items()), columns=['Feature', 'Accuracy'])
    df = df.sort_values('Accuracy', ascending=False)

    plt.figure(figsize=(10, 8))  # Increased figure size for better visibility
    plt.barh(df['Feature'], df['Accuracy'], color='skyblue')
    plt.xlabel('Accuracy')
    plt.ylabel('Feature')
    plt.title('Feature Accuracies')
    plt.xlim([0, 1])
    x_ticks = np.arange(0, 1.05, 0.05)
    plt.xticks(x_ticks)

    plt.tight_layout()  # Adjust layout to prevent cutting off labels

    # Save the figure to a file

    # Get the current time as a string
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path_png = os.path.join(save_dir, f'Accuracies_{time_str}.png')
    df.to_csv(os.path.join(save_dir, f'Accuracies_{time_str}.csv'), index=False)
    plt.savefig(save_path_png)


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


def plot_confusion_matrix(cm, classes, feature):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {feature}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_confusion_matrix(clf, generator):
    # Get y_true and y_pred from batches
    y_true = []
    y_pred = []

    for batch_x, batch_y in generator:
        batch_pred = clf.predict(batch_x)
        y_true.extend(np.argmax(batch_y, axis=1))
        y_pred.extend(np.argmax(batch_pred, axis=1))

    cm = confusion_matrix(y_true, y_pred)

    return cm


def calculate_per_class_accuracy(cm):
    total_samples_per_class = np.sum(cm, axis=1)
    tp_per_class = np.diagonal(cm)
    per_class_acc = (tp_per_class / total_samples_per_class)

    return per_class_acc


def plot_per_class_accuracy(per_class_acc, general_acc, classtypes, feature=''):
    for i, acc in enumerate(per_class_acc):
        print(f'Accuracy for Class {classtypes[i]} in {feature}: {acc:.2f}%')

    plt.bar(classtypes, per_class_acc)
    plt.axhline(y=general_acc, color='r', linestyle='--', label=f'General Accuracy: {general_acc:.2f}%')

    # Set y-ticks at 0.05 intervals
    y_ticks = np.arange(0, 1.05, 0.05)
    plt.yticks(y_ticks)
    # Set y-axis limits to [0, 1]
    plt.ylim(0, 1)

    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Per-Class Accuracy for {feature}')

    plt.legend()
    plt.show()


def save_model(model, feature, acc, working_directory):
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join(working_directory, f'{feature}_model')
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
