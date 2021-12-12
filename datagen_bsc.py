import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


class DataGen(Sequence):
    """
    Attributes
    ----------
    ID_list : ndarray
        a list of the feature filenames
    labels : ndarray
        corresponding labels for ID_list
    batch_size : int
        size of the batch
    dim : tuple
        dimensions of the input
    n_classes : int
        number of directions
    shuffle : bool
        boolean to determine whether to shuffle the dataset between epochs

    Methods
    -------
    on_epoch_end()
        determines the action on epochs end
   """

    def __init__(self, ID_list, labels, batch_size, dim, n_classes, feature,
                 shuffle=True):
        """
        Parameters
        ----------
        ID_list : ndarray
            a list of the feature filenames
        labels : ndarray
            corresponding labels for ID_list
        batch_size : int
        dim : tuple
            dimensions of the input
        n_classes : int
            number of directions
        feature : str
            name of the feature whose data is produced
        shuffle : bool
            boolean to determine whether to shuffle the dataset between epochs
        """

        self.ID_list = ID_list
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.feature = feature
        self.shuffle = shuffle
        self.on_epoch_end()

    def __iter__(self):
        return self

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ID_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ID_list_tmp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(ID_list_tmp):
            # Store sample
            X[i, ] = np.load('./direction-dataset/audio/' + ID + '_'
                             + self.feature + '.npy')
            # Store class
            y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.ID_list) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        ID_list_tmp = [self.ID_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(ID_list_tmp)

        return X, y

    def input_shape(self):
        return self.dim
