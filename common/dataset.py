# Custom Gesture classifier
# Seung-Chan Kim

import numpy as np
import glob
import matplotlib.pyplot as plt

from tqdm import tqdm
#from data_util import *
#from sklearn import preprocessing

class dataset(object):
    def __init__(self,
                 images,
                 labels,
                 nb_per_classes=()
                  ):#samples_in_use = -1
        self._images = images
        self._labels = labels
        self._nb_per_classes = nb_per_classes
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]
        #self._samples_in_use = samples_in_use
        self._idx_figure = 0

        self._mean = 0
        self._scale = 0
        a = images.shape # <type 'tuple'>: (2100, 300, 5)
        #b = images.shape[2]

        self._dim = images.shape[2]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def desc(self):
        print ('======================')
        print ('Data : ', self._images.shape)
        print ('Label : ', self._labels.shape)
        print ('======================')

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def nb_tot_samples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def getMinMax(self, idx):
        a = self.images[idx:idx + 1] # shape : <type 'tuple'>: (1, 30, 3)
        #
        maxv = np.max(a)
        minv = np.min(a)
        return minv, maxv

    

    def next_batch(self, batch_size, shuffle=True, return_stats=False):
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            if return_stats:
                return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0), np.nanstd(np.concatenate((images_rest_part, images_new_part), axis=0), axis=0)
            else:
                return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            if return_stats:
                return self._images[start:end], self._labels[start:end], np.nanstd(self._images[start:end], axis=0)
            else:

                return self._images[start:end], self._labels[start:end]


