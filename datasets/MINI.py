"""
Module for managing the IRIS dataset
"""

import numpy as np

from .Dataset import Dataset

DATASET_SIZE = 1000

dataset_ad_hoc = [[1, 1, 1],
                  [1, 1, -1],
                  [1, -1, -1],
                  [-1, -1, -1]]


class MINI(Dataset):
    """
    This class manage the dataset MINI, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """

    default_params = {
        'pm_one': True
    }

    classification = False  # true if

    implemented_params_keys = ['dataName']  # all the admitted keys

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = True

        self._pm_one = params['pm_one']

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_data()

    def _dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        MINI.check_params_impl(params)

        id = 'MINI'
        if not params['pm_one']:
            id += '-pm%d' % int(params['pm_one'])

        return id

    @staticmethod
    def generate_dataset(nr_of_samples):
        # The dataset is defined in the Tutorial on Helmholtz Machines by Kevin G Kirby
        dataset = []

        dataset += dataset_ad_hoc * nr_of_samples

        dataset = np.asarray(dataset)

        len_of_dataset = len(dataset)

        # make them 3*3 pictures
        return dataset.reshape((len_of_dataset, 3, 1, 1))

    def load_data(self):
        # see https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
        miniDS = self.generate_dataset(DATASET_SIZE)

        len_of_dataset = len(miniDS)

        perm = np.random.permutation(len_of_dataset)

        data = miniDS[perm].astype(np.float32)

        data = self._pm_cast(data)

        partition = int(len_of_dataset / 6)
        # extra is ignored, since the images are too simple
        return data[:len_of_dataset - partition], None, \
               data[len_of_dataset - partition:], None, \
               data[len_of_dataset - partition:], None

    @property
    def color_images(self):
        return 0

    @property
    def image_shape(self):
        return (3, 1, 1)  # the last number is the channel

    # Likelihood always returns the patterns as 0,1, not -1,1
    @property
    def likelihood(self):
        patterns_and_likelihoods = [((np.asarray(i) + 1) / 2, 1/4) for i in dataset_ad_hoc]

        return patterns_and_likelihoods

    def _pm_cast(self, lis):
        if self._pm_one:
            return lis
        else:
            return (np.asarray(lis) + 1) / 2
