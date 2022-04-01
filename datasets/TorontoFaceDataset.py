"""
Module for managing TFD faces dataset
"""

import numpy as np

import os.path
import urllib.request

from .Dataset import NPROCS
from .ImageDataset import ImageDataset  ## TODO  Check wiythout point
from scipy.io import loadmat, savemat

import pdb

IMG_ROWS_SMALL = 24
IMG_COLS_SMALL = 24

IMG_ROWS_MEDIUM = 48
IMG_COLS_MEDIUM = 48

IMG_ROWS_LARGE = 96
IMG_COLS_LARGE = 96

SMALL = "SMALL"
MEDIUM = "MEDIUM"
LARGE = "LARGE"


class TorontoFaceDataset(ImageDataset):
    """
    This class manage the dataset TFD faces, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default one. It then return a unique id identifier

    Parameters
    ---------

    params : dict
        dictionary that can contain
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | params Key  | values    | default v |  id short | Short description                                           |
         +=============+===========+===========+===========+=============================================================+
         |   binary    |  0,1      |  0        | "-c", "-d"| load continuous or binary TFD faces                          |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         |  stochastic |  0,1      |  0        |   "-st"   | sample using Bernoulli from the continuous TFD faces after every|
         |             |           |           |           | epoch during training, see IWAE and LVAE papers that claim  |
         |             |           |           |           | this techniques reduces overfitting; this function only     |
         |             |           |           |           | loads continuous TFD faces to be used later                     |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | data_dir    | path str  | None      |           | path of the dataset. In some cases cannot be set            |
         |             |           |           |           | (non binary mnist only)                                     |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | subsampilng |  integer  | None      |"-subsamp" | Reduce the dataset providing 1 data over `subsamping`       |
         |             |           |           |           | samples                                                     |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | clip_low    | bool      | None      | "-clipL"  | clip the dataset to a minimum value (used to avoid zero     |
         |             |           |           |           | gradient)    (-clipLH in case of also high)                 |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | clip_high   | bool      | None      | "-clipH"  | clip the dataset to a max value                             |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | id_note     | string    | ""        | id_note   | Arbitrary string to append to the id                        |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+

    # TODO: train/test split customizable
    """

    default_params = {
        'binary': 0,
        'stochastic': 0,
        'subsampling': None,
        'clip_high': None,
        'clip_low': None,
        'vect': False,
        'id_note': None,
        'size': "MEDIUM"
    }

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = self._params['binary']

        self.data_dir = "dataset_path/TFD"

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if self._binary_input == 0 or (self._binary_input == 0 and self._params['stochastic'] == 1):
            dtype = 'float32'
        else:
            dtype = 'int32'

        self.img_rows, self.img_cols = image_sizes(size=self._params['size'])

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_data(size=self._params['size'], data_dir=self.data_dir,
                                                            dtype=dtype)
        # choose a subset
        if self._params['subsampling']:
            self._train_set_x, self._train_set_y = \
                self.sub_sample(self._train_set_x, self._train_set_y, self._params['subsampling'])
            self._validation_set_x, self._validation_set_y = \
                self.sub_sample(self._validation_set_x, self._validation_set_y, self._params['subsampling'])
            self._test_set_x, self._test_set_y = \
                self.sub_sample(self._test_set_x, self._test_set_y, self._params['subsampling'])

        # clip
        clip_low = self._params['clip_low']
        clip_high = self._params['clip_high']
        if (clip_low is not None) or (clip_high is not None):
            m = clip_low if clip_low is not None else 0
            M = clip_high if clip_high is not None else 1
            self._train_set_x = np.clip(self._train_set_x, a_min=m, a_max=M)
            self._val_set_x = np.clip(self._val_set_x, a_min=m, a_max=M)
            self._test_set_x = np.clip(self._test_set_x, a_min=m, a_max=M)

        if self._params['vect']:
            self._train_set_x = self._train_set_x.reshape((-1, self.img_rows * self.img_cols))
            self._validation_set_x = self._validation_set_x.reshape((-1, self.img_rows * self.img_cols))
            self._test_set_x = self._test_set_x.reshape((-1, self.img_rows * self.img_cols))

        else:
            self._train_set_x = self._train_set_x.reshape((-1, self.img_rows, self.img_cols, 1))
            self._validation_set_x = self._validation_set_x.reshape((-1, self.img_rows, self.img_cols, 1))
            self._test_set_x = self._test_set_x.reshape((-1, self.img_rows, self.img_cols, 1))

    implemented_params_keys = ['dataName', 'binary', 'stochastic',
                               'position_label', 'subsampling', 'clip_high', 'clip_low',
                               'data_dir', 'id_note']  # all the admitted keys

    def _dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?
        TorontoFaceDataset.check_params_impl(params)

        id = 'TFD'

        # binary or continuous
        id_binary = {0: '-c', 1: '-d'}
        id += id_binary[params['binary']]

        # stochastic
        id += '-st' + str(params["stochastic"])

        # subsampling
        if params['subsampling']:
            id += '-ss' + str(params['subsampling'])

        # clip
        # TODO The parameters of clip should be the values to which you clip
        clip_high = False
        if params['clip_high']:
            id += '-cH'
            clip_high = True

        if params['clip_low']:
            id += '-cL'
            if clip_high:
                id += "H"

        # id note (keep last)
        if params['id_note']:
            id += params['id_note']

        return id

    @staticmethod
    def sub_sample(data_set_x, data_set_y, subsampling):
        """
        return a value every "subsampling"

        :param data_set_x
        :param data_set_y
        :param subsampling: integer < dim(data_set)
        :return: dataset_x, dataset_y
        """

        len_train = len(data_set_x)
        reshuf_index_train = np.random.permutation(len_train)
        new_len_train = int(len_train / subsampling)

        data_set_x = data_set_x[reshuf_index_train[:new_len_train]]
        data_set_y = data_set_y[reshuf_index_train[:new_len_train]]

        return data_set_x, data_set_y

    @staticmethod
    def class_filter(data_set_x, data_set_y, classes, position_label):
        """
        return the dataset with labels in the list classes

        :param data_set_x: data
        :param data_set_y: labels
        :param classes:    list of classes
        :param position_label:  list of classes
        :return: (dataset_x, dataset_y) with filtered elemnts not in classes
        """

        ix_mtch_class_train = np.in1d(data_set_y, classes)
        data_set_x = data_set_x[ix_mtch_class_train]
        data_set_y = data_set_y[ix_mtch_class_train]
        if position_label:

            def replace_with_position(label_set, classes):
                label_set_new = np.copy(label_set)
                for ix, class_ in enumerate(classes): label_set_new[label_set == class_] = ix
                return label_set_new

            data_set_y = replace_with_position(data_set_y, classes)

        return data_set_x, data_set_y

    '''
    def get_data_dict(self):

        if not self._binary_input or (self.params['binary'] and not self.params['stochastic']):
            ds["train_set_y"] = self._train_set_y
            ds["test_set_y"] = self._test_set_y

        return ds
    '''

    @staticmethod
    def load_data(size, data_dir, dtype):

        rows, cols = image_sizes(size)
        fileName = data_dir + '/TFD_' + str(rows) + 'x' + str(cols) + '.mat'

        if not os.path.isfile(fileName):
            # True place of datasets
            # http://www.cs.toronto.edu/~jsusskin/TFD/TFD_48x48.mat
            # http://www.cs.toronto.edu/~jsusskin/TFD/TFD_96x96.mat
            # http://www.cs.toronto.edu/~jsusskin/TFD/TFD_info.mat
            origin = (
                'http://www.cs.toronto.edu/~jsusskin/TFD/TFD_info.mat'
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, data_dir + '/TFD_info.mat')

            if size == SMALL:
                origin = (
                    'http://www.cs.toronto.edu/~jsusskin/TFD/TFD_48x48.mat'
                )
                tempfileName = data_dir + '/TFD_' + str(IMG_ROWS_MEDIUM) + 'x' + str(IMG_COLS_MEDIUM) + '.mat'
                print('Downloading data from %s' % origin)
                urllib.request.urlretrieve(origin, tempfileName)

                import cv2

                mat = loadmat(tempfileName, squeeze_me=True, struct_as_record=False)
                images_small = np.asarray(
                    list(map(lambda img: cv2.resize(img, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC), mat['images'])))
                dic = {"images": images_small, "labs_ex": mat["labs_ex"], "labs_id": mat["labs_id"],
                       "folds": mat["folds"]}
                savemat(fileName, dic)
            elif size == MEDIUM:
                origin = (
                    'http://www.cs.toronto.edu/~jsusskin/TFD/TFD_48x48.mat'
                )
                print('Downloading data from %s' % origin)
                urllib.request.urlretrieve(origin, fileName)
            elif size == LARGE:
                origin = (
                    'http://www.cs.toronto.edu/~jsusskin/TFD/TFD_96x96.mat'
                )
                print('Downloading data from %s' % origin)
                urllib.request.urlretrieve(origin, fileName)
            else:
                raise Exception("No such size for TFD")

        # this needs to be rewritten based on the warning you get in TF
        mat = loadmat(fileName, squeeze_me=True, struct_as_record=False)

        X = mat["images"].reshape((-1, rows, cols))
        X = X / 255. * 2 - 1

        # labels = ['labs_ex', 'labs_id', 'folds'] if needed
        Y = mat["labs_ex"]

        from sklearn.model_selection import train_test_split

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.33, shuffle=True)

        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)
        print("Y_train shape:", Y_train.shape)
        print("Y_val shape:", Y_val.shape)
        print("Y_test shape:", Y_test.shape)

        X_train = X_train.astype(dtype)
        X_val = X_val.astype(dtype)
        X_test = X_test.astype(dtype)

        n_pixels = rows * cols
        train_set_x = X_train.reshape((len(X_train), n_pixels))
        val_set_x = X_val.reshape((len(X_val), n_pixels))
        test_set_x = X_test.reshape((len(X_test), n_pixels))

        return train_set_x, None, \
               val_set_x, None, \
               test_set_x, None

    @property
    def input_size(self):
        return self.img_rows * self.img_cols

    @property
    def output_size(self):
        pass

    @property
    def color_images(self):
        return 0

    @property
    def image_shape(self):
        return (self.img_rows, self.img_cols, 1)  # 1 is the number of channels


def image_sizes(size="MEDIUM"):
    if size == "MEDIUM":
        return IMG_ROWS_MEDIUM, IMG_COLS_MEDIUM
    elif size == "SMALL":
        return IMG_ROWS_SMALL, IMG_COLS_SMALL
    elif size == "LARGE":
        return IMG_ROWS_LARGE, IMG_COLS_LARGE
