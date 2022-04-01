import numpy as np
from skimage.transform import resize

from .Dataset import NPROCS
from .MNIST import MNIST


class miniMNIST(MNIST):
    default_params = {
        'binary':         0,
        'stochastic':     0,
        'classes':        (),  # no classes means all
        'vect':           False,
        'position_label': True,
        'subsampling':    None,
        'clip_high':      None,
        'clip_low':       None,
        'id_note':        None,
        'pm_one':         True,
    }

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                               'position_label', 'subsampling', 'clip_high', 'clip_low',
                               'data_dir', 'id_note', 'vect']  # all the admitted keys

    classification = True  # true if

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = self._params['binary']

        self._pm_one = self._params['pm_one']

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = [np.asarray(v, dtype=np.float32) if i % 2 == 0 else (np.asarray(v, dtype=np.int32) if v is not None else None) for i,v in
                                              enumerate(self.resize_images())] #x has to be float32 and i should be int


        # Number of classes.
        self._num_classes = 10

        # clip
        clip_low = self._params['clip_low']
        clip_high = self._params['clip_high']
        if (clip_low is not None) or (clip_high is not None):
            m = clip_low if clip_low is not None else 0
            M = clip_high if clip_high is not None else 1
            self._train_set_x = np.clip(self._train_set_x, a_min=m, a_max=M)
            self._validation_set_x = np.clip(self._validation_set_x, a_min=m, a_max=M)
            self._test_set_x = np.clip(self._test_set_x, a_min=m, a_max=M)

        if self._params['vect']:
            self._train_set_x = self._train_set_x.reshape((-1, 196))
            self._validation_set_x = self._validation_set_x.reshape((-1, 196))
            self._test_set_x = self._test_set_x.reshape((-1, 196))

        else:
            self._train_set_x = self._train_set_x.reshape((-1, 14, 14, 1))
            self._validation_set_x = self._validation_set_x.reshape((-1, 14, 14, 1))
            self._test_set_x = self._test_set_x.reshape((-1, 14, 14, 1))

    def resize_to_mini(self, ds, size=(14, 14)):
        import multiprocessing
        from functools import partial
        if self._pm_one:
            ds = (ds + 1)/2.0
        # create a new function that multiplies by 2
        pool = multiprocessing.Pool(NPROCS)
        res = partial(resize, output_shape=[*size, 1])
        ds_resized = pool.map(res, ds)
        pool.close()
        # binary mnist
        ds_resized = np.asarray(ds_resized)
        if self._binary_input == 1:
            ds_resized = np.round(ds_resized)

        if self._pm_one:
            ds_resized = ds_resized * 2.0 - 1
        # np.bincount(np.asarray(ds_resized.flatten()+1, dtype=np.int32))
        return ds_resized

    def resize_images(self):
        self._train_set_x = self.resize_to_mini(self._train_set_x)
        self._validation_set_x = self.resize_to_mini(self._validation_set_x)
        self._test_set_x = self.resize_to_mini(self._test_set_x)

        return self._train_set_x, self._train_set_y, \
               self._validation_set_x, self._validation_set_y, \
               self._test_set_x, self._test_set_y

    def _dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?

        miniMNIST.check_params_impl(params)

        id = 'miniMNIST'

        # binary or continuous
        id_binary = {
            0: '-c',
            1: '-d'}
        id += id_binary[params['binary']]

        # stochastic
        id += '-st' + str(params["stochastic"])

        # subclasses
        #
        if ('classes' in params) and (params['classes'] != ()):
            all_dg = list(range(10))  # list of available digits
            # check the list is a list of digits
            if params['classes'] is not None:
                if params['classes'] is not None:
                    assert (set(params['classes']) <= set(all_dg)), \
                        "classes contains labels not present in MNIST"
            id += ('-sc' + ''.join(map(str, params['classes'].sort())))  # append requested classes to the id

            # if position label is not activated
            if not params['position_label']:
                id += 'npl'

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
        if not params['pm_one']:
            id += '-pm%d' % int(params['pm_one'])

        return id
