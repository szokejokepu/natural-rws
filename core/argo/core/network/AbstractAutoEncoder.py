from abc import abstractmethod

from .AbstractGenerativeModel import AbstractGenerativeModel
import tensorflow as tf

class AbstractAutoEncoder(AbstractGenerativeModel):
    default_params= {
        **AbstractGenerativeModel.default_params,
        "denoising" : 0, # enables denoising
        }
    
    def create_id(self):
        _id = '-d' + str(self._opts["denoising"])

        super_id = super().create_id()
        _id += super_id
        return _id
    
    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):
        super().__init__(opts, dirName, check_ops, gpu, seed)
        self.denoising = opts["denoising"]
        
        # dictionaries with train, validation and test nodes
        self.x = None
        self.x_target = None


    def create_input_nodes(self, dataset):
        """
        creates input nodes for an autoencoder from the dataset

        Sets:
            x, x_target
        """
        
        #datasets_nodes, handle, ds_initializers, ds_handles = self.create_datasets_with_handles(dataset)
        self.create_datasets_with_handles(dataset)

        # optionally set y
        if len(self.datasets_nodes)==2:
            self.y = tf.identity(self.datasets_nodes[1], name="y")
            self.y_one_hot = tf.identity(tf.one_hot(self.y, dataset.n_labels), name="y1h")
        else:
            self.y = None
            self.y_one_hot = None

        # (****) see _unpack_data_nodes(self, datasets_nodes) in AbstractGenerativeModel
        # where the order raw_x=raw_x, x_data=pert_x, and x_data_target=aug_x
        raw_x, x_data, x_data_target = self._unpack_data_nodes(self.datasets_nodes) #, perturbed_dataset)

        self.raw_x = raw_x
        self.x = x_data
        self.x_target = x_data_target

    @abstractmethod
    def encode(self, X, sess = None):
        pass

    @abstractmethod
    def decode(self, Z, sess = None):
        pass

    @abstractmethod
    def reconstruct(self, X):
        pass
