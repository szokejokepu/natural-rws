from abc import abstractmethod

from ..TFDeepLearningModel import TFDeepLearningModel
import tensorflow as tf

class AbstractGenerativeModel(TFDeepLearningModel):
    default_params= {
        **TFDeepLearningModel.default_params,
        }

    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):
        super().__init__(opts, dirName, check_ops, gpu, seed)
        
        # dictionaries with train, validation and test nodes
        self.x = None

    def create_input_nodes(self, dataset):
        """
        creates input nodes for an autoencoder from the dataset
        """
        
        datasets_nodes, handle, ds_initializers, ds_handles = self.create_datasets_with_handles(dataset)

        # optionally set y
        if len(datasets_nodes)==2:
            self.y = tf.identity(datasets_nodes[1], name="y")
            self.y_one_hot = tf.identity(tf.one_hot(self.y, dataset.n_labels), name="y1h")
        else:
            self.y = None
            self.y_one_hot = None

        raw_x, x_data, x_data_target  = self._unpack_data_nodes(datasets_nodes)

        self.raw_x = raw_x
        self.x = x_data

    def _unpack_data_nodes(self, datasets_nodes):
        # what I will do next, is to move from
        #     dataset_x, perturbed_dataset_x
        # which are obtained from the dataset, to
        #     source_x, target_x
        # based on the value of perturbed_dataset

        raw = datasets_nodes[0][0]
        target = datasets_nodes[0][1]
        source = datasets_nodes[0][2]

        return raw, source, target

    @abstractmethod
    def generate(self, batch_size=1, sess=None):
        pass
