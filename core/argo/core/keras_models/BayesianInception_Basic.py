import tensorflow as tf
import tensorflow_probability as tfp
from .keras_utils import get_renorm_clipping, get_keras_activation, act_id
from .ArgoKerasModel import ArgoKerasModel
from ..utils.argo_utils import listWithPoints, make_list
from tensorflow.python.util import tf_inspect
class BayesianResNetTf(ArgoKerasModel):
    """Constructs a ResNet model.
    """
    def _id(self):
        _id = 'Incep'
        _id += "_f" + listWithPoints(self._filters)
        _id += "_fl" + str(int(self._flipout))
        _id += "_rn" + str(int(self._renorm))
        _id += "_a" + act_id(self._activation_tuple)
        _id += "_fa" + str(int(self._final_activation))
        pooling_dict = {
            "max": 'M',
            "avg": 'A',
            None: 'N'
             }
        _id += "_p" + pooling_dict[self._pooling_name]
        if self._linear_last is not None:
            _id += "_l" + listWithPoints(self._linear_last)
        return _id
        
    
    
    def __init__(self,
             filters_1x1=[16, 16, 16, 32],
             filters_3x3_reduce=[32, 32, 32, 32],
             filters_3x3=[64, 64, 64, 64],
             filters_5x5_reduce=[16, 16, 16, 32],
             filters_5x5=[32, 32, 32, 32],
             filters_pool_proj=[32, 32, 32, 32],
             strides=[2, 2, 2, 2],
             linear_last=None,
             flipout=True,
             pooling="max",
             renorm=False,
             activation=('relu', {'alpha':0.3}),
             final_activation=True,
             layer_kwargs={},
             layer_kwargs_bayes={}):

        super().__init__(name='incep')

        self._flipout = flipout
        self._filters = filters_1x1
        self._renorm = renorm
        self._pooling_name = pooling
        self._final_activation = final_activation
        

        n_blocks = len(kernels)
        end_activations = [True] * n_blocks

        if linear_last is not None:
            n_lin_layers = len(linear_last)
            linear_last = make_list(linear_last)
            lin_end_activations = [True] *n_lin_layers

            lin_end_activations[-1] = final_activation

        else:
            end_activations[-1] = final_activation
        renorm_clipping = None
        if renorm:
            renorm_clipping = get_renorm_clipping()

        self._linear_last = linear_last
        self._activation_tuple = activation

        pooling_choices = ["max", "avg", None]
        if pooling not in pooling_choices:
            raise ValueError("pooling must be in {:}, instead {:} found.".format(pooling_choices, pooling))
        self._pooling = None
        if pooling == "max":
            self._pooling = tf.keras.layers.MaxPooling2D(2,1)
        elif pooling == "avg":
            self._pooling = tf.keras.layers.AveragePooling2D(2,1)
        self._bn2c = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        self._act2c = get_keras_activation(activation)

        if flipout:
            Conv2D = tfp.layers.Convolution2DFlipout
            Dense = tfp.layers.DenseFlipout
        else:
            Conv2D = tfp.layers.Convolution2DReparameterization
            Dense = tfp.layers.DenseReparameterization

        renorm_clipping = None
        if renorm:
            renorm_clipping = get_renorm_clipping()
        
        self.convstar = Conv2D(
                        16,
                        3,
                        padding='same',
                        strides=1,
                        **layer_kwargs_bayes)

        self.blocks_list = []
       
        for i in range(n_blocks):
            block = Module_Inception(Conv2D,
                            filters_1x1[i],
                            filters_3x3_reduce[i],
                            filters_3x3[i],
                            filters_5x5_reduce[i],
                            filters_5x5[i],
                            filters_pool_proj[i],
                            renorm,
                            pooling,
                            renorm_clipping,
                            activation_name = activation_name,
                            activation_kwargs = activation_kwargs,
                            **layer_kwargs_bayes)
            self.blocks_list.append(block)

        # if logits_size is specified I add an extra Dense layer
        self.blocks_list.append(self._bn2c)
        self.blocks_list.append(self._act2c)
        self.blocks_list.append(self.convstar)
        self.blocks_list.append(self._pooling)
        self.blocks_list.append(tf.keras.layers.Flatten())
        if self._linear_last is not None:
            self.blocks_list.append(tf.keras.layers.Flatten())
            # check that activations are put here
            for ls, act_bool in zip(self._linear_last, lin_end_activations):
                self.blocks_list.append(Dense(ls, **layer_kwargs_bayes))
                if act_bool:
                    self.blocks_list.append(tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping))
                    self.blocks_list.append(get_keras_activation(activation))

        self._layer_call_argspecs = {}
        for layer in self.blocks_list:
            self._layer_call_argspecs[layer.name] = tf_inspect.getfullargspec(layer.call).args

    def _layer_call_kwargs(self, layer, training):
        kwargs = {}
        argspec = self._layer_call_argspecs[layer.name]
        if 'training' in argspec:
            kwargs['training'] = training
    
        return kwargs

    def call(self, inputs, training=False, **extra_kwargs):
        net = inputs
        for layer in self.blocks_list:
            kwargs = self._layer_call_kwargs(layer, training)
            net = layer(net, **kwargs)

        return net

    

class Module_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 filters_1x1,
                 filters_3x3_reduce,
                 filters_3x3,
                 filters_5x5_reduce,
                 filters_5x5,
                 filters_pool_proj,
                 renorm, pooling,renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='Module_Inception')
        def conv2d_(  filters=10,
                      num_row=1,
                      num_col=1,
                      padding='same'):
            return Conv2D(filters,
                        kernel_size=(num_row, num_col),
                        padding=padding,
                        **layer_kwargs_bayes)
        def concatenate():
            return Concatenate(axis=3)

        def batch_bn():
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act():
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(pool_size=2, padding='same'):
            if pooling == "max":
                pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=1,padding=padding)
            elif pooling == "avg":
                pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=1,padding=padding)
            return pool 

        self._conv1a =conv2d_(filters=filters_1x1, num_row=1, num_col=1,padding='same')
        self._batch1a= batch_bn()
        self._act1a= act()
        
        self._conv2a =conv2d_(filters=filters_3x3_reduce, num_row=1, num_col=1,padding='same')
        self._batch2a= batch_bn()
        self._act2a= act()
        self._conv2b =conv2d_(filters=filters_3x3, num_row=3, num_col=3,padding='same')
        self._batch2b= batch_bn()
        self._act2b= act()
        
        self._conv3a =conv2d_(filters=filters_5x5_reduce, num_row=1, num_col=1,padding='same')
        self._batch3a= batch_bn()
        self._act3a= act()
        self._conv3b =conv2d_(filters=filters_5x5, num_row=5, num_col=5,padding='same')
        self._batch3b= batch_bn()
        self._act3b= act()
        
        self._pool_4a=pool_apply(2,padding='same')
        self._conv4a =conv2d_(filters=filters_pool_proj, num_row=1, num_col=1,padding='same')
        self._batch4a= batch_bn()
        self._act4a= act()
        
        self._concatenate_a=concatenate()
        #self._pool_5a=pool_apply(2,strides,padding='same')

    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self._conv1a(input_tensor)
        branch1 = self._act1a(branch1)

        branch2 = self._conv2a(input_tensor)
        branch2 = self._act2a(branch2)
        branch2 = self._conv2b(branch2)
        branch2 = self._act2b(branch2)


        branch3 = self._conv3a(input_tensor)
        branch3 = self._act3a(branch3)
        branch3 = self._conv3b(branch3)
        branch3 = self._act3b(branch3)

        branch4 = self._pool_4a(input_tensor)
        branch4 = self._conv4a(branch4)
        branch4 = self._act4a(branch4)


        branch_T = self._concatenate_a([branch1, branch2, branch3,branch4])
        return branch_T
