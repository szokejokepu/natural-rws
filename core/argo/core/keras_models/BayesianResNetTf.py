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
        _id = 'Res'
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
             filters=[16, 16, 32, 32, 32],  # [32, 64, 64, 128, 128],
             kernels=[3, 3, 3, 3, 3],
             strides=[2, 2, 2, 2, 2],
             linear_last=None,
             flipout=True,
             pooling="max",
             renorm=False,
             activation=('relu', {'alpha':0.3}),
             final_activation=True,
             layer_kwargs={},
             layer_kwargs_bayes={}):

        super().__init__(name='resNTf')

        self._flipout = flipout
        self._filters = filters
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
            self._pooling = tf.keras.layers.MaxPooling2D(3,2)
            self._poolinga = tf.keras.layers.GlobalAveragePooling2D()
        elif pooling == "avg":
            self._pooling = tf.keras.layers.AveragePooling2D(3,2)
            self._poolinga = tf.keras.layers.GlobalAveragePooling2D()
#        self._bn2c = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
#        self._act2c = get_keras_activation(activation)
#         self._bn2d = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
#         self._act2d = get_keras_activation(activation)
        self._bn2e = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)

#         self._act2e = get_keras_activation(activation)

        if flipout:
            Conv2D = tfp.layers.Convolution2DFlipout
            Dense = tfp.layers.DenseFlipout
        else:
            Conv2D = tfp.layers.Convolution2DReparameterization
            Dense = tfp.layers.DenseReparameterization

        renorm_clipping = None
        if renorm:
            renorm_clipping = get_renorm_clipping()
        
        self.convstar1 = Conv2D(
                        32,
                        7,
                        padding='same',
                        strides=2,
                        **layer_kwargs_bayes)
#         self.convstar2 = Conv2D(
#                         64,
#                         3,
#                         padding='same',
#                         strides=1,
#                         **layer_kwargs_bayes)

        self.blocks_list = []
        self.blocks_list.append(self.convstar1)
        self.blocks_list.append(self._bn2e)
        self.blocks_list.append(self._pooling)
        for i in range(n_blocks):
            block = ResnetBlock(Conv2D,
                            filters[i],
                            kernels[i],
                            strides[i],
                            renorm,
                            renorm_clipping,
                            activation,
                            **layer_kwargs_bayes)
            self.blocks_list.append(block)

        # if logits_size is specified I add an extra Dense layer
        self.blocks_list.append(self._poolinga)
        #self.blocks_list.append(self._bn2c)
        #self.blocks_list.append(self._act2c)
#         self.blocks_list.append(self.convstar1)
#         self.blocks_list.append(self._bn2d)
#         self.blocks_list.append(self._act2d)
#         self.blocks_list.append(self.convstar2)
#         self.blocks_list.append(self._bn2e)
#         self.blocks_list.append(self._act2e)
        
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





class ResnetBlock(tf.keras.Model):
    def __init__(self, Conv2D, filters, kernel, stride,
                 renorm, renorm_clipping,
                 activation_tuple= ('relu', {}),
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='resnet_block')
        self.stride=stride
        self.filters=filters
        
        self.conv2a = Conv2D(
                        self.filters,
                        kernel,
                        padding='same',
                        strides=self.stride,
                        **layer_kwargs_bayes)

        self.bn2a = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        self._act2a = get_keras_activation(activation_tuple)

        self.conv2b = Conv2D(
                        self.filters,
                        kernel,
                        padding='same',
                        **layer_kwargs_bayes)

        self.bn2b = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        self._act2b = get_keras_activation(activation_tuple)
        self.projection_shortcut = Conv2D(self.filters,
                             1,
                             padding='valid',
                             strides=self.stride,
                             **layer_kwargs_bayes)

        #self._act2c = None
        #if final_activation:
         #   self._act2c = get_keras_activation(activation_tuple)
        #self.bn2c = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)

    def call(self, input_tensor, training=False, **extra_kwargs):
        x = input_tensor#
        x = self.bn2a(x, training=training)
        x = self._act2a(x)
        if self.stride != 1 or self.filters != x.shape[1]:
            shortcut = self.projection_shortcut(x)
        else:
            shortcut = x
        #x = self.bn2a(x, training=training)
        #x = self._act2a(x)
        x = self.conv2a(x)
        x = self.bn2b(x, training=training)
        x = self._act2b(x)
        x = self.conv2b(x)
        #x = self.bn2c(x, training=training)
        output = x + shortcut
        return output
#     def call(self, input_tensor, training=False, **extra_kwargs):#emod
#         x1 = self.bn2a(input_tensor, training=training)
#         x = self._act2a(x1)
#         x = self.conv2b(x)
#         x = self.bn2b(x, training=training) 
#         x = self._act2a(x)
#         x = self.conv2a(x) 
#        # x = self._act2b(x)
       
#         if self.stride != 1 or self.filters != x1.shape[1]:
#             shortcut = self.projection_shortcut(x1)
#         else:
#             shortcut = input_tensor
#         output = x + shortcut
       
#         return output
