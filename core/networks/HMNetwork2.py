import numpy as np
import tensorflow as tf

from core.argo.core.network.ArgoNetworkWithDefaults import ArgoNetworkWithDefaults
from core.networks.double_nodes.BiasNode import BiasNode
from core.networks.NodeConstructor import NodeConstructor, get_layer_id


class HMNetwork2(ArgoNetworkWithDefaults):
    """
    Network class for managing a VAE network
    """

    # if not present in self._opts, the values specified in default_params will be provided automagically
    # see more in AbstractNetwork.__init__
    default_params = {
        "network_architecture": {
            "layers": [10, 10]
        },
        "samples": 10,
        "pm_one": True,
        "automatic_diff": True,
    }

    def create_id(self):
        layers_ids = "_".join(map(lambda layer: get_layer_id(layer[0], layer[1]), self._layers_spec))

        _id = '-ls' + layers_ids
        _id += '-ad' + ('1' if "automatic_diff" in self._network_architecture and
                               self._network_architecture["automatic_diff"] else '0')

        super_id = super().create_id()
        _id += super_id

        return _id

    def __init__(self, opts, clip_probs=0, pm=True, name="hm_network"):
        """Short summary.

        Args:
            self._opts (dict): parameters of the task.
            name (str): name of the Sonnet module.
        """

        super().__init__(opts, name, None)

        self._network_architecture = self._opts["network_architecture"]
        self._layers_spec = self._network_architecture["layers"]
        self._automatic_diff = self._network_architecture["automatic_diff"] \
            if "automatic_diff" in self._network_architecture else False
        self._clip_probs = clip_probs
        self._pm = pm

        self._regs = self._get_regularizers(self._default_weights_reg, self._default_bias_reg)

        self._init = {"w": self._default_weights_init,
                      "b": self._default_bias_init}
        self._node_constructor = NodeConstructor()

    def _build(self, x, b_size, n_z_samples, b_size_int, n_z_samples_int):
        """
        Args:
            x (tf.tensor): input node.
        """
        self.b_size_int = b_size_int
        self.n_z_samples_int = n_z_samples_int
        self.b_size = b_size
        self.n_z_samples = n_z_samples
        self.mega_batch_size = b_size * n_z_samples
        self._input_orig_shape = x.shape.as_list()[1:]
        self._input_shape = np.prod(self._input_orig_shape)

        self._x = x
        self._x_tiled = tf.tile(self._x, [n_z_samples, 1, 1, 1])

        network_list = [*self._layers_spec]
        with tf.compat.v1.variable_scope("trainable_vars", reuse=True, use_resource=True):
            self._layers = self._node_constructor.construct_layers(self._input_orig_shape,
                                                                   network_list,
                                                                   b_size=self.b_size,
                                                                   n_z_samples=self.n_z_samples,
                                                                   b_size_int=self.b_size_int,
                                                                   n_z_samples_int=self.n_z_samples_int,
                                                                   clip_probs=self._clip_probs,
                                                                   pm=self._pm,
                                                                   initializers=self._init,
                                                                   regularizers=self._regs,
                                                                   automatic_diff=self._automatic_diff)
            self.bias = BiasNode(b_size=self.b_size,
                                 n_z_samples=self.n_z_samples,
                                 b_size_int=self.b_size_int,
                                 n_z_samples_int=self.n_z_samples_int,
                                 size_bottom=np.prod(self._layers[-1].get_output_shape()),
                                 clip_probs=self._clip_probs,
                                 pm=self._pm,
                                 initializers=self._init,
                                 regularizers=self._regs,
                                 automatic_diff=self._automatic_diff)
            self._layers.append(self.bias)

        with tf.compat.v1.variable_scope("WS", reuse=True, use_resource=True):
            self._setup_wake_sleep()

        self.logp_is, self.logp_is_bihm = self._create_importance_sampling_node()

        self.importance_weights, self.importance_weights_bihm = self._get_importance_weights()

        self.x_inferred, self.x_inferred_node = tuple(
            map(self._reshape_to_input_shape, self._generate()))

        self.x_decode, self.x_decode_node = tuple(
            map(self._reshape_to_input_shape, self._decode()))

        self.h_inferred, self._model_latent_mean = self._encode()

    def _setup_wake_sleep(self):
        x_rec_input = self._x_tiled

        def create_lambda(layer, other_func):
            return lambda rec_input: layer(rec_input, other_func)

        constructor_node = (lambda rec_input: self.bias(rec_input))
        for layer in self._layers[:-1][::-1]:
            constructor_node = create_lambda(layer, constructor_node)

        constructor_node(x_rec_input)
        self._prior_samples, self._prior_mean = self.bias.gen()

    def _get_regularizers(self, weights_reg=None, bias_reg=None):
        regs = {}
        if weights_reg:
            regs["w"] = weights_reg
        if bias_reg:
            regs["b"] = bias_reg
        return regs

    def _generate(self):
        x_inferred, x_inferred_node = self._layers[0].gen()
        return x_inferred, x_inferred_node

    def generate(self):
        return self.x_inferred, self.x_inferred_node

    def _decode(self):
        x_inferred, x_inferred_node = self._layers[0].gen()
        return x_inferred, x_inferred_node

    def decode(self):
        return self.x_decode, self.x_decode_node

    def _encode(self):
        return self._layers[-2].rec()

    def encode(self):
        return self.h_inferred, self._model_latent_mean

    def _get_reconstruction_log_joint_likelihood_p(self):
        log_joint_likelihood_p = tf.zeros((self.mega_batch_size))
        for lay in self._layers:
            log_joint_likelihood_p += lay.loss_w

        log_joint_likelihood_p = tf.reduce_mean(log_joint_likelihood_p, axis=0)

        return log_joint_likelihood_p

    def _get_dream_rec_log_joint_conditional_q(self):
        log_joint_likelihood_q = tf.zeros((self.mega_batch_size))
        for lay in self._layers[:-1]:
            log_joint_likelihood_q += lay.loss_s

        log_joint_likelihood_q = tf.reduce_mean(log_joint_likelihood_q, axis=0)

        return log_joint_likelihood_q

    def _create_importance_sampling_node(self):
        ps = tf.zeros((self.mega_batch_size))
        qs = tf.zeros((self.mega_batch_size))

        ps += -self._layers[0].loss_w

        for i in range(len(self._layers) - 1):
            ps += -self._layers[i + 1].loss_w
            qs += -self._layers[i].loss_w_s

        # Reshape
        log_p_all = tf.reshape(ps, [self.n_z_samples, self.b_size])
        log_q_all = tf.reshape(qs, [self.n_z_samples, self.b_size])

        # Approximate log(p(x))
        logp_is = tf.reduce_logsumexp(log_p_all - log_q_all, axis=0)
        logp_is_bihm = tf.reduce_logsumexp((log_p_all - log_q_all)/2, axis=0)

        return logp_is, logp_is_bihm

    def get_prior(self):
        return self.bias, self._prior_samples

    def _get_importance_weights(self):
        with tf.name_scope("reweights"):
            log_probs_p = 0.0
            log_probs_q = 0.0

            log_probs_p += -self._layers[0].loss_w
            for i in range(len(self._layers) - 1):
                log_probs_p += -self._layers[i + 1].loss_w
                log_probs_q += -self._layers[i].loss_w_s

            unnormalized_weight_log = self.get_unnormalized_weigth_log(log_probs_p, log_probs_q)
            unnormalized_weight_log_BiHM = unnormalized_weight_log / tf.constant(2.0)

            unnormalized_weight_log_reduced = tf.reduce_logsumexp(
                tf.reshape(unnormalized_weight_log, [self.n_z_samples, self.b_size]), axis=0)
            unnormalized_weight_log_reduced_BiHM = tf.reduce_logsumexp(
                tf.reshape(unnormalized_weight_log_BiHM, [self.n_z_samples, self.b_size]), axis=0)

            normalized_weights = tf.exp(unnormalized_weight_log - tf.tile(unnormalized_weight_log_reduced,
                                                                          [self.n_z_samples]))
            normalized_weights_BiHM = tf.exp(
                unnormalized_weight_log_BiHM - tf.tile(unnormalized_weight_log_reduced_BiHM,
                                                       [self.n_z_samples]))
        return normalized_weights, normalized_weights_BiHM

    def get_unnormalized_weigth_log(self, log_probs_p, log_probs_q):
        return log_probs_p - log_probs_q

    def _reshape_to_input_shape(self, input):
        return tf.reshape(input, [-1] + self._input_orig_shape)
