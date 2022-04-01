from abc import abstractmethod

from core.networks.nodes.AbstractHMNode import AbstractHMNode
from core.networks.nodes.AbstractStochasticHMNode import AbstractStochasticHMNode


class AbstractNaturalizableStochasticHMNode(AbstractStochasticHMNode):

    @abstractmethod
    def _apply_fisher_multipliers(self, fisher_calculator, next_layer_distr_probs, previous_layer_sample, grads,
                                  global_step, layer, imp_weights=None):
        pass
