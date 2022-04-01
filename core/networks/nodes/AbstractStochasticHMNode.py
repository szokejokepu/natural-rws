from abc import abstractmethod

from core.networks.nodes.AbstractHMNode import AbstractHMNode

WAKE_PHASE = "WAKE"
SLEEP_PHASE = "SLEEP"


def get_all_axis_except_the_first(input):
    return list(range(1, len(input.shape)))


class AbstractStochasticHMNode(AbstractHMNode):

    @property
    @abstractmethod
    def distr(self):
        pass

    def get_loss(self, output):
        return -self.log_probs(output)

    def sample(self, size=()):
        output = self.distr.sample(size)
        return output, self.distr.mean()

    def log_probs(self, samples):
        return self.distr.log_prob(samples)

    def get_probs(self, samples):
        return self.distr.prob(samples)

    def mean(self):
        return self.distr.mean()

    def mode(self):
        return self.distr.mode()

    def shape(self):
        return self.distr.shape()