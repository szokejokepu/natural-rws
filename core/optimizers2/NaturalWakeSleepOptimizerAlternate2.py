from core.networks.nodes.FisherCalculator import FisherCalculator
from core.optimizers2.NaturalWakeSleepOptimizer2 import NaturalWakeSleepOptimizer2


class NaturalWakeSleepOptimizerAlternate2(NaturalWakeSleepOptimizer2):

    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)

        self._k_step_update = optimizer_kwargs["k_step_update"]
        self._saves = {}
