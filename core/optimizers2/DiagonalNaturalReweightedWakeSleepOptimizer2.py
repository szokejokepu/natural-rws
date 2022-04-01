from core.optimizers2.NaturalReweightedWakeSleepOptimizerAlternate2 import NaturalReweightedWakeSleepOptimizerAlternate2


class DiagonalNaturalReweightedWakeSleepOptimizer2(NaturalReweightedWakeSleepOptimizerAlternate2):
    def __init__(self, **optimizer_kwargs):
        super(DiagonalNaturalReweightedWakeSleepOptimizer2, self).__init__(**optimizer_kwargs)
        self._diagonal = True
