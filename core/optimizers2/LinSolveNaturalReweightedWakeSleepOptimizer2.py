from core.optimizers2.NaturalReweightedWakeSleepOptimizerAlternate2 import NaturalReweightedWakeSleepOptimizerAlternate2


class LinSolveNaturalReweightedWakeSleepOptimizer2(NaturalReweightedWakeSleepOptimizerAlternate2):
    def __init__(self, **optimizer_kwargs):
        super(LinSolveNaturalReweightedWakeSleepOptimizer2, self).__init__(**optimizer_kwargs)
        self._linsolve = True
