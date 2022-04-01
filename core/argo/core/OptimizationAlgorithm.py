from abc import ABCMeta, abstractmethod

from .Launchable import Launchable

class OptimizationAlgorithm(Launchable):
    __metaclass__ = ABCMeta

    def __init__(self, opts, dirName, seed=0):
        Launchable.__init__(self, opts, dirName, seed)
        
    @abstractmethod
    def minimize(self):
        pass
