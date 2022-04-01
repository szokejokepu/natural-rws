import sys
import os

from .ArgoLauncher import ArgoLauncher
import os.path

from .utils.argo_utils import load_class

class OptimizationLauncher(ArgoLauncher):
    
    def __init__(self, conf): # was function_conf
        super().__init__() #function_conf)

    def _load_launchableClass(self, model_params):
        return self._load_model_class(model_params["algorithm"])
    
    def execute(self, model, opts, dataset=None):
        model.minimize(dataset, opts)

    # this should be moved to an intermediate class, since it may be common to different
    # launchers which deals with functions

    def load_data(self, function_conf):

        return Function(function_conf)


class Function():
    
    def __init__(self, function_conf):
        self._function = function_conf["function"]
        self._params_original = function_conf.copy()
        
    @property
    def id(self):
        return self._function

    @property
    def f(self):
        function = load_class(self._function)
        return function
