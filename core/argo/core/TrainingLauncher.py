from .ArgoLauncher import ArgoLauncher

import pdb

class TrainingLauncher(ArgoLauncher):
            
    def execute(self, model, opts, **kwargs):
        model.train()

    def initialize(self, model, dataset, config):
        super().initialize(model,dataset, config)
        model.create_session(config)