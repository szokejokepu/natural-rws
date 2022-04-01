import os

import matplotlib
import numpy as np

from core.argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook

matplotlib.use('Agg')

from datasets.Dataset import TRAIN, VALIDATION

from core.argo.core.argoLogging import get_logger

tf_logging = get_logger()

SUMMARIES_KEY = "3by3"


class MutualInformationHook(EveryNEpochsTFModelHook):
    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName,
                 dataset_keys=[TRAIN, VALIDATION],
                 plot_offset=0,
                 extra_feed_dict={}
                 ):
        dirName = dirName + '/mutual_information'

        super().__init__(model,
                         period,
                         time_reference,
                         dataset_keys,
                         dirName=dirName,
                         plot_offset=plot_offset,
                         extra_feed_dict=extra_feed_dict)

        tf_logging.info("Create MutualInformationHook to save the bottom layer parameters")

        fileName = "mutual_information"

    """
    Hook for importance sampling estimation
    """

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for MutualInformationHook")
        gen_variables = run_context.session.run(
            [self._model._network._layers[i].gen_distr.trainable_variables for i in range(len(self._model._network._layers))],
            feed_dict={**self._extra_feed_dict}
        )
        rec_variables = run_context.session.run(
            [self._model._network._layers[i].rec_distr.trainable_variables for i in range(len(self._model._network._layers) - 1)],
            feed_dict={**self._extra_feed_dict}
        )
        # print(gen_variables, rec_variables)
        self._save_variables(gen_variables, "gen_variables")
        self._save_variables(rec_variables, "rec_variables")

    def _save_variables(self, variables, name):
        for i, var in enumerate(variables):
            with open(os.path.join(self._dirName, name + "_L{}_B.npy".format(i)), 'wb') as f:
                # print("Saving {}".format(f))
                np.save(f, var[0])
            if len(var)>1:
                with open(os.path.join(self._dirName, name + "_L{}_W.npy".format(i)), 'wb') as f:
                    # print("Saving {}".format(f))
                    np.save(f, var[1])

    def after_run(self, run_context, run_values):
        super(EveryNEpochsTFModelHook, self).after_run(run_context, run_values)

        if self._trigged_for_step:

            # update time and reset triggered for step
            self.update_time()

            self.do_when_triggered(run_context, run_values)
