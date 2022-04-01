import multiprocessing
import os
import traceback
from time import strftime, gmtime

# this is the class of the consumer
class ProcessTask(multiprocessing.Process):

    def __init__(self, tasks_queue, gpu_allocation_queue, launcher): #, return_queue
        super(ProcessTask, self).__init__()

        print("Creating consumer " + str(os.getpid()))

        self.tasks_queue = tasks_queue
        self.gpu_allocation_queue = gpu_allocation_queue
        self.lock = None
        self.dependencies = None
        self.launcher = launcher

    def set_dependencies(self, dependencies, lock):
        self.dependencies = dependencies
        self.lock = lock

    def run(self):
        while True:
            # this accessed in a safe way by the ProcessTask
            task_and_config = self.tasks_queue.get()

            if task_and_config is None:
                # Poison pill means shutdown
                print("Exiting from consumer " + str(os.getpid()))
                self.tasks_queue.task_done()
                break

            (task, config) = task_and_config

            gpu =  self.gpu_allocation_queue.get()
            print("Running on GPU " + str(gpu))

            executed = False
            #import pdb;pdb.set_trace()
            try:
                executed = self.launcher.task_execute(task, config, gpu,
                                                      self.dependencies,
                                                      self.lock,
                                                      "Consumer " + str(os.getpid()))
            except Exception as exc:
                errfile = self.launcher._launchable.dirName + '/error.log'

                errstream = open(errfile, 'a')
                errtime = strftime("%Y-%m-%d %H:%M:%S\n", gmtime())
                errstream.write("\nError occurred at: " + errtime)
                errstream.write("Failed to execute job: \n"+ str(exc) + "\n")
                trace = traceback.format_exc()
                errstream.write(trace)
                errstream.close()

                print("Failed to execute job: \n"+ str(exc) + "\n")

            self.gpu_allocation_queue.put(gpu)
            self.tasks_queue.task_done()
            print("Consumer PID = " + str(os.getpid()) + " has processed the job")