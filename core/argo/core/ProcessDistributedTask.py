import multiprocessing
import os

# this is the class of the consumer when we run on distributed envorinments, such as the UBB cluster
class ProcessDistributedTask(multiprocessing.Process):

    def __init__(self, tasks_queue, gpu_allocation_queue, launcher, node_number, ip): #, return_queue
        super(ProcessDistributedTask, self).__init__()

        print("Creating consumer from " + str(os.getpid()))

        self.tasks_queue = tasks_queue
        self.gpu_allocation_queue = gpu_allocation_queue
        #self.return_queue = return_queue

        #self.task_execute = task_execute
        self.launcher = launcher

        self.node_number = node_number
        self.ip = ip

    def run(self):
        while True:
            task_and_config = self.tasks_queue.get()

            if task_and_config is None:
                # Poison pill means shutdown
                print("Exiting from consumer " + str(os.getpid()))
                self.tasks_queue.task_done()
                break

            (task, config) = task_and_config

            gpu =  self.gpu_allocation_queue.get()
            print("Planned to run on GPU " + str(gpu) + " at " + self.ip)

            # replace in config specific inforation about the node/GPU
            config["nodes"] = [config["nodes"][self.node_number]]
            config["nodes"][0]["used_GPUs"] = {gpu}
            config["nodes"][0]["cores_per_GPU"] = 1

            #self.task_execute(task, config, gpu, "Consumer " + str(os.getpid()))
            self.launcher.task_execute_remote(task, config, gpu, self.ip, "Consumer " + str(os.getpid()))
            self.gpu_allocation_queue.put(gpu)

            self.tasks_queue.task_done()
            print("Consumer done PID = " + str(os.getpid()))