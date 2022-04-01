# Helmholtz Machine
This project is based on licensed through the MIT license:
[argo](https://github.com/rist-ro/argo)

Argo is a library for deep learning algorithms based on TensorFlow 1 and Sonnet. The library allows you to train different models (feed-forwards neural networks for regression and classification problems, autoencoders and variational autoencoders, Bayesian neural networks, Helmholtz machines, etc) by specifying their parameters as well as the network topologies in a configuration file. The models can then be trained in parallel in presence of multiple GPUs. The library is easy to expand for alternative models and training algorithms, as well as for different network topologies. 


## Installation

Requirements (stable):
* tensorflow-datasets      1.2.0    
* tensorflow-estimator     1.14.0   
* tensorflow-gpu           1.14.0   
* tensorflow-metadata      0.14.0   
* tensorflow-probability   0.7.0    
* sonnet 1.32
* torchfile
* seaborn     
* matplotlib
* numpy

Or:
```bash
pip install -r requirements.txt
```

## Before you run 
Create two shortcuts (symbolic link) in the root of this project:
1. "datasets" 
A path to the datasets used for testing
1. "save_path"
A path where the results of the experiments will be saved.

## How to run the code:
To run the examples provided in the framework (or new ones) one can choose between three separate modes of running:

1. single:
Runs a single instance of the configuration file
    ```bash
    python argo/runTraining.py configFile.conf single
    ```
1. pool:
Runs a muliple experiments (if defined) from the configuration file
    ```bash
    python argo/runTraining.py configFile.conf pool
    ```

```bash
python argo/runTraining.py examples/ThreeByThree.conf single
```

How to run the code on multiple GPUs:
```bash
python3 argo/runTrainingHM.py configFile.conf single/pool
 ```
 
See ConfOptions.conf in examples/ for details regarding meaning of
parameters and logging options.

## Plotting
To run plotting example to make a summary on multiple experiments run:

```bash
python core/argo/plotMatplotlibCurves.py plots/basic.conf
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
The library has been developed in the context of the DeepRiemann project, co-funded by the European Regional Development Fund and the Romanian Government
through the Competitiveness Operational Programme 2014-2020, Action 1.1.4, project ID P_37_714, SMIS code 103321, contract no. 136/27.09.2016.

