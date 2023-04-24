# Robust Federated Learning

Class project for Security and Privacy in Machine Learning

## Implementation Status

### Dataset:
- MNIST
- CIFAR

### Models:
- MLP
- ConvNet

### Aggregators:
- FedAvg
- RFA (Robust Federated Aggregation using geometric mean)
    - Whole RFA
    - Per-Cluster RFA
- ClusterAgg (Using UMAP and Affinity Propagation)
    - Median
    - Average
<!-- - Krum -->


## Installation

### Making virtual environment:

This project is built using python 3.10. Conda is highly recommended.

#### Using conda

1. First create a conda environment with the specified python version (`-y` flag is not necessary, it just means we are saying `y` to everything in the installation step)

```
conda create -n ${ENV_NAME} -y
```

And activate by running

```
conda activate ${ENV_NAME}
```

2. Install torch and torchvision through the conda channel (taken from [here](https://pytorch.org/get-started/locally/))

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

3. Install UMAP
```
conda install umap-learn -c conda-forge
```

4. Install the rest of pakages through pip

```
pip install numpy pandas matplotlib seaborn
```


Alternatively, you can re-use the `environment.yml` file to re-create the conda environment. If you are on a GPU enabled device, you may need to re-install pytorch and torchvision for the specific cuda version. Check [here](https://pytorch.org/get-started/locally/) for more instructions.

To recreate an environment from the conda file, run 

```
conda create -n {ENV_NAME} --file=environment.yml
```


#### Using python venv
run `python3 -m venv venv` in the root folder of the project.

To activate this environment, run `source venv/bin/activate` in a bash terminal or `venv/bin/Activate.ps1`.

To install the dependencies, run `pip install -r requirements.txt` after activating the environment.

Or you can create a conda environment with the specified python version by running `conda create -n ${ENV_NAME}` and run `conda activate ${ENV_NAME}` to activate it.


### Running the code:

After activating the environment, you can run the following command from the root folder of the project to run an experiment.
Notice that the aggregator configuration and base configuration are separated into two files to avoid having to write repetive settings.
You can use our pre-set `.yaml` files in the root directory of make your own to run the experiments.

```
python src/main.py -c={CONFIG_FILE}
```

#### CLI Usage

Running `python src/main.py --help` should display this message.

```
usage: main.py [-h] -c CONFIG_FILE [-s] [-i] [-l STALENESS_LAMBDA] [-g STALENESS_GAMMA] [-e SIM_EPOCH] [-o]

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Location of the configuration file to load. Must contain learning, aggregator and client
                        settings
  -s, --use_staleness   When specified, staleness_lambda and staleness_lambda will be used to weight delayed updates.
  -i, --ignore_delays
  -l STALENESS_LAMBDA, --staleness_lambda STALENESS_LAMBDA
                        Value of lambda to use in staleness weighting. Increasing this value will increase the weight
                        assigned towards on-time updates.
  -g STALENESS_GAMMA, --staleness_gamma STALENESS_GAMMA
                        Value of gamma to use in staleness weighting. Increasing this value will increase the
                        influence of the staleness mechanism.
  -e SIM_EPOCH, --sim_epoch SIM_EPOCH
                        Number of global epochs to use for each simluation.
  -o, --overwrite       When specified, the report file will be overwritten. If not, simulations with existing reports
                        will be skipped.
```

### Viewing the results

The plots are generated from the `report.csv` files under the ouput directory. The notebook used for plot generation can be found in [notebooks/view_results.ipynb](notebooks/view_results.ipynb)