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


## Usage

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

```
python src/main.py \
    --agg_config={AGGREGATOR_CONFIG_FILE} \
    --client_config={CLIENTS_CONFIG_FILE} \
    --data_config={DATASET_CONFIG_FILE}
```

You can also refer to the sample scripts in [scripts](scripts) for examples.


## Related Works

[related_works.md](related_works.md)


## Our Proposed Aggregations:


### ClusterAgg





### RandomAgg

