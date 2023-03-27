# Robust Federated Learning

Class project for Security and Privacy in Machine Learning



## Making virtual environment:

This project is built using python 3.10. Conda is highly recommended.

### Using conda

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

3. Install the rest of pakages through pip

```
pip install numpy pandas scikit-learn matplotlib seaborn
```


### Using python venv
run `python3 -m venv venv` in the root folder of the project.

To activate this environment, run `source venv/bin/activate` in a bash terminal or `venv/bin/Activate.ps1`.

To install the dependencies, run `pip install -r requirements.txt` after activating the environment.

Or you can create a conda environment with the specified python version by running `conda create -n ${ENV_NAME}` and run `conda activate ${ENV_NAME}` to activate it.



## Running the code:

After activating the environment, move into `src` and run 
```
python main.py configurations/{CONFIG_NAME}.yaml
```

Example configuration can be found in [src/configurations/MNIST_MLP_FedAvg/no_straggle.yaml](src/configurations/MNIST_MLP_FedAvg/no_straggle.yaml)


## Related Works

[related_works.md](related_works.md)


## Our Proposed Framework

