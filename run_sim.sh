# python src/main.py -C configurations/MNIST/MLP/FedAvg/no_straggle.yaml
# python src/main.py -C configurations/MNIST/MLP/RFA/whole/no_straggle.yaml

# python src/main.py -C configurations/MNIST/MLP/FedAvg/yes_straggle.yaml
# python src/main.py -C configurations/MNIST/MLP/RFA/whole/yes_straggle.yaml

python src/main.py -C configurations/MNIST/MLP/ClusterAgg/average/no_straggle.yaml
python src/main.py -C configurations/MNIST/MLP/ClusterAgg/average/yes_straggle.yaml

# python src/main.py -C configurations/MNIST/MLP/ClusterAgg/median/no_straggle.yaml
# python src/main.py -C configurations/MNIST/MLP/ClusterAgg/median/yes_straggle.yaml
