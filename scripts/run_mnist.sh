# bash scripts/fedavg.sh configurations/datasets/mnist_0_1.yaml
bash scripts/rfa_whole.sh configurations/datasets/mnist_0_1.yaml
bash scripts/rfa_component.sh configurations/datasets/mnist_0_1.yaml
bash scripts/clusteragg_average.sh configurations/datasets/mnist_0_1.yaml
bash scripts/clusteragg_median.sh configurations/datasets/mnist_0_1.yaml