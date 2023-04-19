#!/bin/sh

for i in {1..7}
do
	python src/main.py --agg_config="configurations/aggregators/random_sample_simple_$i.yml" \
		--client_config="configurations/clients/no_poison_no_straggle.yaml" \
		--data_config="configurations/datasets/mnist_0_1.yaml"
done
