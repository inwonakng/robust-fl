#/usr/bin/bash

learning_config=configurations/learning/mnist_0_1.yaml

aggregator_settings=(
    "fedavg"
    "rfa_whole"
    "rfa_component"
    "random_sample_simple_0_3"
    "random_sample_simple_0_5"
    "random_sample_simple_0_7"
    "clusteragg_kmeans_3_average"
    "clusteragg_kmeans_5_average"
    "clusteragg_kmeans_8_average"
    "clusteragg_kmeans_3_median"
    "clusteragg_kmeans_5_median"
    "clusteragg_kmeans_8_median"
    "clusteragg_agglo_average"
    "clusteragg_agglo_median"
    "clusteragg_meanshift_average"
    "clusteragg_meanshift_median"
)

client_settings=(
    "no_poison_no_straggle"
    "yes_poison_no_straggle"
    "no_poison_yes_straggle"
    "yes_poison_yes_straggle"
)

for agg_config in "${aggregator_settings[@]}"
    do 
    for client_config in "${client_settings[@]}"
        do
            python src/main.py \
                --agg_config="configurations/aggregators/${agg_config}_staleness.yaml" \
                --client_config="configurations/clients/${client_config}.yaml" \
                --learning_config=$learning_config
        done
    done