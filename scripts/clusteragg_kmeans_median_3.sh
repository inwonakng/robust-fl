python src/main.py \
    --agg_config=configurations/aggregators/clusteragg_kmeans_median_3.yaml \
    --client_config=configurations/clients/no_poison_no_straggle.yaml \
    --learning_config=$1
python src/main.py \
    --agg_config=configurations/aggregators/clusteragg_kmeans_median_3.yaml \
    --client_config=configurations/clients/yes_poison_no_straggle.yaml \
    --learning_config=$1
python src/main.py \
    --agg_config=configurations/aggregators/clusteragg_kmeans_median_3.yaml \
    --client_config=configurations/clients/no_poison_yes_straggle.yaml \
    --learning_config=$1
python src/main.py \
    --agg_config=configurations/aggregators/clusteragg_kmeans_median_3.yaml \
    --client_config=configurations/clients/yes_poison_yes_straggle.yaml \
    --learning_config=$1