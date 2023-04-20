python src/main.py \
    --agg_config=configurations/aggregators/random_sample_simple_0_5.yaml \
    --client_config=configurations/clients/no_poison_no_straggle.yaml \
    --learning_config=$1
python src/main.py \
    --agg_config=configurations/aggregators/random_sample_simple_0_5.yaml \
    --client_config=configurations/clients/yes_poison_no_straggle.yaml \
    --learning_config=$1
python src/main.py \
    --agg_config=configurations/aggregators/random_sample_simple_0_5.yaml \
    --client_config=configurations/clients/no_poison_yes_straggle.yaml \
    --learning_config=$1
python src/main.py \
    --agg_config=configurations/aggregators/random_sample_simple_0_5.yaml \
    --client_config=configurations/clients/yes_poison_yes_straggle.yaml \
    --learning_config=$1
