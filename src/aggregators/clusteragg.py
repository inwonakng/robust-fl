from typing import List, Union
import torch
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import AffinityPropagation
from hdbscan import HDBSCAN


from models import Trainer
from update import Update
from aggregators import Aggregator
from .utils import geometric_median,weighted_average

REDUCER_MAPPING = dict(
    PCA = PCA,
    UMAP = UMAP
)

DETECTOR_MAPPING = dict(
    HDBSCAN = HDBSCAN,
    AffinityPropagation = AffinityPropagation
)

class ClusterAgg(Aggregator):
    def __init__(
        self,
        reducer_args: dict,
        cluster_detector_args: dict,
        agg_mode:str = 'average', 
    ) -> None:
        """Creates an instance of RFA aggregator that uses geometric mean.

        Args:
            agg_mode (str, optional): _description_. Defaults to 'average'.
            use_clusters (bool, optional): _description_. Defaults to False.
        """

        super(ClusterAgg, self).__init__()
        self.agg_mode = agg_mode

        self.dim_reducer = self._initiate_reducer(**reducer_args)
        self.cluster_detector = self._initiate_cluster_detector(**cluster_detector_args)

    def _initiate_reducer(
        self, 
        reducer_type: str,
        **reducer_args: dict,
    ) -> Union[UMAP, PCA]:
        return REDUCER_MAPPING[reducer_type](**reducer_args)
    
    def _initiate_cluster_detector(
        self,
        detector_type:str,
        **detector_args:dict,
    ) -> Union[HDBSCAN, AffinityPropagation]:
        return DETECTOR_MAPPING[detector_type](**detector_args)

    def combine_weights(
        self, 
        model_weights: torch.Tensor,
        update_weights: torch.Tensor,
    ):
        if self.agg_mode == 'average':
            combined = weighted_average(model_weights, update_weights)
        elif self.agg_mode == 'median':
            combined = geometric_median(model_weights, update_weights)
        else:
            raise Exception('unknown final aggregation method!')
        return combined

    def aggregate(
        self,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()
        model_weights, update_weights = self.parse_updates(updates)
        update_weights = torch.tensor(update_weights).to(model_weights[0][0].device)
        final_agg = []

        for component in map(torch.stack,zip(*model_weights)): 
            reduced = self.dim_reducer.fit_transform(component.flatten(1).cpu())
            component_clusters = self.cluster_detector.fit_predict(reduced)

            cluster_ids = np.unique(component_clusters)
            if len(cluster_ids) == 1 and cluster_ids[0] == -1:
                raise Exception('Cluster detector failed to find any valid clusters!')

            cluster_medians = torch.stack([
                self.combine_weights(
                    component[component_clusters == c], 
                    update_weights[component_clusters == c],
                )
                # ignore clusters labeld as -1
                for c in np.unique(component_clusters) if c > -1
            ])
            cluster_weights = torch.tensor(
                np.unique(
                    component_clusters[component_clusters > -1], 
                    return_counts = True
                )[1]
            ).to(cluster_medians.device)
            component_agg = self.combine_weights(cluster_medians, cluster_weights)         
            final_agg.append(component_agg)

        for i,k in enumerate(new_global_state.keys()):
            new_global_state[k] = final_agg[i]

        return new_global_state
