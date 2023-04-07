from typing import List, Union
import torch
import numpy as np
from sklearn.cluster import AffinityPropagation
from umap import UMAP

from models import Trainer
from update import Update
from aggregators import Aggregator
from .utils import geometric_median,weighted_average

class ClusterAgg(Aggregator):
    def __init__(
        self,
        agg_mode:str = 'average', 
        umap_n_components:int = 10,
    ) -> None:
        """Creates an instance of RFA aggregator that uses geometric mean.

        Args:
            agg_mode (str, optional): _description_. Defaults to 'average'.
            use_clusters (bool, optional): _description_. Defaults to False.
        """

        super(ClusterAgg, self).__init__()
        self.agg_mode = agg_mode

        self.dim_reducer = UMAP(n_components = umap_n_components)
        self.cluster_detector = AffinityPropagation(random_state = 0)

    def combine_points(
        self, 
        points: torch.Tensor,
        weights: torch.Tensor,
    ):
        
        if self.agg_mode == 'average':
            combined = weighted_average(points, weights)
        elif self.agg_mode == 'median':
            combined = geometric_median(points, weights)
        else:
            raise Exception('unknown final aggregation method!')
        return combined

    def aggregate(
        self,
        global_model: Trainer,
        updates:List[Update],
    ) -> dict:
        new_global_state = global_model.get_state()

        points, update_weights, update_delays = [], [], []
        for u in updates:
            points.append(list(u.new_state.values()))
            update_weights.append(u.train_size)
            update_delays.append(u.counter)

        update_weights = torch.tensor(update_weights).to(points[0][0].device)
        final_agg = []

        for component in map(torch.stack,zip(*points)): 
            reduced = self.dim_reducer.fit_transform(component.flatten(1).cpu())
            component_clusters = self.cluster_detector.fit_predict(reduced)
            cluster_medians = torch.stack([
                self.combine_points(
                    component[component_clusters == c], 
                    update_weights[component_clusters == c],
                )
                for c in np.unique(component_clusters)
            ])
            cluster_weights = torch.tensor(np.unique(component_clusters, return_counts = True)[1]).to(cluster_medians.device)
            component_agg = self.combine_points(cluster_medians, cluster_weights)         
            final_agg.append(component_agg)

        for i,k in enumerate(new_global_state.keys()):
            new_global_state[k] = final_agg[i]

        return new_global_state