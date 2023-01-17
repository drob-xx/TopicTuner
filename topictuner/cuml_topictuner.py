from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from topictuner.topictuner import TopicModelTuner
import numpy as np
from typing import List


class cumlTopicModelTuner(TopicModelTuner):
    '''
    classdocs
    '''


    def __init__(
        self,
        embeddings: np.ndarray = None,  # pre-generated embeddings
        embedding_model=None,  # set for alternative transformer embedding model
        docs: List[
            str
        ] = None,  # can be set here or when embeddings are created manually
        reducer_random_state=None,
        reducer_components=5,
        reduced_embeddings=None,
        viz_reduction=None,
        verbose: int = 0, 
    ):  
        '''
        Constructor
        '''
        hdbscan_model = HDBSCAN()
        umap_model = UMAP(n_neighbors=15,
                         n_components=5,
                         min_dist=0.0,
                         metric='cosine')

        TopicModelTuner.__init__(
            self,
            embeddings=embeddings,
            embedding_model=embedding_model,
            docs=docs,
            reducer_random_state=reducer_random_state,
            reduced_embeddings=reduced_embeddings,
            viz_reduction=viz_reduction,
            verbose=verbose,
            reducer_model=umap_model,
            hdbscan_model=hdbscan_model,
        )
        
    def getBERTopicModel(self, min_cluster_size: int = None, min_samples: int = None):
        """
        Returns a BERTopic model with the specified HDBSCAN parameters. The user is left
        to specify their chosen best settings after running a series of parameters searches.

        This function is necessary because any given HDBSCAN parameters will return somewhat different
        results when clustering a given UMAP reduction, simply using the parameters derived from a tuned
        TMT model will not produce the same results for a new BERTopic instance.

        The reason for this is that BERTopic re-runs UMAP each time fit() is
        called. Since different runs of UMAP will have  different characteristics, to recreate
        the desired results in the new BERTopic model we need to use the same random seed for the BERTopic's UMAP
        as was used in the TMT model.
        """

        min_cluster_size, min_samples = self._check_CS_SS(min_cluster_size, min_samples, True)

        hdbscan_params = deepcopy(self.hdbscan_params)
        hdbscan_params["min_cluster_size"] = min_cluster_size
        hdbscan_params["min_samples"] = min_samples

        hdbscan_model = HDBSCAN(**hdbscan_params)

        return BERTopic(
            umap_model=deepcopy(self.reducer_model), hdbscan_model=hdbscan_model
        )