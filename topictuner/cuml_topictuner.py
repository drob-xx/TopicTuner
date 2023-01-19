from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from topictuner.topictuner import TopicModelTuner
import numpy as np
from typing import List
from copy import copy, deepcopy
from bertopic import BERTopic


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
                         metric='cosine',
                         init='random') # added b/c of cuML UMAP bug - https://github.com/rapidsai/cuml/issues/5099#issuecomment-1396382450 

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
            reducer_components=reducer_components
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

        hdbscan_params = copy(self.hdbscan_params)
        hdbscan_params["min_cluster_size"] = min_cluster_size
        hdbscan_params["min_samples"] = min_samples

        hdbscan_model = HDBSCAN(**hdbscan_params)

        return BERTopic(
            umap_model=deepcopy(self.reducer_model),
            hdbscan_model=hdbscan_model,
            embedding_model=self.embedding_model,  
        )