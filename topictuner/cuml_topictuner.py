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