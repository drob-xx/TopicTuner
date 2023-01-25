from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from topictuner.topictuner import TopicModelTuner
import numpy as np
from typing import List
from copy import copy, deepcopy
from bertopic import BERTopic
from random import randrange
from loguru import logger



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
        reducer_model=None,
        reducer_random_state=None,
        reducer_components: int=5,
        reduced_embeddings=None,
        hdbscan_model=None,
        viz_reduction=None,
        viz_reducer=None,
        verbose: int = 0, 
    ):  
        '''
        Constructor
        '''
        
        self.reducer_model = (
            reducer_model  
        )        
        
        if reducer_random_state != None:
            self.__reducer_random_state = np.uint64(reducer_random_state)
        else:
            self.__reducer_random_state = np.uint64(randrange(1000000))
        
        if self.reducer_model == None:  
            # Use default BERTopic params
            self.reducer_model = UMAP(
                n_components=reducer_components,
                metric="cosine",
                n_neighbors=5,
                min_dist=0.0,
                verbose=verbose,
                random_state=self.__reducer_random_state,
                init="random"
            )
        
        TopicModelTuner.__init__(
            self,
            embeddings=embeddings,
            embedding_model=embedding_model,
            docs=docs,
            reducer_random_state=self.__reducer_random_state,
            reducer_model=self.reducer_model,
            reduced_embeddings=reduced_embeddings,
            viz_reduction=viz_reduction,
            viz_reducer=viz_reducer,
            verbose=verbose,
            hdbscan_model=hdbscan_model,
            reducer_components=reducer_components
        )
        
        logger.warning("Due to a bug in the cuML implementation of UMAP the UMAP init parameter is set to \'random\'")

    @property
    def reducer_random_state(self):
        return self.__reducer_random_state
            
    @reducer_random_state.setter 
    def reducer_random_state(self, rv : np.uint64):
        if self.reducer_model != None :
            self.__reducer_random_state = rv
            self.reducer_model.init = 'random'  # added b/c of cuML UMAP bug - https://github.com/rapidsai/cuml/issues/5099#issuecomment-1396382450
            self.reducer_model.random_state = np.uint64(rv)
        
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
        
        reducer_model = deepcopy(self.reducer_model)
        reducer_model.random_state = self.reducer_random_state 


        return BERTopic(
            umap_model=reducer_model,
            hdbscan_model=hdbscan_model,
            embedding_model=self.embedding_model,  
        )