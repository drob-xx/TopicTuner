from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml import TSNE
# from topictuner.topictuner import TopicModelTuner
from topictuner import TopicModelTuner
import numpy as np
from typing import List
from copy import copy, deepcopy
from bertopic import BERTopic
from random import randrange
from loguru import logger


class cumlTopicModelTuner(TopicModelTuner):
    """
    classdocs
    """

    def __init__(
        self,
        embeddings: np.ndarray = None,  # pre-generated embeddings
        embedding_model=None,  # set for alternative transformer embedding model
        docs: List[
            str
        ] = None,  # can be set here or when embeddings are created manually
        reducer_model=None,
        reducer_random_state=None,
        reducer_components: int = 5,
        reduced_embeddings=None,
        hdbscan_model=None,
        viz_reduction=None,
        viz_reducer=None,
        verbose: int = 0,
    ):
        """
        Constructor
        """

        self.reducer_model = reducer_model
        
        if reducer_random_state != None:
            self.__reducer_random_state = np.uint64(reducer_random_state)
        else:
            self.__reducer_random_state = np.uint64(randrange(1000000))            

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
            reducer_components=reducer_components,
        )
        
        
        if self.reducer_model == None:
            # Use default BERTopic params
            self.reducer_model = self._getUMAP()

        logger.warning(
            "Due to a bug in the cuML implementation of UMAP the UMAP init parameter is set to 'random'"
        )

    @property
    def reducer_random_state(self):
        return self.__reducer_random_state
    
    @reducer_random_state.setter
    def reducer_random_state(self, rv: np.uint64):
        if self.reducer_model != None:
            self.__reducer_random_state = rv
            self.reducer_model.init = "random"  # added b/c of cuML UMAP bug - https://github.com/rapidsai/cuml/issues/5099#issuecomment-1396382450
            self.reducer_model.random_state = np.uint64(rv)

    # def getHDBSCAN(self, min_cluster_size: int = None, min_samples: int = None):
    #     """
    #     Exposed for convenience, returns a parameterized HDBSCAN model per
    #     the current version in BaseHDBSCANTuner (with the params other than
    #     min_cluster_size and min_samples)
    #     """
    #
    #     min_cluster_size, min_samples = self._check_CS_SS(
    #         min_cluster_size, min_samples, True
    #     )
    #
    #     if self.hdbscan_model is None:
    #         hdbscan_params = deepcopy(self.hdbscan_params)
    #     else:
    #         hdbscan_params = self.hdbscan_model.get_params()
    #
    #     hdbscan_params["min_cluster_size"] = min_cluster_size
    #     hdbscan_params["min_samples"] = min_samples
    #
    #     return self._getHDBSCAN(hdbscan_params)
        #
        #
        # if self.hdbscan_model == None:
        #     hdbscan_params = deepcopy(self.hdbscan_params)
        #     hdbscan_params["min_cluster_size"] = min_cluster_size
        #     hdbscan_params["min_samples"] = min_samples
        #     hdbscan_model = self._getHDBSCAN(hdbscan_params)
        # else:
        #     hdbscan_model = deepcopy(self.hdbscan_model)
        #     hdbscan_model.min_cluster_size = min_cluster_size
        #     hdbscan_model.min_samples = min_samples
        # return deepcopy(hdbscan_model)


    # def getBERTopicModel(self, min_cluster_size: int = None, min_samples: int = None):
    #     """
    #     Returns a BERTopic model with the specified HDBSCAN parameters. The user is left
    #     to specify their chosen best settings after running a series of parameters searches.
    #
    #     This function is necessary because any given HDBSCAN parameters will return somewhat different
    #     results when clustering a given UMAP reduction, simply using the parameters derived from a tuned
    #     TMT model will not produce the same results for a new BERTopic instance.
    #
    #     The reason for this is that BERTopic re-runs UMAP each time fit() is
    #     called. Since different runs of UMAP will have  different characteristics, to recreate
    #     the desired results in the new BERTopic model we need to use the same random seed for the BERTopic's UMAP
    #     as was used in the TMT model.
    #     """
    #
    #     min_cluster_size, min_samples = self._check_CS_SS(
    #         min_cluster_size, min_samples, True
    #     )
    #
    #     hdbscan_params = copy(self.hdbscan_params)
    #     hdbscan_params["min_cluster_size"] = min_cluster_size
    #     hdbscan_params["min_samples"] = min_samples
    #
    #     hdbscan_model = self._getHDBSCAN(hdbscan_params)
    #
    #     reducer_model = deepcopy(self.reducer_model)
    #     reducer_model.random_state = self.reducer_random_state
    #
    #     return BERTopic(
    #         umap_model=reducer_model,
    #         hdbscan_model=hdbscan_model,
    #         embedding_model=self.embedding_model,
    #     )
        
    # def createVizReduction(self, method="UMAP"):
    #     """
    #     Uses the reducer to create a 2D reduction of the embeddings to use for a scatter-plot representation
    #     """
    #     if not np.all(self.embeddings):
    #         raise AttributeError(
    #             "No embeddings, either set embeddings= or call createEmbeddings()"
    #         )
    #     if method == "UMAP":
    #         self.viz_reducer = deepcopy(self.reducer_model)
    #         self.viz_reducer.n_components = 2
    #         self.viz_reducer.fit(self.embeddings)
    #     else:  # Only TSNE is supported
    #         self.viz_reducer = TSNE(
    #             n_components=2,
    #             verbose=self.verbose,
    #             random_state=self.__reducer_random_state,
    #         )
    #         self.viz_reducer.fit(self.embeddings)
    #     self.viz_reduction = self.viz_reducer.embedding_
        
    def _getTSNE(self):
        return TSNE(
                n_components=2,
                verbose=self.verbose,
                random_state=self.__reducer_random_state,
            )

    def _getUMAP(self):
        return UMAP(
                n_components=self.reducer_components,
                metric="cosine",
                n_neighbors=5,
                min_dist=0.0,
                verbose=self.verbose,
                random_state=self.__reducer_random_state,
                init="random", # bug in cuML UMAP requires this work-around for now - could have problematic implications
                hash_input=True, # so that umap_model.embedding_ == output from umap_model.transform() which is what BERTopic uses 
            )
        
    def _getHDBSCAN(self, params):
        return HDBSCAN(**params)
    
    
