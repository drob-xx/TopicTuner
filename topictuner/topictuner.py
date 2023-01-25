from collections import namedtuple
from copy import copy, deepcopy
from random import randrange
from typing import List

import joblib
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from umap import UMAP

from topictuner import BaseHDBSCANTuner

paramPair = namedtuple("paramPair", "cs ss")

class TopicModelTuner(BaseHDBSCANTuner):
    """
    TopicModelTuner (TMT) is a class facilitate the interactive optimization of HDBSCAN's
    min_clust_size and min_sample parameters in the context BERTopic.

    The convenience function wrapBERTopicModel() returns a TMT instance initialized
    with the provided BERTopic model's embedding model, HDBSCAN and UMAP instances and
    parameters.

    Alternatively a new TMT instance can be created from scratch and, in either case,
    once the optimized parameters have been identified, calling getBERTopicModel()
    returns a configured BERTopic instance with the desired parameters.

    TMT is a subclass of BaseHDBSCANTuner. BaseHDBSCANTuner provides the basic HDBSCAN related
    functionality and TMT adds the BERTopic specific pieces.

    TMT does not explicitly provide functionality for testing alternative UMAP parameters or
    HDBSCAN parameters other than min_cluster_size or min_samples. However both the HDBSCAN
    and UMAP models are exposed within the class and can be set to any parameters desired.
    """

    def __init__(
        self,
        embeddings: np.ndarray = None,
        embedding_model=None,
        docs: List[
            str
        ] = None,  
        reducer_model=None,  
        reducer_random_state=None,
        reducer_components: int=5,
        reduced_embeddings=None,
        hdbscan_model=None,  
        viz_reduction=None,
        viz_reducer=None,
        verbose: int = 0, 
    ):  
        """
        Unless explicitly set, TMT Uses the same default param defaults for the embedding model 
        as well as HDBSCAN and UMAP parameters as are used in the BERTopic defaults. 
        
        - 'all-MiniLM-L6-v2' sentence transformer as the default language model embedding.
        - UMAP - metric='cosine', n_neighbors=5, min_dist=0.0
        - HDBSCAN - metric='euclidean', cluster_selection_method='eom', prediction_data=True,
                    min_cluster_size = 10.
        
        Options include:
        
        - Using your own embeddings by setting embeddings after creating an instance
        - Using different UMAP settings or a different dimensional reduction method by setting reducer_model
        - Using different HDBSCAN parameters by setting hdbscan_model
        
        These can be set in the constructor or after instantiation by setting the instance variables 
        before generating the embeddings or reduction.
        
        Unlike BERTopic, TMT has an option for saving both embeddings and the doc corpus - or optionally
        omitting them.
        """

        BaseHDBSCANTuner.__init__(
            self,
            hdbscan_model=hdbscan_model,
            target_vectors=reduced_embeddings,  
            viz_reduction=viz_reduction,
            verbose=verbose,
        )

        # Set the default BERTopic params
        self.hdbscan_params = {
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
            "min_cluster_size": 10,
        }

        self.embeddings = embeddings
        self.reducer_components = reducer_components
        self.docs = docs
        self.viz_reducer = viz_reducer

        if embedding_model == None:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.embedding_model = embedding_model

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
                n_components=self.reducer_components,
                metric="cosine",
                n_neighbors=5,
                min_dist=0.0,
                verbose=self.verbose,
                random_state=self.__reducer_random_state,
            )

    @property
    def reducer_random_state(self):
        return self.__reducer_random_state
    
    @reducer_random_state.setter
    def reducer_random_state(self, rv : np.uint64):
        if self.reducer_model != None :
            self.__reducer_random_state = rv 
            self.reducer_model.random_state = np.uint64(rv)  # added b/c of cuML UMAP bug - https://github.com/rapidsai/cuml/issues/5099#issuecomment-1396382450

    @staticmethod
    def wrapBERTopicModel(BERTopicModel: BERTopic):
        """
        This is a helper function which returns a TMT instance using the values from a BERTopic instance.
        """

        return TopicModelTuner(
            embedding_model=BERTopicModel.embedding_model,
            reducer_model=BERTopicModel.umap_model,
            hdbscan_model=BERTopicModel.hdbscan_model,
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

        reducer_model = deepcopy(self.reducer_model)
        reducer_model.random_state = self.reducer_random_state 

        return BERTopic(
            umap_model=reducer_model, 
            hdbscan_model=hdbscan_model,
            embedding_model=self.embedding_model,            
        )

    def createEmbeddings(self, docs: List[str] = None):
        """
        Create embeddings
        """
        # if self.embeddings != None :
        if np.any(self.embeddings):
            raise AttributeError(
                "Embeddings already created, reset by setting embeddings=None"
            )
        if (np.all(self.docs == None)) and (np.all(docs == None)):
            raise AttributeError("Docs not specified, set docs=")
        
        if np.all(docs)!=None:
            self.docs = docs

        self.embeddings = self.embedding_model.encode(self.docs)

    def reduce(self):
        """
        Reduce dimensionality of the embeddings
        """
        if not np.any(self.embeddings):
            raise AttributeError(
                "No embeddings set, call createEmbeddings() or set embeddings="
            )
        self.reducer_model.fit(self.embeddings)
        self.target_vectors = self.reducer_model.embedding_

    def createVizReduction(self, method="UMAP"):
        """
        Uses the reducer to create a 2D reduction of the embeddings to use for a scatter-plot representation
        """
        if not np.all(self.embeddings):
            raise AttributeError(
                "No embeddings, either set embeddings= or call createEmbeddings()"
            )
        if method == "UMAP":
            self.viz_reducer = deepcopy(self.reducer_model)
            self.viz_reducer.n_components = 2
            self.viz_reducer.fit(self.embeddings)
        else:  # Only TSNE is supported
            self.viz_reducer = TSNE(
                n_components=2,
                verbose=self.verbose,
                random_state=self.__reducer_random_state,
            )
            self.viz_reducer.fit(self.embeddings)
        self.viz_reduction = self.viz_reducer.embedding_

    def getVizCoords(self):
        """
        Returns the X,Y coordinates for use in plotting a visualization of the embeddings.
        """
        if not np.any(self.viz_reduction):
        # if self.viz_reduction == None:
            raise AttributeError(
                "Visualization reduction not performed, call createVizReduction first"
            )
        return self.viz_reducer.embedding_[:, 0], self.viz_reducer.embedding_[:, 1]

    def save(self, fname, save_docs=True, save_embeddings=True, save_viz_reducer=True):
        """
        Saves the TMT object. User can choose whether or not to save docs, embeddings and/or
        the viz reduction
        """

        docs = self.docs
        embeddings = self.embeddings
        viz_reduction = self.viz_reducer
        with open(fname, "wb") as file:
            if not save_docs:
                self.docs = None
            if not save_embeddings:
                self.embeddings = None
            if not save_viz_reducer:
                self.viz_reduction = None
            joblib.dump(self, file)
        self.docs = docs
        self.embeddings = embeddings
        self.viz_reduction = viz_reduction

    @staticmethod
    def load(fname):
        """
        Restore a saved TMT object from disk
        """

        with open(fname, "rb") as file:
            restored = joblib.load(file)
            # restored._paramPair = namedtuple("paramPair", "cs ss")
            return restored
