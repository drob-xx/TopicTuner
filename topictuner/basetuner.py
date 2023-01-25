from collections import namedtuple
from random import randrange
from textwrap import wrap
from typing import List
from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm

paramPair = namedtuple("paramPair", "cs ss")

class BaseHDBSCANTuner(object):
    """
    A base class with the HDBSCAN functionality without any references to BERTopic.
    In the future this may be broken out for HDBSCAN tuning outside of BERTopic.
    """
    def __init__(
        self,
        hdbscan_model=None,  #: an HDBSCAN instance
        target_vectors=None,  # vectors to be clustered
        viz_reduction=None,  # 2D reduction of the target_vectors
        verbose: int = 0,
    ):
        self.hdbscan_model = hdbscan_model
        self.target_vectors = target_vectors 
        self.verbose = verbose
        self.hdbscan_params = {}
        self.ResultsDF = None  # A running collection of all the parameters and results if a DataFrame
        self.viz_reduction = viz_reduction
        
        self._paramPair = paramPair # a type 
        self.__bestParams = paramPair(None, None)
        
        if self.hdbscan_model == None:
            self.hdbscan_params = {  # default BERTopic Params
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "prediction_data": True,
                "min_cluster_size": 10,
            }
        else:
            self.hdbscan_params = None
            
    """
    Placeholder
    """

    @property
    def bestParams(self):
        """
        The best min_cluster_size and min_samples values. These are set
        by the user, not automatically. They are used in various places as 
        default values (if set).
        """
        return self.__bestParams
    
    @bestParams.setter
    def bestParams(self, params):
        if not ((type(params) == tuple) or (type(params) == paramPair)) :
            raise ValueError("bestParams must be a tuple or parampair")
        self._check_CS_SS(params[0], params[1])
        self.__bestParams = paramPair(params[0], params[1])

    def getHDBSCAN(self, min_cluster_size: int = None, min_samples: int = None):
        """
        Exposed for convenience, returns a parameterized HDBSCAN model per
        the current version in BaseHDBSCANTuner (with the params other than 
        min_cluster_size and min_samples)
        """
        
        min_cluster_size, min_samples = self._check_CS_SS(min_cluster_size, min_samples, True)

        if self.hdbscan_model == None:
            hdbscan_params = deepcopy(self.hdbscan_params)
            hdbscan_params["min_cluster_size"] = min_cluster_size
            hdbscan_params["min_samples"] = min_samples
            hdbscan_model = HDBSCAN(**hdbscan_params)
        else:
            hdbscan_model = deepcopy(self.hdbscan_model)
            hdbscan_model.min_cluster_size = min_cluster_size
            hdbscan_model.min_samples = min_samples
        return deepcopy(hdbscan_model)

    def runHDBSCAN(self, min_cluster_size: int = None, min_samples: int = None):
        """
        Cluster the target embeddings (these will be the reduced embeddings when
        run as a TMT instance. Per HDBSCAN, min_samples must be more than 0 and less than
        or equal to min_cluster_size.
        """
        min_cluster_size, min_samples = self._check_CS_SS(min_cluster_size, min_samples, True)
        hdbscan_model = self.getHDBSCAN(min_cluster_size, min_samples)
        hdbscan_model.fit_predict(self.target_vectors)
        return hdbscan_model.labels_

    def randomSearch(
        self,
        cluster_size_range: List[int],
        min_samples_pct_range: List[float],
        iters=20,
    ):
        """
        Run a passel (# of iters) of HDBSCAN within a given range of parameters.
        cluster_size_range is a list of ints and min_samples_pct_range is a list of percentage
        values in decimal form e.g. [.1, .25, .50, .75, 1].

        This function will randomly select a min_cluster_size and a min_samples percent
        value from the supplied values. The min_samples percent will be used to calculate
        the min_samples parameter to be used. That value will be rounded up to 1 if less than 1
        and cannot be larger than the selected cluster_size. So if the random cluster size is 10 and 
        the random percent is .75 then the min_cluster_size=10 and min_samples=8.

        All of the search results will be added to ResultsDF and a separate DataFrame containing
        just the results from just this search will be returned by this method.
        """
        
        if len(cluster_size_range) == 0 or len(min_samples_pct_range) == 0 :
            raise ValueError('cluster_size_range and min_samples_pct_range cannot be empty')
        if [0, 1] in cluster_size_range:
            raise ValueError("min_cluster_size must be more than 1")
        for x in min_samples_pct_range:
            if x > 1 :
                raise ValueError('min_samples calculated as percent of cluster_size, must be less than or equal to 1')

        searchParams = self._genRandomSearchParams(
            cluster_size_range, min_samples_pct_range, iters
        )
        return self._runTests(searchParams)

    def pseudoGridSearch(self, cluster_sizes: List[int], min_samples: List[float]):
        """
        Note that this is not a really a grid search. It will search for all cluster_sizes, but only
        search for the percent values of those cluster sizes. For example if cluster_sizes were
        [*range(100,102)] and min_samples [val/100 for val in [*range(10, 101 ,10)]], a clustering for
        each percentage value in min_samples for each value in cluster_sizes would be run
        for a total of 20 clusterings (cluster sizes 100 and 101 * percent values of those for 10%, 20%, 30%,...100%).
        """
        
        if len(cluster_sizes) == 0 or len(min_samples) == 0 :
            raise ValueError('cluster sizes and min_samples cannot be empty')
        
        if [0, 1] in cluster_sizes:
            raise ValueError(
                "min_cluster_size must be more than 1"
            )
        for x in min_samples:
            if x > 1 :
                raise ValueError(
                    "min_samples calculated as percent of cluster_size, must be less than or equal to 1"
            )

        searchParams = self._genGridSearchParams(cluster_sizes, min_samples)
        return self._runTests(searchParams)

    def gridSearch(self, searchRange: List[int]):
        """
        For any n (int) in searchRange, generates all possible min_samples values (1 to n) and performs
        the search.
        """
        if [0,1] in searchRange:
            raise ValueError('Cluster sizes must be > 1')
        if len(searchRange) == 0  :
            raise ValueError('Search range cannot be empty')
        if [0, 1] in searchRange:
            ValueError("min_cluster_size must be more than 1")

        cs_list, ss_list = [], []
        for cs_val in searchRange:
            for ss_val in [*range(1, cs_val + 1)]:
                cs_list.append(cs_val)
                ss_list.append(ss_val)
        return self.simpleSearch(cs_list, ss_list)    
    
    def simpleSearch(self, cluster_sizes: List[int], min_samples: List[int]):
        """
        A clustering for each value in cluster_sizes will be run using the corresponding min_samples
        value. For example if cluster_sizes was [10, 10, 12, 18] and min_samples was [2, 8, 12, 9],
        then searches for the pairs 10,2, 10,8, 12,12, and 18,9 would be performed.
        The len of each list must be the same. Each cluster_size must be > 0 and min_samples must
        be >0 and <= cluster_size.
        """
        if len(cluster_sizes) == 0 or len(min_samples) == 0 :
            raise ValueError('cluster sizes and min_samples cannot be empty')
        
        if len(cluster_sizes) != len(min_samples):
            raise ValueError(
                "Length of cluster sizes and samples sizes lists must match"
            )
        
        if [0,1] in cluster_sizes:
            raise ValueError('Cluster sizes must be > 1')
        
        for x in range(len(cluster_sizes)):
            if (not cluster_sizes[x] > 1) or (min_samples[x] > cluster_sizes[x]):
                raise ValueError(
                    "min_cluster_size must be more than one and min_samples less than or equal to cluster size."
                ) 
        return self._runTests(
            [self._paramPair(cs, ss) for cs, ss in zip(cluster_sizes, min_samples)]
        )

    def visualizeSearch(self, resultsDF: pd.DataFrame = None):
        """
        Returns a plotly fig of a parrallel coordinates graph of the searches performed on this instance.
        """

        if not np.any(resultsDF):
            resultsDF = self.ResultsDF
        return px.parallel_coordinates(
            resultsDF,
            color="number_uncategorized",
            labels={
                "min_cluster_size": "min_cluster_size",
                "min_samples": "min_samples",
                "number_of_clusters": "number_of_clusters",
                "number_uncategorized": "number_uncategorized",
            },
        )

    def summarizeResults(self, summaryDF: pd.DataFrame = None):
        """
        Takes DataFrame of results and returns a DataFrame containing only one record for
        each value of number of clusters. Returns the record with the lowest number of
        uncategorized documents. By default runs against self.ResultsDF - the aggregation of all
        searches run for this model.
        """

        if not np.any(summaryDF):
            summaryDF = self.ResultsDF
        if not np.any(summaryDF):
            raise ValueError(
                "No searches run on this TMT instance, or DF to summarize is None"
            )
        resultSummaryDF = pd.DataFrame()
        for num_clusters in set(summaryDF["number_of_clusters"].unique()):
            resultSummaryDF = pd.concat(
                [
                    resultSummaryDF,
                    summaryDF[summaryDF["number_of_clusters"] == num_clusters]
                    .sort_values(by="number_uncategorized")
                    .iloc[[0]],
                ]
            )
        resultSummaryDF.reset_index(inplace=True, drop=True)
        return resultSummaryDF.sort_values(by=["number_of_clusters"])

    def clearSearches(self):
        """
        A convenience function that resets the saved searches
        """
        self.ResultsDF = None

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
                n_components=2, verbose=self.verbose, random_state=self.__reducer_random_state
            )
            self.viz_reducer.fit(self.embeddings)
        self.viz_reduction = self.viz_reducer.embedding_

    def getVizCoords(self):
        """
        Returns the X,Y coordinates for use in plotting a visualization of the embeddings.
        """
        if self.viz_reducer == None:
            raise AttributeError(
                "Visualization reduction not performed, call createVizReduction first"
            )
        return self.viz_reducer.embedding_[:, 0], self.viz_reducer.embedding_[:, 1]

    def visualizeEmbeddings(
        self, 
        min_cluster_size: int = None, 
        min_samples: int = None, 
        width: int=800, 
        height: int=800, 
        markersize: int=5,
        opacity: float=0.50,
    ):
        """
        Visualize the embeddings, clustered according to the provided HDBSCAN parameters.
        If docs has been set then the first 400 chars of each document will be shown as a
        hover over each data point.

        Returns a plotly fig object
        """
        min_cluster_size, min_samples = self._check_CS_SS(min_cluster_size, min_samples, True)

        fig = go.Figure()
        VizDF = pd.DataFrame()
        VizDF["x"], VizDF["y"] = self.getVizCoords()
        VizDF['topics'] = self.runHDBSCAN(min_cluster_size, min_samples)
        if np.any(self.docs != None):
          wrappedText = ["<br>".join(wrap(txt[:400], width=60)) for txt in self.docs]
          VizDF['wrappedText'] = ["Topic #: "+str(topic)+"<br><br>"+text for topic, text in zip(VizDF['topics'], wrappedText)]
        else :
          VizDF['wrappedText'] = ["Topic #: "+str(topic) for topic in self.runHDBSCAN()]
        
        for topiclabel in set(VizDF['topics']): 
          topicDF = VizDF.loc[VizDF['topics']==topiclabel]
          fig.add_trace(
            go.Scattergl(
              x=topicDF["x"],
              y=topicDF["y"],
              mode='markers',
              name=str(topiclabel)+" ("+str(topicDF.shape[0])+")",
              text=topicDF['wrappedText'],
              hovertemplate = "%{text}<extra></extra>",
          ))
        
        fig.update_traces(marker=dict(size=markersize, 
                                      opacity=opacity,))
        fig.update_layout(width=width, height=height, legend_title_text='Topics')
        
        return fig


    def _genRandomSearchParams(
        self, cluster_size_range, min_samples_pct_range, iters=20
    ):
        searchParams = []
        for _ in range(iters):
            searchParams.append(
                self._returnParamsFromCSandPercent(
                    cluster_size_range[randrange(len(cluster_size_range))],
                    min_samples_pct_range[randrange(len(min_samples_pct_range))],
                )
            )
        return searchParams

    def _genGridSearchParams(self, cluster_sizes, min_samples_pct_range):
        searchParams = []
        for cluster_size in cluster_sizes:
            for min_samples_pct in min_samples_pct_range:
                searchParams.append(
                    self._returnParamsFromCSandPercent(cluster_size, min_samples_pct)
                )
        return searchParams

    def _returnParamsFromCSandPercent(self, cluster_size, min_samples_pct):
        min_samples = int(cluster_size * min_samples_pct)
        if min_samples < 1:
            min_samples = 1
        return self._paramPair(cluster_size, min_samples)

    def _runTests(self, searchParams):
        """
        Runs a passel of HDBSCAN clusterings for searchParams
        """
        if self.verbose > 1:
            results = [
                (params.cs, params.ss, self.runHDBSCAN(params.cs, params.ss))
                for params in tqdm(searchParams)
            ]
        else:
            results = [
                (params.cs, params.ss, self.runHDBSCAN(params.cs, params.ss))
                for params in searchParams
            ]
        RunResultsDF = pd.DataFrame()
        RunResultsDF["min_cluster_size"] = [tupe[0] for tupe in results]
        RunResultsDF["min_samples"] = [tupe[1] for tupe in results]
        RunResultsDF["number_of_clusters"] = [
            len(pd.Series(tupe[2]).value_counts()) for tupe in results
        ]
        uncategorized = []
        for aDict in [pd.Series(tupe[2]).value_counts().to_dict() for tupe in results]:
            if -1 in aDict.keys():
                uncategorized.append(aDict[-1])
            else:
                uncategorized.append(0)
        RunResultsDF["number_uncategorized"] = uncategorized
        self.ResultsDF = pd.concat([self.ResultsDF, RunResultsDF])
        self.ResultsDF.reset_index(inplace=True, drop=True)

        return RunResultsDF

    def _check_CS_SS(self, min_cluster_size: int, min_samples: int, useBestParams: bool = False):
        if min_cluster_size == None :
            if useBestParams and ( self.__bestParams.cs != None) :
                min_cluster_size = self.__bestParams.cs 
                min_samples = self.__bestParams.ss
            else :
                raise ValueError("Cannot set min_cluster_size==None")
        if min_cluster_size == 1:
            raise ValueError("min_cluster_size must be more than 1")
        if min_samples > min_cluster_size:
            raise ValueError("min_samples must be equal or less than min_cluster_size")                

        return min_cluster_size, min_samples


