from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

from copy import copy
from random import randrange
from collections import namedtuple
from tqdm.notebook import tqdm
from textwrap import wrap
from typing import List
import numpy as np
import joblib
import pandas as pd
import plotly.express as px


class TopicModelTuner(object):
    '''
    TopicModelTuner (TMT) is a tool to optimize the min_clust_size and min_sample parameters 
    of HDBSCAN. It is a compliment to the BERTopic default configuration and assumes that 
    HDBSCAN is being used to cluster UMAP reductions of embeddings. 
    
    The convenience function TMT.wrapBERTopicModel() returns a TMT instance initialized 
    with the provided BERTopic model's embedding model, HDBSCAN and UMAP instances and
    parameters. 
    
    Alternatively a new TMT instance can be created from scratch and, in either case, 
    once the optimized parameters have been identified, calling TMT.getBERTopicModel()
    returns a configured BERTopic instance with the desired parameters.
    
    TMT does not explicitly provide functionality for testing alternative UMAP parameters or
    HDBSCAN parameters other than min_cluster_size or min_samples. However both the HDBSCAN
    and UMAP models are exposed within the class and can be set to any parameters desired. 
    '''
    
    def __init__(self, 
                 embeddings: np.ndarray = None, #: pre-generated embeddings
                 embedding_model = None, #: set for alternative transformer embedding model
                 docs: List[str] = None, #: can be set here or when embeddings are created manually
                 reducer_model = None, #: a UMAP instance
                 hdbscan_model = None, #: an HDBSCAN instance
                 reducer_components: int = 5, #: for UMAP
                 verbose: int = 2): #: for UMAP
      '''
      Unless explicitly set, TMT Uses the same default parame defaults for the embedding model 
      as well as HDBSCAN and UMAP parameters as are used in the BERTopic defaults. 

      - 'all-MiniLM-L6-v2' sentence transformer as the default language model embedding.
      - UMAP - metric='cosine', n_neighbors=5, min_dist=0.0
      - HDBSCAN - metric='euclidean', cluster_selection_method='eom', prediction_data=Truej. 
                  min_samples=sample_size and cluster_size are the parameters being optimized
                  so those values are left to the user.
      
      Options include:

      - Using your own embeddings by setting TMT.embeddings after creating an instance
      - Using different UMAP settings or a different dimensional reduction method by setting TMT.reducer_model
      - Using different HDBSCAN parameters by setting TMT.hdbscan_model

      These can be set in the constructor or after instantiation by setting the instance variables 
      before generating the embeddings or reductions.

      Unlike BERTopic, TMT has an option for saving both embeddings and the doc corpus - or optionally
      omitting them.
      '''
      
      self.verbose=verbose 
      self.embeddings = embeddings 
      self.reducer_components=reducer_components 
      self.viz_reducer = None # A reducer (defaults to UMAP to create a 2D reduction for a
                              # scatter plot visualization of the embeddings
      self.reducer_model = reducer_model # Used to reduce the embeddings (if necessary)
      self.verbose = verbose
      self.hdbscan_model = hdbscan_model
      self.ResultsDF = None # A running collection of all the parameters and results if a DataFrame
      self.docs = docs
      self._paramPair = namedtuple('paramPair', 'cs ss') # Used internally to enhance readability

      if embedding_model == None :
          self.model = SentenceTransformer('all-MiniLM-L6-v2')
      else :
          self.model=embedding_model
          
      self.docs = docs
      if reducer_model == None : 
          self.reducer_model = UMAP(n_components=self.reducer_components, 
                             metric='cosine', 
                             n_neighbors=5, 
                             min_dist=0.0, 
                             verbose=self.verbose)

    @staticmethod
    def wrapBERTopicModel(BERTopicModel : BERTopic, verbose: int = 2 ) :
      '''
      This is a helper function which returns a TMT instance using the values from a BERTopic instance.
      '''
  
      return TopicModelTuner(embedding_model=BERTopicModel.embedding_model,
                               reducer_model=BERTopicModel.umap_model,
                               hdbscan_model=BERTopicModel.hdbscan_model,
                               verbose=verbose)

    def getBERTopicModel(self, min_cluster_size : int, min_samples : int):
      '''
      Returns a BERTopic model with the specified HDBSCAN parameters. The user is left
      to specify their chosen best settings after running a series of parameters searches.
      
      This function is necessary to return a BERTopic instance with a static UMAP reduction. 
      Since any given HDBSCAN parameters will return somewhat different results when clustering
      a given UMAP reduction, simply using the parameters derived from a tuning session derived
      from the UMAP reduction in the TMT instance will not produce the same results for a new 
      BERTopic instance. The reason for this is that BERTopic re-runs UMAP each time fit() is 
      called. Since different runs of UMAP will have  different characteristics, to recreate
      the desired results in the new BERTopic model it is necessary to use the same UMAP embeddings 
      as used for tuning. The BERTopic instance returned by this function uses UMAP_facade (see below)
      so that instead of re-runing UMAP it returns the pre-processed reduction instead.
      '''
      hdbscan_model = HDBSCAN(metric='euclidean',
                                      cluster_selection_method='eom',
                                      prediction_data=True,
                                      min_cluster_size=min_cluster_size,
                                      min_samples=min_samples,
                                      )

      return BERTopic(umap_model=UMAP_facade(self.reducer_model.embedding_),
                      hdbscan_model=hdbscan_model)

    
    def createEmbeddings(self, docs : List[str] = None) :
      '''
      Create embeddings 
      '''
      if self.embeddings != None :
        raise AttributeError('Embeddings already created, reset by setting TMT.embeddings=None')
        
      if (self.docs == None) and (docs == None) :
        raise AttributeError('Docs not specified, set docs=')
        
      if docs != None :
        self.docs=docs
             
      self.embeddings = self.model.encode(self.docs)
    
        
    def reduce(self) :
      '''
      Reduce dimensionality of the embeddings
      '''
      try :
        if self.embeddings == None :
          raise AttributeError('No embeddings set, call TMT.createEmbeddings() or set TMT.embeddings directly')
      except ValueError as e :
          pass # embeddings already set
      self.reducer_model.fit(self.embeddings)
    
    def createVizReduction(self) :
      '''
      Uses the reducer to create a 2D reduction of the embeddings to use for a scatter-plot representation
      '''
      try :
        if self.embeddings == None :
          raise AttributeError('No embeddings, either set TMT.embeddings= or call TMT.createEmbeddings()')
      except ValueError as e :
          pass # embeddings already set

      self.viz_reducer = copy(self.reducer_model)
      self.viz_reducer.n_components = 2
      self.viz_reducer.fit(self.embeddings)

    def getVizCoords(self) :
      '''
      Returns the X,Y coordinates for use in plotting a visualization of the embeddings.
      '''
      if self.viz_reducer == None :
          raise AttributeError('Visualization reduction not performed, call createVizReduction first')

      return self.viz_reducer.embedding_[:,0], self.viz_reducer.embedding_[:,1]

      
    def visualizeEmbeddings(self, min_cluster_size: int, sample_size: int) :
      '''
      Visualize the embeddings, clustered according to the provided HDBSCAN parameters.
      If TMT.docs has been set then the first 400 chars of each document will be shown as a 
      hover over each data point.
  
      Returns a plotly fig object
      '''
      topics = self.runHDBSCAN(min_cluster_size, sample_size)

      VizDF = pd.DataFrame()
      VizDF['x'], VizDF['y'] = self.getVizCoords()

      if self.docs != None :
          wrappedtext = ['<br>'.join(wrap(txt[:400], width=60)) for txt in self.docs]
          VizDF['text'] = wrappedtext
          hover_data = {'text': True}
      else :
          hover_data = None
          
      fig = px.scatter(VizDF,
               x='x',
               y='y',
               hover_data = hover_data,
               color=[str(top) for top in topics])

      return fig    

    def runHDBSCAN(self, min_cluster_size: int, sample_size) :
      '''
      Cluster reduced embeddings. sample_size must be more than 0 and less than
      or equal to min_cluster_size.
      '''

      if self.hdbscan_model == None :
          hdbscan_model = HDBSCAN(metric='euclidean',
                                      cluster_selection_method='eom',
                                      prediction_data=True,
                                      min_samples=sample_size,
                                      min_cluster_size=min_cluster_size,
                                      )
      else :
          hdbscan_model = copy(self.hdbscan_model)
          hdbscan_model.min_samples = sample_size
          hdbscan_model.min_cluster_size = min_cluster_size
  
      return hdbscan_model.fit_predict(self.reducer_model.embedding_)  


        
    def _runTests(self, searchParams):
      '''
      Runs a passel of HDBSCAN clusterings for searchParams
      '''
      if self.verbose > 1 :
        results = [(params.cs, params.ss, self.runHDBSCAN(params.cs, params.ss)) for params in tqdm(searchParams)] 
      else :
        results = [(params.cs, params.ss, self.runHDBSCAN(params.cs, params.ss)) for params in searchParams] 
      RunResultsDF = pd.DataFrame()
      RunResultsDF['min_cluster_size'] = [tupe[0] for tupe in results]
      RunResultsDF['sample_size'] = [tupe[1] for tupe in results]
      RunResultsDF['number_of_clusters'] = [len(pd.Series(tupe[2]).value_counts()) for tupe in results]
      uncategorized = []
      for aDict in [pd.Series(tupe[2]).value_counts().to_dict() for tupe in results] :
        if -1 in aDict.keys() :
            uncategorized.append(aDict[-1])
        else:
            uncategorized.append(0)
      RunResultsDF['number_uncategorized'] = uncategorized
      self.ResultsDF = pd.concat([self.ResultsDF, RunResultsDF])
      self.ResultsDF.reset_index(inplace=True, drop=True)
  
      return RunResultsDF

    def _returnParamsFromCSandPercent(self, cluster_size, sample_size_pct) :
        sample_size = int(cluster_size * sample_size_pct)
        if sample_size < 1 :
          sample_size = 1
        return self._paramPair(cluster_size, sample_size)
      
  
    def _genRandomSearchParams(self, cluster_size_range, sample_size_pct_range, iters=20) :
      searchParams = []
      for _ in range(iters) :
        searchParams.append(
          self._returnParamsFromCSandPercent(cluster_size_range[randrange(len(cluster_size_range))],
                                        sample_size_pct_range[randrange(len(sample_size_pct_range))]
                                       )
        )
      return searchParams

    def _genGridSearchParams(self, cluster_sizes, sample_size_pct_range) :
      searchParams = []
      for cluster_size in cluster_sizes :
        for sample_size_pct in sample_size_pct_range :
          searchParams.append(self._returnParamsFromCSandPercent(cluster_size, sample_size_pct))
      return searchParams
  
    def randomSearch(self, cluster_size_range: List[int], sample_size_pct_range: List[float], iters=20) :
      '''
      Run a passel of HDBSCAN within a given range of parameters.
      cluster_size_range is a list of ints and sample_size_pct_range is a list of percentage
      values in decimal form e.g. [.1, .25, .50, .75, 1]. 
      
      This function will randomly select a min_cluster_size  and a sample_size percent
      value from the supplied values. The sample_size percent will be used to calculate
      the sample_size parameter to be used. That value will be rounded up to 1 if less than 1
      and cannot be larger than the selected cluster_size.

      All of the search results will be added to TMT.ResultsDF and a separate DataFrame containing
      just the results from this search will be returned to the caller.
      '''
      searchParams = self._genRandomSearchParams(cluster_size_range, sample_size_pct_range, iters)
      return self._runTests(searchParams)

    def gridSearch(self, cluster_sizes: List[int], sample_sizes: List[float]) :
      '''
      Note that this is not a really a grid search. Rather this function will use each value
      in cluster_sizes to initiate a clustering on each percent value in sample_sizes. For 
      example if the values are [*range(100,102)] and [val/100 for val in [*range(10, 101 ,10)]], a clustering for 
      each percentage value in sample_sizes for each value in cluster_sizes would be run
      for a total of 20 clusterings.
      '''
      searchParams = self._genGridSearchParams(cluster_sizes, sample_sizes)
      return self._runTests(searchParams)

    def simpleSearch(self, cluster_sizes: List[int], sample_sizes: List[int]) :
      '''
      A clustering for each value in cluster_sizes will be run using the corresponding sample_sizes 
      value. The len of each list must be the same. Each cluster_size must be > 0 and sample_size must 
      be >0 and <= cluster_size.
      '''
      if len(cluster_sizes) != len(sample_sizes) :
        raise ValueError('Length of cluster sizes and samples sizes lists must match')
      return self._runTests([self._paramPair(cs,ss) for cs, ss in zip(cluster_sizes, sample_sizes)])

    def completeGridSearch(self, searchRange: List[int]) :
      '''
      For any n (int) in searchRange, generates all possible sample_size values (1 to n) and performs
      the search.
      '''
      cs_list, ss_list = [], []
      for cs_val in searchRange :
          for ss_val in [*range(1,cs_val+1)] :
            cs_list.append(cs_val)
            ss_list.append(ss_val)
      self.simpleSearch(cs_list, ss_list) 


                   
    def visualizeSearch(self, resultsDF: pd.DataFrame) :
      '''
      Creates a plotly parrallel coordinates graph of the searches contained in the DataFrame. 
      Returns a plotly fig object.
      '''
    
      # if self.reducer_model.embedding_.sum() == 0  :
      #     raise AttributeError('Reducer not run yet, call createReduction() first')

      return px.parallel_coordinates(resultsDF,
                                    color="number_uncategorized", 
                                    labels={"min_cluster_size": "min_cluster_size",
                                            "sample_size": "sample_size", 
                                            "number_of_clusters": "number_of_clusters",
                                            "number_uncategorized": "number_uncategorized", },)
  
    def summarizeResults(self, summaryDF : pd.DataFrame = None) :
      '''
      Takes DataFrame of results and returns a DataFrame containing only one record for 
      each value of number of clusters. Returns the record with the lowest number of 
      uncategorized documents. By default runs against self.ResultsDF - the aggregation of all
      searches run for this model.
      '''
      if summaryDF == None :
        summaryDF = self.ResultsDF
      resultSummaryDF = pd.DataFrame()
      for num_clusters in set(summaryDF['number_of_clusters'].unique()) :
            resultSummaryDF = pd.concat([resultSummaryDF, summaryDF[summaryDF['number_of_clusters']==num_clusters].sort_values(by='number_uncategorized').iloc[[0]]])
      resultSummaryDF.reset_index(inplace=True, drop=True)
      return resultSummaryDF

    def save(self, 
             fname,
             save_docs=True,
             save_embeddings=True,
             save_viz_reducer=True) :
      '''
      Saves the TMT object. User can choose whether or not to save docs, embeddings and/or
      the viz reduction
      '''
      
      docs = self.docs
      embeddings = self.embeddings
      viz_reduction = self.viz_reducer
      self._paramPair = None 
      with open(fname, 'wb') as file :
          if not save_docs :
              self.docs = None
          if not save_embeddings :
              self.embeddings = None
          if not save_viz_reducer :
              self.viz_reduction = None
          joblib.dump(self, file)
          
      self.docs = docs
      self.embeddings = embeddings
      self.viz_reduction = viz_reduction
      self._paramPair = namedtuple('paramPair', 'cs ss') 
     
    @staticmethod    
    def load(fname) :
      '''
      Restore a saved TMT object from disk
      '''
      
      with open(fname, 'rb') as file :    
        restored = joblib.load(file)
        restored._paramPair = namedtuple('paramPair', 'cs ss') 
        return restored


class UMAP_facade :
    '''
    By default BERTopic re-runs UMAP each time fit() is called,
    this class allows for a BERTopic instance to have a fixed UMAP embedding. 
    Necessary to achieve the best optimized HDBSCAN results because each 
    UMAP reduction will vary and require slightly different HDBSCAN parameters.
    '''
    def __init__(self, umap_embeddings):
        '''
        Pass in the umap_embeddings you want to optimize HDBSCAN for. 
        ''' 

        self.embeddings_ = umap_embeddings
        
    def fit(self, x, y) :
        '''
        Does nothing except conform to the interface BERTopic expects.
        '''
        
        return self
    
    def transform(self, embeddings=None) :
        '''
        Returns fixed embeddings,
        NOTE RE EMBEDDINGS PARAM
        ''' 
        return self.embeddings_ 