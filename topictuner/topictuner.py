from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN         
from sentence_transformers import SentenceTransformer
from copy import copy, deepcopy
from collections import namedtuple
from tqdm.notebook import tqdm
from textwrap import wrap
from typing import List
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
from random import randrange
from sklearn.manifold import TSNE

paramPair = namedtuple('paramPair', 'cs ss')

class BaseHDBSCANTuner(object):
    def __init__(self,
                 HDBSCAN_model = None, #: an HDBSCAN instance
                 target_vectors = None, # vectors to be clustered
                 viz_reduction = None, # 2D reduction of the target_vectors
                 verbose: int = 0): 
        
        self.hdbscan_model = HDBSCAN_model
        self.target_vectors = target_vectors
        self.verbose = verbose
        self.hdbscan_params = {}
        self.ResultsDF = None # A running collection of all the parameters and results if a DataFrame
        self.viz_reduction = viz_reduction, # a 2D reduction of the embeddings used for visualization

        
        # self._paramPair = namedtuple('paramPair', 'cs ss') # Used internally to enhance readability
        self._paramPair = paramPair        
        self.__bestParams = self._paramPair(None, None)
    
    def _check_CS_SS(self, min_cluster_size: int, min_samples: int):

      if min_cluster_size == None :
          raise ValueError('Cannot set min_cluster_size==None')
      if min_cluster_size == 1 :
          raise ValueError('min_cluster_size must be more than 1')
      if (min_samples > min_cluster_size) :
        raise ValueError('min_samples must be equal or less than min_cluster_size')

      # if (self.bestParams.cs == None) & (self.bestParams.ss == None) :
      #     raise ValueError('min_cluster_size and min_samples must be set (no default (best) values set for this model)')

      # if min_cluster_size == None : 
      #   min_cluster_size = self.bestParams.cs # go ahead and set - bestParams is either None or a better value
      #   min_samples = self.bestParams.ss # go ahead and set - if min_cluster_size was None then this has to be None or a better value

      return min_cluster_size, min_samples

    @property
    def bestParams(self):
        return self.__bestParams

    # @bestParams.setter
    # def bestParams(self, min_cluster_size: int, sample_size: int=None):
    #    self.__bestParams.cs, self.__bestParams.ss = self._check_CS_SS(min_cluster_size, sample_size)

    @bestParams.setter
    def bestParams(self, params):
        cs, ss = self._check_CS_SS(params[0], params[1])
        self.__bestParams = self._paramPair(cs,ss)
            
    def runHDBSCAN(self, min_cluster_size: int = None, min_samples: int = None) :
      '''
      Cluster reduced embeddings. min_samples must be more than 0 and less than
      or equal to min_cluster_size.
      '''
      min_cluster_size, min_samples = self._check_CS_SS(min_cluster_size, min_samples)
          
      if self.hdbscan_model == None :
        hdbscan_params = copy(self.hdbscan_params)
        hdbscan_params['min_cluster_size'] = min_cluster_size
        hdbscan_params['min_samples'] = min_samples
        hdbscan_model = HDBSCAN(**hdbscan_params)
      else :
        hdbscan_model = copy(self.hdbscan_model)
        hdbscan_model.min_cluster_size = min_cluster_size
        hdbscan_model.min_samples = min_samples
  
      return hdbscan_model.fit_predict(self.target_vectors)  

    def randomSearch(self, cluster_size_range: List[int], min_samples_pct_range: List[float], iters=20) :
      '''
      Run a passel of HDBSCAN within a given range of parameters.
      cluster_size_range is a list of ints and min_samples_pct_range is a list of percentage
      values in decimal form e.g. [.1, .25, .50, .75, 1]. 
      
      This function will randomly select a min_cluster_size  and a min_samples percent
      value from the supplied values. The min_samples percent will be used to calculate
      the min_samples parameter to be used. That value will be rounded up to 1 if less than 1
      and cannot be larger than the selected cluster_size.

      All of the search results will be added to TMT.ResultsDF and a separate DataFrame containing
      just the results from this search will be returned to the caller.
      '''
      searchParams = self._genRandomSearchParams(cluster_size_range, min_samples_pct_range, iters)
      return self._runTests(searchParams)

    def psuedoGridSearch(self, cluster_sizes: List[int], min_samples: List[float]) :
      '''
      Note that this is not a really a grid search. Rather this function will use each value
      in cluster_sizes to initiate a clustering on each percent value in min_sampless. For 
      example if the values are [*range(100,102)] and [val/100 for val in [*range(10, 101 ,10)]], a clustering for 
      each percentage value in min_sampless for each value in cluster_sizes would be run
      for a total of 20 clusterings (cluster sizes 100 and 101 * percent values of those for 10%, 20%, 30%,...100%).
      '''
      searchParams = self._genGridSearchParams(cluster_sizes, min_samples)
      return self._runTests(searchParams)

    def simpleSearch(self, cluster_sizes: List[int], min_sampless: List[int]) :
      '''
      A clustering for each value in cluster_sizes will be run using the corresponding min_sampless 
      value. The len of each list must be the same. Each cluster_size must be > 0 and min_samples must 
      be >0 and <= cluster_size.
      '''
      if len(cluster_sizes) != len(min_sampless) :
        raise ValueError('Length of cluster sizes and samples sizes lists must match')
      return self._runTests([self._paramPair(cs,ss) for cs, ss in zip(cluster_sizes, min_sampless)])

    def gridSearch(self, searchRange: List[int]) :
      '''
      For any n (int) in searchRange, generates all possible min_samples values (1 to n) and performs
      the search.
      '''
      cs_list, ss_list = [], []
      for cs_val in searchRange :
          for ss_val in [*range(1,cs_val+1)] :
            cs_list.append(cs_val)
            ss_list.append(ss_val)
      return self.simpleSearch(cs_list, ss_list) 
                   
    def visualizeSearch(self, resultsDF: pd.DataFrame = None) :
      '''
      Returns a plotly fig of a parrallel coordinates graph of the searches performed on this instance. 
      '''
        
      if not np.any(resultsDF) :
          resultsDF = self.ResultsDF

      return px.parallel_coordinates(resultsDF,
                                    color="number_uncategorized", 
                                    labels={"min_cluster_size": "min_cluster_size",
                                            "min_samples": "min_samples", 
                                            "number_of_clusters": "number_of_clusters",
                                            "number_uncategorized": "number_uncategorized", },)
  
    def summarizeResults(self, summaryDF : pd.DataFrame = None) :
      '''
      Takes DataFrame of results and returns a DataFrame containing only one record for 
      each value of number of clusters. Returns the record with the lowest number of 
      uncategorized documents. By default runs against self.ResultsDF - the aggregation of all
      searches run for this model.
      '''

      if not np.any(summaryDF) :
          summaryDF = self.ResultsDF

      if not np.any(summaryDF) :
          raise ValueError("No searches run on this TMT instance, or DF to summarize is None")
        
      resultSummaryDF = pd.DataFrame()
      for num_clusters in set(summaryDF['number_of_clusters'].unique()) :
        resultSummaryDF = pd.concat([resultSummaryDF, summaryDF[summaryDF['number_of_clusters']==num_clusters].sort_values(by='number_uncategorized').iloc[[0]]])
      resultSummaryDF.reset_index(inplace=True, drop=True)
      return resultSummaryDF.sort_values(by=['number_of_clusters'])
  
    def clearSearches(self):
        """
        A convenience function that resets the saved searches
        """
        self.ResultsDF = None
        
    def createVizReduction(self, method='UMAP') :
      '''
      Uses the reducer to create a 2D reduction of the embeddings to use for a scatter-plot representation
      '''
      if not np.all(self.embeddings) :
        raise AttributeError('No embeddings, either set TMT.embeddings= or call TMT.createEmbeddings()')

      if method == 'UMAP' :
        self.viz_reducer = copy(self.reducer_model)
        self.viz_reducer.n_components = 2
        self.viz_reducer.fit(self.embeddings)
      else : # Only TSNE is supported
        self.viz_reducer = TSNE(n_components=2, verbose=self.verbose, random_state=self.random_state)
        self.viz_reducer.fit(self.embeddings)

      self.viz_reduction = self.viz_reducer.embedding_


    def getVizCoords(self) :
      '''
      Returns the X,Y coordinates for use in plotting a visualization of the embeddings.
      '''
      if self.viz_reducer == None :
          raise AttributeError('Visualization reduction not performed, call createVizReduction first')
 
      return self.viz_reducer.embedding_[:,0], self.viz_reducer.embedding_[:,1]

      
    def visualizeEmbeddings(self, min_cluster_size: int = None, min_samples: int = None) :
      '''
      Visualize the embeddings, clustered according to the provided HDBSCAN parameters.
      If TMT.docs has been set then the first 400 chars of each document will be shown as a 
      hover over each data point.
  
      Returns a plotly fig object
      '''  
      
      topics = self.runHDBSCAN(min_cluster_size, min_samples)

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
        
  
    def _genRandomSearchParams(self, cluster_size_range, min_samples_pct_range, iters=20) :
        searchParams = []
        for _ in range(iters) :
            searchParams.append(
              self._returnParamsFromCSandPercent(cluster_size_range[randrange(len(cluster_size_range))],
                                            min_samples_pct_range[randrange(len(min_samples_pct_range))]
                                           )
        )
        return searchParams
    
    def _genGridSearchParams(self, cluster_sizes, min_samples_pct_range) :
      searchParams = []
      for cluster_size in cluster_sizes :
        for min_samples_pct in min_samples_pct_range :
          searchParams.append(self._returnParamsFromCSandPercent(cluster_size, min_samples_pct))
      return searchParams

    def _returnParamsFromCSandPercent(self, cluster_size, min_samples_pct) :
        min_samples = int(cluster_size * min_samples_pct)
        if min_samples < 1 :
          min_samples = 1
        return self._paramPair(cluster_size, min_samples)
       
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
      RunResultsDF['min_samples'] = [tupe[1] for tupe in results]
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
        
class TopicModelTuner(BaseHDBSCANTuner):
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
                 embeddings: np.ndarray = None, # pre-generated embeddings
                 embedding_model = None, # set for alternative transformer embedding model
                 docs: List[str] = None, # can be set here or when embeddings are created manually
                 reducer_model = None, # a UMAP instance
                 reducer_random_state = None,
                 reducer_components = 5, 
                 reduced_embeddings = None,
                 hdbscan_model = None, # an HDBSCAN instance
                 viz_reduction = None,
                 verbose: int = 0): #: for UMAP
        
      BaseHDBSCANTuner.__init__(self, 
                   HDBSCAN_model = hdbscan_model,
                   target_vectors = reduced_embeddings, # Set the reduced embeddings to be clustered 
                   viz_reduction = viz_reduction,
                   verbose = verbose)

      '''
      Unless explicitly set, TMT Uses the same default param defaults for the embedding model 
      as well as HDBSCAN and UMAP parameters as are used in the BERTopic defaults. 

      - 'all-MiniLM-L6-v2' sentence transformer as the default language model embedding.
      - UMAP - metric='cosine', n_neighbors=5, min_dist=0.0
      - HDBSCAN - metric='euclidean', cluster_selection_method='eom', prediction_data=True,
                  min_cluster_size = 10.
      
      Options include:

      - Using your own embeddings by setting TMT.embeddings after creating an instance
      - Using different UMAP settings or a different dimensional reduction method by setting TMT.reducer_model
      - Using different HDBSCAN parameters by setting TMT.hdbscan_model

      These can be set in the constructor or after instantiation by setting the instance variables 
      before generating the embeddings or reduction.

      Unlike BERTopic, TMT has an option for saving both embeddings and the doc corpus - or optionally
      omitting them.
      '''
      
      self.embeddings = embeddings 
      self.reducer_model = reducer_model # Used to reduce the embeddings (if necessary)
      self.reducer_components = reducer_components
      self.reducer_random_state = reducer_random_state
      self.docs = docs

      self.viz_reducer = None # A reducer (defaults to UMAP to create a 2D reduction for a
                              # scatter plot visualization of the embeddings


      
      if embedding_model == None :
          self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
      else :
          self.embedding_model=embedding_model
    
      if reducer_model == None : # Use default BERTopic params 
          self.reducer_model = UMAP(n_components=self.reducer_components, 
                                    metric='cosine', 
                                    n_neighbors=5, 
                                    min_dist=0.0, 
                                    verbose=self.verbose)

      # Default BERTopic params
      self.hdbscan_params = {'metric': 'euclidean',
                             'cluster_selection_method': 'eom',
                             'prediction_data': True,
                             'min_cluster_size': 10,
                             }

    @staticmethod
    def wrapBERTopicModel(BERTopicModel : BERTopic) :
      '''
      This is a helper function which returns a TMT instance using the values from a BERTopic instance.
      '''
  
      return TopicModelTuner(embedding_model=BERTopicModel.embedding_model,
                               reducer_model=BERTopicModel.umap_model,
                               hdbscan_model=BERTopicModel.hdbscan_model,
                            )

    def getBERTopicModel(self, min_cluster_size : int = None, min_samples : int = None):
      '''
      Returns a BERTopic model with the specified HDBSCAN parameters. The user is left
      to specify their chosen best settings after running a series of parameters searches.
      
      This function is necessary because any given HDBSCAN parameters will return somewhat different 
      results when clustering a given UMAP reduction, simply using the parameters derived from a tuned
      TMT model will not produce the same results for a new BERTopic instance. 
      
      The reason for this is that BERTopic re-runs UMAP each time fit() is 
      called. Since different runs of UMAP will have  different characteristics, to recreate
      the desired results in the new BERTopic model we need to use the same random seed for the BERTopic's UMAP
      as was used in the TMT model.
      '''
      
      min_cluster_size, min_samples = self._check_CS_SS(min_cluster_size, min_samples)

      hdbscan_params = copy(self.hdbscan_params)
      hdbscan_params['min_cluster_size'] = min_cluster_size
      hdbscan_params['min_samples'] = min_samples
        
      hdbscan_model = HDBSCAN(**hdbscan_params)

      return BERTopic(umap_model=deepcopy(self.reducer_model),
                      hdbscan_model=hdbscan_model)

    
    def createEmbeddings(self, docs : List[str] = None) :
      '''
      Create embeddings 
      '''
      # if self.embeddings != None :
      if np.any(self.embeddings) :
        raise AttributeError('Embeddings already created, reset by setting TMT.embeddings=None')
        
      if (self.docs == None) and (docs == None) :
        raise AttributeError('Docs not specified, set docs=')
        
      self.docs=docs
             
      self.embeddings = self.embedding_model.encode(self.docs)
    
        
    def reduce(self) :
        '''
        Reduce dimensionality of the embeddings
        '''
        if not np.any(self.embeddings) :
          raise AttributeError('No embeddings set, call TMT.createEmbeddings() or set TMT.embeddings=')
        
        if self.reducer_model.random_state != None :  
          self.reducer_random_state = self.reducer_model.random_state # The reducer has already set a random_state
        else :  
          random_state = self.reducer_random_state # could be None
          if random_state != None :  # Reducer random state was not set but tmt random state was
              self.reducer_model.random_state = random_state  # set reducer model random state
          else : # set new random state and capture it
              self.reducer_model.random_state = randrange(1000000) # new value
              self.reducer_random_state = self.reducer_model.random_state # sync TMT random state
        
        self.reducer_model.fit(self.embeddings)    
        self.target_vectors = self.reducer_model.embedding_ 
    
    def createVizReduction(self, method='UMAP') :
      '''
      Uses the reducer to create a 2D reduction of the embeddings to use for a scatter-plot representation
      '''
      if not np.all(self.embeddings) :
        raise AttributeError('No embeddings, either set TMT.embeddings= or call TMT.createEmbeddings()')

      if method == 'UMAP' :
        self.viz_reducer = copy(self.reducer_model)
        self.viz_reducer.n_components = 2
        self.viz_reducer.fit(self.embeddings)
      else : # Only TSNE is supported
        self.viz_reducer = TSNE(n_components=2, verbose=self.verbose, random_state=self.reducer_random_state)
        self.viz_reducer.fit(self.embeddings)
        
      self.viz_reduction = self.viz_reducer.embedding_

    def getVizCoords(self) :
      '''
      Returns the X,Y coordinates for use in plotting a visualization of the embeddings.
      '''
      if self.viz_reducer == None :
          raise AttributeError('Visualization reduction not performed, call createVizReduction first')
 
      return self.viz_reducer.embedding_[:,0], self.viz_reducer.embedding_[:,1]

      
    def visualizeEmbeddings(self, min_cluster_size: int = None, min_samples: int = None) :
      '''
      Visualize the embeddings, clustered according to the provided HDBSCAN parameters.
      If TMT.docs has been set then the first 400 chars of each document will be shown as a 
      hover over each data point.
  
      Returns a plotly fig object
      '''  
      
      topics = self.runHDBSCAN(min_cluster_size, min_samples)

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
