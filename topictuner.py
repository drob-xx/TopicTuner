'''
Created on Jul 19, 2022

@author: Dan
'''

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

from copy import copy
from random import randrange
from tqdm.notebook import tqdm
from textwrap import wrap


import numpy as np
import joblib
import pandas as pd

import plotly.express as px


class TopicModelTuner(object):
    '''
    TopicModelTuner - TMT - wraps a BERTopic instance and provides tools for 
    optimizing the min_clust_size and min_sample parameters of an HDBSCAN
    clustering instance when applied against a give UMAP reduction.
    
    A TMT instance can be initialized in a 'stand-alone' mode - without reliance on
    an already existing BERTopic instance, or it can be bootstrapped from an 
    existing BERTopic instance - either one already fit, or not. In either case, after 
    obtaining suitable parameters its getBERTopicModel() function can be called
    to return a functioning BERTopic instance with the desired parameters.
    
    TMT does not explicitly support tuning the UMAP instance, but could certainly be 
    used in a situation where multiple UMAP instances have been generated with 
    differing parameters. In this case TMT would be used to produce an individual
    set of HDBSCAN parameters for each UMAP instance which could then be compared for 
    the best results. 
    '''
    
    def __init__(self, embeddings=None, 
                 embedding_model=None, 
                 docs=None, 
                 reducer_model=None, 
                 hdbscan_model=None,
                 reducer_components=5,
                 verbose=2):
      '''
      The default initialization does not assume a BERTopic model explicitly, but the class is 
      currently written implicitly assuming the BERTopic, default, topic model pattern - 
      1) the 'all-MiniLM-L6-v2' sentence transformer as the default language model embedding.
      2) UMAP as the reduction method to 5 features using the same parameters as BERTopic defaults to.
      3) HDBSCAN as the clustering mechansim

      Options include:

      - Using your own embeddings by setting embeddings=
      - Using different UMAP settings or a different dimensional reduction method by setting reducer_model=
      - Using different HDBSCAN parameters by setting hdbscan_model=

      Unlike BERTopic, TMT has an option for saving both embeddings and the doc corpus - this is optional.
      '''
      
      self.verbose=verbose
      self.embeddings = embeddings
      self.reducer_components=reducer_components
      self.viz_reducer = None
      self.reducer_model = reducer_model
      self.verbose = verbose
      self.hdbscan_model = hdbscan_model
      self.ResultsDF = None
      self.docs = docs

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
          # this is a kludge.  See below for test to see if UMAP model has been fitted.
          self.reducer_model.embedding_ = np.array([0,0])

    @staticmethod
    def wrapBERTopicModel(BERTopicModel : BERTopic, verbose=2) :
      '''
      This is a helper function which returns a TMT instance using the values from a BERTopic Model.
      placeholder(FIXED)
      '''
  
      return TopicModelTuner(embedding_model=BERTopicModel.embedding_model,
                               reducer_model=BERTopicModel.umap_model,
                               hdbscan_model=BERTopicModel.hdbscan_model,
                               verbose=verbose)

    def getBERTopicModel(self, min_cluster_size, min_samples):
      '''
      Returns a BERTopic model with the specified HDBSCAN parameters.
      Since most, if not all, tuning will involve testing a range
      of parameters, the user is left to specify their chosen best settings.
      
      The substantial reason for this function is to create a UMAP_facade object
      as an argument to the BERTopic constructor, since any given HDBSCAN parameter
      set will be specific to a particular run of UMAP. 
      '''
      hdbscan_model = HDBSCAN(metric='euclidean',
                                      cluster_selection_method='eom',
                                      prediction_data=True,
                                      min_cluster_size=min_cluster_size,
                                      min_samples=min_samples,
                                      )

      return BERTopic(umap_model=UMAP_facade(self.reducer_model.embedding_),
                      hdbscan_model=hdbscan_model)
      
    

    def getBERTopicModel(self, min_cluster_size, min_samples):
      '''
      Returns a BERTopic model with the specified HDBSCAN parameters, regardles
      of whether or not the TMT instance was created using the default __init__() or 
      the helper wrapBERTopicModel() funciton.
      
      The user is left to specify their chosen best settings after running a series of 
      candidate parameters.
      
      The reason for this function is to return a BERTopic instance with create a 
      UMAP_facade object as an argument to the BERTopic initialization, since any 
      given HDBSCAN parameter set will be specific to a particular run of UMAP. 
      Just using a tuned HDBSCAN
      instance (or simply using the parameters derived from a tuning session) will 
      not provide the best results in a new BERTopic instance. This is because BERTopic
      will re-run UMAP each time fit() is called. Different runs of UMAP will have slightly 
      different characteristics and will therefore perform differently with different 
      HDBSCAN settings. HDBSCAN on the other hand will always return the same values assuming
      that all the paremeters, and the data being clustered, are the same.
      '''
      hdbscan_model = HDBSCAN(metric='euclidean',
                                      cluster_selection_method='eom',
                                      prediction_data=True,
                                      min_cluster_size=min_cluster_size,
                                      min_samples=min_samples,
                                      )

      return BERTopic(umap_model=UMAP_facade(self.reducer_model.embedding_),
                      hdbscan_model=hdbscan_model)

    
    def createEmbeddings(self, docs=None) :
      '''
      Create embeddings using the embedding model specified during initiliazation.
      '''
      if self.embeddings != None :
      raise AttributeError('Embeddings already created, reset by setting embeddings=None')
        
      if (self.docs == None) and (docs == None) :
        raise AttributeError('Docs not specified, set docs=)
        
      if docs != None :
        self.docs=docs
             
      self.embeddings = self.model.embedding_model.encode(self.docs)
    
        
    def reduce(self) :
      '''
      Reduce dimensionality of the embeddings
      '''
      if self.embeddings == None :
        raise AttributeError('No embeddings, either set via embeddings= or call createEmbeddings()')
        
      self.reducer_model.fit(self.embeddings)
    
    def createVizReduction(self) :
      '''
      Uses the reducer to create a 2D reduction of the embeddings to use for a scatter-plot representation
      '''

      if self.embeddings == None :
          raise AttributeError('No embeddings set: either set via embeddings= or call createEmbeddings()')

      self.viz_reducer = copy(self.reducer_model)
      self.viz_reducer.n_components = 2
      self.viz_reducer.fit(self.embeddings)


    def getVizCoords(self) :
        '''
        Returns the X,Y coordinates for use in plotting
        '''
        if self.viz_reducer == None :
            raise AttributeError('Visualization reduction not performed, call createVizReduction first')

        return self.viz_reducer.embedding_[:,0], self.viz_reducer.embedding_[:,1]

    def visualizeEmbeddings(self, min_cluster_size, min_sample_size) :
      '''
      Visualize the embeddings, clustered according to the provided parameters.
      If docs has been set then the first 400 chars of each document will be 
      shown as a hover over each data point.
  
      Returns a plotly fig object
      '''
      topics = self.runHDBSCAN(min_cluster_size, min_sample_size)

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
             path='./',
             save_docs=True,
             save_embeddings=True,
             save_viz_reduction=True) :
        '''
        Saves the TMT object. User can choose whether or not to save docs, embeddings and/or
        the viz reduction
        '''
        
        docs = self.docs
        embeddings = self.embeddings
        viz_reduction = self.viz_reduction
        with open(path, 'wb') as file :
            if not save_docs :
                self.docs = None
            if not save_embeddings :
                self.embeddings = None
            if not save_viz_reduction :
                self.viz_reduction = None
            joblib.dump(self, file)
            
        self.docs = docs
        self.embeddings = embeddings
        self.viz_reduction = viz_reduction
     
    @staticmethod    
    def load(path='./') :
      '''
      Restore a saved TMT object from disk
      '''
      
      with open(path, 'rb') as file :    
          return joblib.load(file)

    def runHDBSCAN(self, min_cluster_size, sample_size) :
      '''
      Cluster reduced embeddings. sample_size must be more than 0 and less than
      or equal to min_cluster_size.
      '''

        if self.hdbscan_model == None :
            hdbscan_model = HDBSCAN(metric='euclidean',
                                        cluster_selection_method='eom',
                                        prediction_data=True,
                                        min_samples=min_sample_size,
                                        min_cluster_size=min_cluster_size,
                                        )
        else :
            hdbscan_model = copy(self.hdbscan_model)
            hdbscan_model.min_samples = min_sample_size
            hdbscan_model.min_cluster_size = min_cluster_size
    
        return hdbscan_model.fit_predict(self.reducer_model.embedding_)  
        
    def _runTests(self, embedding, cluster_size_range, sample_size_pct_range, iters=20 ):
      '''
      Internal call to run a passel of HDBSCAN within a given range of parameters.
      cluster_size_range is a list of ints and sample_size_pct_range is a list of percentages e.g.
      [.1, .25, .50, .75, 1]. One of the percent values will be randomly chosen and 
      multiplied by the randomly chosen cluster_size_range to produce a min_samples
      value for HDBSCAN. The calulated min_samples value must be larger than 0 and not 
      greater than the selected cluster_size_range value.
      '''
      results = []
      for _ in tqdm(range(iters)) :
          min_cluster_size = cluster_size_range[randrange(len(cluster_size_range))]
          min_sample_size = int(min_cluster_size * (sample_size_pct_range[randrange(len(sample_size_pct_range))]))
          results.append((min_cluster_size, min_sample_size, self.runHDBSCAN(min_cluster_size, min_sample_size)))
      RunResultsDF = pd.DataFrame()
      RunResultsDF['min_cluster_size'] = [tupe[0] for tupe in results]
      RunResultsDF['min_sample_size'] = [tupe[1] for tupe in results]
      RunResultsDF['number_of_clusters'] = [len(pd.Series(tupe[2]).value_counts()) for tupe in results]
      uncategorized = []
      for aDict in [pd.Series(tupe[2]).value_counts().to_dict() for tupe in results] :
          if -1 in aDict.keys() :
              uncategorized.append(aDict[-1])
          else:
              uncategorized.append(0)
      RunResultsDF['number_uncategorized'] = uncategorized
  
      # if self.ResultsDF == None :
      #     self.ResultsDF = pd.DataFrame()
  
      self.ResultsDF = pd.concat([self.ResultsDF, RunResultsDF])
      self.ResultsDF.reset_index(inplace=True, drop=True)
  
      return RunResultsDF


    def evalParams(self, cluster_size_range, sample_size_range, iters = 20) :
      '''
      Runs iters number of randomly generated cluster size and sample range pairs
      against the embeddings. Note that sample sizes have to be a percentage value of 
      a given cluster_size and cannot be 0. 

      Returns both a plotly fig with a parrallel_coordinates chart of the run as well
      as a summarized version of the data produced by the run. Each run will produce
      its own summary table, but a history of all runs is also stored in the objects ResultsDF
      attribute.
      '''
    
        if self.reducer_model.embedding_.sum() == 0  :
            raise AttributeError('Reducer not run yet, call createReduction() first')

        ResultsDF = self._runTests(self.reducer_model.embedding_ , cluster_size_range, sample_size_range, iters=iters)
        fig = px.parallel_coordinates(ResultsDF,
                                      color="number_uncategorized", 
                                      labels={"min_cluster_size": "min_cluster_size",
                                              "min_sample_size": "min_sample_size", 
                                              "number_of_clusters": "number_of_clusters",
                                              "number_uncategorized": "number_uncategorized", },)

        resultSummaryDF = self.summarizeResults(ResultsDF)
        return fig, resultSummaryDF
    
    def summarizeResults(self, summaryDF : pd.DataFrame) :
      '''
      Pass this a DataFrame with run results - either the summary DF from a given
      run or the historical table run and it will return a table where each record 
      represents contains the smallest number of uncategorized documents for a given
      number of clusters.
      '''
      resultSummaryDF = pd.DataFrame()
      for num_clusters in set(summaryDF['number_of_clusters'].unique()) :
            resultSummaryDF = pd.concat([resultSummaryDF, summaryDF[summaryDF['number_of_clusters']==num_clusters].sort_values(by='number_uncategorized').iloc[[0]]])
      resultSummaryDF.reset_index(inplace=True, drop=True)
      return resultSummaryDF


class UMAP_facade :
    '''
    Since by default BERTopic re-runs UMAP each time fit() is called,
    this class allows for a BERTopic instance to have fixed UMAP embedding. 
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