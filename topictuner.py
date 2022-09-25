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
        
    
    def __init__(self, embeddings=None, 
                 embedding_model=None, 
                 docs=None, 
                 reducer_model=None, 
                 hdbscan_model=None,
                 reducer_components=5,
                 verbose=2):
        '''
        When initializing 
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
        as an argument to the BERTopic initialization, since any given HDBSCAN parameter
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

    
    def createEmbeddings(self, docs=None) :
        if self.embeddings != None :
            raise AttributeError('Embeddings already created, reset with embeddings=None')
        
        if (self.docs == None) and (docs == None) :
            raise AttributeError('Docs not specified, call createEmbeddings(docs)')
        
        if docs != None :
            self.docs=docs
             
        self.embeddings = self.model.embedding_model.encode(self.docs)
    
        
    def reduce(self) :
        
        # if self.embeddings == None :
        #     raise AttributeError('No embeddings, either set via embeddings= or call createEmbeddings()')
        
        self.reducer_model.fit(self.embeddings)
    
    def createVizReduction(self) :

        # if self.embeddings == None :
        #     raise AttributeError('No embeddings not set: either set via embeddings= or call createEmbeddings()')

        self.viz_reducer = copy(self.reducer_model)
        self.viz_reducer.n_components = 2
        self.viz_reducer.fit(self.embeddings)


    def getVizCoords(self) :
        
        if self.viz_reducer == None :
            raise AttributeError('Visualization reduction not performed, call createVizReduction first')

        return self.viz_reducer.embedding_[:,0], self.viz_reducer.embedding_[:,1]

    def visualizeEmbeddings(self, min_cluster_size, min_sample_size) :

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
        with open(path, 'rb') as file :    
            return joblib.load(file)

    def runHDBSCAN(self, min_cluster_size, min_sample_size) :

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
        results = []
        for _ in tqdm(range(iters)) :
            min_cluster_size = cluster_size_range[randrange(len(cluster_size_range))]
            min_sample_size = int(min_cluster_size * (sample_size_pct_range[randrange(len(sample_size_pct_range))]))
            # results.append((min_cluster_size, min_sample_size, RunHDBSCAN(model, embedding, min_cluster_size, min_sample_size)))
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