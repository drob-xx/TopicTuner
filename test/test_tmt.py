'''
Created on Jan 4, 2023

@author: Dan
'''

# from topictuner import TopicModelTuner as TMT

# https://github.com/lmcinnes/umap/issues/153


import pytest
from topictuner import TopicModelTuner as TMT
from sklearn.datasets import fetch_20newsgroups
from loguru import logger
import sys
import numpy as np
import pandas as pd

logger.remove(0)
logger.add(sys.stderr, format = "{time} : {level} : {message} ")
logger.add('test_results.txt')

@pytest.fixture(scope="module")
def documents():
    return fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:200]
    
@pytest.fixture(scope="module")
def tmt_instance(documents):
    logger.info('Creating TMT object')
    tmt = TMT()
    tmt.reducer_random_state = 73433
    tmt.bestParams = (6, 1)
    logger.info('Running createEmbeddings')
    tmt.createEmbeddings(documents)
    logger.info('Reducing')
    tmt.reduce()
    tmt.createVizReduction()
    return tmt

def test_bestParams(tmt_instance):
        with pytest.raises(ValueError) : # error no vals set, no bestParams
            new_instance = TMT()
            new_instance._check_CS_SS(None, None, True)
        tmt_instance.bestParams = (22, 3)
        cs, ss = tmt_instance._check_CS_SS(None, None, True)
        assert(cs == 22)
        assert(ss == 3)
        tmt_instance.bestParams = tmt_instance._paramPair(22, 3)
        cs, ss = tmt_instance._check_CS_SS(None, None, True)
        assert(cs == 22)
        assert(ss == 3)
        with pytest.raises(ValueError) :
            tmt_instance.bestParams = ('foo')
        with pytest.raises(ValueError) :
            tmt_instance.bestParams = (3)
        tmt_instance.bestParams = (4,3)
        assert(tmt_instance.bestParams[0] == 4)
        assert(tmt_instance.bestParams[1] == 3)
        assert(tmt_instance.bestParams.cs == 4)
        assert(tmt_instance.bestParams.ss == 3)
        tmt_instance.bestParams = tmt_instance._paramPair(4,3)
        assert(tmt_instance.bestParams[0] == 4)
        assert(tmt_instance.bestParams[1] == 3)
        assert(tmt_instance.bestParams.cs == 4)
        assert(tmt_instance.bestParams.ss == 3)

def test_randomSearch(tmt_instance):

    logger.info('Running randomSearch')
    tmt_instance.clearSearches()
    with pytest.raises(ValueError):
        tmt_instance.randomSearch([1], [.1, .25])
    with pytest.raises(ValueError):
        tmt_instance.randomSearch([0], [.1, .25])
    with pytest.raises(ValueError):
        tmt_instance.randomSearch([2], [.1, 1.1])
    with pytest.raises(ValueError):
        tmt_instance.randomSearch([], [.1, .25])
    with pytest.raises(ValueError):
        tmt_instance.randomSearch([2, 3], [])
    with pytest.raises(ValueError):
        tmt_instance.randomSearch([], [])
    search_resultsDF = tmt_instance.randomSearch([*range(5,51)], [.1, .25, .5, .75, 1])
    assert(search_resultsDF.shape[0] == 20)
    logger.info('Completed randomSearch')
    
def test_psuedoGridSearch(tmt_instance):
    logger.info('Running psuedoGridSearch')
    tmt_instance.clearSearches()
    with pytest.raises(ValueError):
        tmt_instance.psuedoGridSearch([1], [.1, .25])
    with pytest.raises(ValueError):
        tmt_instance.psuedoGridSearch([0], [.1, .25])
    with pytest.raises(ValueError):
        tmt_instance.psuedoGridSearch([2], [.1, 1.1])
    with pytest.raises(ValueError):
        tmt_instance.psuedoGridSearch([], [.1, .25])
    with pytest.raises(ValueError):
        tmt_instance.psuedoGridSearch([3, 5], [])
    with pytest.raises(ValueError):
        tmt_instance.psuedoGridSearch([], [])
    search_resultsDF = tmt_instance.psuedoGridSearch([*range(2,11)], [.1, .25, .5, .75, 1])
    assert(search_resultsDF.shape[0] == 45)
    
def test_simpleSearch(tmt_instance):
    logger.info('Running simpleSearch')
    tmt_instance.clearSearches()
    with pytest.raises(ValueError):
        tmt_instance.simpleSearch([1], [.1])
    with pytest.raises(ValueError):
        tmt_instance.simpleSearch([0], [.1])
    with pytest.raises(ValueError):
        tmt_instance.simpleSearch([2], [.1, 1.1])
    with pytest.raises(ValueError):
        tmt_instance.simpleSearch([2, 2], [.1])
    with pytest.raises(ValueError):
        tmt_instance.simpleSearch([2, 2], [.1, 1.1])
    with pytest.raises(ValueError):
        tmt_instance.simpleSearch([], [.1, 1.1])
    with pytest.raises(ValueError):
        tmt_instance.simpleSearch([2, 3], [])
    with pytest.raises(ValueError):
        tmt_instance.simpleSearch([], [])

    csizes = []
    ssizes = []
    for csize in range(2,11) :
      for ssize in range(1, csize+1) :
        csizes.append(csize)
        ssizes.append(ssize) 
    search_resultsDF = tmt_instance.simpleSearch(csizes, ssizes)
    assert(search_resultsDF.shape[0] == 54)
      
def test_gridSearch(tmt_instance):
    logger.info('Running gridSearch')
    tmt_instance.clearSearches()
    with pytest.raises(ValueError):
        tmt_instance.gridSearch([0])
    with pytest.raises(ValueError):
        tmt_instance.gridSearch([1])
    with pytest.raises(ValueError):
        tmt_instance.gridSearch([])
    search_resultsDF = tmt_instance.gridSearch([*range(2,11)])
    assert(search_resultsDF.shape[0] == 54)
      
def test_visualizeSearch(tmt_instance):
    logger.info('Running visualizeSearch')
    tmt_instance.clearSearches()
    search_resultsDF = tmt_instance.gridSearch([*range(2,11)])
    assert(search_resultsDF.shape[0] == 54)
    fig = tmt_instance.visualizeSearch(search_resultsDF)
    assert(len(fig.to_dict()['data'][0]['dimensions'][0]['values']) == 54)
    # defaults to ResultsDF 
    fig = tmt_instance.visualizeSearch()
    assert(len(fig.to_dict()['data'][0]['dimensions'][0]['values']) == 54)
    
def test_summarizeResults(tmt_instance):
    logger.info('Running summarizeResults')
    tmt_instance.clearSearches()
    search_resultsDF = tmt_instance.gridSearch([*range(2,11)])
    # will use ResultsDF
    assert(np.all(tmt_instance.summarizeResults()['min_cluster_size'].isin([*range(2,11)])) == True)
    assert(np.all(tmt_instance.summarizeResults(search_resultsDF)['min_cluster_size'].isin([*range(2,11)])) == True)
    
def test_createVizReduction(tmt_instance):
    logger.info('Running createVizReduction')
    tmt_instance.createVizReduction('UMAP')
    assert(tmt_instance.viz_reduction.shape[0] == 200)
    tmt_instance.createVizReduction('TSNE')
    assert(tmt_instance.viz_reduction.shape[0] == 200)
    
def test_visualizeEmbeddings(tmt_instance):
    logger.info('Running visualizeEmbeddings')
    tmt_instance.bestParams = tmt_instance._paramPair(22, 3)
    fig = tmt_instance.visualizeEmbeddings()
    del(fig)
    fig = tmt_instance.visualizeEmbeddings(6,1)
    del(fig)

def test_get_wrap_BERTopicModel(tmt_instance):
    logger.info('Running get_wrap_BERTopicModel')
    tmt_instance.bestParams = tmt_instance._paramPair(22, 3)
    btModel = tmt_instance.getBERTopicModel()
    assert(btModel.hdbscan_model.min_cluster_size == 22)
    assert(btModel.hdbscan_model.min_samples == 3)
    btModel = tmt_instance.getBERTopicModel(6, 1)
    hdbscan_model = tmt_instance.getHDBSCAN(6, 1)
    hdbscan_model.fit_predict(tmt_instance.target_vectors)
    tmtLabels = hdbscan_model.labels_
    # btModel.fit_transform(documents)
    documents = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:200]
    btModel.fit_transform(documents)

    assert(len(btModel.topics_) == len(tmtLabels))
    assert(pd.Series(tmtLabels).value_counts()[-1] == pd.Series(btModel.topics_).value_counts()[-1])
    newDocs = ["doc one", "doc two", "doc three"]
    preds = btModel.transform(newDocs)
    assert(len(preds[1]) == 3)
    tmtModel = TMT.wrapBERTopicModel(btModel)
    assert(str(type(tmtModel)) == "<class 'topictuner.topictuner.TopicModelTuner'>")
    assert(str(type(btModel)) == "<class 'bertopic._bertopic.BERTopic'>")

def test_save_load(tmt_instance):
    logger.info('Running save_load')
    tmt_instance.save('tmt_instance')
    tmtModel = TMT.load('tmt_instance')
    assert(str(type(tmtModel)) == "<class 'topictuner.topictuner.TopicModelTuner'>")
    
    
    
    