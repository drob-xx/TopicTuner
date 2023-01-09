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

logger.remove(0)
logger.add(sys.stderr, format = "<red>[{level}]</red> Message : <green>{message}</green> @ {time}", colorize=True)
logger.add('test_results.txt')

@pytest.fixture(scope="module")
def documents():
    return fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:200]
    
@pytest.fixture(scope="module")
def tmt_instance(documents):
    logger.info('Creating TMT object')
    tmt = TMT()
    tmt.reducer_random_state = 73433
    tmt._setBestParams(6, 1)
    logger.info('Running createEmbeddings')
    tmt.createEmbeddings(documents)
    logger.info('Reducing')
    tmt.reduce()
    return tmt


def test_randomSearch(tmt_instance):

    logger.info('Running randomSearch')
    tmt_instance.ResultsDF = None
    search_resultsDF = tmt_instance.randomSearch([*range(5,51)], [.1, .25, .5, .75, 1])
    assert(search_resultsDF.shape[0] == 20)
    
def test_psuedoGridSearch(tmt_instance):
    logger.info('Running psuedoGridSearch')
    tmt_instance.ResultsDF = None
    search_resultsDF = tmt_instance.psuedoGridSearch([*range(2,11)], [.1, .25, .5, .75, 1])
    assert(search_resultsDF.shape[0] == 45)
    assert(tmt_instance.summarizeResults()['number_of_clusters'].mean() == 21.6)
    assert(tmt_instance.summarizeResults()['number_uncategorized'].mean() == 18.6)
    
def test_simpleSearch(tmt_instance):
    logger.info('Running simpleSearch')
    csizes = []
    ssizes = []
    for csize in range(131,132) :
      for ssize in range(1, csize+1) :
        csizes.append(csize)
        ssizes.append(ssize) 
    tmt_instance.simpleSearch(csizes, ssizes)
      
def test_gridSearch(tmt_instance):
    logger.info('Running gridSearch')
    tmt_instance.gridSearch([*range(131,134)])  

def test_visualizeSearch(tmt_instance):
    logger.info('Running visualizeSearch')
    
def test_summarizeResults(tmt_instance):
    logger.info('Running summarizeResults')
    
def test_wrapBERTopicModel(tmt_instance):
    logger.info('Running wrapBERTopicModel')
    
def test_getBERTopicModel(tmt_instance):
    logger.info('Running getBERTopicModel')
    
def test_createVizReduction(tmt_instance):
    logger.info('Running createVizReduction')
    
def test_visualizeEmbeddings(tmt_instance):
    logger.info('Running visualizeEmbeddings')

def test_save_load(tmt_instance):
    logger.info('Running save_load')
    
    
    