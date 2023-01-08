'''
Created on Jan 4, 2023

@author: Dan
'''

# from topictuner import TopicModelTuner as TMT

import pytest
from topictuner import TopicModelTuner as TMT
from sklearn.datasets import fetch_20newsgroups

@pytest.fixture(scope="module")
def documents():
    print('fetching docs')
    return fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:200]
    
@pytest.fixture(scope="module")
def tmt_instance(documents):
    print('begining test')
    tmt = TMT()
    tmt.reducer_random_state = 73433
    tmt._setBestParams(6, 1)
    print('createEmbeddings')
    tmt.createEmbeddings(documents)
    print('reduce')
    tmt.reduce()
    return tmt


def test_randomSearch(tmt_instance):

    print('randomSearch')
    tmt_instance.ResultsDF = None
    search_resultsDF = tmt_instance.randomSearch([*range(5,51)], [.1, .25, .5, .75, 1])
    assert(search_resultsDF.shape[0] == 20)
    
def test_psuedoGridSearch(tmt_instance):
    print('psuedoGridSearch')
    tmt_instance.ResultsDF = None
    search_resultsDF = tmt_instance.psuedoGridSearch([*range(2,11)], [.1, .25, .5, .75, 1])
    assert(search_resultsDF.shape[0] == 45)
    print(tmt_instance.summarizeResults()['number_of_clusters'].mean())
    print(tmt_instance.summarizeResults()['number_uncategorized'].mean())
    assert(tmt_instance.summarizeResults()['number_of_clusters'].mean() > 17.9)
    assert(tmt_instance.summarizeResults()['number_uncategorized'].mean() > 15.8)
    
# test_tmt_features()
    
       
        
