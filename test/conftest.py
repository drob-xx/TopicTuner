'''
Created on Jan 7, 2023

@author: Dan
'''

import pytest
from topictuner import TopicModelTuner as TMT
from sklearn.datasets import fetch_20newsgroups

@pytest.fixture()
def documents():
    print('fetching docs')
    return fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:200]
    
@pytest.fixture()
def tmt_instance():
    print('begining test')
    tmt = TMT()
    tmt.reducer_random_state = 73433
    tmt._setBestParams(6, 1)
    print('createEmbeddings')
    tmt.createEmbeddings(documents)
    print('reduce')
    tmt.reduce()
    return tmt

    