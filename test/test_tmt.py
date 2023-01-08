'''
Created on Jan 4, 2023

@author: Dan
'''

# from topictuner import TopicModelTuner as TMT


def test_randomSearch():

    print('randomSearch')
    tmt_instance.ResultsDF = None
    search_resultsDF = tmt_instance.randomSearch([*range(5,51)], [.1, .25, .5, .75, 1])
    assert(search_resultsDF.shape[0] == 20)
    
def test_psuedoGridSearch():
    print('psuedoGridSearch')
    tmt_instance.ResultsDF = None
    search_resultsDF = tmt.psuedoGridSearch([*range(2,11)], [.1, .25, .5, .75, 1])
    assert(search_resultsDF.shape[0] == 45)
    assert(tmt.summarizeResults()['number_of_clusters'].mean() > 17.9)
    assert(tmt.summarizeResults()['number_uncategorized'].mean() > 15.8)
    
# test_tmt_features()
    
       
        
