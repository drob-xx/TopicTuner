# TopicTuner &#8212; Tune BERTopic HDBSCAN Models

## The Problem
Out of the box, BERTopic relies upon HDBSCAN to cluster topics. Two of the most important HDBSCAN parameters, 
min_cluster_size and sample_size will almost always have a dramatic effect on cluster formation. They will
determine the number of clusters created and the number of uncategorized records assigned to the -1 category. The resultant topic model will,
more often than not, be constructed from a large and unmanageable number
of topics. Additionally, the default setting often lead to a large percentage of documents not being assigned to a 
topic and therefore not contributing to the topic model itself.

## The Solution
TopicTuner provides a TopicModelTuner class&#8201;&#8212;&#8201;a convenience wrapper for BERTopic Models that efficiently manages 
the process of discovering optimized min_cluster_size and sample_size parameters, providing:

- Random and grid search functionality to discover optimized parameters for a given BERTopic model.
- A datastore that records all searches for a given model.
- Visualizations to assist in parameter tuning and selection.
- 'Import/Export' functionality to both wrap existing BERTopic models and to provide a BERTopic model tuned with the 
optimized parameters.
- Save and Load for persistance.

To get you started this release includes both a demo notebook and [API documentation](http://htmlpreview.github.io/?https://github.com/drob-xx/TopicTuner/blob/main/doc/topictuner.html)

