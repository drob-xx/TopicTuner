# TopicTuner &#8212; Tune BERTopic HDBSCAN Models

To install from PyPi :
>pip install topicmodeltuner

## The Problem
Out of the box, [BERTopic](https://github.com/MaartenGr/BERTopic) relies upon [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) to cluster topics. Two of the most important HDBSCAN parameters, `min_cluster_size` and `sample_size` will almost always have a dramatic effect on cluster formation. They dictate the number of clusters created including the -`1` or *uncategorized* cluster. While with some datasets a large number of uncategorized documents may be the *right* clustering, in practice BERTopic will essentially discard a large percentage of *"good"* documents and not use them for cluster formation and topic formation. 

HDBSCAN is quite sensitive to the values of these two parameters relative to the text being clustered. This means that when using the BERTopic default value of `min_topic_size=10` (which is assigned to HDBSCAN's `min_cluster_size`) the default parameters will more often than not result in an unmanageable number of topics; as well as a sub-optimal number of uncategorized documents. Additionally, documents assigned to the -1 category will not be used to determine topic vocabularly results. 

## The Solution
TopicTuner provides a TopicModelTuner class&#8201;&#8212;&#8201;a convenience wrapper for BERTopic Models that efficiently manages the process of discovering optimized min_cluster_size and sample_size parameters, providing:

- Random and grid search functionality to quickly discover optimized parameters for a given BERTopic model.
- An internal datastore that records all searches for a given model, making parameter selection fast and easy.
- Visualizations to assist in parameter tuning and selection.
- Two way Import/Export functionality so that you can start from scratch, or with an existing BERTopic model and export a BERTopic model with optimized parameters at the end of your session.
- Save and Load for persistance.

To get you started this release includes both a [demo notebook](https://githubtocolab.com/drob-xx/TopicTuner/blob/main/TopicTunerDemo.ipynb) and [API documentation](https://drob-xx.github.io/TopicTuner)
