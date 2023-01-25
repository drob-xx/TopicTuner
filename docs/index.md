# API Overview

This solution exposes three classes. BaseHDBSCANTuner provides the core HDBSCAN parameter tuning functionality. TopicModelTuner extends the base to provide BERTopic specific functionality like the import and export of BERTopic models. cumlTopicModelTuner overrides TopicModelTuner to provide for cuML specific UMAP and HDBSCAN implementations.

At this point the BaseHDBSCANTuner class is a preliminary implementation and has not been tested "stand-alone" and apart from TopicModelTuner. Use TopicModelTuner objects for tuning and cumlTopicModelTuner if you have set up a cuML environment


