dataset-generator
===============

### Dependencies
Tested with Python 3.6, and requires the following packages, which are available via PIP:

* Required: [numpy >= 1.11.3](http://www.numpy.org/)
* Required: [matplotlib >= 2.0.0](https://matplotlib.org/)

### Dataset Format
Before applying a dataset generator, the first step is to ensure that the corpus is stored in a suitable format so that the ground truth labels can be extracted. An example of this structure can be seen in the [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/).

### Applying Concept Shift Generator
	python concept-shift.py --input dataset_folder -k 5 --window_size 100 --min_topics 3 --shift_prob 0.05 --output path/dataset_name
	
The generator will produce and store the dataset in the dataset_name folder. An example concept shift dataset can be found in data/shift.

A plot of the distribution of the topics over each time window is also generated and stored as a pdf file.

### Applying Concept Drift Generator

    python concept-drift.py --input dataset_folder -k 5 --window_size 100 --decrease_windows 5 --increase_windows 10 --min_topics 3 --drift_prob 0.05 --output path/dataset_name

The generator will produce and store the dataset in the dataset_name folder. An example concept drift dataset can be found in data/drift.

*Currently this generator requires the number of decrease_windows to be less than the number of increase_windows*.

An overview of the dataset is generated which includes the number of topics, the number of documents, whether drift was activated and the distribution of topics in a time window. This is stored as a csv file.

### Parameters

**input**: an existing dataset with ground truth topic annotations.

**k**: number of starting topics.

**window_size**: number of documents in each time window.

**min-topics**: minimum number of topics present before ending.

**shift-prob**: the probability of a concept shift occurring.

**drift-prob:** the probability of a concept drift occurring.

**increase-window**: number of windows for a topic to gradually disappear.

**decrease-windows**: number of windows for a topic to gradually appear.



