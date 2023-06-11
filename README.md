# Detecting Hateful Users on Twitter with Graph Machine Learning

Hateful speech on online platforms is a prominent and widespread issue that can lead to real-world violence and discrimination, making its mitigation crucial for creating safe online environments. In this project, we approach this problem by implementing automated systems that detect hateful users with the help of Graph Machine Learning. We focus on a retweet graph collected in October 2017, with a sample of users being labeled as hateful or normal. We study the network by analyzing its degree and feature distributions, their correlations with labels, and the occurrence of triadic motifs, and we compare the graph with theoretical random models. Subsequently, we train various predictive models on the task of node prediction, including logistic regression, label propagation, GraphSAGE, and Graph Attention Networks (GAT). Our results indicate that GraphSAGE performs the best in terms of F1 score, outperforming all baselines and demonstrating its effectiveness in generalizing to unseen nodes and running on large graphs. Overall, our findings highlight the potential of Graph Machine Learning for fighting hateful speech on social media platforms.

## Authors
This project was carried out by Francesco Salvi and Giacomo Orsi, in the context of the course "Network Machine learning" at EPFL. Data is provided by the paper *Characterizing and Detecting Hateful Users on Twitter* by Ribeiro et al. (2018) [1](https://arxiv.org/abs/1803.08977).

## Report
[Here](report.pdf) you can find a detailed report where we: 
- analyze the degree distribution, the distribution of features, and their correlation with the labels. We also extract additional insights through the use of motifs, studying the occurrence of triads in the network
- discuss prediction models in the task of binary node classification. We explore simple baselines and more sophisticated models, including logistic regression, label propagation, GraphSAGE, and Graph Attention Networks (GAT)

## Code 
The results of the report are generated by the following files:
- [exploration.ipynb](src/exploration.ipynb): contains the code for the data exploration
- [exploitation.ipynb](src/exploitation.ipynb): contains the code for the predictive models
- [collect_motifs.py](src/collect_motifs.py): contains the code for the motif analysis

### Getting Started with Code
------------
1. Create virtual environment
```
conda env create -f environment.yml
```

2. Download the data from [Kaggle](https://www.kaggle.com/datasets/manoelribeiro/hateful-users-on-twitter) and place it into the ```data/``` folder

Regarding the code for predictive models, we recommend you run the notebook [exploitation.ipynb](src/exploitation.ipynb) on Google Colab, as it requires a GPU to run the GraphSAGE and GAT models. The notebook contains cells with the packages that have to be installed on Colab.

## Media
The folder [media](media/) contains the images used in the report.

