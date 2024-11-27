# Conceptualizing Concept Drift
This repository contains code for the paper:  
**Conceptualizing Concept Drift, Isaac Roberts, Fabian Hinder Valerie Vaquet, Alexander Schulz, Barbara Hammer, submitted to European Symposium on Artificial Neural Networks(ESANN) 2025
**

## Step 1: Technical prerequisites

This experiment builds mainly on the following code:
- [CRAFT in Xplique]((https://github.com/deel-ai/xplique)) - A framework for automatically extracting Concept Activation Vectors which explain deep
  neural networks. Link to Paper [CRAFT](https://arxiv.org/abs/2211.10154)
- [Model-based Drift Explanations]((https://github.com/FabianHinder/DRAGON)) - Repo for example Model-based explanations. Link to Paper [Model-based Drift Explanantions](https://www.sciencedirect.com/science/article/pii/S0925231223007634)


- In this experiment we use CRAFT in combination with [Pytorch](https://pytorch.org/).\ 
We suggest to setup a local [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)-environment
using **Python 3.10** and install the repository as follows:

```bash
git clone https://github.com/robertsi20/ConceptualizingConceptDrift.git
pip install -r requirements.txt
```




# Environment Setup
pip install fastai datasets jupyter xplique timm

# Data acquisition
mkdir data/
wget -O data/ninco.tar.gz https://zenodo.org/record/8013288/files/NINCO_all.tar.gz?download=1
mkdir data/ninco_data
tar -xf data/ninco.tar.gz -C data/ninco_data

# Experiments
Experiment Results for Streams D1 and D2

# Case Study 

