# Conceptualizing Concept Drift
This repository contains code for the paper:  
**Conceptualizing Concept Drift, Isaac Roberts, Fabian Hinder, Valerie Vaquet, Alexander Schulz, Barbara Hammer, submitted to European Symposium on Artificial Neural Networks(ESANN) 2025
**

# Step 1: Technical prerequisites

This experiment builds mainly on the following code:
- [CRAFT in Xplique]((https://github.com/deel-ai/xplique)) - A framework for automatically extracting Concept Activation Vectors which explain deep
  neural networks. Link to Paper [CRAFT](https://arxiv.org/abs/2211.10154)
- [Model-based Drift Explanations]((https://github.com/FabianHinder/DRAGON)) - Repository of example Model-based explanations. Link to Paper [Model-based Drift Explanantions](https://www.sciencedirect.com/science/article/pii/S0925231223007634)

In this experiment, we use CRAFT in combination with [Pytorch](https://pytorch.org/).We suggest to setup a local [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)-environment
using **Python 3.10** and install the repository as follows:
```bash
git clone https://github.com/robertsi20/ConceptualizingConceptDrift.git
pip install fastai datasets jupyter xplique timm
```


# Step 2: Data acquisition
We construct two Datastreams for our experiments and Case Study:
- [NINCO](https://github.com/j-cb/NINCO) - The NINCO (No ImageNet Class Objects) dataset consists of 64 OOD classes with a total of 5879 samples. The OOD classes were selected to have no categorical overlap with any classes of ImageNet-1K.  Link to Paper [NINCO]((https://arxiv.org/abs/2306.00826))
- [Subset of ImageNet](https://www.image-net.org/) - ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images.  Link to Paper [ImageNet](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)
    - To create the subset:
        - Obtain permission for research use on the ImageNet Website
        - then used [ImageNet-Dataset-Downloader](https://github.com/mf1024/ImageNet-Datasets-Downloader) to construct a datset consisting of:
            - red foxes
            - gray foxes
            - arctic foxes
            - timber wolves
            - red wolves
            - white wolves

To proceed with the NINCO datset:
```bash
mkdir data/
wget -O data/ninco.tar.gz https://zenodo.org/record/8013288/files/NINCO_all.tar.gz?download=1
mkdir data/ninco_data
tar -xf data/ninco.tar.gz -C data/ninco_data
```

# Step 3a: Experiments
To see the results of the experiments, one can either run the experiment notebooks (assuming one has acquired the data), or can directly load the csv files which contain the results over 50 runs used in the paper. 
The scripts contained in the notebooks construct the drift from the datasets obtained above. 

# Step 3b: Case Study 
To see examples of the produced explanantions, please follow along in the case study notebook.

