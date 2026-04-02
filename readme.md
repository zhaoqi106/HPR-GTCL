# Graph Transformer Architecture with Semantic Enhancement and Contrastive Learning for Herbal Prescription Recommendation

## Overview

This repository contains the complete code and data for the HPR-GTCL model, a novel approach for herbal prescription recommendation using Graph Transformer and Contrastive Learning. The model leverages both symptom-herb relationships and textual embeddings to generate accurate herbal prescriptions.

## 1. Requirements

The project requires the following dependencies:

```
numpy==1.25.0
pandas==3.0.1
scikit_learn==1.8.0
torch==2.3.0+cu118
```

## 2. Data Structure

### Dataset1

- **Dataset1.csv**: Main dataset containing processed symptoms and herbs. Symptoms are ID'd from 0 to 389 (total 390 symptoms), and herbs are ID'd from 390 onwards, corresponding to their order in the respective text files.
- **symptom_contains.txt**: List of all symptoms appearing in Dataset1.
- **herbs_contains.txt**: List of all herbs appearing in Dataset1.
- **cities files**: Adjacency matrix files constructed based on co-occurrence frequencies.
- **.npy files**: Vector representations of symptoms and herbs encoded using fine-tuned BERT.

### Dataset2

- **Dataset2.csv**: Main dataset containing processed symptoms and herbs. Symptoms are ID'd from 0 to 559 (total 560 symptoms), and herbs are ID'd from 560 onwards, corresponding to their order in the respective text files.
- **symptom_contains.txt**: List of all symptoms appearing in Dataset2.
- **herbs_contains.txt**: List of all herbs appearing in Dataset2.
- **cities files**: Adjacency matrix files constructed based on co-occurrence frequencies.
- **.npy files**: Vector representations of symptoms and herbs encoded using fine-tuned BERT.

##### Fine-tuned BERT:Fine-tuned BERT configuration file. embedding.py is used to convert symptom and herb text into corresponding embeddings.

## 3. Code Structure

- **utils.py**: Contains basic utility functions for data processing, evaluation metrics, and helper functions used throughout the project.
- **dataloader.py**: Defines data loading functions and custom dataset classes for efficient batch processing of symptom-herb pairs.
- **model.py**: Encapsulates the complete HPR-GTCL model architecture, including Graph Transformer layers and contrastive learning components.
- **train.py**: Main training and evaluation script for Dataset1.
- **train2.py**: Main training and evaluation script for Dataset2.

## 4. Getting Started

#### Clone the repository:

```
git clone https://github.com/zhaoqi106/HPR-GTCL.git
cd HPR-GTCL
```

#### Install the required dependencies:

```
pip install numpy==1.25.0 pandas==3.0.1 scikit_learn==1.8.0
pip install torch==2.3.0+cu118
```

## 5.Running the Model

All paths are set as relative paths, so you can directly run the code to reproduce our results.

**For Dataset1:**

```
python train.py
```

**For Dataset2:**

```
python train2.py
```

#### The Fine-tuned BERT can be obtained from https://pan.baidu.com/s/1AglsAhlrO5gNiKijxd0K8Q?pwd=st5e

