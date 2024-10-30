# Graph Neural Networks for Football Pass Map Analysis

## Project Overview

This project uses Graph Neural Networks (GNNs) to analyze and predict football game outcomes by modeling pass maps. Specifically, we aim to predict the second-half pass patterns and final game outcomes by learning from first-half pass maps. Our work leverages GNNs to explore connections and patterns within pass networks, potentially offering insights into team strategies and game dynamics.

## Goals and Methodology

We have implemented two GNN-based models:

1. **2nd Half Predictor**: This model predicts the second-half pass map based on the first-half data.
2. **Outcome Predictor**: This model aims to predict the final game result using data from the first half and from the predicted second half.

### Data Source

The pass map data is sourced from [StatsBomb](https://statsbomb.com/) and includes game-by-game passing information for each team. The data is processed to create a graph representation of the field with 21 regions, where each node represents a region of the field, and edges represent pass frequencies between these regions.

### Data Representation

- The field is divided into 21 regions, each representing a node in the graph.
- Each game is represented by a 21x21x9 matrix, where:
  - The 21x21 matrix contains pass counts between regions.
  - The 9 represents 5-minute intervals, totaling the first 45 minutes.
- For modeling purposes, the home and away teams' data are concatenated to form a 42x42x9 matrix as input to the GNN.

### Model Architecture
We have implemented four GNN-based models:

1. **GAT**
2. **SAGE**
3. **GCN**
4. **Temporal Convolution**

## Authors
- Yahav Cohen
- Yael Hari
- Yonatan Sabag

## Usage
We ran the project by executing:

```bash
python train.py ```

Each run generates two output files:

A loss.txt file logging the model's loss over epochs.
A balanced_accuracy.txt file recording the balanced accuracy for each model.
After running the four models, run the visualization script to display the graph.

```bash
python visualization.py
```

