# Graph Neural Networks for Football Pass Map Analysis

## Project Overview

This project uses Graph Neural Networks (GNNs) to analyze and predict football game outcomes by modeling pass maps. Specifically, we aim to predict the second-half pass patterns and final game outcomes by learning from first-half pass maps. Our work leverages GNNs to explore connections and patterns within pass networks, potentially offering insights into team strategies and game dynamics.

## Authors
- Yahav
- Yael Hari
- Yonatan Sabag

## Goals and Methodology

We have implemented two GNN-based models:

1. **2nd Half VAE**: This model predicts the second-half pass map based on the first-half data.
2. **2nd Half Live Predictor**: This model aims to predict the final game result using data from the first half.

### Data Source

The pass map data is sourced from [StatsBomb](https://statsbomb.com/) and includes game-by-game passing information for each team. The data is processed to create a graph representation of the field with 21 regions, where each node represents a region of the field, and edges represent pass frequencies between these regions.

### Data Representation

- The field is divided into 21 regions, each representing a node in the graph.
- Each game is represented by a 21x21x9 matrix, where:
  - The 21x21 matrix contains pass counts between regions.
  - The 9 represents 5-minute intervals, totaling the first 45 minutes.
- For modeling purposes, the home and away teams' data are concatenated to form a 42x42x9 matrix as input to the GNN.

### Model Architecture

Each GNN model contains:

- **Graph Nodes**: Represent the 21 field regions with pass information between them.
- **GNN Layers**: Two convolutional layers to learn spatial-temporal pass patterns.

## Requirements



