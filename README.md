
# Graph Neural Networks for Football Pass Map Analysis

This project aims to analyze and predict football game outcomes by modeling pass maps using Graph Neural Networks (GNNs). Using data from StatsBomb, we created a GNN model that processes pass maps and predicts game outcomes based on passes between different sectors of the field.

## Table of Contents
- [Project Overview](#project-overview)
- [Models](#models)
- [Data Preprocessing](#data-preprocessing)
- [Requirements](#requirements)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

Our analysis involves representing each football game as a graph with nodes corresponding to 21 field sectors. We use two models for this analysis:
1. **2nd Half VAE** - predicts the pass map for the second half.
2. **2nd Half Live Predictor** - predicts the final game result based on pass map patterns.

## Models

### 2nd Half VAE
This Variational Autoencoder (VAE) model divides the game into 21 nodes, each representing a field sector, and predicts the pass map for the second half based on the first half's pass map.

### 2nd Half Live Predictor
This model is used to predict the final game result using the live pass map data during the game.

## Data Preprocessing

The StatsBomb data provides detailed pass data. Each match is split into 21 sectors (nodes), with each layer representing a 5-minute interval for each half. The input matrix is prepared as follows:
1. **Input Matrix (42x42x9)**: Home and away team matrices are concatenated, resulting in a 42x42x9 matrix.
2. **Node Features**: Each node represents passes between two sectors with a dimension of 9 representing time.

## Requirements

Ensure you have the necessary packages installed:
```bash
pip install -r requirements.txt
```

Packages include:
- PyTorch
- DGL (Deep Graph Library)
- StatsBombPy (for data acquisition)
- NumPy
- Pandas

## Usage

To run the project, use the `train.py` script:

```bash
python train.py 
```

### Example Usage

To train the 2nd Half VAE model for 50 epochs with a batch size of 32:
```bash
python train.py --model vae --epochs 50 --batch_size 32 --learning_rate 0.001
```


## Acknowledgments

- **StatsBomb** for providing open-access football data.
- **Deep Graph Library (DGL)** for enabling GNN model development.
