# Fraud Detection GNN

A Graph Neural Network model for detecting fraudulent transactions on the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). The model treats the transaction network as a graph and uses node classification to identify illicit transactions, addressing the class imbalance inherent in fraud detection tasks.

## Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (optional, CPU training is supported)
- Conda (recommended for environment management)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/gnn-fraud-detection.git
cd gnn-fraud-detection
```

2. Create and activate the Conda environment:

```bash
conda create -n fraud-gnn python=3.10
conda activate fraud-gnn
```

3. Install PyTorch and PyTorch Geometric separately before the rest of the dependencies (refer to [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for the correct CUDA version):

```bash
pip install torch>=2.1.0
pip install torch-geometric>=2.4.0
```

4. Install remaining dependencies:

```bash
pip install -r requirements.txt
```

5. Download the Elliptic dataset from Kaggle and place the three CSV files into `data/raw/`:

```
data/raw/
    elliptic_txs_classes.csv
    elliptic_txs_edgelist.csv
    elliptic_txs_features.csv
```

## Usage

All scripts must be run from inside the `src/` directory.

```bash
cd src
```

**Training:**

Trains the model for 100 epochs, prints Precision/Recall/F1 every 10 epochs, and saves weights to `src/fraud_gnn_weights.pth`.

```bash
python train.py
```

**Analysis (requires a trained checkpoint):**

Loads saved weights and prints the top-K highest-risk transactions from the test set, along with their AI-assigned fraud probability and ground-truth label.

```bash
python analyze.py
```

Example output:

```
Node ID    | Fraud Probability    | Actual Label
157681     |              100.00% | Gian lận
136312     |              100.00% | Gian lận
158303     |              100.00% | Gian lận
140762     |              100.00% | Gian lận
142234     |              100.00% | Gian lận
148166     |              100.00% | Gian lận
162513     |              100.00% | Gian lận
166581     |              100.00% | Gian lận
147116     |              100.00% | Gian lận
142515     |              100.00% | Gian lận
```

## Architecture / Project Structure

```
gnn-fraud-detection/
    data/
        raw/            # Raw CSV files from Elliptic dataset (not tracked by git)
        processed/      # Preprocessed graph objects (generated at runtime)
    notebooks/          # Exploratory analysis notebooks
    src/
        model.py        # FraudGNN: 2-layer GraphSAGE (SAGEConv + ReLU + Dropout)
        dataset.py      # load_elliptic_data(): builds PyG Data object, time-based train/test split
        train.py        # Training loop with weighted CrossEntropyLoss
        evaluate.py     # Computes Precision, Recall, F1 on the test set
        analyze.py      # Generates ranked fraud report from model confidence scores
        predict.py      # (placeholder)
    requirements.txt
```

**Model architecture:**

The `FraudGNN` model is a 2-layer GraphSAGE network:

```
Input features (166-dim) -> SAGEConv(128) -> ReLU -> Dropout(0.5) -> SAGEConv(2) -> Logits
```

**Train/test split:**

The dataset is split by time step. Time steps <= 34 are used for training; time steps > 34 are used for testing. Nodes with class label `unknown` are excluded from loss computation and evaluation metrics.

**Class imbalance handling:**

`CrossEntropyLoss` is applied with weights `[0.3, 0.7]` to up-weight the minority fraud class.

## Results

> **Note:** The model is currently under active optimization. The results below are preliminary.

Training configuration: GraphSAGE 2-layer + BatchNorm, hidden dim 128, Adam lr=0.01 + `weight_decay=0` + `ReduceLROnPlateau(factor=0.5, patience=10)`, early stopping (patience=20 evaluations), **dynamic class weights** computed from training set ratio (`[1.0, 7.63]`), train/test split at time step 34.

Metrics are evaluated on the test set (time steps > 34, excluding `unknown` labels) every 10 epochs:

| Epoch | Loss   | Precision | Recall | F1-Score |
|------:|-------:|----------:|-------:|---------:|
| 0     | 0.8632 | 0.0773    | 0.9898 | 0.1434   |
| 10    | 0.2623 | 0.1998    | 0.7655 | 0.3169   |
| 20    | 0.1777 | 0.2325    | 0.7276 | 0.3524   |
| 30    | 0.1333 | 0.3079    | 0.6704 | 0.4220   |
| 40    | 0.0997 | 0.3609    | 0.6574 | 0.4660   |
| 50    | 0.0768 | 0.4163    | 0.6223 | 0.4989   |
| 60    | 0.0610 | 0.4684    | 0.6020 | 0.5269   |
| 70    | 0.0494 | 0.4985    | 0.5965 | 0.5431   |
| 80    | 0.0406 | 0.5510    | 0.5688 | 0.5597   |
| 90    | 0.0337 | 0.6038    | 0.5559 | 0.5788   |
| 100   | 0.0281 | 0.6586    | 0.5540 | 0.6018   |
| 110   | 0.0235 | 0.6865    | 0.5439 | 0.6069   |
| 120   | 0.0199 | 0.7250    | 0.5429 | 0.6209   |
| 130   | 0.0171 | 0.7484    | 0.5383 | 0.6262   |
| 140   | 0.0148 | 0.7484    | 0.5411 | 0.6281   |
| **150**   | **0.0130** | **0.7571**    | **0.5383** | **0.6292** |
| 160   | 0.0116 | 0.7558    | 0.5374 | 0.6282   |
| 170   | 0.0104 | 0.7465    | 0.5355 | 0.6237   |
| 299   | 0.0043 | 0.7345    | 0.5235 | 0.6113   |

Best checkpoint saved at epoch 150: **F1 = 0.6292**, Precision = 0.7571, Recall = 0.5383.

Compared to the baseline run (static class weight `[0.3, 0.7]`, best F1 = 0.5586), switching to dynamic class weight `[1.0, 7.63]` improved best F1 by **+0.07**. The model continues to show a precision/recall trade-off after peak F1. Further optimization is ongoing.

## License

This project is released under the MIT License.