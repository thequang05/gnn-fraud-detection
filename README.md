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
3142       |              97.83%  | Gian lận
8801       |              95.11%  | Gian lận
...
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

Training was run for 200 epochs on CPU with the configuration described above (GraphSAGE 2-layer, hidden dim 128, Adam lr=0.01, weighted CrossEntropyLoss `[0.3, 0.7]`, train/test split at time step 34).

Metrics are evaluated on the test set (time steps > 34, excluding `unknown` labels) every 10 epochs:

| Epoch | Loss   | Precision | Recall | F1-Score |
|------:|-------:|----------:|-------:|---------:|
| 0     | 1.0509 | 0.0000    | 0.0000 | 0.0000   |
| 10    | 0.2675 | 0.2066    | 0.5965 | 0.3069   |
| 20    | 0.1944 | 0.2646    | 0.6371 | 0.3739   |
| 30    | 0.1417 | 0.3189    | 0.6131 | 0.4196   |
| 40    | 0.1114 | 0.4112    | 0.6048 | 0.4895   |
| 50    | 0.0900 | 0.4221    | 0.6030 | 0.4966   |
| 60    | 0.0740 | 0.4859    | 0.5900 | 0.5329   |
| 70    | 0.0621 | 0.5300    | 0.5706 | 0.5496   |
| 80    | 0.0529 | 0.5469    | 0.5549 | 0.5509   |
| 90    | 0.0455 | 0.5702    | 0.5476 | 0.5586   |
| 100   | 0.0393 | 0.5600    | 0.5254 | 0.5422   |
| 110   | 0.0344 | 0.5820    | 0.5208 | 0.5497   |
| 120   | 0.0303 | 0.5861    | 0.5060 | 0.5431   |
| 130   | 0.0268 | 0.5930    | 0.5005 | 0.5428   |
| 140   | 0.0239 | 0.5837    | 0.4894 | 0.5324   |
| 150   | 0.0213 | 0.5868    | 0.4746 | 0.5248   |
| 160   | 0.0191 | 0.5870    | 0.4672 | 0.5203   |
| 170   | 0.0172 | 0.5873    | 0.4598 | 0.5158   |
| 180   | 0.0156 | 0.5835    | 0.4580 | 0.5132   |
| 190   | 0.0141 | 0.5754    | 0.4543 | 0.5077   |
| 199   | 0.0130 | 0.5671    | 0.4488 | 0.5010   |

The model reaches peak F1 around epoch 90 (F1 = 0.5586), after which overfitting causes recall to degrade while precision stays roughly flat. Early stopping or regularization adjustments are being investigated.

## License

This project is released under the MIT License.