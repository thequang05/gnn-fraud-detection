# GNN Fraud Detection - Elliptic Bitcoin Dataset

> **Phát hiện giao dịch gian lận (Fraud Detection) trên mạng Bitcoin bằng Graph Neural Network (GraphSAGE)**

---

## 📌 Giới thiệu

Dự án này áp dụng **Graph Neural Network (GNN)** để phân loại các giao dịch Bitcoin là *hợp lệ* (licit) hoặc *gian lận* (illicit) dựa trên **Elliptic Bitcoin Dataset** — một trong những dataset đồ thị tài chính thực tế lớn nhất được công bố công khai.

### Tại sao dùng GNN?
Trong tài chính, các giao dịch không tồn tại độc lập — chúng kết nối với nhau thành một **mạng lưới dòng tiền**. GNN khai thác cấu trúc đồ thị này, cho phép mô hình học không chỉ từ đặc trưng của từng giao dịch mà còn từ **hành vi của các giao dịch lân cận**, từ đó phát hiện các mô hình rửa tiền phức tạp mà các phương pháp ML truyền thống bỏ sót.

---

## 📊 Dataset

| Thuộc tính    | Giá trị |
|---------------|---------|
| Tên           | [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |
| Nodes         | 203,769 giao dịch |
| Edges         | 234,355 dòng tiền |
| Features/Node | 166 đặc trưng |
| Illicit nodes | ~4,545 (~2%) |

**Đặt file vào** `data/raw/`:
```
data/raw/
├── elliptic_txs_features.csv
├── elliptic_txs_edgelist.csv
└── elliptic_txs_classes.csv
```

---

## 🏗️ Kiến trúc

```
Elliptic Graph → GraphSAGE (3 layers, hidden=128) → Binary Classifier
                  ↑ BatchNorm + ReLU + Dropout(0.5)
```

**Mô hình**: GraphSAGE (Hamilton et al., 2017) — phù hợp với đồ thị lớn và học inductive.

---

## 📂 Cấu trúc thư mục

```
gnn_fraud_detection/
├── data/
│   ├── raw/               ← Dữ liệu gốc (KHÔNG sửa trực tiếp)
│   └── processed/         ← Graph đã xử lý
├── notebooks/
│   └── 01_eda_elliptic.ipynb  ← Phân tích khám phá dữ liệu
├── src/
│   ├── dataset.py         ← Chuyển CSV → PyG Data object
│   ├── model.py           ← Kiến trúc GraphSAGE
│   ├── train.py           ← Vòng lặp huấn luyện
│   ├── evaluate.py        ← Precision / Recall / F1 / AUC
│   └── predict.py         ← Inference & xuất kết quả CSV
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn chạy

### 1. Cài đặt môi trường
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tải dataset
Tải từ [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) và đặt vào `data/raw/`.

### 3. Huấn luyện
```bash
python -m src.train --epochs 200 --hidden 128 --layers 3 --lr 0.001
```

### 4. Dự đoán
```bash
python -m src.predict --checkpoint best_model.pt --output predictions.csv
```

### 5. Khám phá dữ liệu
```bash
jupyter notebook notebooks/01_eda_elliptic.ipynb
```

---

## 📈 Kết quả kỳ vọng

| Metric    | GraphSAGE (baseline) |
|-----------|---------------------|
| F1-Score  | ~0.86               |
| Precision | ~0.88               |
| Recall    | ~0.84               |
| AUC-ROC   | ~0.97               |

> **Lưu ý**: Dataset rất mất cân bằng (2% illicit). F1-Score là metric quan trọng nhất.

---

## 🛠️ Công nghệ sử dụng

- **PyTorch** & **PyTorch Geometric** — Deep learning trên đồ thị
- **Pandas / NumPy** — Xử lý dữ liệu
- **scikit-learn** — Đánh giá mô hình
- **NetworkX** — Phân tích đồ thị
- **Jupyter** — Phân tích khám phá dữ liệu

---

## 📚 Tài liệu tham khảo

- Hamilton et al. (2017). [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (GraphSAGE)
- Weber et al. (2019). [Anti-Money Laundering in Bitcoin](https://arxiv.org/abs/1908.02591) (Elliptic Dataset)