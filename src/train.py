import torch
import torch.nn as nn
import torch.optim as optim
from model import FraudGNN
from dataset import load_elliptic_data
from evaluate import evaluate_model
if __name__ == "__main__":
    data = load_elliptic_data()
    model = FraudGNN(in_channels=data.x.shape[1], hidden_channels=128, out_channels=2)
    train_labels = data.y[data.train_mask & (data.y != -1)]
    n_licit   = (train_labels == 0).sum().item()
    n_illicit = (train_labels == 1).sum().item()
    class_weight = torch.tensor([1.0, n_licit / n_illicit])
    print(f"Class weights: [1.0, {n_licit / n_illicit:.2f}]  (n_licit={n_licit}, n_illicit={n_illicit})")
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    epochs = 300
    best_f1 = 0.0
    patience = 20
    no_improve_count = 0
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        train_condition = data.train_mask
        loss = criterion(out[train_condition], data.y[train_condition])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            precision, recall, f1 = evaluate_model(model, data)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:>3}/{epochs-1} | Loss: {loss.item():.4f} | '
                  f'Precision: {precision:.4f} | Recall: {recall:.4f} | '
                  f'F1: {f1:.4f} | LR: {current_lr:.5f}')
            scheduler.step(f1)
            if f1 > best_f1:
                best_f1 = f1
                no_improve_count = 0
                torch.save(model.state_dict(), 'fraud_gnn_weights.pth')
                print(f'  -> Best model saved (F1={best_f1:.4f})')
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f'Early stopping at epoch {epoch}. Best F1: {best_f1:.4f}')
                    break