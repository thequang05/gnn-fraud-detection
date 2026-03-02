import torch
import torch.nn as nn
import torch.optim as optim
from model import FraudGNN
from dataset import load_elliptic_data
from evaluate import evaluate_model
if __name__== "__main__":
    data=load_elliptic_data()
    model = FraudGNN(in_channels=data.x.shape[1],hidden_channels=128,out_channels=2)

    criterion=nn.CrossEntropyLoss(ignore_index=-1,weight=torch.tensor([0.3, 0.7]))
    optimizer=optim.Adam(model.parameters(),lr=0.01)

    epochs=100
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out=model(data.x,data.edge_index)
        train_condition = data.train_mask
        loss = criterion(out[train_condition], data.y[train_condition])
        loss.backward()
        optimizer.step()
        if epoch%10==0 or epoch == epochs - 1:
            precision, recall, f1 = evaluate_model(model, data)
            print(f'Epoch {epoch:>3}/{epochs-1} | Loss: {loss.item():.4f} | '
                  f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}')
    torch.save(model.state_dict(), 'fraud_gnn_weights.pth')
