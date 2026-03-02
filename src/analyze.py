import torch
import torch.nn.functional as F
from model import FraudGNN
from dataset import load_elliptic_data

def generate_business_report(model,data,top_k=5):
    model.eval()

    with torch.no_grad():
        logits=model(data.x,data.edge_index)
        probabilities=F.softmax(logits,dim=1)
        risk_scores=probabilities[:,1]
    test_nodes_idx = torch.where(data.test_mask & (data.y != -1))[0]
    test_risk_scores = risk_scores[test_nodes_idx]
    test_true_labels = data.y[test_nodes_idx]   
    top_scores, top_relative_indices = torch.topk(test_risk_scores, top_k)
    for i in range(top_k):
        real_node_idx=test_nodes_idx[top_relative_indices[i]].item()
        ai_score = top_scores[i].item() * 100
        actual = "Gian lận" if test_true_labels[top_relative_indices[i]] == 1 else "Bình thường"
        print(f"{real_node_idx:<10} | {ai_score:>20.2f}%      | {actual:<20}")
if __name__ == "__main__":
    data = load_elliptic_data()
    model = FraudGNN(in_channels=data.x.shape[1], hidden_channels=128, out_channels=2)
    model.load_state_dict(torch.load('fraud_gnn_weights.pth', weights_only=True))
    generate_business_report(model, data, top_k=10)   