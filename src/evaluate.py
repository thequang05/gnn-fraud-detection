import torch
from sklearn.metrics import f1_score,precision_score,recall_score
def evaluate_model(model,data):
    model.eval()
    with torch.no_grad():
        out=model(data.x,data.edge_index)
        predictions=out.argmax(dim=1)
        test_condition = data.test_mask & (data.y != -1)
        y_true = data.y[test_condition].numpy()
        y_pred = predictions[test_condition].numpy()
        precision=precision_score(y_true,y_pred)
        recall=recall_score(y_true,y_pred)
        f1=f1_score(y_true,y_pred)
        return precision,recall,f1