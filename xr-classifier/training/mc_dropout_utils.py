# mc_dropout_utils.py
import torch
import numpy as np

def enable_dropout(model):
    """Enable dropout layers during inference"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def mc_dropout_predict(model, dl, T=20):
    """Run T stochastic forward passes with dropout enabled"""
    model.eval()
    enable_dropout(model)

    all_preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds_T = []
            for _ in range(T):
                preds = torch.softmax(model(xb), dim=1)
                preds_T.append(preds.unsqueeze(0))
            preds_T = torch.cat(preds_T, dim=0)  # [T, N, C]
            all_preds.append(preds_T)
    return torch.cat(all_preds, dim=1).cpu().numpy()
