# mc_dropout_utils.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def enable_dropout(model):
    "Enable dropout layers in a model (keeps other layers in current state)."
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

def mc_dropout_predict(model, dataloader, T=20, device='cpu'):
    """
    model: nn.Module (assumed to return logits)
    dataloader: PyTorch/fastai test_dl (yields (xb, yb) or xb)
    returns: preds shape [T, N, C] (numpy)
    """
    model.to(device)
    # keep batchnorm in eval mode
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

    enable_dropout(model)  # dropout active
    preds_list = []
    with torch.no_grad():
        for t in range(T):
            batch_preds = []
            for batch in tqdm(dataloader, desc=f"MC pass {t+1}/{T}", leave=False):
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                xb = xb.to(device)
                out = model(xb)  # logits
                p = torch.softmax(out, dim=1).cpu()
                batch_preds.append(p)
            preds_list.append(torch.cat(batch_preds, dim=0))
    preds = torch.stack(preds_list, dim=0)  # [T, N, C]
    return preds.numpy()
