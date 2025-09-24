# calibration_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def compute_ece(probs, labels, n_bins=15):
    """
    probs: numpy array shape (N, C) - probabilities after softmax
    labels: numpy array shape (N,) integer class labels
    returns: ECE scalar
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0:
            continue
        acc = accuracies[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(acc - conf)
    return ece

def plot_reliability_diagram(probs, labels, save_path, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    avg_conf = []
    avg_acc = []
    counts = []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        counts.append(mask.sum())
        if mask.sum() == 0:
            avg_conf.append(0.0)
            avg_acc.append(0.0)
        else:
            avg_conf.append(confidences[mask].mean())
            avg_acc.append(accuracies[mask].mean())

    # plot
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.plot(avg_conf, avg_acc, marker='o')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

class ModelWithTemperature(nn.Module):
    """
    Wrap a model and learn a scalar temperature on validation logits
    (On Calibration of Modern Neural Networks - Guo et al.)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # initialize temperature > 0
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(self, valid_loader, device="cpu", max_iter=50):
        self.to(device)
        nll_criterion = torch.nn.CrossEntropyLoss().to(device)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                label = label.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)

        if len(logits_list) == 0:
            print("⚠️ Warning: No logits collected for calibration. Skipping temperature scaling.")
            return 1.0  # default temperature

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        nll_criterion = nn.CrossEntropyLoss().to(device)

        # Optimize temperature (LBFGS)
        self.temperature.data = torch.ones(1).to(device) * float(self.temperature.data.item())
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def _eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return float(self.temperature.item())
