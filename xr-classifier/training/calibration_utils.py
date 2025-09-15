# calibration_utils.py
import torch
import torch.nn as nn

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # scale logits by learned temperature
        return logits / self.temperature

    def set_temperature(self, valid_dl):
        self.eval()
        nll_criterion = nn.CrossEntropyLoss()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for xb, yb in valid_dl:
                logits = self.model(xb)
                logits_list.append(logits)
                labels_list.append(yb)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        return self
