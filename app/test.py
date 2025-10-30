import torch
import torch.nn as nn
import mlflow.pytorch

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.pytorch.log_model(model, "model", registered_model_name="IrisClassifier_PyTorch")

print("✅ Model registered as IrisClassifier_PyTorch")
