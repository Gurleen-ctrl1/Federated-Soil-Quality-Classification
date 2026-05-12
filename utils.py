import torch

def calculate_model_drift(model1, model2):
    """Calculate the average L2 distance between two model state_dicts."""
    distance = 0.0
    for k in model1.keys():
        distance += torch.norm(model1[k] - model2[k]).item()
    return distance / len(model1)
