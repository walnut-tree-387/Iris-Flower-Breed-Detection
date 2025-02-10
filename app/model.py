import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#Model Definition
class FlowerPredictorModel(nn.Module) :
    def __init__(self, in_features = 4, h1 = 11, h2 = 13, out_features = 3) :
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x) :
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
        

# Load the trained model
def load_model():
    model = FlowerPredictorModel()  
    model.load_state_dict(torch.load('app/flower_model.pth')) 
    model.eval()
    return model

# Predict the species of the Iris flower
def predict_species(features):
    flower_names = ['Versicolor', 'Setosa', 'Virginica']
    model = load_model()
    prediction = model.forward(features)
    max_index = torch.argmax(prediction, dim=1)
    max_index = max_index.item()
    return flower_names[max_index]
