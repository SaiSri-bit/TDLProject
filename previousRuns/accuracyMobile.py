## Data set is from https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
## Simply line 27 to have different activation functions

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("Train.csv")
X = data.drop(["price_range"], axis=1)
y = data["price_range"]
    
le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long) # Use y_test.value if it cant be represented a



# Define model
class PriceRangeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,activator):
        super(PriceRangeClassifier, self).__init__()
        if activator == 0:
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(), 
                nn.Linear(hidden_size, output_size)
            )
        elif activator == 1:
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),  
                nn.Linear(hidden_size, output_size)
            )
        elif activator == 2:
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Sigmoid(),  
                nn.Linear(hidden_size, output_size)
            )
        elif activator == 2:
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Sigmoid(),  
                nn.Linear(hidden_size, output_size)
            )
        elif activator == 3:
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Softmax(dim=1),  
                nn.Linear(hidden_size, output_size)
            )
        elif activator == 4:
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(),  
                nn.Linear(hidden_size, output_size)
            )

    
    def forward(self, x):
        return self.model(x)
    
bettiDF = pd.DataFrame(columns=["ReLU", "Tanh", "Sigmoid", "Softmax", "Leaky ReLU"])

# Load and preprocess the test data
test_data = pd.read_csv("Test.csv")
test_trueResults = test_data["price_range"]  # Save IDs for reference if needed
test_features = test_data.drop(["price_range"], axis=1)

# Standardize the test data
test_features = scaler.transform(test_features)
test_features = torch.tensor(test_features, dtype=torch.float32)

# DataFrame to store Betti numbers for TDA
# DataFrame to store Betti 1 values
errror_results_df = pd.DataFrame(columns=["ReLU", "Tanh", "Sigmoid", "Softmax", "Leaky ReLU"])

for k in range(200):  # Iterate over experiments
    print(f"Iteration {k}")
    row_data = {}  # Dictionary to store Betti 1 for the current iteration
    for j in range(5):  # Iterate over activation functions
        print(f"Activation Function: {j}")
        
        # Define and train the model
        input_size = X_train.shape[1]
        hidden_size = 16
        output_size = 4
        model = PriceRangeClassifier(input_size, hidden_size, output_size, j)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 50
        batch_size = 32
        for epoch in range(num_epochs):
            for i in range(0, X_train.size(0), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Make predictions on the test data
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features)  # Raw logits from the model
            predicted_classes = torch.argmax(test_outputs, dim=1)
        predicted_categories = le.inverse_transform(predicted_classes.numpy())
        correct_predictions = (predicted_categories == test_trueResults).sum()
        accuracy = correct_predictions / len(predicted_categories)
        activation_name = ["ReLU", "Tanh", "Sigmoid", "Softmax", "Leaky ReLU"][j]
        row_data[activation_name] = accuracy
    errror_results_df = pd.concat([errror_results_df, pd.DataFrame([row_data])], ignore_index=True)
errror_results_df.to_csv("accuracy_ResultsTrial2.csv", index_label="Iteration")
