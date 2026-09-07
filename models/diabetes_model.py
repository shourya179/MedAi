import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# load data
df = pd.read_csv('data/diabetes.csv')

# features and target
X = df.drop('outcome', axis=1).values
y = df['outcome'].values

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# save scaler for later use in web app
os.makedirs('models/saved', exist_ok=True)
with open('models/saved/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

# define network
class DiabetesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 32),    # 8 features in
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)      # 1 output
        )

    def forward(self, x):
        return self.network(x)

# train
model     = DiabetesNet()
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training diabetes model...")
for epoch in range(2000):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = (model(X_test) >= 0.5).float()
            acc = accuracy_score(y_test.numpy(), test_pred.numpy())
            print(f"Epoch:{epoch} - Loss:{loss.item():.4f} - Accuracy:{acc:.4f}")

# final evaluation
model.eval()
with torch.no_grad():
    final_pred = (model(X_test) >= 0.5).float()
    print("\nFinal Classification Report:")
    print(classification_report(
        y_test.numpy(),
        final_pred.numpy(),
        target_names=['No Diabetes', 'Diabetes']
    ))

# save model
torch.save(model.state_dict(), 'models/saved/diabetes_model.pth')
print("\nModel saved to models/saved/diabetes_model.pth")
print("Scaler saved to models/saved/scaler.pkl")