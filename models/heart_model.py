import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
import torch.optim.adam

#load data
df = pd.read_csv('data/heart.csv')

#features
x = df.drop('target',axis=1).values
y = df['target'].values

#splite
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#scale
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#save scaler
os.makedirs('models/saved', exist_ok=True)
with open('models/saved/heart_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#convert to tensor

x_train = torch.tensor(x_train,dtype=torch.float32)
x_test = torch.tensor(x_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)

class HeartNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self,x):
        return self.network(x)


#train 
model = HeartNet()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr =0.01)

#loop
for epoch in range(500):
    model.train()
    pred = model(x_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = (model(x_test) >= 0.5).float()
            acc = accuracy_score(y_test.numpy(), test_pred.numpy())
            print(f"Epoch:{epoch} - Loss:{loss.item():.4f} - Accuracy:{acc:.4f}")

#final report
model.eval()
with torch.no_grad():
    final_pred = (model(x_test) >= 0.5).float()
    print("\nFinal Classification Report:")
    print(classification_report(
        y_test.numpy(),
        final_pred.numpy(),
        target_names=['No Disease', 'Heart Disease']
    ))

#save modle 
torch.save(model.state_dict(),'models/saved/heart_model.pth')
print("\nModel saved! ✅")
print("Scaler saved! ✅")