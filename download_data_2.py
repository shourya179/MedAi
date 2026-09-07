import pandas as pd
import os

os.makedirs('data', exist_ok=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

cols = ['age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv(url, names=cols)

# clean — replace ? with NaN and drop
df = df.replace('?', float('nan'))
df = df.dropna()

# convert target to binary (0=no disease, 1=disease)
df['target'] = (df['target'] > 0).astype(int)

df.to_csv('data/heart.csv', index=False)

print("Shape:", df.shape)
print("\nTarget distribution:")
print(df['target'].value_counts())
print("\nFirst 5 rows:")
print(df.head())