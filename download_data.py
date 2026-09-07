import pandas as pd
import os

# create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

cols = ['pregnancies', 'glucose', 'blood_pressure',
        'skin_thickness', 'insulin', 'bmi',
        'diabetes_pedigree', 'age', 'outcome']

df = pd.read_csv(url, names=cols)
df.to_csv('data/diabetes.csv', index=False)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nTarget distribution:")
print(df['outcome'].value_counts())