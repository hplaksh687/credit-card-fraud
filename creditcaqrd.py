import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("creditcard.csv")
print(df.head())
print("Shape of dataset:", df.shape)
print("\nDataset Info:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


print("\nClass distribution:")
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

def flag_transaction(transaction_data):
    prediction = model.predict(transaction_data)

    if prediction[0] == 1:
        return "🚨 FRAUD TRANSACTION"
    else:
        return "✅ NORMAL TRANSACTION"
    
    sample_transaction = X_test.iloc[0].values.reshape(1, -1)
result = flag_transaction(sample_transaction)

print("Transaction Status:", result)

