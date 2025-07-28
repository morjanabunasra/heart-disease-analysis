import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

file_path = r"C:\Users\marja\Downloads\archive (1)\heart_cleveland_upload.csv"
df = pd.read_csv(file_path)

print(df.columns)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.countplot(x='condition', data=df)
plt.title('Distribution of Target Variable (condition)')
plt.show()

sns.histplot(data=df, x='age', hue='condition', multiple='stack')
plt.title('Age Distribution by Condition')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(10,6))
scatter = plt.scatter(
    x=df['age'],
    y=df['chol'],
    s=df['trestbps'],
    c=df['condition'],
    cmap='coolwarm',
    alpha=0.6,
    edgecolors='w',
    linewidth=0.5
)
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Bubble Chart: Age vs Cholesterol with Blood Pressure and Condition')
plt.colorbar(scatter, label='Condition (0=No disease, 1=Disease)')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='thalach', hue='condition', data=df, palette='coolwarm')
plt.title('Scatter Plot: Max Heart Rate vs Age by Condition')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate Achieved')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='condition', y='chol', data=df, hue='condition', palette='Set2', legend=False)
plt.title('Boxplot of Cholesterol by Condition')
plt.xlabel('Condition (0=No disease, 1=Disease)')
plt.ylabel('Cholesterol')
plt.show()

selected_cols = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak', 'condition']
sns.pairplot(df[selected_cols], hue='condition', palette='coolwarm', diag_kind='kde')
plt.suptitle('Pairplot of Selected Variables by Condition', y=1.02)
plt.show()

X = df.drop('condition', axis=1)
y = df['condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))

