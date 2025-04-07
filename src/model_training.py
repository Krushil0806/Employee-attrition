#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
import joblib


# In[2]:


data=pd.read_csv("C:/Users/Krushil Dobariya/PYTHON/PROJECTS/AIML PROJECT/Data/Preprocess-Dataset.csv")


# In[3]:


data.head(1)


# In[4]:


data.columns


# In[5]:


X = data.drop(columns=['Attrition','Unnamed: 0'])
y = data['Attrition']


# In[66]:


X.columns


# In[6]:


X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train1)
X_test = scaler.transform(X_test1)


# In[7]:


models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}


# In[8]:


results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    results[model_name] = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": cr
    }


# In[74]:


for model_name, model in models.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"C:/Users/Krushil Dobariya/PYTHON/PROJECTS/AIML PROJECT/Images/{model_name}_confusion_matrix.png", format='png')
    plt.close()


# In[9]:


for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Confusion Matrix:\n{result['confusion_matrix']}")
    print(f"Classification Report:\n{result['classification_report']}")
    print("-" * 50)


# In[10]:


best_model_name = max(results, key=lambda model: results[model]['accuracy'])
best_model_accuracy = results[best_model_name]['accuracy']

print(f"The best model based on accuracy is: {best_model_name}")
print(f"Accuracy: {best_model_accuracy:.4f}")


# In[52]:


best_model = RandomForestClassifier(random_state=42)
best_model.fit(X, y)


# In[64]:


joblib.dump(best_model,"C:/Users/Krushil Dobariya/PYTHON/PROJECTS/AIML PROJECT/models/model(rf).pkl")


# In[ ]:




