#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[2]:


data = pd.read_csv("C:/Users/Krushil Dobariya/PYTHON/PROJECTS/AIML PROJECT/Data/Employee-Attrition.csv")


# In[3]:


data.head(2)


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isna().sum()


# In[7]:


data.duplicated().sum()


# In[8]:


data.drop(['DailyRate','EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis=1, inplace=True)


# In[9]:


data.head(2)


# In[11]:


data['Attrition']=data['Attrition'].map({'Yes':1,'No':0})
def map_categorical_features(data):
    mappings = {
        "BusinessTravel": {"Travel_Rarely": 0, "Travel_Frequently": 1, "Non-Travel": 2},
        "Department": {"Sales": 0, "Research & Development": 1, "Human Resources": 2},
        "EducationField": {
            "Life Sciences": 0, "Other": 1, "Medical": 2, "Marketing": 3,
            "Technical Degree": 4, "Human Resources": 5
        },
        "Gender": {"Male": 0, "Female": 1},
        "JobRole": {
            "Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2,
            "Manufacturing Director": 3, "Healthcare Representative": 4, "Manager": 5,
            "Sales Representative": 6, "Research Director": 7, "Human Resources": 8
        },
        "MaritalStatus": {"Single": 0, "Married": 1, "Divorced": 2},
        "OverTime": {"Yes": 1, "No": 0}
    }
    
    for col, mapping in mappings.items():
        data[col] = data[col].map(mapping)
    
    return data


# In[13]:


data = map_categorical_features(data)
data.head(1)


# In[17]:


plt.figure(figsize=(10, 8))
sns.heatmap(data.iloc[:, :10].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()


# In[23]:


def chi_square_test(data, column):
    crosstab = pd.crosstab(data[column], data['Attrition'])
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    print(f"Chi-Square Test for {column}:")
    print(f"Chi2 Statistic = {chi2:.4f}, p-value = {p:.4f}\n")
    return p


# In[25]:


categorical_columns = ['Department', 'JobRole', 'Gender']
for col in categorical_columns:
    chi_square_test(data, col)


# In[31]:


plt.figure(figsize=(15, 5))

for i, col in enumerate(categorical_columns):
    plt.subplot(1, 3, i+1) 
    sns.countplot(x=data[col], hue=data['Attrition'], palette='coolwarm', data=data)
    plt.xticks()
    plt.title(f'Attrition by {col}')

plt.tight_layout()
plt.show()


# In[35]:


drop_columns = [
    'Gender', 'PerformanceRating', 'BusinessTravel', 'HourlyRate', 
    'PercentSalaryHike', 'Education', 'TrainingTimesLastYear', 
    'WorkLifeBalance', 'RelationshipSatisfaction'
]
data= data.drop(columns=drop_columns)


# In[37]:


data.columns


# In[39]:


data.to_csv("C:/Users/Krushil Dobariya/PYTHON/PROJECTS/AIML PROJECT/Data/Preprocess-Dataset.csv")


# In[ ]:




