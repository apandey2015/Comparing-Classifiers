#!/usr/bin/env python
# coding: utf-8

# # Practical Application III: Comparing Classifiers
# 
# **Overview**: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.  
# 
# 

# ### Getting Started
# 
# Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  We will make use of the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.
# 
# 

# ### Problem 1: Understanding the Data
# 
# To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the **Materials and Methods** section of the paper.  How many marketing campaigns does this data represent?

# In[6]:


print("The dataset collected is related to 17 campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts.") 


# ### Problem 2: Read in the Data
# 
# Use pandas to read in the dataset `bank-additional-full.csv` and assign to a meaningful variable name.

# In[36]:


import pandas as pd
import numpy as np 
import pandas as pd 
import time

from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC



# In[37]:


df = pd.read_csv('data/bank-additional-full.csv', sep = ';')


# In[9]:


df.head()


# ### Problem 3: Understanding the Features
# 
# 
# Examine the data description below, and determine if any of the features are missing values or need to be coerced to a different data type.
# 
# 
# ```
# Input variables:
# # bank client data:
# 1 - age (numeric)
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# # related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: 'cellular','telephone')
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# # other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# # social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# 20 - nr.employed: number of employees - quarterly indicator (numeric)
# 
# Output variable (desired target):
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
# ```
# 
# 

# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


plt.figure(figsize=(6,4))
sns.countplot(data=df, x="y")
plt.title("Distribution of Marketing Response (Yes/No)")
plt.show()


# In[11]:


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="y", y="age")
plt.title("Age vs. Response to Marketing Campaign")
plt.show()


# In[12]:


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="y", y="duration")
plt.title("Call Duration vs. Response")
plt.show()


# ### Problem 4: Understanding the Task
# 
# After examining the description and data, your goal now is to clearly state the *Business Objective* of the task.  State the objective below.

# In[13]:


df.info()


# In[14]:


print("Business Objective: To enhance the bank’s strategic marketing capabilities by developing a data-driven model that predicts a customer’s likelihood of subscribing to a term deposit following a telephone marketing campaign. By using machine learning techniques, the bank seeks to optimize customer targeting, improve campaign efficiency, and increase conversion rates.")


# ### Problem 5: Engineering Features
# 
# Now that you understand your business objective, we will build a basic model to get started.  Before we can do this, we must work to encode the data.  Using just the bank information features, prepare the features and target column for modeling with appropriate encoding and transformations.

# In[10]:


df["y"] = df["y"].map({"yes": 1, "no": 0})

bank_features = [
    "age", "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "day_of_week",
    "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

X = df[bank_features]
y = df["y"]


# In[11]:


# Identify numeric and categorical columns

numeric_cols = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

categorical_cols = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "day_of_week", "poutcome"
]


# In[12]:


# Preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)


# ### Problem 6: Train/Test Split
# 
# With your data prepared, split it into a train and test set.

# In[13]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42, stratify=y)


# ### Problem 7: A Baseline Model
# 
# Before we build our first model, we want to establish a baseline.  What is the baseline performance that our classifier should aim to beat?

# In[14]:


baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)

print("\n=== BASELINE MODEL (DummyClassifier) ===")
print("Training Accuracy:", baseline.score(X_train, y_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[15]:


baseline_accuracy = baseline.score(X_test, y_test)

plt.figure(figsize=(6,4))
plt.bar(["Dummy (Most Frequent)"], [baseline_accuracy])
plt.ylim(0,1)
plt.title("Baseline Model Accuracy (DummyClassifier)")
plt.ylabel("Accuracy")
plt.show()


# In[16]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Pred No", "Pred Yes"],
            yticklabels=["Actual No", "Actual Yes"])
plt.title("DummyClassifier Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# ### Problem 8: A Simple Model
# 
# Use Logistic Regression to build a basic model on your data.  

# In[17]:


log_reg_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("logreg", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42, stratify=y)

log_reg_model.fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test)

print("Simple Logistic Regression Model")




# ### Problem 9: Score the Model
# 
# What is the accuracy of your model?

# In[18]:


print("Training Accuracy:", log_reg_model.score(X_train, y_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred))


# ### Problem 10: Model Comparisons
# 
# Now, we aim to compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models.  Using the default settings for each of the models, fit and score each.  Also, be sure to compare the fit time of each of the models.  Present your findings in a `DataFrame` similar to that below:
# 
# | Model | Train Time | Train Accuracy | Test Accuracy |
# | ----- | ---------- | -------------  | -----------   |
# |     |    |.     |.     |

# In[38]:


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True)
}
   
results = []

for name, model in models.items():
    
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    
    train_time = end_time - start_time
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append([name, train_time, train_acc, test_acc, roc_auc])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Train Time", "Train Accuracy", "Test Accuracy", "ROC-AUC"]
)

results_df


# In[25]:


plt.figure(figsize=(10, 6))

x = range(len(results_df))

plt.bar(x, results_df["Train Accuracy"], width=0.4, label="Train Accuracy")
plt.bar([i + 0.4 for i in x], results_df["Test Accuracy"], width=0.4, label="Test Accuracy")

plt.xticks([i + 0.2 for i in x], results_df["Model"], rotation=20)
plt.ylabel("Accuracy")
plt.title("Model Comparison: Train vs Test Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


# In[26]:


plt.figure(figsize=(10, 6))

plt.bar(results_df["Model"], results_df["Train Time"])

plt.ylabel("Seconds")
plt.title("Model Training Time Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# In[27]:


plt.figure(figsize=(8, 5))

sorted_df = results_df.sort_values("Test Accuracy", ascending=False)

plt.bar(sorted_df["Model"], sorted_df["Test Accuracy"])
plt.ylabel("Test Accuracy")
plt.title("Model Test Accuracy (Best to Worst)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# In[28]:


print("Among the four models evaluated, SVM achieved the highest test accuracy, but at a very high computational cost. Logistic Regression delivered nearly identical performance with much faster training and greater interpretability, making it the most practical choice. Decision Trees showed overfitting, and KNN underperformed in generalization. Overall, Logistic Regression offers the best balance for this task.")


# ### Problem 11: Improving the Model
# 
# Now that we have some basic models on the board, we want to try to improve these.  Below, we list a few things to explore in this pursuit.
# 
# - More feature engineering and exploration.  For example, should we keep the gender feature?  Why or why not?
# - Hyperparameter tuning and grid search.  All of our models have additional hyperparameters to tune and explore.  For example the number of neighbors in KNN or the maximum depth of a Decision Tree.  
# - Adjust your performance metric

# In[29]:


print("Gender Feature: No, we should not keep the gender feature in the final model as because it offers minimal benefit while posing fairness and interpretability concerns.")


# In[40]:


models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.1, 1, 10]}
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [5, 15, 25]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [3, 5, 8]}
    ),
    "SVM": (
        SVC(probability=True),
        {"model__C": [0.1, 1, 10]}
    )
}


# In[41]:


tuned_models = {}
results_tuned = []

for name, (model, param_grid) in models.items():
    
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",   # better metric
        cv=3,
        n_jobs=-1
    )
    
    start = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - start
    
    best_model = grid.best_estimator_
    
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    results_tuned.append([
        name,
        train_time,
        best_model.score(X_train, y_train),
        best_model.score(X_test, y_test),
        roc_auc_score(y_test, y_prob)
    ])
    
    tuned_models[name] = best_model


# In[42]:


tuned_results_df = pd.DataFrame(
    results_tuned,
    columns=["Model", "Train Time", "Train Accuracy", "Test Accuracy", "ROC-AUC"]
)

tuned_results_df


# In[43]:


roc_comparison_df = results_df[["Model", "ROC-AUC"]].merge(
    tuned_results_df[["Model", "ROC-AUC"]],
    on="Model",
    suffixes=(" (Default)", " (Tuned)")
)


# In[44]:


x = np.arange(len(roc_comparison_df))
width = 0.35

plt.figure()
plt.bar(x - width/2, roc_comparison_df["ROC-AUC (Default)"], width)
plt.bar(x + width/2, roc_comparison_df["ROC-AUC (Tuned)"], width)

plt.xticks(x, roc_comparison_df["Model"], rotation=20)
plt.xlabel("Model")
plt.ylabel("ROC-AUC")
plt.title("ROC-AUC: Default vs Tuned Models")
plt.legend(["Default", "Tuned"])
plt.show()


# In[45]:


print("The tuned models were compared based on accuracy and training time. The SVM achieved the best test performance but required the most training time. Logistic Regression provided strong, stable results with fast training and clear interpretability. The Decision Tree showed overfitting, and KNN performed the weakest. Overall, SVM is best for accuracy, while Logistic Regression is best for efficiency and interpretability. Hyperparameter tuning improves ROC–AUC for most models, particularly Decision Trees, but Logistic Regression remains the most robust and efficient classifier")


# In[46]:


print("Other Feature engineering and next steps: The most useful features captured contact history, how often customers were contacted, their financial burden, the contact channel, and age group, improving accuracy and interpretability without added complexity.")


# ##### Questions

# In[47]:


print("How would model performance change if predictions were made before any customer contact?")
print("How frequently should the model be retrained as new campaign data becomes available?")


# In[ ]:




