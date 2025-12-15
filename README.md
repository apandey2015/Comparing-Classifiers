# Practical Application III: Comparing Classifiers
Overview: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. We will utilize a dataset related to marketing bank products over the telephone.

# Getting Started
Our dataset comes from the UCI Machine Learning repository link. The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. We will make use of the article accompanying the dataset here for more information on the data and features.

# Problem 1: Understanding the Data
To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the Materials and Methods section of the paper. How many marketing campaigns does this data represent?
The dataset collected is related to 17 campaigns that occurred between May 2008 and November 2010, corresponding to a total of 79354 contacts.

# Problem 2: Read in the Data
Use pandas to read in the dataset bank-additional-full.csv and assign to a meaningful variable name.

# Problem 3: Understanding the Features
Examine the data description below, and determine if any of the features are missing values or need to be coerced to a different data type.

Input variables:
bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

<img width="558" height="391" alt="image" src="https://github.com/user-attachments/assets/2a3b989d-8fa3-41b2-b3e0-2cd921135256" />
<img width="695" height="468" alt="image" src="https://github.com/user-attachments/assets/0420a0b9-82b3-462e-af3d-b83744bf7d65" />
<img width="704" height="468" alt="image" src="https://github.com/user-attachments/assets/4e130fe7-74aa-47fc-bf8c-511f3f8f4178" />

# Problem 4: Understanding the Task
After examining the description and data, your goal now is to clearly state the Business Objective of the task. State the objective below.

Business Objective: To enhance the bank’s strategic marketing capabilities by developing a data-driven model that predicts a customer’s likelihood of subscribing to a term deposit following a telephone marketing campaign. By using machine learning techniques, the bank seeks to optimize customer targeting, improve campaign efficiency, and increase conversion rates

# Problem 5: Engineering Features
Now that you understand your business objective, we will build a basic model to get started. Before we can do this, we must work to encode the data. Using just the bank information features, prepare the features and target column for modeling with appropriate encoding and transformations.

# Problem 6: Train/Test Split
With your data prepared, split it into a train and test set.

# Problem 7: A Baseline Model
Before we build our first model, we want to establish a baseline. What is the baseline performance that our classifier should aim to beat?
<img width="607" height="312" alt="image" src="https://github.com/user-attachments/assets/2dcb9b01-e8c6-4d73-9886-f681b9e6113b" />
<img width="523" height="468" alt="image" src="https://github.com/user-attachments/assets/7bf88b0f-c5fb-49c4-ab39-5febfe4f5589" />

# Problem 8: A Simple Model¶
Use Logistic Regression to build a basic model on your data.

# Problem 9: Score the Model
What is the accuracy of your model?
<img width="406" height="65" alt="image" src="https://github.com/user-attachments/assets/f2e2121d-d14e-44b6-b7da-59b82b620c20" />

# Problem 10: Model Comparisons
Now, we aim to compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models. Using the default settings for each of the models, fit and score each. Also, be sure to compare the fit time of each of the models. Present your findings in a DataFrame similar to that below:
<img width="621" height="168" alt="image" src="https://github.com/user-attachments/assets/a67d267e-7fae-4640-bdd5-a5fcecf53137" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/8a448e45-a902-4b44-8b2a-9cdff21f42d8" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/729e2ef6-46c0-48e9-a5da-8a85398eab89" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/fb2ba4df-26a0-479c-96c7-b27778c37f50" />
Observation: Among the four models evaluated, SVM achieved the highest test accuracy, but at a very high computational cost. Logistic Regression delivered nearly identical performance with much faster training and greater interpretability, making it the most practical choice. Decision Trees showed overfitting, and KNN underperformed in generalization. Overall, Logistic Regression offers the best balance for this task.

Problem 11: Improving the Model
Now that we have some basic models on the board, we want to try to improve these. Below, we list a few things to explore in this pursuit.

More feature engineering and exploration. For example, should we keep the gender feature? Why or why not?
Hyperparameter tuning and grid search. All of our models have additional hyperparameters to tune and explore. For example the number of neighbors in KNN or the maximum depth of a Decision Tree.
Adjust your performance metric

Gender Feature: No, we should not keep the gender feature in the final model as because it offers minimal benefit while posing fairness and interpretability concerns.
<img width="637" height="166" alt="image" src="https://github.com/user-attachments/assets/e3bdd201-f4e5-43e3-ba81-ab637ad9c766" />
<img width="567" height="497" alt="image" src="https://github.com/user-attachments/assets/239e4047-7640-46ee-b2fc-d29bd6b6bd63" />

The tuned models were compared based on accuracy and training time. The SVM achieved the best test performance but required the most training time. Logistic Regression provided strong, stable results with fast training and clear interpretability. The Decision Tree showed overfitting, and KNN performed the weakest. Overall, SVM is best for accuracy, while Logistic Regression is best for efficiency and interpretability. Hyperparameter tuning improves ROC–AUC for most models, particularly Decision Trees, but Logistic Regression remains the most robust and efficient classifier

Other Feature engineering and next steps: The most useful features captured contact history, how often customers were contacted, their financial burden, the contact channel, and age group, improving accuracy and interpretability without added complexity.

# Questions:
How would model performance change if predictions were made before any customer contact?
How frequently should the model be retrained as new campaign data becomes available?








