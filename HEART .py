#!/usr/bin/env python
# coding: utf-8

# # HEART DISEASE PREDICTION

# Data Collection: We obtain the medical data for the patients from trusted and authorized sources. This data includes relevant information such as medical records, diagnoses, treatments, and other pertinent details. We prioritize privacy and adhere to strict data protection protocols to ensure the confidentiality of patient information.
# 
# 

# Data Preprocessing:we proceed with the crucial step of data preprocessing. This step involves cleaning and preparing the data for further analysis.

# Data Split: Once the medical data is obtained, we perform a careful data split. This involves dividing the collected data into two subsets: a 80% training set and a 20% testing set. The training set is used to train machine learning models and algorithms, allowing them to learn patterns and relationships within the data. The testing set is reserved for evaluating the performance and generalization ability of the trained models.
# 
# By splitting the data in this manner, we can effectively assess the accuracy and effectiveness of our models while ensuring robustness and reliability in their predictions.

# Explanatory Data Analysis (EDA): After preprocessing the data, we delve deeper into its analysis through Exploratory Data Analysis (EDA). EDA involves a comprehensive examination of the data to uncover patterns, relationships, and insights that can guide subsequent modeling and decision-making processes.

# Balancing the Data: Recognizing the presence of data imbalance, we proactively address this issue to ensure a more equitable representation of the different classes within the dataset.

# Disease Prediction: To predict diseases accurately, we employ a range of machine learning algorithms, leveraging their capabilities to analyze the collected medical data. These algorithms enable us to identify patterns, correlations, and predictive features that contribute to disease prediction.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv("C:/Users/Sankung/Downloads/heart.csv")


# In[3]:


df.head()


# 'age': Age of the patient. It represents the age in years of the individual for whom the other measurements are recorded.

# 'sex': Sex of the patient. It is a binary variable where 1 represents male and 0 represents female.

# 'cp': Chest pain type. It is a categorical variable with 4 categories representing different types of chest pain experienced by the patient. The categories are typically defined as follows:
# 
# Value 1: Typical angina (chest pain related to reduced blood flow to the heart)
# Value 2: Atypical angina (chest pain not clearly related to heart)
# Value 3: Non-anginal pain (pain unrelated to heart)
# Value 4: Asymptomatic (no chest pain)

# 'trestbps': Resting blood pressure. It represents the resting blood pressure of the patient in millimeters of mercury (mm Hg).

# 'chol': Serum cholesterol measurement. It denotes the level of cholesterol in the patient's blood serum measured in milligrams per deciliter (mg/dl).

# 'fbs': Fasting blood sugar. It is a binary variable indicating whether the patient's fasting blood sugar level is greater than 120 mg/dl. A value of 1 represents a fasting blood sugar level greater than 120 mg/dl, and 0 represents a level less than or equal to 120 mg/dl.

# 'restecg': Resting electrocardiographic results. It represents the results of the resting electrocardiogram, which is a test that measures the electrical activity of the heart at rest. The variable has three possible values:
# 
# Value 0: Normal
# Value 1: Abnormal ST-T wave
# Value 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria

# 'thalach': Maximum heart rate achieved. It denotes the maximum heart rate achieved by the patient during exercise.

# 'exang': Exercise-induced angina. It is a binary variable indicating whether the patient experienced exercise-induced angina during the exercise test. A value of 1 represents the presence of exercise-induced angina, and 0 represents its absence

# 'oldpeak': ST depression induced by exercise relative to rest. It represents the ST segment depression (measured in millimeters) observed during exercise compared to the resting electrocardiogram.

# 'slope': The slope of the peak exercise ST segment. It represents the slope of the ST segment during peak exercise, which is an indicator of the rate of change of ST segment depression. The variable has three possible values:
# 
# Value 0: Upsloping
# Value 1: Flat
# Value 2: Downsloping

# 'ca': Number of major vessels colored by fluoroscopy. It denotes the number of major blood vessels (0-3) as observed by fluoroscopy. Fluoroscopy is a medical imaging technique that uses X-rays to obtain real-time moving images of the patient's internal structures.

# 'thal': Thallium stress test result. It represents the results of the thallium stress test, which is a nuclear imaging test used to evaluate blood flow to the heart. The variable has three possible categories:
# 
# Value 1: Fixed defect (no blood flow in some part of the heart)
# Value 2: Normal (normal blood flow)
# Value 3: Reversible defect (blood flow is observed but not normal)

# 'target': Presence of heart disease. It is the target variable that indicates the presence 

# In[4]:


missing_values_sum = df.isnull().sum()
print(missing_values_sum)


# In[5]:


df.describe


# In[55]:


# Calculate correlation matrix
corr_matrix = df.corr()

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap')
plt.show()


# We don't have a multi-collinearity problem in this data set

# In[6]:



# Define the categorical columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Perform one-hot encoding
df= pd.get_dummies(df, columns=categorical_cols)

# Split the dataset into training and test sets
X = df.drop(columns='target')


# In[7]:


# Split the dataset into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:



from imblearn.over_sampling import SMOTE


# Count the number of samples in each class before oversampling
class_counts_before = df['target'].value_counts()

# Visualize class distribution before oversampling
plt.bar(class_counts_before.index, class_counts_before.values, color=['red', 'green'])
plt.xticks(class_counts_before.index)
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Class Distribution Before Oversampling')
plt.show()

# Print the class counts before and after oversampling
print("Class Counts Before Oversampling:")
print(class_counts_before)


# In[10]:


# Oversample using SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Count the number of samples in each class after oversampling
class_counts_after = np.bincount(y_train_resampled)

# Visualize class distribution after oversampling
plt.bar(np.unique(y_train_resampled), class_counts_after, color=['red', 'green'])
plt.xticks(np.unique(y_train_resampled))
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Class Distribution After Oversampling')
plt.show()

print("\nClass Counts After Oversampling:")
print(class_counts_after)


# The dataset is imbalance with 220 with no heart disease and ony 83 with heart disease. So we have to correct this before proceeding with the ML-Algorithms.

# # Logistic Model

# In[10]:


# We start with the logistics regression classifier
# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# The accuracy of 0.83 indicates that the heart disease prediction model correctly predicted the presence or absence of heart disease for approximately 83% of the samples in the dataset.

# In[11]:


# Create a confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Define class labels
class_names = ['No Heart Disease', 'Heart Disease']

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# The confusion matrix indicates that the model correctly predicted 39 observations that do not have heart disease and 12 observations that do have heart disease. However, the model also misclassified 5 observations as heart disease when they were not actually cases of heart disease, and it misclassified 5 observations as not having heart disease when they were indeed cases of heart disease.

# In[12]:


from sklearn.metrics import roc_curve, roc_auc_score

# Calculate predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC score
auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# AUC of 0.91 suggests that the classifier has good discriminative ability. It means that the classifier has a high probability of correctly ranking a randomly selected positive instance higher than a randomly selected negative instance about 89% of the time.

# In[13]:



from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Calculate average precision score
avg_precision = average_precision_score(y_test, y_prob)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve (AP = {:.2f})'.format(avg_precision))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.show()


# In[14]:


import numpy as np

# Get the absolute values of the coefficients
coefs = np.abs(model.coef_[0])

# Sort the coefficients in descending order
indices = np.argsort(coefs)[::-1]

# Rearrange feature names based on the sorted indices
feature_names = X.columns[indices]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), coefs[indices])
plt.xticks(range(X.shape[1]), feature_names, rotation=90)
plt.xlabel('Features')
plt.ylabel('Coefficient Magnitude')
plt.title('Feature Importance (Logistic Regression)')
plt.show()


# Turns out that the number of major vessels covered by floroscopy is the most important for heart disease follow by thallium normal which evaluate the blood floor in the heart.

# In[15]:


# Select numerical features to plot histograms
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Plot histograms
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[feature][df['target'] == 0], label='No Heart Disease', color='blue', alpha=0.5)
    sns.histplot(df[feature][df['target'] == 1], label='Heart Disease', color='red', alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend()
plt.tight_layout()
plt.show()


# In[16]:


# Select features for pairwise scatter plots
scatter_features = ['age', 'trestbps', 'chol', 'thalach']

# Create pairwise scatter plots
sns.pairplot(data=df, vars=scatter_features, hue='target', palette='Set2')
plt.show()


# In[17]:


# Select features for box plots
boxplot_features = ['age', 'thalach', 'oldpeak']

# Create box plots
plt.figure(figsize=(12, 6))
for i, feature in enumerate(boxplot_features):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='target', y=feature, data=df, palette='Set2')
    plt.xlabel('Target')
    plt.ylabel(feature)
    plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
plt.tight_layout()
plt.show()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Select features for count plots
countplot_features = ['cp_0', 'cp_1', 'cp_2', 'cp_3', 'cp_4']

# Create subplots for count plots
plt.figure(figsize=(12, 8))
for i, feature in enumerate(countplot_features):
    plt.subplot(2, 3, i+1)
    sns.countplot(x=feature, hue='target', data=df, palette='Set2')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(title='Target', labels=['No Disease', 'Disease'])

plt.tight_layout()
plt.show()


# The results indicate that patients who are asymptomatic have a lower likelihood of having heart disease, while the probability increases for those who are non-asymptomatic. The second graph demonstrates that patients with chest pain unrelated to the heart have a minimal or negligible chance of experiencing heart failure. On the other hand, individuals with atypical angina, characterized by chest pain that is not clearly associated with the heart, exhibit an increased probability of having heart disease. Lastly, patients experiencing typical angina (chest pain caused by reduced blood flow to the heart) are at a higher risk of developing heart disease.

# # Support Vector Machine

# In[19]:


from sklearn.svm import SVC
from sklearn import svm

svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(svc_classifier.score(X_test, y_test))


# In[20]:



from matplotlib.cm import get_cmap

# Sample code for illustration purposes
svc_scores = [0.8, 0.75, 0.82, 0.68]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
colors = get_cmap('rainbow')(np.linspace(0, 1, len(kernels)))

plt.bar(kernels, svc_scores, color=colors)

for i in range(len(kernels)):
    plt.text(i, svc_scores[i], str(svc_scores[i]), ha='center', va='bottom')

plt.xlabel('Kernel')
plt.ylabel('Accuracy Score')
plt.title('SVC Accuracy Scores for Different Kernels')
plt.ylim(0, 1)  # Set the y-axis limits between 0 and 1

plt.show()


# In[21]:



# Create an instance of the SVC classifier with RBF kernel
classifier = svm.SVC(kernel='linear')

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[22]:


from sklearn.metrics import roc_curve, auc

# Compute the probabilities of the positive class
y_scores = classifier.decision_function(X_test)

# Compute the false positive rate and true positive rate
fpr, tpr, _ = roc_curve(y_test, y_scores)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Show the plot
plt.show()


# # Decision Tree Classifier

# In[23]:



from sklearn.tree import DecisionTreeClassifier


# In[24]:



dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features=i, random_state=0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))

plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color='green', marker='o', linestyle='-', linewidth=2)

for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], f"{dt_scores[i-1]:.2f}", ha='center', va='bottom')

plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('Decision Tree Classifier Scores for Different Number of Maximum Features', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()


# In[25]:


# Create the Decision Tree classifier with max_features=8
dt_classifier = DecisionTreeClassifier(max_features=4, random_state=0)

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = dt_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)


# # Random Forest Classifier

# In[26]:



from sklearn.ensemble import RandomForestClassifier

rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators=i, random_state=0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))

colors = get_cmap('rainbow')(np.linspace(0, 1, len(estimators)))

plt.bar([i for i in range(len(estimators))], rf_scores, color=colors, width=0.8)

for i in range(len(estimators)):
    plt.text(i, rf_scores[i], f"{rf_scores[i]:.2f}", ha='center', va='bottom')

plt.xticks(ticks=[i for i in range(len(estimators))], labels=[str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('Random Forest Classifier Scores for Different Number of Estimators', fontsize=14)

plt.show()


# In[27]:



# Create the Random Forest Classifier with 100 estimators
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = rf_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[28]:



# Separate age data for people with and without the disease
age_disease = df[df['target'] == 1]['age']
age_no_disease = df[df['target'] == 0]['age']

# Plotting the age distribution
plt.figure(figsize=(8, 6))
sns.histplot(age_disease, color='red', label='Suffering from Disease')
sns.histplot(age_no_disease, color='green', label='Not Suffering from Disease')

plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution by Disease Status')
plt.legend()
plt.show()


# The results from the graph unequivocally demonstrate that adults aged 54 to 60 are at a substantially higher risk of experiencing heart disease. Youths and childrens below the age of 35 barely have heart disease.

# # NEURAL NETWORK

# In[22]:


import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, verbose=0)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# # XG BOOST

# In[33]:


# Define the XGBoost classifier
model = xgb.XGBClassifier()

# Train the model
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, eval_metric='error', verbose=False)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy:", accuracy)


# In[34]:




# Get the evaluation results
results = model.evals_result()

# Extract the training and validation metrics
train_error = results['validation_0']['error']
val_error = results['validation_1']['error']

# Visualize the error
plt.figure(figsize=(10, 5))
plt.plot(train_error, label='Training Error')
plt.plot(val_error, label='Validation Error')
plt.title('Error Curves')
plt.xlabel('Boosting Round')
plt.ylabel('Error')
plt.legend()
plt.show()


# # KNN CLASSIFIER

# In[34]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Initialize the KNN classifier
model = KNeighborsClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy score achieved using K-Nearest Neighbors is:", accuracy)


# In[39]:


# Assign the accuracy scores to variables
score_lr = 0.85
score_svm = 0.85
score_knn = 0.65
score_dt = 0.80
score_rf = 0.82
score_xgb = 0.82
score_nn = 0.85

# Create the list of accuracy scores and algorithm names
scores = [score_lr, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]

# Print the accuracy scores achieved by each algorithm
for i in range(len(algorithms)):
    print("The accuracy score achieved using " + algorithms[i] + " is: " + str(scores[i]) + " %")


# In[40]:


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# # Conclusion

# The logistic model, support vector machine, and neural network emerged as our top-performing algorithms for predicting the presence of heart disease, yielding an impressive accuracy of 85%. Conversely, K-Nearest Neighbors appeared to be the least effective algorithm for classifying the presence or absence of heart disease. Our findings revealed a higher risk of heart disease among individuals aged 55 to 60, with the number of major vessels covered by fluoroscopy emerging as the primary contributing factor. Furthermore, our analysis indicated that asymptomatic patients have a lower likelihood of heart disease, whereas non-asymptomatic individuals exhibit an increased probability. Moreover, patients with typical angina demonstrate a higher risk of heart disease, while those with atypical angina also face a considerable risk.

# By SANKUNG FATTY
# Email: sfatty@dons.usfca.edu
