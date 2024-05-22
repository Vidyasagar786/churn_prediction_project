import numpy as np
import pandas as pd
import os
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, classification_report, recall_score, confusion_matrix,
roc_auc_score, precision_score, f1_score, roc_curve, auc )
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier

data_path="Telco-Customer-Churn.csv"
df=pd.read_csv(data_path)

# Convert TotalCharges to numeric, filling NaN values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)

# Convert SeniorCitizen to object
df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

# Replace 'No phone service' and 'No internet service' with 'No' for certain columns
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for column in columns_to_replace:
    df[column] = df[column].replace('No internet service', 'No')

# Convert 'Churn' categorical variable to numeric
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

#stratified shuffle split
strat_split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)

train_index, test_index=next(strat_split.split(df, df["Churn"]))

#creating train &test sets
start_train_set=df.loc[train_index]
start_test_Set=df.loc[test_index]
x_train=start_train_set.drop("Churn", axis=1)
y_train=start_train_set["Churn"].copy()

x_test=start_test_Set.drop("Churn", axis=1)
y_test=start_test_Set["Churn"].copy()

#CATBOOST

categorical_columns=df.select_dtypes(include=['object']).columns.tolist()

cat_model=CatBoostClassifier( random_state=0, scale_pos_weight=3 )
cat_model.fit(x_train, y_train, cat_features=categorical_columns, eval_set=(x_test, y_test))

#prediction
y_pred=cat_model.predict(x_test)
#eval metrics
accuracy, recall, roc_auc, precision=[round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

#df to store the results
model_name=['catboostmodel']
result=pd.DataFrame({'Accuracy':accuracy, 'Recall':recall, 'Roc_Auc':roc_auc, 'Precision':precision}, index=model_name)
print(result)

#save model
model_dir="/churn_prediction_project/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_pkl_path = os.path.join(model_dir, "catboost_model.pkl")
with open(model_pkl_path, 'wb') as file:
    pickle.dump(cat_model, file)