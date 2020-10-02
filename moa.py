#https://www.kaggle.com/c/lish-moa/data
#https://www.kaggle.com/c/lish-moa/overview/description
#%%
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

data_folder = "data/"
#Description of task	Predicting a receptor respons based on gene expression, cell viability, drug, dose, and treatment type
#cp_type	        trt_cp (treatment), ctl_vehicle (control group)
#cp_time	        treatment duration (24, 48, 72 hours)
#cp_dose	        dose of drug (high, low)
#c-	                cell viability data
#g-	                gene expression data

#%%
#------------------------ Preprocessing of Data ------------------------#
#Get X and y data
X = pd.read_csv(data_folder + "train_features.csv")
y = pd.read_csv(data_folder + "train_targets_scored.csv")
print("X, y shape before id remove: ", X.shape, y.shape)
#%%
sns.displot(y.sum(axis=1))
print(y.sum(axis=1).value_counts().sort_index(axis=0))
print(100-((303+55+13+6)/len(y)*100), " percent has 0,1 or 2 labels")
#%%


#%%
#Subset dataframe sig_id from dataframe
X = X.iloc[:, 1:]
y = y.iloc[:, 1:]
print("X, y shape after id remove: " ,X.shape, y.shape)

#Encode treatment duration (cp_time), dosing (cp_dose), sampel type (cp_type)
enc_time_df = pd.get_dummies(X['cp_time'])
enc_dose_df = pd.get_dummies(X['cp_dose'])
enc_type_df = pd.get_dummies(X['cp_type'])
X = X.drop(["cp_time", "cp_dose", "cp_type"], axis=1)

#Normalize cell and gene columns between 0 and 1
X=(X-X.min())/(X.max()-X.min())
print("Min/Max per column of X --> Confirming that cell and gene columns are normalized")
print(pd.concat([X.min(),X.max()],axis=1))

X = pd.concat([enc_dose_df, enc_time_df, enc_type_df, X],axis=1)
print("X shape after encoding and dropping: " , X.shape)
print("X column names after encoding ", X.columns)

#------------------------ Exporatory Data Analysis ------------------------#

#Creates a variable that encodes no target prediction
y_binary = (y_load.sum(axis=1) == 0).astype(int)
print("y shape before y_binary concat ", y.shape)
y = pd.concat([y, y_binary], axis=1)
print("y shape after y_binary concat ", y.shape)

#Percentage of rows 
print("Rows with no target post-control group removal: ", np.sum(y_binary),  len(y_binary), " percentage no target: ", np.sum(y_binary)/len(y_binary)*100)
#%%

#for col in y_train.columns:
 #   print(col)
#------------------------ Splitting data ------------------------#
#Train and validation data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)
print("X_val shape: ", X_test.shape)
print("y_val shape: ", y_test.shape)
#%%
#------------------------ Multilabel classification ------------------------#
#Multilabel classification baseline model
model = Sequential()
model.add(Dense(16, activation='elu')) 
model.add(Dense(16, activation='elu')) 
model.add(Dense(207, activation='sigmoid')) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=64, epochs=5)
#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=64)
#Predict on test set to get final results
y_pred = model.predict(X_test)

#%%
#Get y pred without dummy column
print(np.array(y_test)[:,:206].shape)
print(y_pred[:,:206].shape)
#%%
print("Shape of predicted values: ", y_pred.shape)
print("Shape of y_test values: ", y_test.shape)
#%%
#Get max probabilities for each row of predicted matrix
for i in range(0, y_pred.shape[0]):
    print("Unique values/row, for row nr:",i, ": ", np.unique(y_pred[i]), np.unique(np.array(y_test.iloc[i])))

#Test set on which te results must be tested and handed in
X_test = pd.read_csv(data_folder + "test_features.csv")


#%%
#Targets

train_target_nonscored = pd.read_csv(data_folder + "train_targets_nonscored.csv")
print(train_target_nonscored.shape)
print(train_target_scored.shape)
#%%
print(train_target_scored)
for col in train_target_scored:
    print(train_target_scored[col].value_counts())
#%%
for col in train_target_nonscored:
    print(train_target_nonscored[col].value_counts())
#%%
print(train_feat.shape, test_feat.shape, train_target_scored.shape)
#%%
print(len(np.unique(train_feat["sig_id"])), np.unique(train_feat["sig_id"]))
print(len(np.unique(train_feat["cp_type"])), np.unique(train_feat["cp_type"]))
#%%
sns.distplot(train_feat["g-0"], color='green')
sns.distplot(train_feat["c-0"], color='red')
#%%
print(train_feat["cp_type"].value_counts())
print(train_feat["cp_dose"].value_counts())

print(train_feat[train_feat['c-0'] < -5].shape)

# %%



#%%