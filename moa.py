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


#------------------------ Preprocessing of Data ------------------------#
#Get X and y data
X_load = pd.read_csv(data_folder + "train_features.csv")
y_load = pd.read_csv(data_folder + "train_targets_scored.csv")
print("X, y shape before id remove: ", X_load.shape, y_load.shape)

#Subset dataframe sig_id from dataframe
X_load = X_load.iloc[:, 1:]
y_load = y_load.iloc[:, 1:]
print("X, y shape after id remove: " ,X_load.shape, y_load.shape)

#Subset X and y so only cp_type ==  trt_cp (treatment) lines are in the dataset
filt_cp_type = X_load['cp_type'] == "trt_cp"
X = X_load[filt_cp_type]
y = y_load[filt_cp_type]
print("X, y shape after control group removal: " ,X.shape, y.shape)

#Encode treatment duration (cp_time), dosing (cp_dose)
enc_time_df = pd.get_dummies(X['cp_time'])
enc_dose_df = pd.get_dummies(X['cp_dose'])
X = X.drop(["cp_time", "cp_dose", "cp_type"], axis=1)

#Normalize cell and gene columns between 0 and 1
X=(X-X.min())/(X.max()-X.min())
print("Min/Max per column of X --> Confirming that cell and gene columns are normalized")
print(pd.concat([X.min(),X.max()],axis=1))

X = pd.concat([enc_dose_df, enc_time_df, X],axis=1)
print("X shape after encoding and dropping: " , X.shape)
print("X column names after encoding ", X.columns)
#%%
#------------------------ Exporatory Data Analysis ------------------------#
#Count how many rows have 0 targets as predictions
def count_no_target(target_matrix):
    no_target = 0
    for i in range(0, target_matrix.shape[0]):
        #If unique array == 1 only 0 is found in row so no target
        if len(np.unique(np.array(target_matrix.iloc[i, :]))) == 1:
            no_target += 1
    return no_target

#39% and 34% of rows should predict 0
print("Rows with no target pre-control group removal: ", count_no_target(y_load)/len(y_load))
print("Rows with no target post-control group removal: ", count_no_target(y)/len(y))

#Build neural net that classifies (binary) if row needs a target --> optimize this accuracy

#Build neural net that predicts the rest of the rows

#Combine prediction matrices

#------------------------ Splitting data & Modeling ------------------------#
#%%
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
#Multilabel classification baseline model
model = Sequential()
model.add(Dense(16, activation='relu')) 
model.add(Dense(16, activation='relu')) 
model.add(Dense(206, activation='softmax')) 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=1, epochs=2)
#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=1)
#Predict on test set to get final results
y_pred = model.predict(X_test)
#%%
print("Shape of predicted values: ", y_pred.shape)
print("Shape of y_test values: ", y_test.shape)
#%%
#Get max probabilities for each row of predicted matrix
for i in range(0, y_pred.shape[0]):
    print("Unique values/row, for row nr:",i, ": ", np.unique(y_pred[i]), np.unique(np.array(y_test.iloc[i])))


#%%

#model.evaluate()
# Early stopping 
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)  
# Callbacks
#mc = ModelCheckpoint('models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)   # ?

#hs = model.fit(X_train, y_train, batch_size=128, epochs=40, validation_data=(X_val, y_val), verbose=1, callbacks=[es,mc])



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