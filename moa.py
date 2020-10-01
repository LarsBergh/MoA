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

#Get X and y data
X = pd.read_csv(data_folder + "train_features.csv")
y = pd.read_csv(data_folder + "train_targets_scored.csv")
print("X, y shape before id remove: ", X.shape, y.shape)
#Subset dataframe sig_id from dataframe
X = X.iloc[:, 1:]
y = y.iloc[:, 1:]
print("X, y shape after id remove: " ,X.shape, y.shape)
#Subset X and y so only cp_type ==  trt_cp (treatment) lines are in the dataset
filt_cp_type = X['cp_type'] == "trt_cp"
X = X[filt_cp_type]
y = y[filt_cp_type]
print("X, y shape after control group removal: " ,X.shape, y.shape)
#Encode treatment duration (cp_time), dosing (cp_dose)
enc_time_df = pd.get_dummies(X['cp_time'])
enc_dose_df = pd.get_dummies(X['cp_dose'])
X = X.drop(["cp_time", "cp_dose", "cp_type"], axis=1)
X = pd.concat([enc_dose_df, enc_time_df, X],axis=1)
print("X shape after encoding and dropping: " , X.shape)
print("X column names after encoding ", X.columns)

#Train and validation data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
#%%
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)


#%%
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


#%%
#Print min and max gene expression/cell viability for 2 columns
print("Max, Min gene expression: ", train_feat["g-0"].max(), train_feat["g-0"].min())
print("Max, Min cell viability: ", train_feat["c-0"].max(), train_feat["c-0"].min())
#%%
sns.distplot(train_feat["g-0"], color='green')
sns.distplot(train_feat["c-0"], color='red')
#%%

print(train_feat["cp_type"].value_counts())
print(train_feat["cp_dose"].value_counts())

print(train_feat[train_feat['c-0'] < -5].shape)

# %%



#%%
# Binary Classification Recurrent Neural Network
model = Sequential()
model.add(Dense(16, activation='relu')) 
model.add(Dense(16, activation='relu')) 
model.add(Dense(1, activation='softmax')) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["acc"]) 
hs = model.fit(X_train, y_train, batch_size=128, epochs=2)
print(model.summary())
#%%

#model.evaluate()
# Early stopping 
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)  
# Callbacks
#mc = ModelCheckpoint('models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)   # ?

#hs = model.fit(X_train, y_train, batch_size=128, epochs=40, validation_data=(X_val, y_val), verbose=1, callbacks=[es,mc])

#%%