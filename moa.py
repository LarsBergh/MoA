#https://www.kaggle.com/c/lish-moa/data
#https://www.kaggle.com/c/lish-moa/overview/description
#%%
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
#%%
data_folder = "data/"

#Description of task	Predicting a receptor respons based on gene expression, cell viability, drug, dose, and treatment type
#g-	                gene expression data
#c-	                cell viability data
#cp_type	        samples
#cp_vehicle	        compound delivery method (how it is taken)
#ctrl_pertubations	control group that causes no effect
#cp_time	        treatment duration (24, 48, 72 hours)
#cp_dose	        dose of drug (high, low)

#Features
train_feat = pd.read_csv(data_folder + "train_features.csv")
test_feat = pd.read_csv(data_folder + "test_features.csv")

#Targets
train_target_scored = pd.read_csv(data_folder + "train_targets_scored.csv")
train_target_nonscored = pd.read_csv(data_folder + "train_targets_nonscored.csv")
#%%
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
for col in train_feat:
    print(col)
#%%
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
# Binary Classification Recurrent Neural Network
model = Sequential()
model.add(Dense(16, activation='relu')) 
model.add(Dense(16, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["acc"]) 

# Early stopping 
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)  
# Callbacks
mc = ModelCheckpoint('models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)   # ?


print(model.summary())
hs = model.fit(X_train, y_train, batch_size=128, epochs=40, validation_data=(X_val, y_val), verbose=1, callbacks=[es,mc])
#%%
t = tf.zeros([5,5,5,5])
t = tf.reshape(t, [625])
print(t)
# %%

# %%
