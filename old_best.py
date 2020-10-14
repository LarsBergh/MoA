#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, ActivityRegularization
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l1, l2, L1L2
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#------------------------ Loading data ------------------------#
#data_folder = "/kaggle/input/lish-moa/"
#output_folder = "/kaggle/working/"
#https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/
#https://developer.nvidia.com/rdp/cudnn-download
data_folder = "data/"
output_folder = "output/"

X = pd.read_csv(data_folder + "train_features.csv")
y = pd.read_csv(data_folder + "train_targets_scored.csv")
X_submit = pd.read_csv(data_folder + "test_features.csv")

#Description of task	Predicting a receptor respons based on gene expression, cell viability, drug, dose, and treatment type
#cp_type	        trt_cp (treatment), ctl_vehicle (control group)
#cp_time	        treatment duration (24, 48, 72 hours)
#cp_dose	        dose of drug (high, low)
#c-	                cell viability data
#g-	                gene expression data

#------------------------ Subsetting data ------------------------#
#Create subsets for train data
print("X, y, X_submit shape before id remove: ", X.shape, y.shape, X_submit.shape)
y_cols = y.columns
X = X.iloc[:, 1:]
y = y.iloc[:, 1:]

#if os.path.exists("/kaggle/working/submission.csv"):
 #   os.remove("/kaggle/working/submission.csv")


#get subsets for submit data
X_id_submit = X_submit.iloc[:, 0]
X_submit = X_submit.iloc[:, 1:]
print("X, y, X_submit shape after id remove: " ,X.shape, y.shape, X_submit.shape)

#------------------------ Exporatory Data Analysis ------------------------#
#Show distribution of amount of labels per row
#sns.displot(y.sum(axis=1))
print(y.sum(axis=1).value_counts().sort_index(axis=0))
print(100-((303+55+13+6)/len(y)*100), " percent has 0,1 or 2 labels")


#------------------------ Encoding and scaling dataframe columns ------------------------#
def encode_scale_df(df):
    print("df before ecode/scale ", df)

    #Encode variables
    enc_time_df = pd.get_dummies(df['cp_time'])
    enc_dose_df = pd.get_dummies(df['cp_dose'])
    enc_type_df = pd.get_dummies(df['cp_type'])

    #Drop encoded variable columns for replacement
    df = df.drop(["cp_time", "cp_dose", "cp_type"], axis=1)

    #Scale all variables that are left (all numerical)
    df=(df-df.min())/(df.max()-df.min())
    print("Min/Max per column of X --> Confirming that cell and gene columns are normalized")
    print(pd.concat([df.min(),df.max()],axis=1))

    #Compile new df from encoded vars & scaled vars
    df = pd.concat([enc_dose_df, enc_time_df, enc_type_df, df],axis=1)
    print("X shape after encoding and dropping: " , df.shape)
    print("X column names after encoding ", df.columns)
    print("df after ecode/scale", df)
    return df

#Scale and encode X_submit and X dataframe
X_submit = encode_scale_df(X_submit)
X = encode_scale_df(X)

#Creates a variable that encodes no target prediction as extra column
y_binary = (y.sum(axis=1) == 0).astype(int)
y = pd.concat([y, y_binary], axis=1)

#------------------------ Splitting data ------------------------#
#Train and validation data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

#Print resulting shapes of splitted datasets
print("X_train, y_train shape: ", X_train.shape, y_train.shape)
print("X_val, y_val shape: ", X_val.shape, y_val.shape)
print("X_test, y_test shape: ", X_test.shape, y_test.shape)



model = Sequential()
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='elu')) 
model.add(Dropout(0.15))
model.add(Dense(64, activation='elu')) 
model.add(Dense(207, activation='softmax')) 
opti = SGD(lr=0.05, momentum=0.98)
model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=4, epochs=50)

#Get validation loss/acc
results = model.evaluate(X_val, y_val, batch_size=1)

#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)
#%%
#Create dataframe and CSV for submission
submit_df = np.concatenate((np.array(X_id_submit).reshape(-1,1), y_submit[:,:206]), axis=1)
pd.DataFrame(submit_df).to_csv(path_or_buf=output_folder + "submission.csv", index=False, header=y_cols)