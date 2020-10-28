#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import seaborn as sns
import matplotlib.pyplot as plt
<<<<<<< HEAD
import kerastuner as kt
from kerastuner.tuners import RandomSearch
=======
>>>>>>> 1af8e2f23b5a28801e0f985d41036753ce6904d7
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout, Activation, ActivityRegularization, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2, L1L2
<<<<<<< HEAD
from tensorflow.keras.optimizers import SGD, Adam
=======
from tensorflow.keras.optimizers import SGD
>>>>>>> 1af8e2f23b5a28801e0f985d41036753ce6904d7
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K

#------------------------ Loading data ------------------------#
is_kaggle = False
if is_kaggle == True:
    data_folder = "/kaggle/input/lish-moa/"
    output_folder = "/kaggle/working/"
    if os.path.exists("/kaggle/working/submission.csv"):
        os.remove("/kaggle/working/submission.csv")
else:
    data_folder = "data/"
    output_folder = "output/"

#Description of task	Predicting a receptor respons based on gene expression, cell viability, drug, dose, and treatment type
#cp_type	        trt_cp (treatment), ctl_vehicle (control group)
#cp_time	        treatment duration (24, 48, 72 hours)
#cp_dose	        dose of drug (high, low)
#c-	                cell viability data
#g-	                gene expression data

X = pd.read_csv(data_folder + "train_features.csv")
y = pd.read_csv(data_folder + "train_targets_scored.csv")
X_submit = pd.read_csv(data_folder + "test_features.csv")

#------------------------ Subsetting data ------------------------#
#Create subsets for train data
print("X, y, X_submit shape before id remove: ", X.shape, y.shape, X_submit.shape)
y_cols = y.columns

X.drop("sig_id", axis=1, inplace=True)
y.drop("sig_id", axis=1, inplace=True)

#get subsets for submit data
X_id_submit = X_submit["sig_id"]
X_submit.drop("sig_id", axis=1, inplace=True)

print("X, y, X_submit shape after id remove: " ,X.shape, y.shape, X_submit.shape)

#------------------------ Exporatory Data Analysis ------------------------#
#Show distribution of amount of labels per row
#sns.displot(y.sum(axis=1))
print(y.sum(axis=1).value_counts().sort_index(axis=0))
print(100-((303+55+13+6)/len(y)*100), " percent has 0,1 or 2 labels")

#------------------------ Parameters ------------------------#
#Encoding type --> "map", "dummy"
ENC_TYPE = "map"
<<<<<<< HEAD

#Scaling type --> "standardize", "normalize", "quantile_normal", "quantile_uniform", "power", "robust"
SC_TYPE = "quantile_uniform"

#------------------------ Encoding and scaling dataframe columns ------------------------#
def scale_df(df, scaler_type):
    df_other = df.iloc[:, :3]
    df = df.iloc[:, 3:]
    
    #https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
    scaler = {
        "standardize" : StandardScaler(), # gives values 0 mean and unit variance
        "normalize" : MinMaxScaler(), # scales values between 0 and 1
        "quantile_normal" : QuantileTransformer(output_distribution="normal"), #non-linear, maps probability density to uniform distribution
        "quantile_uniform" : QuantileTransformer(output_distribution="uniform"), #matches values to gaussian distribution
        "power": PowerTransformer(method="yeo-johnson"), # stabilizes variance, minimizes skewness, applies zero mean unit variance also
        "robust" : RobustScaler() #scaling robust to outliers. removes median, scales data to IQR 
    }

    sc = scaler[scaler_type]
    print("Scaler used: ", sc)
    return pd.concat([df_other, pd.DataFrame(sc.fit_transform(df), columns=df.columns)], axis=1)

def encode_df(df, encoder_type):
    cols = ["cp_type", "cp_time", "cp_dose"]
    enc = pd.DataFrame()

    if encoder_type == "dummy":
        enc = pd.get_dummies(df[cols], columns=cols)

    elif encoder_type == "map":           
        enc_type = df['cp_type'].map({"ctl_vehicle": 0, "trt_cp": 1})
        enc_time = df['cp_time'].map({24: 0, 48: 0.5, 72: 1})
        enc_dose = df['cp_dose'].map({'D1': 0, 'D2': 1})
        enc = pd.concat([enc_type, enc_time, enc_dose], axis=1)

    df = df.drop(cols, axis=1)

    return pd.concat([enc, df],axis=1)


print("X before scaling it with", SC_TYPE, X)

#Scaling values
X_submit = scale_df(df=X_submit, scaler_type=SC_TYPE)
X = scale_df(df=X, scaler_type=SC_TYPE)

print("X after scaling it with", ENC_TYPE, X)

#Encoding numerical and categorical vars
X = encode_df(df=X, encoder_type=ENC_TYPE)
X_submit = encode_df(df=X_submit, encoder_type=ENC_TYPE)

print("X after encoding it with", ENC_TYPE, X)
#%%
def plot_scaled_df(df, scaler_type):
    # Create four polar axes and access them through the returned array
    fig, axs = plt.subplots(3,3, figsize=(20,20))
    count = 0
    fig.suptitle("Columns scaled with " + SC_TYPE, fontsize=22)
    for i in range(0, 3):
        for j in range(0, 3):
            axs[i, j].hist(x=X.loc[:, X.columns[count+ list(X.columns).index("g-0")]], bins=50)
            axs[i, j].set_title("Distribution G-" + str(count))
            count += 1

plot_scaled_df(df=X, scaler_type=SC_TYPE)
=======

#Scaling type --> "standardize", "normalize", "quantile_normal", "quantile_uniform", "power", "robust"
SC_TYPE = "quantile_uniform"

#------------------------ Encoding and scaling dataframe columns ------------------------#
def scale_df(df, scaler_type):
    df_other = df.iloc[:, :3]
    df = df.iloc[:, 3:]
    
    #https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
    scaler = {
        "standardize" : StandardScaler(), # gives values 0 mean and unit variance
        "normalize" : MinMaxScaler(), # scales values between 0 and 1
        "quantile_normal" : QuantileTransformer(output_distribution="normal"), #non-linear, maps probability density to uniform distribution
        "quantile_uniform" : QuantileTransformer(output_distribution="uniform"), #matches values to gaussian distribution
        "power": PowerTransformer(method="yeo-johnson"), # stabilizes variance, minimizes skewness, applies zero mean unit variance also
        "robust" : RobustScaler() #scaling robust to outliers. removes median, scales data to IQR 
    }

    sc = scaler[scaler_type]
    print("Scaler used: ", sc)
    return pd.concat([df_other, pd.DataFrame(sc.fit_transform(df), columns=df.columns)], axis=1)

def encode_df(df, encoder_type):
    cols = ["cp_type", "cp_time", "cp_dose"]
    enc = pd.DataFrame()

    if encoder_type == "dummy":
        enc = pd.get_dummies(df[cols], columns=cols)

    elif encoder_type == "map":           
        enc_type = df['cp_type'].map({"ctl_vehicle": 0, "trt_cp": 1})
        enc_time = df['cp_time'].map({24: 0, 48: 0.5, 72: 1})
        enc_dose = df['cp_dose'].map({'D1': 0, 'D2': 1})
        enc = pd.concat([enc_type, enc_time, enc_dose], axis=1)

    df = df.drop(cols, axis=1)

    return pd.concat([enc, df],axis=1)

#Scale and encode X_submit and X dataframe
print("X before scaling it with", SC_TYPE, X)

X_submit = scale_df(df=X_submit, scaler_type=SC_TYPE)
X = scale_df(df=X, scaler_type=SC_TYPE)

print("X after scaling it with", ENC_TYPE, X)

X = encode_df(df=X, encoder_type=ENC_TYPE)
X_submit = encode_df(df=X_submit, encoder_type=ENC_TYPE)

print("X after encoding it with", ENC_TYPE, X)
#%%
# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(3,3, figsize=(20,20))
count = 0
fig.suptitle("Columns scaled with " + SC_TYPE, fontsize=22)
for i in range(0, 3):
    for j in range(0, 3):
        axs[i, j].hist(x=X.loc[:, X.columns[count+ list(X.columns).index("g-0")]], bins=50)
        axs[i, j].set_title("Distribution G-" + str(count))
        count += 1
>>>>>>> 1af8e2f23b5a28801e0f985d41036753ce6904d7
#%%
#------------------------ Splitting data ------------------------#
#Train and validation data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

#Print resulting shapes of splitted datasets
print("X_train, y_train shape: ", X_train.shape, y_train.shape)
print("X_val, y_val shape: ", X_val.shape, y_val.shape)
print("X_test, y_test shape: ", X_test.shape, y_test.shape)

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units',min_value=32,max_value=512,step=32), activation='relu'))
    model.add(Dense(206, activation='softmax'))                                    
    model.compile(optimizer=Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])),loss='binary_crossentropy',metrics=['binary_crossentropy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='binary_crossentropy',
    max_trials=5,
    executions_per_trial=1,
    directory='output',
    project_name='MoA target')

tuner.search(X, y, epochs=5, validation_data=(X_val, y_val))
models = tuner.get_best_models(num_models=2)
tuner.results_summary()
#%%
y_pred1 = models[0].predict(X_test)
y_pred2 = models[1].predict(X_test)
print(y_pred1, y_pred2)

#%%
model = Sequential()
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='selu')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu')) 
model.add(Dense(206, activation='softmax')) 

opti = SGD(lr=0.05, momentum=0.98)
early_stop = EarlyStopping(monitor='val_loss', patience=1, mode='auto')

model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=4, epochs=25, validation_data=(X_val, y_val), callbacks=[early_stop])
 
#Get validation loss/acc
results = model.evaluate(X_test, y_test, batch_size=1)
print(results)
#Predict on test set to get final results
y_pred = model.predict(X_test)

#Predict values for submit
y_submit = model.predict(X_submit)

#Create dataframe and CSV for submission
submit_df = np.concatenate((np.array(X_id_submit).reshape(-1,1), y_submit[:,:206]), axis=1)
pd.DataFrame(submit_df).to_csv(path_or_buf=output_folder + "submission.csv", index=False, header=y_cols)

#%%

<<<<<<< HEAD























def create_model(lay, acti, neur):
    model = Sequential()

    for i in range(0,lay):
        model.add(BatchNormalization())
        #Add Alpha dropout (only works well with exponentials)
        if acti=="elu":
            model.add(AlphaDropout(0.2))
        #Else add regular dropout
        else:
            model.add(Dropout(0.15))
        model.add(Dense(neur, activation=acti))

=======
def create_model(lay, acti, neur):
    model = Sequential()

    for i in range(0,lay):
        model.add(BatchNormalization())
        #Add Alpha dropout (only works well with exponentials)
        if acti=="elu":
            model.add(AlphaDropout(0.2))
        #Else add regular dropout
        else:
            model.add(Dropout(0.15))
        model.add(Dense(neur, activation=acti))

>>>>>>> 1af8e2f23b5a28801e0f985d41036753ce6904d7
     
    model.add(Dense(206, activation='softmax')) 
    opti = AdamW(lr=0.001, weight_decay=0.0001)
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
    return model

#Get original model --> 25 epoch, batch 4, elu, 64 neurons, 0.15 dropout

elu_model = create_model(lay=2, acti="elu", neur=64)

early_stop = EarlyStopping(monitor='val_loss', patience=2, mode='auto')   
elu_model.fit(X_train, y_train, batch_size=64, epochs=25, validation_data=(X_val, y_val), callbacks=[early_stop])

#Get validation loss/acc
results = elu_model.evaluate(X_test, y_test, batch_size=1)

#Predict on test set to get final results
y_pred = elu_model.predict(X_test)

#Predict values for submit
y_submit = elu_model.predict(X_submit)

#Create dataframe and CSV for submission
submit_df = np.concatenate((np.array(X_id_submit).reshape(-1,1), y_submit[:,:206]), axis=1)
pd.DataFrame(submit_df).to_csv(path_or_buf=output_folder + "submission.csv", index=False, header=y_cols)

#%%
<<<<<<< HEAD
model = Sequential()
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='selu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='selu')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu')) 
model.add(Dense(206, activation='softmax')) 

opti = SGD(lr=0.05, momentum=0.98)
early_stop = EarlyStopping(monitor='val_loss', patience=1, mode='auto')

model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
model.fit(X_train, y_train, batch_size=4, epochs=25, validation_data=(X_val, y_val), callbacks=[early_stop])
 
=======
>>>>>>> 1af8e2f23b5a28801e0f985d41036753ce6904d7
