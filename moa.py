#https://www.kaggle.com/c/lish-moa/data
#https://www.kaggle.com/c/lish-moa/overview/description
#%%
#------------------------ Loading libraries ------------------------#
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, ActivityRegularization
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l1, l2, L1L2
from tensorflow.keras.optimizers import SGD
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#------------------------ Loading data ------------------------#
#data_folder = "/kaggle/input/lish-moa/"
#output_folder = "/kaggle/working/"
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
sns.displot(y.sum(axis=1))
print(y.sum(axis=1).value_counts().sort_index(axis=0))
print(100-((303+55+13+6)/len(y)*100), " percent has 0,1 or 2 labels")

#------------------------ Encoding and scaling dataframe columns ------------------------#
def pca(df, df_sub, var_req, pca_type):
    #Get subset on gene or cell data
    pca_cols = [x.startswith("g-") if pca_type == "gene" else x.startswith("c-") for x in df.columns]

    #Get df subset based on pca cols
    pca = df.loc[:,pca_cols]
    pca_sub = df_sub.loc[:,pca_cols]

    #Get PCA dataframe based on gene/cell dataframe  
    pca_fit = PCA(n_components=pca.shape[1], random_state=0).fit(pca)
    pca_fit_sub = PCA(n_components=pca_sub.shape[1], random_state=0).fit(pca_sub)

    #Get variance explained and tot variance
    var_pca = pca_fit.explained_variance_
    var_pca_sub = pca_fit_sub.explained_variance_
    tot_var = np.sum(var_pca)

    #Loop over variance until total variance exceeds required variance
    cols = []
    for pc in range(1, len(var_pca)): 

        if pca_type == "gene":
            cols.append('g-' + str(pc))

        elif pca_type == "cell":
            cols.append('c-' + str(pc))

        expl_var = np.sum(var_pca[:pc])/tot_var   
        if expl_var > var_req:
            break

    #Return PCA df
    X_pca = pd.DataFrame(PCA(n_components=pc, random_state=0).fit_transform(pca),columns=cols)
    X_sub_pca = pd.DataFrame(PCA(n_components=pc, random_state=0).fit_transform(pca_sub),columns=cols)
    return X_pca, X_sub_pca, pc

#Apply PCA on gene/cell columns of X and X_submit
g_df, g_df_sub, g_comp = pca(df=X, df_sub=X_submit, var_req=0.99, pca_type="gene")
c_df, c_df_sub, c_comp = pca(df=X, df_sub=X_submit, var_req=0.99, pca_type="cell")

print("Gene df", g_df.shape, g_df_sub, " amount of components with 80% var explained: " , g_comp)
print("Cell df", c_df.shape,  c_df_sub.shape, " amount of components with 90% var explained: " , c_comp)

def encode_scale_df(df, cols):
    print("df before ecode/scale ", df)

    #Encode variables
    enc = pd.get_dummies(df, columns=cols)

    #Drop encoded variable columns for replacement
    df = df.drop(cols, axis=1)

    #Scale all variables that are left (all numerical)
    df=(df-df.min())/(df.max()-df.min())
    print("Min/Max per column of X --> Confirming that cell and gene columns are normalized")
    print(pd.concat([df.min(),df.max()],axis=1))

    #Compile new df from encoded vars & scaled vars
    df = pd.concat([enc, df],axis=1)
    print("X shape after encoding and dropping: " , df.shape)
    print("X column names after encoding ", df.columns)
    print("df after ecode/scale", df)
    return df

main_cols = ["cp_time", "cp_dose", "cp_type"]
X = pd.concat([X[main_cols], g_df, c_df],axis=1)
X_submit = pd.concat([X_submit[main_cols], g_df_sub, c_df_sub],axis=1)

#Scale and encode X_submit and X dataframe
X = encode_scale_df(df=X, cols=main_cols)
X_submit = encode_scale_df(df=X_submit, cols=main_cols)

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

#------------------------ Ensure model reproducibility ------------------------
#Start tensorflow session and set np and tensorflow seeds for this session
np.random.seed(0)
tf.random.set_seed(0)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
tf.compat.v1.keras.backend.set_session(sess)

#------------------------ Model creation ------------------------#
def create_model(X, y, lay, acti_hid, acti_out, neur, drop, epo, opti):
    #Print model parameters
    print("Creating model with:")
    print("Activation hidden: ", acti_hid)
    print("Activation output: ", acti_out)
    print("Hidden layer count: ", lay)
    print("Neuron count per layer: ", neur)
    print("Dropout value: ", drop)
    print("Epoch count: ", epo)

    #Create model
    model = Sequential()
    
    #Create layers based on count with specified activations and dropouts
    for lay in range(0,lay):
        model.add(Dense(neur, activation=acti_hid))
        model.add(Dropout(drop))

    #Add output layer
    model.add(Dense(207, activation=acti_out)) 

    #Define optimizer and loss
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 

    #Fit and return model
    model.fit(X, y, batch_size=8, epochs=epo)
    return model


#------------------------ Random search ------------------------#
def random_search(model_count):
    best_loss = 100000
    best_params = {}

    """layers = np.array([x for x in range(1, 5)]) #Layers 1 to 5
    activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "linear"] #All acti except for "exponential" because gives NA loss
    dropout = [x for x in np.round(np.arange(0.1, 1, 0.1),1)] #dropout 0.1 to 0.9
    neurons = [8, 16, 32]
    epochs = [1,2,3,4]"""

    layers = [2,3,4] #Layers 1 to 5
    acti_hid = ["elu"] #All acti except for "exponential" because gives NA loss
    acti_out = ["softmax"] #All acti except for "exponential" because gives NA loss
    dropout = [0.2] #dropout 0.1 to 0.9
    neurons = [16, 32, 64]
    epochs = [10, 20, 30]
    optimizers = ["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd", SGD(lr=0.05, momentum=0.98)]

    for model in range(0, model_count):
        #Randomly select parameters from parameter lists
        params = {}

        para_dic = {"lay": layers, "acti_hid": acti_hid, 
                    "acti_out": acti_out, "neur": neurons, 
                    "drop": dropout, "epo": epochs,
                    "opti": optimzers}

        for key, parameters in para_dic.items():
            params[key] = random.choice(parameters)

        #Create model and fit on data on randomly selected parameters
        model = create_model(X=X_train, y=y_train, 
                            lay=params["lay"], 
                            acti_hid=params["acti_hid"], 
                            acti_out=params["acti_out"], 
                            neur=params["neur"], 
                            drop=params["drop"], 
                            epo=params["epo"],
                            opti=params["optimizers"])

        #Get validation loss/acc
        val_loss = model.evaluate(X_val, y_val, batch_size=8)
        test_loss = model.evaluate(X_test, y_test, batch_size=8)

        if test_loss[0] < best_loss:
            print("Better loss found. From ", best_loss ," to ", test_loss[0])
            best_loss = test_loss[0]
            best_params = params
            best_model = model

    return (best_params, best_loss, model)

#Run random search 
best_params, best_loss, model = random_search(model_count=1)
print("Best params: ", best_params, " Best loss: ", best_loss)

#Calculate Binary Crossentropy of model without last column
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
#%%
#------------------------ Calculate log loss -----------------------------#
val_loss = 0
test_loss = 0
for col in range(len(y_val.columns)):
    val_loss += log_loss(np.array(y_val.iloc[:,col]), y_val_pred[:,col], labels=[0,1]) / y_val.shape[1]
    test_loss += log_loss(np.array(y_test.iloc[:,col]), y_test_pred[:,col], labels=[0,1]) / y_test.shape[1]

print("Val loss: ",val_loss, "Test loss: ", test_loss)
#%%
#Predict values for submit
y_submit = model.predict(X_submit)

#Create dataframe and CSV for submission
submit_df = np.concatenate((np.array(X_id_submit).reshape(-1,1), y_submit[:,:206]), axis=1)
pd.DataFrame(submit_df).to_csv(path_or_buf=output_folder + "submission.csv", index=False, header=y_cols)


# %%
