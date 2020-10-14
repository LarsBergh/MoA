#https://www.kaggle.com/c/lish-moa/data
#https://www.kaggle.com/c/lish-moa/overview/description

#Description of task	Predicting a receptor respons based on gene expression, cell viability, drug, dose, and treatment type
#cp_type	        trt_cp (treatment), ctl_vehicle (control group)
#cp_time	        treatment duration (24, 48, 72 hours)
#cp_dose	        dose of drug (high, low)
#c-	                cell viability data
#g-	                gene expression data

#%%
#google colab
#evulationary models
#Building 

#%%
#------------------------ Loading libraries ------------------------#
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l1, l2, L1L2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K

#------------------------ Parameters and constants ------------------------#
#is kaggle
is_kaggle = False

#Determines how much variance must be explained by gene and cell columns for PCA
C_VAR_REQ = None
G_VAR_REQ = None

#Amount of gene/cell PCA components to use.
C_PCA_REQ = 5 #Max 100
G_PCA_REQ = 30 #Max 772

L1_REG = 0.000000001

#Determines folds in K-fold cross validation
N_FOLD = 2

L_SPEED = 0.001
W_DECAY = 0.00001

RANDOM_STATE = None

"""fils n_param_sets of dictionaries with parameters that can be used for model building"""

"""layers = np.array([x for x in range(1, 5)]) #Layers 1 to 5
activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "linear"] #All acti except for "exponential" because gives NA loss
dropout = [x for x in np.round(np.arange(0.1, 1, 0.1),1)] #dropout 0.1 to 0.9
neurons = [8, 16, 32]
epochs = [1,2,3,4]
optimizers = ["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd", SGD(lr=0.05, momentum=0.98),Adam(learning_rate=L_SPEED)]
"""
#"selu", softmax, 3, layers, 64 neuron, 25 epoch, adam

#Apply grid search if true, random search if false
APPLY_GRID = False
N_RAND_MODELS = 1

#Define lists with parameter for random/grid search
layers = [3] #Layers 1 to 5
acti_hid = ["elu"] #All acti except for "exponential" because gives NA loss
acti_out = ["softmax"] #All acti except for "exponential" because gives NA loss
dropout = [0.15] #dropout 0.1 to 0.9
neurons = [64]
epochs = [7]
optimizers = [SGD(lr=0.05, momentum=0.95)]

#Create dictionary of the given parameters
PARAM_DIC = {"lay": layers, "acti_hid": acti_hid, 
            "acti_out": acti_out, "neur": neurons, 
            "drop": dropout, "epo": epochs,
            "opti": optimizers}

#------------------------ FUNCTIONS - select grid parameters  ------------------------#
def select_grid_parameters(param_dic):
 
    #Create list for parameter sets
    params_set_list = []

    for lay in param_dic["lay"]:
        for acti_hid in param_dic["acti_hid"]:
            for acti_out in param_dic["acti_out"]:
                for neur in param_dic["neur"]:
                    for drop in param_dic["drop"]:
                        for epo in param_dic["epo"]:
                            for opti in param_dic["opti"]:
                                params_set_list.append([lay, acti_hid, acti_out, neur, drop, epo, opti])    
    return pd.DataFrame(params_set_list, columns=param_dic.keys())

  
#------------------------ FUNCTIONS - select random parameters  ------------------------#
def select_random_parameters(n_param_sets, param_dic):
    
    #Create list for parameter sets
    par_total_list = []

    for i in range(n_param_sets):
            
        par_sub_list = []

        #Randomly select parameters from parameter lists
        for key, parameters in param_dic.items():
            par_sub_list.append(random.choice(parameters))
            
        #Append dictionary of random parameters to list
        print(par_sub_list)
        par_total_list.append(par_sub_list)
    
    return pd.DataFrame(par_total_list, columns=param_dic.keys())


#------------------------ FUNCTIONS - reduce dimensionality  ------------------------#
def pca(df, df_sub, pca_type, var_req=None, num_req=None):
    #Get subset on gene or cell data
    pca_cols = [x.startswith("g-") if pca_type == "gene" else x.startswith("c-") for x in df.columns]

    #Get df subset based on pca cols
    pca = df.loc[:,pca_cols]
    pca_sub = df_sub.loc[:,pca_cols]

    #Get PCA dataframe based on gene/cell dataframe  
    pca_fit = PCA(n_components=pca.shape[1], random_state=RANDOM_STATE).fit(pca)
    pca_fit_sub = PCA(n_components=pca_sub.shape[1], random_state=RANDOM_STATE).fit(pca_sub)

    #Get variance explained and tot variance
    var_pca = pca_fit.explained_variance_
    var_pca_sub = pca_fit_sub.explained_variance_
    tot_var = np.sum(var_pca)

    #Loop over variance until total variance exceeds required variance
    cols = []
    comp = None
    
    if num_req != None: 
        for_len = num_req
        comp = num_req

    elif var_req != None:
        for_len = len(var_pca)

    for pc in range(0, for_len): 

        if pca_type == "gene":
            cols.append('g-' + str(pc))

        elif pca_type == "cell":
            cols.append('c-' + str(pc))

        if var_req != None: 
            expl_var = np.sum(var_pca[:pc])/tot_var  

            if expl_var > var_req:
                comp = pc
                break

    #Return PCA df
    X_pca = pd.DataFrame(PCA(n_components=comp, random_state=RANDOM_STATE).fit_transform(pca),columns=cols)
    X_sub_pca = pd.DataFrame(PCA(n_components=comp, random_state=RANDOM_STATE).fit_transform(pca_sub),columns=cols)

    return X_pca, X_sub_pca

#------------------------ FUNCTIONS - Encode & scale dataframe vars  ------------------------#
def encode_scale_df(df, cols):
    
    #Create encode df and drop encoded vars from df (all except for cp_type)
    enc = pd.get_dummies(df[cols], columns=cols)

    #Drop encoded vars from df
    df = df.drop(cols, axis=1)

    #Scale all variables that are left (all numerical)
    df=(df-df.min())/(df.max()-df.min())

    #Compile new df from encoded vars & scaled vars
    df = pd.concat([enc, df],axis=1)

    return df

#------------------------ FUNCTIONS - Create DL model  ------------------------#
def create_model(X_train, X_val, y_train, y_val, lay, acti_hid, acti_out, neur, drop, epo, opti):
    #Print model parameters
    print("Creating model with:")
    print("Activation hidden: ", acti_hid)
    print("Activation output: ", acti_out)
    print("Hidden layer count: ", lay)
    print("Neuron count per layer: ", neur)
    print("Dropout value: ", drop)
    print("Epoch count: ", epo)
    print("Optimizer: ", opti)

    
    #Create model
    model = Sequential()
    
    #Create layers based on count with specified activations and dropouts
    for l in range(0,lay):
        model.add(BatchNormalization())
        model.add(Dense(neur, activation=acti_hid, activity_regularizer=L1L2(L1_REG)))
        
        #Add dropout except for last layer
        if l != lay - 1:
            model.add(Dropout(drop))

    #Add output layer
    model.add(Dense(206, activation=acti_out)) 

    #Define optimizer and loss
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=["acc"]) 
    
    #Define callbacks
    hist = History()
    early_stop = EarlyStopping(monitor='val_loss', patience=2, mode='auto')

    #Fit and return model and loss history
    model.fit(X_train, y_train, batch_size=64, epochs=epo, validation_data=(X_val, y_val), callbacks=[early_stop, hist])
    print(model.summary())
    return model, hist

#------------------------ FUNCTIONS - Apply K-fold on model given params  ------------------------#
def k_fold(X, y, n_fold, params):
    print(X.shape, y.shape)
    #Split data into train and test. train will be splitted later in K-fold
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print("X_train, X_test, y_train, y_test shape: ", X.shape, X_test.shape, y.shape, y_test.shape)


    best_loss = 100000
    best_params = {}

    #Loop over parameter rows
    for i in range(params.shape[0]):  
        
        loss, acc, val_loss, val_acc, test_loss, = 0, 0, 0, 0, 0
        loss_history = []
        best_loss = 100000

        #For each parameter sec Kfold
        for train_i, test_i in KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE).split(X):
                
            #Define train and test for Kfolds
            X_train = X.iloc[train_i,:]
            X_val = X.iloc[test_i, :]
            y_train = y.iloc[train_i, :]
            y_val = y.iloc[test_i, :]

            #Create a model for each split 
            model, hist = create_model(
                    X_train=X_train, X_val=X_val, 
                    y_train=y_train, y_val=y_val, 
                    lay=params.loc[i, "lay"], 
                    acti_hid=params.loc[i, "acti_hid"], 
                    acti_out=params.loc[i, "acti_out"], 
                    neur=params.loc[i, "neur"], 
                    drop=params.loc[i, "drop"], 
                    epo=params.loc[i, "epo"],
                    opti=params.loc[i, "opti"])

            #Get total losses/acc of all k-folds
            h = hist.history
            loss += h["loss"][-1]
            acc += h["acc"][-1]
            val_loss += h["val_loss"][-1]
            val_acc += h["val_acc"][-1]
            test_loss += model.evaluate(X_test, y_test, batch_size=1)[0]

        #Find best loss and model
        if test_loss/n_fold < best_loss:
            best_model = model
            best_hist = hist.history


        #Add average model performance across all k-folds      
        params.loc[i, "loss"] = loss/n_fold
        params.loc[i, "acc"] = acc/n_fold
        params.loc[i, "val_loss"] = val_loss/n_fold
        params.loc[i, "val_acc"] = val_acc/n_fold
        params.loc[i, "test_loss"] = test_loss/n_fold   
            
    return params, best_model, best_hist


#------------------------ Paths & Loading data ------------------------#
if is_kaggle == True:
    data_folder = "/kaggle/input/lish-moa/"
    output_folder = "/kaggle/working/"
    if os.path.exists("/kaggle/working/submission.csv"):
        os.remove("/kaggle/working/submission.csv")
else:
    data_folder = "data/"
    output_folder = "output/"

X= pd.read_csv(data_folder + "train_features.csv")
y = pd.read_csv(data_folder + "train_targets_scored.csv")
X_submit = pd.read_csv(data_folder + "test_features.csv")



#------------------------ Splitting id & data columns ------------------------#
print("X, y, X_submit shape before id remove: ", X.shape, y.shape, X_submit.shape)
y_cols = y.columns
X = X.iloc[:, 1:]
y = y.iloc[:, 1:]

X_id_submit = X_submit.iloc[:, 0]
X_submit = X_submit.iloc[:, 1:]

print("X, y, X_submit shape after id remove: " ,X.shape, y.shape, X_submit.shape)



#------------------------ Exporatory Data Analysis ------------------------#
#Show distribution of amount of labels per row
if is_kaggle == False:
    sns.displot(y.sum(axis=1))
print(y.sum(axis=1).value_counts().sort_index(axis=0))
print(100-((303+55+13+6)/len(y)*100), " percent has 0,1 or 2 labels")



#------------------------ Encoding and scaling dataframe columns ------------------------#
#Apply PCA on gene/cell columns of X and X_submit
print("Before PCA shape", X.shape, y.shape)

g_df, g_df_sub = pca(df=X, df_sub=X_submit, pca_type="gene", var_req=None, num_req=G_PCA_REQ)
c_df, c_df_sub = pca(df=X, df_sub=X_submit, pca_type="cell", var_req=None, num_req=C_PCA_REQ)

print("Gene df", g_df.shape, "Gene df submit: ", g_df_sub.shape, " with", G_VAR_REQ, "% var explained: ")
print("Cell df", c_df.shape, "Cell df",  c_df_sub.shape, " with", C_VAR_REQ, "% var explained: ")

#Combine main columns with PCA columns into 1 dataframe for X and X_submit
main_cols = ["cp_time", "cp_dose", "cp_type"]
X = pd.concat([X[main_cols], g_df, c_df],axis=1)
X_submit = pd.concat([X_submit[main_cols], g_df_sub, c_df_sub],axis=1)

#Scale and encode X_submit and X dataframe/ remove control group rows
X = encode_scale_df(df=X, cols=main_cols)
X_submit = encode_scale_df(df=X_submit, cols=main_cols)
print("after ecoding and scaling ", X.shape, y.shape)



#------------------------ Modeling ------------------------#
#Ensure model reproducability
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
tf.compat.v1.keras.backend.set_session(sess)


#Select random/grid parameters
grid_params = select_grid_parameters(PARAM_DIC)
rand_params = select_random_parameters(N_RAND_MODELS, PARAM_DIC)

#Apply K_fold on grid/random parameters
if APPLY_GRID == True:
    model_log, best_model, best_hist = k_fold(X=X, y=y, n_fold=N_FOLD, params=grid_params)
else:
    model_log, best_model, best_hist = k_fold(X=X, y=y, n_fold=N_FOLD, params=rand_params)

#------------------------ Evaluation ------------------------#
#Find best & worst model
print(model_log.iloc[model_log["test_loss"].argmin()])
print(model_log.iloc[model_log["test_loss"].argmax()])

#Plot training and validation loss
if is_kaggle == False:
    plt.clf()
    loss_df = pd.DataFrame(data=best_hist["loss"], columns=["loss"])
    val_loss_df = pd.DataFrame(data=best_hist["val_loss"], columns=["val_loss"])
    total_loss_df = pd.concat([loss_df, val_loss_df], axis=1, keys=["loss", "val_loss"])
    sns.lineplot(data=total_loss_df)

#Predict values for submit df
y_submit = best_model.predict(X_submit)

#Create dataframe and CSV for submission
submit_df = np.concatenate((np.array(X_id_submit).reshape(-1,1), y_submit), axis=1)
pd.DataFrame(submit_df).to_csv(path_or_buf=output_folder + "submission.csv", index=False, header=y_cols)
# %%
