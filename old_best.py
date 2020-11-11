#%%
import sys
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import seaborn as sns
import matplotlib as m
import matplotlib.pyplot as plt
import sklearn as sk
import kerastuner as kt

from PIL import Image
from os import path
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout, Activation, ActivityRegularization, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.regularizers import l1, l2, L1L2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from tensorflow_addons.optimizers import AdamW

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler

#------------------------ Package versions ------------------------#
#Print versions for "Language and Package section of report"
print("numpy version: ", np.__version__)
print("pandas version: ", pd.__version__)
print("tensorflow version: ", tf.__version__)
print("tensorflow addons version: ", tfa.__version__)
print("seaborn version: ", sns.__version__)
print("matplotlib version: ", m.__version__)
print("sklearn version: ", sk.__version__)
print("kerastuner version: ", kt.__version__)

#------------------------ Model settings ------------------------#
is_kaggle = False #Set to true if making upload to kaggle

compute_baseline = False #Set to true to ONLY compute baseline model without scaled variables. Does encode first 3 columns based on ENC_TYPE
plot_graps = False #Set to true if plots should be created

N_FOLDS = 2 #Determines how many folds are used for K-fold

apply_pca = False #apply principal component analysis to reduce dimensionality 

#Determines how much variance must be explained by gene and cell columns for PCA
C_VAR_REQ = None
G_VAR_REQ = None

#Amount of gene/cell PCA components to use.
C_PCA_REQ = 5 #Max 100
G_PCA_REQ = 30 #Max 772

create_random_param_models = False #Create extra models instead of using existing models
N_RAND_MODELS = 8

use_preset_params = True #Uses pre-defined perameters in the code
n_ensemble_models = 5 #Define how many out of the best models should be used in ensemble
n_ensemble_w = 3 #Defines how many of the top row weight array should be used in ensemble

#Encoding type --> "map", "dummy"
ENC_TYPE = "map"

#Scaling type --> "standardize", "normalize", "quantile_normal", "quantile_uniform", "power", "robust"
SC_TYPE = "robust"

#Set random seed
RANDOM_STATE = 0

print("Printing model settings....")
print("Plot graphs: ", plot_graps)
print("Compute baseline: ", compute_baseline)
print("Encoding type: ", ENC_TYPE)
print("Scaling type: ", SC_TYPE)

#------------------------ Loading data ------------------------#
#Description of task	Predicting a receptor respons based on gene expression, cell viability, drug, dose, and treatment type
#cp_type	        trt_cp (treatment), ctl_vehicle (control group)
#cp_time	        treatment duration (24, 48, 72 hours)
#cp_dose	        dose of drug (high, low)
#c-	                cell viability data
#g-	                gene expression data
#%%
if is_kaggle == True:
    data_folder = "/kaggle/input/lish-moa/"
    output_folder = "/kaggle/working/"
    if os.path.exists("/kaggle/working/submission.csv"):
        os.remove("/kaggle/working/submission.csv")
else:
    data_folder = "data/"
    output_folder = "output/"

X = pd.read_csv(data_folder + "train_features.csv")
y = pd.read_csv(data_folder + "train_targets_scored.csv")
X_submit = pd.read_csv(data_folder + "test_features.csv")

#------------------------ Classes ------------------------#
class Preprocessor:
    def __init__(self, X, X_submit, y, encode_cols):
        self.X = X
        self.X_submit = X_submit
        self.y = y
        self.encode_cols = encode_cols
        self.X_id_submit = X_submit["sig_id"]
        self.y_cols = y.columns

    def print_example_rows():
        print("Printing example rows of raw data....")
        print(X.loc[[0,1,2,23810,23811],["sig_id", "cp_type", "cp_time", "cp_dose", "g-0", "c-0"]])

    def drop_id(self):
        print("Dropping id column....")
        print("X, y, X_submit shape before id remove: ", X.shape, y.shape, X_submit.shape)
        self.X.drop("sig_id", axis=1, inplace=True)
        self.X_submit.drop("sig_id", axis=1, inplace=True)
        self.y.drop("sig_id", axis=1, inplace=True)
        print("X, y, X_submit shape after id remove: ", X.shape, y.shape, X_submit.shape)

    def encode_df(self, encoder_type):
        #Encode columns as dummy variables
        if encoder_type == "dummy":            
            self.X = pd.concat([self.X.get_dummies(self.X[self.encode_cols], columns=self.encode_cols), self.X[~self.encode_cols]],axis=1)
            self.X.drop(self.encode_cols, axis=1)

            self.X_submit = pd.concat([self.X_submit.get_dummies(self.X_submit[self.encode_cols], columns=self.encode_cols), self.X_submit[~self.encode_cols]],axis=1)
            self.X_submit.drop(self.encode_cols, axis=1)

        #Map values to encodable columns
        elif encoder_type == "map":           
            self.X['cp_type'] = self.X['cp_type'].map({"ctl_vehicle": 0, "trt_cp": 1})
            self.X['cp_time'] = self.X['cp_time'].map({24: 0, 48: 0.5, 72: 1})
            self.X['cp_dose'] = self.X['cp_dose'].map({'D1': 0, 'D2': 1})

            self.X_submit['cp_type'] = self.X_submit['cp_type'].map({"ctl_vehicle": 0, "trt_cp": 1})
            self.X_submit['cp_time'] = self.X_submit['cp_time'].map({24: 0, 48: 0.5, 72: 1})
            self.X_submit['cp_dose'] = self.X_submit['cp_dose'].map({'D1': 0, 'D2': 1})

    def scale_df(self, scaler_type):
        X_cat = self.X[self.encode_cols]
        X_scale = self.X[X.columns.difference(self.encode_cols)]
        
        X_submit_cat  = self.X_submit[self.encode_cols]
        X_submit_scale = self.X_submit[X_submit.columns.difference(self.encode_cols)]

        #https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-col_plot-all-scaling-py
        scaler = {
            "standardize" : StandardScaler(), # gives values 0 mean and unit variance
            "normalize" : MinMaxScaler(), # scales values between 0 and 1
            "quantile_normal" : QuantileTransformer(output_distribution="normal"), #non-linear, maps probability density to uniform distribution
            "quantile_uniform" : QuantileTransformer(output_distribution="uniform"), #matches values to gaussian distribution
            "power": PowerTransformer(method="yeo-johnson"), # stabilizes variance, minimizes skewness, applies zero mean unit variance also
            "robust" : RobustScaler() #scaling robust to outliers. removes median, scales data to IQR 
        }

        sc = scaler[scaler_type]       
        self.X = pd.concat([X_cat, pd.DataFrame(sc.fit_transform(X_scale), columns=X_scale.columns)], axis=1)
        self.X_submit = pd.concat([X_submit_cat, pd.DataFrame(sc.fit_transform(X_submit_scale), columns=X_submit_scale.columns)], axis=1)
        print("Scaler used: ", sc, " to scale X and X_submit")

class Plotter():
    def __init__(self, X, y, plot_path):
        self.X = X
        self.y = y
        self.plot_path = plot_path

    def plot_sum_per_target_count(self):
    
        #Count amount of targets per row and sum by target count
        label_per_row = self.y.sum(axis=1).value_counts().sort_index(axis=0)
        print(100-((303+55+13+6)/len(self.y)*100), " percent has 0,1 or 2 labels")
        print("% 0, 1, 2 labels: ", label_per_row[0]/sum(label_per_row),label_per_row[1]/sum(label_per_row),label_per_row[2]/sum(label_per_row))

        #Plot sum of label counts across all rows 
        fig, axs = plt.subplots(1,1, figsize=(7,5))
        target_counts = pd.concat([pd.Series([x for x in range(8)]), label_per_row], axis=1, keys=["targets per drug", "amount of drugs"]).fillna(0)
        row_plot = sns.barplot(data= target_counts, x="targets per drug", y="amount of drugs")
        row_plot.set_title('Amount of targets per drug admission')
        
        #Add values of bar chart on top of bars
        for index, row in target_counts.iterrows():
            row_plot.text(row.name,row["amount of drugs"] + 40, int(row["amount of drugs"]), color='black', ha="center")
        
        #Adjust layout and save
        plt.tight_layout()
        row_plot.figure.savefig(self.plot_path + "sum_per_target_count.jpg")

    #Show distribution of amount of labels per COLUMN
    def plot_sum_per_target(self):

        #Create df with label counts per column
        count_target_df = pd.DataFrame(self.y.sum(axis=0).sort_values(ascending=False), columns=["target count"])
        
        #print top 50 targets as pecentage of total targets
        tot_label = count_target_df["target count"].sum()
        top_50_label = count_target_df["target count"][:50].sum()
        bottom_50_label = count_target_df["target count"][-50:].sum()
        print("Top 50 targets have " + str((top_50_label/tot_label)*100) + " percent of all labels")
        print("Bottom 50 targets have " + str((bottom_50_label/tot_label)*100) + " percent of all labels")

        #Sub
        count_target_df_50 = count_target_df.iloc[:50,:]
        count_target_df_50['target name'] = count_target_df_50.index
        
        #Plot target sum across all drug administrations
        fig, axs = plt.subplots(1,1, figsize=(15,7))

        col_plot = sns.barplot(data=count_target_df_50, x="target name", y="target count")
        col_plot.set_title('Top 50 targets count across all drug admissions')
        col_plot.set_xticklabels(col_plot.get_xticklabels(),rotation=45,ha="right",rotation_mode='anchor', fontsize=8)
        count = 0
        
        #Add values of bar chart on top of bars
        for index, row in count_target_df_50.iterrows():
            col_plot.text(count,row["target count"] + 6, int(row["target count"]), color='black', ha="center")
            count += 1

        #Adjust layout and save
        plt.tight_layout()
        col_plot.figure.savefig(self.plot_path + "sum_per_target.jpg")

    #Plots cell and gene distributions
    def plot_gene_cell_dist(self):

        # Get 4 random cell viability and 4 random gene expression cols
        cols = ["g-0","g-175","g-363","g-599", "c-4", "c-33", "c-65", "c-84"]

        # Create four polar axes and access them through the returned array
        fig, axs = plt.subplots(2,4, figsize=(20,10))

        #Loop over plot grid
        count = 0
        for i in range(0, 2):
            for j in range(0, 4):

                #Color first and last four plots differently (seperate cell and gene by color)
                if count >= 4:
                    axs[i, j].hist(x=self.X.loc[:, cols[count]], bins=50, color="#E1812B")
                else:
                    axs[i, j].hist(x=self.X.loc[:, cols[count]], bins=50, color="#3174A1")

                axs[i, j].set_title("Distribution " + cols[count])
                count += 1

        #Adjust format of plot and save
        plt.tight_layout()
        fig.savefig(self.plot_path + "genes_cells-dist.jpg")

    #Plots skew and kurtotis for gene and cell data
    def skew_kurtosis(self):

        #Calculate skewness and kurtosis
        kurtosis = self.X.loc[:,"g-0":].kurtosis(axis=0)
        skew = self.X.loc[:,"g-0":].skew(axis=0)

        #Split kurtosis and skew values into bins
        bin_skew = [-np.inf, -2, 2, np.inf]
        lab_skew = ['skew left', 'normally distributed', 'skew right']
        
        bin_kurt = [-np.inf, -2, 2, np.inf]
        lab_kurt = ['platykurtic', 'mesokurtic', 'leptokurtic']

        #Create skew label and kurtosis label columns
        skew_labeled = pd.cut(skew, bins=bin_skew, labels=lab_skew, ordered=False)
        kurtosis_labeled = pd.cut(kurtosis, bins=bin_kurt, labels=lab_kurt, ordered=False)
        
        #Create full skew, kurtosis dataframe
        skew_kurt_df = pd.concat([skew, skew_labeled, kurtosis, kurtosis_labeled], 
                            keys=["skewness", "skewness columns per group", "kurtosis", "kurtosis columns per group"], axis=1)

        #Split into cell and gene skew/kurtosis df
        self.gene_df = skew_kurt_df.loc["g-0":"g-771", :]
        self.cell_df = skew_kurt_df.loc["c-0":"c-99",:]

    #Plot skew and kurtosis for gene/cell cols
    def plot_skew_kurtosis(self):
        self.img_paths = []
        df, color = None, None
        
        for g_c in ["gene", "cell"]:
            for s_k in ["skewness", "kurtosis"]:
                
                if g_c == "gene":
                    df = self.gene_df
                    color="#3174A1"

                else: 
                    df = self.cell_df
                    color="#E1812B"
                
                fig, axs = plt.subplots(1,2, figsize=(10,5))
                axs[0].hist(x=df[s_k], bins=50, color=color)
                axs[0].set_title(g_c + " " + s_k + " values")
                axs[0].set_xlabel(s_k + " value")
                axs[0].set_ylabel("Amount of " + g_c + " columns")

                bar = df[s_k + " columns per group"].value_counts().reset_index()
                print(bar.columns)
                bar_skew = sns.barplot(data=bar, x="index", y=s_k + " columns per group", ax=axs[1], color=color)
                bar_skew.set_xticklabels(bar_skew.get_xticklabels(),rotation=15,ha="right",rotation_mode='anchor')
                axs[1].set_title(g_c + " " + s_k + " values")
                axs[1].set_xlabel('')
                plt.tight_layout()
                img_path = self.plot_path + g_c + "_" + s_k + ".jpg"
                fig.savefig(img_path)
                self.img_paths.append(img_path)

    def combine_graphs(self):
        #https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
        figs = [Image.open(x) for x in self.img_paths]
        widths, heights = zip(*(i.size for i in figs))

        total_width = widths[0] * 2
        max_height = heights[0] * 2

        x_off = figs[0].size[0]
        y_off = figs[0].size[1]

        new_im = Image.new('RGB', (total_width, max_height))
        count = 0

        for im in figs:
            if count == 0:
                new_im.paste(im, (0,0))
            elif count == 1:
                new_im.paste(im, (x_off,0))
            elif count == 2:
                new_im.paste(im, (0,y_off))
            elif count == 3:
                new_im.paste(im, (x_off,y_off))
            count += 1
            
        new_im.save(self.plot_path + 'total_skew_kurt.jpg')


class ModelBuilder():
    def __init__(self, X, y, X_submit, random_state, is_kaggle):
        self.X = X
        self.y = y
        self.X_submit = X_submit
        self.random_state = random_state
        self.is_kaggle = is_kaggle

        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        tf.compat.v1.keras.backend.set_session(sess)
    
    def select_random_parameters(self, n_param_sets):
        
        #Defines the allowed search space
        layers = [3,4,5]
        acti_hid = ["elu", "relu", "sigmoid", "softplus", "softsign"] 
        dropout = [0.15, 0.20, 0.25]
        neurons = [64, 96, 128]
        optimizers = ["nadam", "adam", SGD(lr=0.05, momentum=0.95), SGD(lr=0.05, momentum=0.98), "rmsprop"]

        #Create dictionary of the given parameters
        param_dic = {"lay": layers, "acti_hid": acti_hid, "neur": neurons, 
                    "drop": dropout, "opti": optimizers}
                    
        #Create list for parameter sets
        rand_params = []

        #Loop over all parameter sets and save dictionaries of random parameters to a list
        for i in range(n_param_sets):
            rand_param_dic = {}

            #Randomly select parameters from parameter lists
            for key, parameters in param_dic.items():
                rand_param_dic[key] = random.choice(parameters)
                
            #Append dictionary of random parameters to list
            if rand_param_dic not in rand_params:
                rand_params.append(rand_param_dic)
        
        self.rand_params = rand_params

    def create_baseline(self):
        """Creates a baseline model to which more advanced models can be compared"""

        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=self.random_state)

        print("Creating baseline model....")
        model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu')) 
        model.add(Dense(206, activation='softmax')) 
        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["binary_crossentropy"]) 
        model.fit(X_train, y_train, batch_size=4, epochs=15, validation_data=(X_val, y_val))
        results = model.evaluate(X_test, y_test, batch_size=1)

    def create_model(self, X_train, y_train, X_val, y_val, X_test, y_test, param_dic):
        """Creates a multilayer perceptron with the given parameter dictionary and data"""

        #Print model parameters
        print("Creating model with:")
        print("Hidden layer count: ", param_dic["lay"])
        print("Activation hidden: ", param_dic["acti_hid"])
        print("Neuron count per layer: ", param_dic["neur"])
        print("Dropout value: ", param_dic["drop"])
        print("Optimizer: ", param_dic["opti"])

        #Create model
        model = Sequential()
        
        #Create layers based on count with specified activations and dropouts
        for l in range(0,param_dic["lay"]):
            model.add(BatchNormalization())
            model.add(Dropout(param_dic["drop"]))
            model.add(Dense(param_dic["neur"], activation=param_dic["acti_hid"]))            

        #Add output layer
        model.add(Dense(206, activation="softmax")) 

        #Define optimizer and loss
        model.compile(optimizer=param_dic["opti"], loss='binary_crossentropy', metrics=["acc"]) 
        
        #Define callbacks
        hist = History()
        early_stop = EarlyStopping(monitor='val_loss', patience=2, mode='auto')

        #Fit and return model and loss history
        model.fit(X_train, y_train, batch_size=16, epochs=40, validation_data=(X_val, y_val), callbacks=[early_stop, hist])
        test_loss = model.evaluate(X_test, y_test, batch_size=1)[0]
        return model, test_loss, hist.history


    def k_fold(self, n_folds, params, n_ensemble_models):
        """Runs n_fold cross validation by creating multiple models with the create model function"""

        print("Splitting data....")
        X, X_test, y, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)

        #Define matrices for the average target predictions (for testing and submit)
        self.average_pred = np.zeros(y_test.shape)
        self.average_pred_submit = np.zeros((self.X_submit.shape[0], 206))
        self.y_test = y_test

        #Set loss parameter  
        test_loss = 0

        #Split data into train and test. train will be splitted later in K-fold
        for train_i, test_i in KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state).split(self.X):
                
            #Define train and test for Kfolds
            X_train = self.X.iloc[train_i,:]
            X_val = self.X.iloc[test_i, :]
            y_train = self.y.iloc[train_i, :]
            y_val = self.y.iloc[test_i, :]

            #Create a model for each split 
            model, test_loss, hist = self.create_model(X_train=X_train, X_val=X_val, 
                                                y_train=y_train, y_val=y_val, 
                                                X_test=X_test, y_test=y_test,
                                                param_dic=params)                                    

        #Add average model performance across all k-folds      
        params["test_loss"] = test_loss/n_folds   

        self.average_pred += np.array(model.predict(X_test)/n_ensemble_models)
        self.average_pred_submit += np.array(model.predict(X_submit)/n_ensemble_models)
        self.history = hist

    def create_model_ensemble(self, n_ensemble_models, n_folds, use_preset_params):
        """Creates an ensemble model by applying K-fold cross validation to a given list of parameters"""

        model_list = []

        #Load model parameters and losses
        if self.is_kaggle == False:
            with open(output_folder + 'random_models.pickle', 'rb') as f:
                model_list = pickle.load(f)
            #print(len(model_list), " amount of models in model object...", "Models in model list: ", model_list)

        #Set of current best parameters
        if use_preset_params == True:
            model_1_params = {'lay': 4, 'acti_hid': 'softplus', 'neur': 96, 'drop': 0.15, 'opti': 'adam'} 
            model_2_params = {'lay': 3, 'acti_hid': 'elu', 'neur': 96, 'drop': 0.2, 'opti': SGD(lr=0.05, momentum=0.95)} 
            model_3_params = {'lay': 5, 'acti_hid': 'elu', 'neur': 64, 'drop': 0.15, 'opti': 'adam'} 
            model_4_params = {'lay': 4, 'acti_hid': 'elu', 'neur': 64, 'drop': 0.1, 'opti': 'nadam'}
            model_5_params = {'lay': 3, 'acti_hid': 'elu', 'neur': 96, 'drop': 0.15, 'opti': 'adam'}
            model_params = [model_1_params, model_2_params, model_3_params, model_4_params, model_5_params]
            print("Number of models to be created: Parameter sets: ",len(model_params), " n_folds: ", n_folds, " total: ", len(model_params) * n_folds )
        
        else:
            model_params = [row[0] for row in model_list][:n_ensemble_models]
        
        #For each set of parameters, create a model and make an average prediction across all ensemble models
        for model_i, params in enumerate(model_params):
            print("Running K_fold on model: ", model_i + 1, "/", len(model_params))
            self.k_fold(n_folds=n_folds, params=params, n_ensemble_models=n_ensemble_models)


    #------------------------ Calculating row weights ------------------------#
    def get_weight_params(self, n_param_sets):
        """Returns parameter sets. Either random parameters from the select random parameter function or params that have been proven in the preset."""
        param_sets = []
        
        #If true, use some predefined parameters that have been found to work
        if use_preset_params == True:
            param_set_1 = {'lay': 4, 'acti_hid': 'softplus', 'neur': 128, 'drop': 0.15, 'opti': 'nadam'}
            param_set_2 = {'lay': 3, 'acti_hid': 'softsign', 'neur': 128, 'drop': 0.25, 'opti': 'adam'}
            param_set_3 = {'lay': 4, 'acti_hid': 'elu', 'neur': 128, 'drop': 0.15, 'opti': 'adam'}
            param_sets = [param_set_1, param_set_2, param_set_3]

        #if you should not use pre-set parameters, generate random parameter sets
        else:
            param_sets = self.select_random_parameters(n_param_sets=n_param_sets)
    
        return param_sets

    def predict_weights(self, param_sets):
        """Predicts row weights for the prediction matrices by training a model that minimizes the absolute error of the amount of targets per row in X_train"""
        
        y_pred_weights, y_submit_weights = [], []

        #Loop over all sets of parameters in the param_sets list
        for param_dic in param_sets:
            
            #Defines training data. y is targets per row, X is preprocessed X data
            targets_per_row = self.y.sum(axis=1)
            X_train, X_val, y_train, y_val = train_test_split(self.X, targets_per_row, test_size=0.2, random_state=self.random_state)
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=self.random_state)

            #Creates MLP to minimze absolute error between predicted amount of targets per row and actual
            print("Creating target prediction model....")
            model = Sequential()
            for i in range(3):
                model.add(BatchNormalization())
                model.add(Dropout(0.2))
                model.add(Dense(64, activation='selu'))
            model.add(Dense(1, activation='linear')) 
            model.compile(optimizer="adam", loss='mae', metrics=["mae"]) 

            #Define early stop and fit model on data and evaluate results
            early_stop = EarlyStopping(monitor='val_mae', patience=3, mode='auto')
            model.fit(X_train, y_train, batch_size=8, epochs=30, validation_data=(X_val, y_val), callbacks=early_stop)
            model.evaluate(X_test, y_test, batch_size=1)
            
            #Uses trained model to predict the amount of targets per row for X data and submit data
            y_pred_weights.append(model.predict(X_test).clip(min=0))
            y_submit_weights.append(model.predict(self.X_submit).clip(min=0))
            
        return y_pred_weights, y_submit_weights

    def calc_average_row_weight(self, pred_w, submit_w, param_sets, n_ensemble_w):
        """Calculates the average row weights, based on lists of row weights and the amount of ensembles to be used for the model"""

        #Find best binary cross-entropy loss with various predicted weight matrices
        weight_bce = []

        print("BCE on test set before row weights", self.calc_bce(self.y_test, self.average_pred))

        for i in range(len(pred_w)):

            #Calculate weighted predictions for each predicted weight array
            av_pred_weighted = self.average_pred * pred_w[i]
            av_pred_submit_weighted = self.average_pred_submit * submit_w[i]

            #Cap values in the prediction matrix to 1
            av_pred_weighted[av_pred_weighted > 1] = 1
            av_pred_submit_weighted[av_pred_submit_weighted > 1] = 1

            #Find weights with best Binary Cross Entropy on test set
            bce = self.calc_bce(self.y_test, av_pred_weighted)
            weight_bce.append([bce, pred_w[i], param_sets[i], submit_w[i]])

            #Calculate bce after transformation with weights
            print("BCE on test set after row weights", bce)

        weight_bce.sort(key=lambda x: x[0])
         
        #Calculate average row weight across best performing weight arrays
        w_average, w_average_submit = 0, 0
        for i in range(n_ensemble_w):
            w_average += (weight_bce[i][0]/n_ensemble_w)
            w_average_submit += (weight_bce[i][3]/n_ensemble_w)

        self.w_average = w_average
        self.w_average_submit = w_average_submit

    def compute_best_matrix(self):
        """Computes the best prediction matrix based on the ensemble weights and ensemble predictions"""
        self.best_mat = self.average_pred * self.w_average
        self.best_submit_mat = self.average_pred_submit * self.w_average_submit

    def calc_bce(self, y_pred, y_true):
        """Calculates Binary Crossentropy for predicted and true matrices"""
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(y_pred, y_true).numpy()

    def best_matrix_to_csv(self, submit_id_col, y_cols):
        """Writes the submit prediction matrix to a csv file"""
        submit_df = np.concatenate((np.array(submit_id_col).reshape(-1,1), self.best_submit_mat), axis=1)
        pd.DataFrame(submit_df).to_csv(path_or_buf=output_folder + "submission.csv", index=False, header=y_cols)

#------------------------ Init preprocessor ------------------------#
pre = Preprocessor(X=X, X_submit=X_submit, y=y, encode_cols=["cp_type", "cp_time", "cp_dose"])
pre.drop_id()

#------------------------ Plot graphs ------------------------#
if plot_graps == True:
    plotter = Plotter(X=pre.X, y=pre.y, plot_path="figs/")

    #Create histogram of 8 columns pre scaling
    plotter.plot_gene_cell_dist()

    #Plot target distributions across columns and with rows summed by target counts
    plotter.plot_sum_per_target_count()
    plotter.plot_sum_per_target()

    #Get skew and kurtosis values for cell viability and gene expression
    plotter.skew_kurtosis()

    #Plot skew and kurtosis for gene expression and cell viability
    plotter.plot_skew_kurtosis()

    #Combine the various skew and kurtosis images into 1
    plotter.combine_graphs()

#------------------------ Encode and scale X data ------------------------#
pre.encode_df(encoder_type=ENC_TYPE)
pre.scale_df(scaler_type=SC_TYPE)

#------------------------ Build models ------------------------#
modelbuilder = ModelBuilder(X=pre.X, y=pre.y, X_submit=pre.X_submit, 
                            random_state=RANDOM_STATE, is_kaggle=is_kaggle)

if compute_baseline == True:
    modelbuilder.create_baseline()
    sys.exit()

#Create an K-folded ensemble model of either preset params or read_params
modelbuilder.create_model_ensemble(n_ensemble_models=n_ensemble_models, n_folds=N_FOLDS, use_preset_params=use_preset_params)

param_sets = modelbuilder.get_weight_params(n_param_sets=15)

pred_w, submit_w = modelbuilder.predict_weights(param_sets=param_sets)

modelbuilder.calc_average_row_weight(pred_w=pred_w, submit_w=submit_w, param_sets=param_sets, n_ensemble_w=n_ensemble_w)

modelbuilder.compute_best_matrix()

best_bce = modelbuilder.calc_bce(y_true=np.array(modelbuilder.y_test).astype(float), y_pred=modelbuilder.best_mat)
print("Best Binary crossentropy is: ", best_bce)
modelbuilder.best_matrix_to_csv(submit_id_col=pre.X_id_submit, y_cols=pre.y_cols)

#%%
#------------------------ Exporatory Data Analysis ------------------------#
#Show distribution of amount of labels per ROW
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


def create_more_random_models(n_rand_models):
    rand_model_path = output_folder + 'random_models.pickle'

    param_sets = select_random_parameters(n_param_sets=n_rand_models)

    model_list = []

    #Save randomly generated models
    if path.exists(rand_model_path):
        
        with open(rand_model_path, 'rb') as f:
            model_list = pickle.load(f)
        print(len(model_list), " model already existed. Adding to existing models...")
    else:
        print("No models exist yet, creating model file...")

    #Loop over created param sets and create new model if param combination does not exist
    for params in param_sets:
        if params not in [row[0] for row in model_list]:
            model, loss = create_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, param_dic=params)    
            model_list.append([params, loss])

    #Sort models based on loss
    model_list.sort(key=lambda x: x[1])

    #Save model parameters and losses
    with open(rand_model_path, 'wb') as f:
        pickle.dump(model_list, f)

#=====================================================================================#
#================================= Execute main code =================================#
#=====================================================================================#

"""if apply_pca == True:
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


#Create extra randomly generated models to add to saved models object for later selection
if create_random_param_models == True:
    create_more_random_models(n_rand_models=N_RAND_MODELS)


#Create ensemble model with either pre-sets or loaded models
av_pred, av_pred_submit, y_test = create_model_ensemble(model_list=model_list)


#Use parameters to create row weight arrays for data and submit
y_pred_weight, y_submit_weight, param_sets = predict_row_weight(X=X, y=y, X_submit=X_submit, param_sets=param_dic)

weight_average, weight_average_submit = calc_average_row_weight(av_pred=av_pred, av_pred_submit=av_pred_submit, 
                                                                y_pred_weight=y_pred_weight, y_submit_weight=y_submit_weight)

best_pred = av_pred * weight_average
best_submit = av_pred_submit * weight_average_submit
print("average of ", n_ensemble_w, " gives best BCE on y_test set: ", calc_bce(y_test, best_pred))"""
