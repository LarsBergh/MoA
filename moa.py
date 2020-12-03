#%%
#Import libraries
#from classes import Preprocessor, Plotter, ModelBuilder
import sys
import time
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler
#%%
#------------------------ Classes ------------------------#
class Preprocessor:
    def __init__(self, X, X_submit, y, encode_cols):
        self.X = X
        self.X_submit = X_submit
        self.y = y
        self.encode_cols = encode_cols
        self.X_id_submit = X_submit["sig_id"]
        self.y_cols = y.columns

    def upsample(self, min_target_count, random_state):
        total_df = pd.concat([self.X, self.y], axis=1)
        sampled_df = pd.DataFrame()
        #Loop over all y cols except for sig_id
        for target in total_df.columns[-206:]:
            
            #get array of positive found classes per type of class
            filt = total_df[target] == 1
            target_amount = filt.sum()
            
            #Calculates size of sample to take
            amount_to_sample = min_target_count - target_amount
            if amount_to_sample < 0:
                amount_to_sample = 0
            else:
                #Sample the dataframe based on amount of rows needed to get to 100
                sample = total_df[filt].sample(amount_to_sample, replace=True)
                
                #Combine sample with rest of sampled data
                sampled_df = pd.concat([sampled_df, sample], axis=0)

        total_and_sample_df = pd.concat([total_df, sampled_df], axis=0).reset_index(drop=True)

        #Seperate into X and y after upsampling
        self.X = total_and_sample_df.iloc[:,:-207]
        self.y = total_and_sample_df.iloc[:,-207:]

    def print_example_rows():
        print("Printing example rows of raw data....")
        print(X.loc[[0,1,2,23810,23811],["sig_id", "cp_type", "cp_time", "cp_dose", "g-0", "c-0"]])

    def drop_id(self):
        print("Dropping id column....")
        print("X, y, X_submit shape before id remove: ", self.X.shape, self.y.shape, self.X_submit.shape)
        self.X.drop("sig_id", axis=1, inplace=True)
        self.X_submit.drop("sig_id", axis=1, inplace=True)
        self.y.drop("sig_id", axis=1, inplace=True)
        print("X, y, X_submit shape after id remove: ", self.X.shape, self.y.shape, self.X_submit.shape)

    def pca(self, c_req, g_req):
    
        pca_df_total = pd.DataFrame()
        X_pca = pd.DataFrame()

        for i, pca_type in enumerate(["gene", "cell"]):
            pca_cols = [x.startswith("g-") if pca_type == "gene" else x.startswith("c-") for x in self.X.columns]

            #Get df subset based on pca cols
            pca = self.X.loc[:,pca_cols]

            #Get PCA dataframe based on gene/cell dataframe  
            pca_fit = PCA(n_components=pca.shape[1], random_state=RANDOM_STATE).fit(pca)

            #Generate column names and pca components
            cols = []

            if pca_type == "gene":
                for comp in range(0, g_req): 
                    cols.append('g-' + str(comp))
                X_pca = pd.DataFrame(PCA(n_components=g_req, random_state=RANDOM_STATE).fit_transform(pca),columns=cols)

            elif pca_type == "cell":
                for comp in range(0, c_req): 
                    cols.append('c-' + str(comp))
                X_pca = pd.DataFrame(PCA(n_components=c_req, random_state=RANDOM_STATE).fit_transform(pca),columns=cols)

            #Return PCA component dataframe for gene/cell columns
            pca_df_total = pd.concat([pca_df_total, X_pca],axis=1)
                
        #Add non_pca cols to df
        main_cols = ["cp_time", "cp_dose", "cp_type"]
        self.X = pd.concat([self.X[main_cols], pca_df_total],axis=1)
        print(self.X.columns)

    def encode_df(self, encoder_type):
        if encoder_type == "map":           
            self.X['cp_type'] = self.X['cp_type'].map({"ctl_vehicle": 0, "trt_cp": 1})
            self.X['cp_time'] = self.X['cp_time'].map({24: 0, 48: 0.5, 72: 1})
            self.X['cp_dose'] = self.X['cp_dose'].map({'D1': 0, 'D2': 1})

            self.X_submit['cp_type'] = self.X_submit['cp_type'].map({"ctl_vehicle": 0, "trt_cp": 1})
            self.X_submit['cp_time'] = self.X_submit['cp_time'].map({24: 0, 48: 0.5, 72: 1})
            self.X_submit['cp_dose'] = self.X_submit['cp_dose'].map({'D1': 0, 'D2': 1})

    def scale_df(self, scaler_type):
        if scaler_type != "None":
            X_cat = self.X[self.encode_cols]
            X_scale = self.X[self.X.columns.difference(self.encode_cols)]
            X_submit_cat  = self.X_submit[self.encode_cols]
            X_submit_scale = self.X_submit[self.X_submit.columns.difference(self.encode_cols)]

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
        print("Scaler used: ", scaler_type, " to scale X and X_submit")

class Plotter():
    def __init__(self, X, y, plot_path):
        self.X = X
        self.y = y
        self.plot_path = plot_path

    def plot_sum_per_target_count(self):
    
        #Count amount of targets per row and sum by target count
        label_per_row = self.y.sum(axis=1).value_counts().sort_index(axis=0)
        print(100-((303+55+13+6)/len(self.y)*100), " percent has 0,1 or 2 labels")
        zero_label_percentage = label_per_row[0]/sum(label_per_row)
        one_label_percentage = label_per_row[1]/sum(label_per_row)
        rest_label_percentage = 1 - zero_label_percentage - one_label_percentage
        print("% 0, 1, 2+ labels: ", zero_label_percentage, one_label_percentage, rest_label_percentage)

        #Plot sum of label counts across all rows 
        fig, axs = plt.subplots(1,1, figsize=(7,5))
        target_counts = pd.concat([pd.Series([x for x in range(8)]), label_per_row], axis=1, keys=["targets per drug", "amount of drugs"]).fillna(0)
        row_plot = sns.barplot(data= target_counts, x="targets per drug", y="amount of drugs")
        row_plot.set_title('Amount of targets per drug admission', fontsize=18)
        axs.set_xlabel("Targets per drug", fontsize=14)
        axs.set_ylabel("Amount of drugs", fontsize=14)

        #Add values of bar chart on top of bars
        for index, row in target_counts.iterrows():
            row_plot.text(row.name,row["amount of drugs"] + 40, int(row["amount of drugs"]), color='black', ha="center", fontsize=14)
        
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

        #Get 
        count_target_df_50 = count_target_df.iloc[:50,:]
        count_target_df_50['target name'] = count_target_df_50.index
        
        #Plot target sum across all drug administrations
        fig, axs = plt.subplots(1,1, figsize=(15,7))

        col_plot = sns.barplot(data=count_target_df_50, x="target name", y="target count")
        col_plot.set_title('Top 50 targets count across all drug admissions', fontsize=18)
        col_plot.set_xticklabels(col_plot.get_xticklabels(), rotation=45, ha="right", rotation_mode='anchor', fontsize=12)
        axs.set_xlabel("Target name", fontsize=14)
        axs.set_ylabel("Target amount", fontsize=14)
        plt.setp(axs.get_yticklabels(), fontsize=14)
        count = 0

        #Add values of bar chart on top of bars
        for index, row in count_target_df_50.iterrows():
            col_plot.text(count,row["target count"] + 6, int(row["target count"]), color='black', ha="center", fontsize=10)
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
                
                plt.setp(axs[i, j].get_xticklabels(), fontsize=14)
                plt.setp(axs[i, j].get_yticklabels(), fontsize=14)
                axs[i, j].set_xlabel(xlabel="Value", fontsize=14)
                axs[i, j].set_ylabel(ylabel="Amount of rows", fontsize=14)
                axs[i, j].set_title("Distribution of column " + cols[count], fontsize=18)
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
        print("Cell df: ", self.cell_df)
        print("Gene df: ", self.gene_df)
        print("Df X: ", self.X)

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
                axs[0].set_title(g_c + " " + s_k + " values", fontsize=18)
                axs[0].set_xlabel(s_k + " value", fontsize=14)
                axs[0].set_ylabel("Amount of " + g_c + " columns", fontsize=14)
                plt.setp(axs[0].get_xticklabels(), fontsize=14)
                plt.setp(axs[0].get_yticklabels(), fontsize=14)

                bar = df[s_k + " columns per group"].value_counts().reset_index()
                bar_skew = sns.barplot(data=bar, x="index", y=s_k + " columns per group", ax=axs[1], color=color)
                bar_skew.set_xticklabels(bar_skew.get_xticklabels(),rotation=15,ha="right",rotation_mode='anchor', fontsize=14)
                axs[1].set_title(g_c + " " + s_k + " values", fontsize=18)
                axs[1].set_xlabel('')
                axs[1].set_ylabel(s_k + " columns per group", fontsize=14)
                plt.setp(axs[1].get_xticklabels(), fontsize=14)
                plt.setp(axs[1].get_yticklabels(), fontsize=14)
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

#%%
class ModelBuilder():
    def __init__(self, X, y, X_submit, random_state, is_kaggle):
        self.X = X
        self.y = y
        self.X_submit = X_submit
        self.random_state = random_state
        self.is_kaggle = is_kaggle
        
        X, X_test, y, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
        self.y_test = y_test
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
        tf.compat.v1.keras.backend.set_session(sess)
    
    def select_random_parameters(self, n_param_sets):
        
        #Defines the allowed search space
        layers = [2,3,4, 5]
        acti_hid = ["elu", "relu", "sigmoid", "softplus", "softsign"] 
        dropout = [0.15, 0.20, 0.25, 0.3, 0.35, 0.4]
        neurons = [64, 96, 128, 160, 192, 224]
        optimizers = ["nadam", "adam", SGD(lr=0.05, momentum=0.98), AdamW(weight_decay=0.0001), "rmsprop"]

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
        
        return rand_params

    def create_more_random_models(self, n_folds, random_state, n_rand_models, output_folder, weight_or_matrix):
        rand_model_path = output_folder
        model_list = []

        #Set path for random model objects
        if weight_or_matrix == "weight":
            rand_model_path += 'random_weights.pickle'

        elif weight_or_matrix == "matrix":
            rand_model_path += 'random_models.pickle'

        #Save randomly generated models
        if path.exists(rand_model_path):
            print(rand_model_path)
            model_list = pickle.load(open(rand_model_path, 'rb'))
            print(len(model_list), weight_or_matrix, " model already existed. Adding to existing models...")
        else:
            print("No models exist yet, creating model file...")

        param_sets = self.select_random_parameters(n_param_sets=n_rand_models)
        
        tf.random.set_seed(random_state)
       
        loss = 0

        #Loop over created param sets and create new model if param combination does not exist
        for params in param_sets:         

            if params not in [row[0] for row in model_list]:
                if weight_or_matrix == "matrix":
                    loss, average_pred = self.k_fold_model(n_folds=n_folds, model_params=params)

                elif weight_or_matrix == "weight":
                    loss, average_weight = self.k_fold_weights(n_folds=n_folds, row_weight_params=params)
                    
                model_list.append([params, loss])

        #Sort models based on loss
        model_list.sort(key=lambda x: x[1])

        #Save model parameters and losses
        pickle.dump(model_list, open(rand_model_path, 'wb'))           

    def upsample(self, X_train, y_train, min_target_count, random_state):
        #Upsample the training data
        total_df = pd.concat([X_train, y_train], axis=1)
        sampled_df = pd.DataFrame()

        #Loop over all y cols except for sig_id
        for target in total_df.columns[-206:]:
            #get array of positive found classes per type of class
            filt = total_df[target] == 1
            target_amount = filt.sum()
            
            #Calculates size of sample to take
            amount_to_sample = min_target_count - target_amount                
            if amount_to_sample > 0 and amount_to_sample != min_target_count:

                #Sample the dataframe based on amount of rows needed to get to 100
                sample = total_df[filt].sample(amount_to_sample, replace=True, random_state=self.random_state)
                
                #Combine sample with rest of sampled data
                sampled_df = pd.concat([sampled_df, sample], axis=0)

                total_and_sample_df = pd.concat([total_df, sampled_df], axis=0).reset_index(drop=True)
                
        #Seperate into X_trian and y_train after upsampling for the model creation
        X_train = total_and_sample_df.iloc[:,:-206]
        y_train = total_and_sample_df.iloc[:,-206:]
        return  X_train, y_train


    def create_baseline(self, random_state, n_folds, use_upsampling=None):
        """Creates a baseline model to which more advanced models can be compared"""

        tf.random.set_seed(random_state)
        print("baseline created with X shape: ", self.X.shape, self.X.iloc[:5,:])

        print("Splitting data to aquire test set....")
        X, X_test, y, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)

        test_loss = 0

        #Split data into train and test. train will be splitted later in K-fold
        for train_i, test_i in KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state).split(X):
                
            #Define train and test for Kfolds
            X_train = X.iloc[train_i,:]
            X_val = X.iloc[test_i, :]
            y_train = y.iloc[train_i, :]
            y_val = y.iloc[test_i, :]

            if use_upsampling == True:
                X_train, y_train = self.upsample(X_train=X_train, y_train=y_train, 
                                                min_target_count=100, random_state=self.random_state)

            print("Creating baseline model....")
            model = Sequential()
            model.add(Dense(64, activation='relu'))
            model.add(Dense(64, activation='relu')) 
            model.add(Dense(206, activation='softmax')) 
            model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["binary_crossentropy"]) 

            early_stop = EarlyStopping(monitor='val_loss', patience=1, mode='auto')
            model.fit(X_train, y_train, batch_size=4, epochs=40, validation_data=(X_val, y_val), callbacks=[early_stop])
            results = model.evaluate(X_test, y_test, batch_size=8)
            test_loss += results[0]/n_folds

        return test_loss


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
        model.fit(X_train, y_train, batch_size=8, epochs=40, validation_data=(X_val, y_val), callbacks=[early_stop, hist])
        test_loss = model.evaluate(X_test, y_test, batch_size=8)[0]
        return model, test_loss, hist.history
        

    def create_row_model(self, X_train, y_train, X_val, y_val, X_test, y_test, param_dic):
        """Creates a multilayer perceptron with the given parameter dictionary and data"""
        print(param_dic)
        #Print model parameters
        print("Creating row model with:")
        print("Hidden layer count: ", param_dic["lay"])
        print("Activation hidden: ", param_dic["acti_hid"])
        print("Neuron count per layer: ", param_dic["neur"])
        print("Dropout value: ", param_dic["drop"])
        print("Optimizer: ", param_dic["opti"])

        #Creates MLP to minimze absolute error between predicted amount of targets per row and actual
        print("Creating target row prediction model....")
        model = Sequential()
        
        #Create layers based on count with specified activations and dropouts
        for l in range(0,param_dic["lay"]):
            model.add(BatchNormalization())
            model.add(Dropout(param_dic["drop"]))
            model.add(Dense(param_dic["neur"], activation=param_dic["acti_hid"]))            

        #Add output layer
        model.add(Dense(1, activation='linear')) 

        #Define optimizer and loss
        model.compile(optimizer=param_dic["opti"], loss='mae', metrics=["mae"]) 
        
        #Define callbacks
        hist = History()
        early_stop = EarlyStopping(monitor='val_mae', patience=3, mode='auto')

        #Fit and return model and loss history
        model.fit(X_train, y_train, batch_size=8, epochs=40, validation_data=(X_val, y_val), callbacks=[early_stop, hist])
        test_loss = model.evaluate(X_test, y_test, batch_size=8)[0]    
        
        return model, test_loss, hist.history

    def k_fold_model(self, n_folds, model_params):
        """Runs n_fold cross validation by creating multiple models with the create model function"""

        print("Splitting data to aquire test set....")
        X, X_test, y, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)

        #Define matrices for the average target predictions (for testing and submit)
        self.average_pred = np.zeros(y_test.shape)
        self.average_pred_submit = np.zeros((self.X_submit.shape[0], 206))
        self.y_test = y_test

        #Set loss parameter  
        av_loss = 0

        #Split data into train and test. train will be splitted later in K-fold
        for train_i, test_i in KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state).split(X):
                
            #Define train and test for Kfolds
            X_train = X.iloc[train_i,:]
            X_val = X.iloc[test_i, :]
            y_train = y.iloc[train_i, :]
            y_val = y.iloc[test_i, :]

            #Create a model for each split (validation and test untouched by upsampling)
            model, test_loss, hist = self.create_model(X_train=X_train, X_val=X_val, 
                                                y_train=y_train, y_val=y_val, 
                                                X_test=X_test, y_test=y_test,
                                                param_dic=model_params)                                    
            
            av_loss += test_loss/n_folds
            self.average_pred += np.array(model.predict(X_test)/n_folds)
            self.average_pred_submit += np.array(model.predict(X_submit)/n_folds)
            self.history = hist
        return av_loss, self.average_pred
    #------------------------ Calculating row weights ------------------------#
    def k_fold_weights(self, n_folds, row_weight_params):
        """Predicts row weights for the prediction matrices by training a model that minimizes the absolute error of the amount of targets per row in X_train"""

        print("Splitting data to aquire test set....")
        targets_per_row = self.y.sum(axis=1)
        X, X_test, y, y_test = train_test_split(self.X, targets_per_row, test_size=0.2, random_state=self.random_state)

        self.y_pred_weights = np.zeros((y_test.shape[0], 1))
        self.y_submit_weights = np.zeros((self.X_submit.shape[0], 1))
        
        av_loss = 0

        #Split data into train and test. train will be splitted later in K-fold
        for train_i, test_i in KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state).split(X):
                
            #Define train and test for Kfolds
            X_train = X.iloc[train_i,:]
            X_val = X.iloc[test_i, :]
            y_train = y.iloc[train_i]
            y_val = y.iloc[test_i]

            #Create model to predict amount of targets per row              
            model, test_loss, hist  = self.create_row_model(X_train=X_train, X_val=X_val, 
                                                                   y_train=y_train, y_val=y_val, 
                                                                   X_test=X_test, y_test=y_test,
                                                                   param_dic=row_weight_params)  

            #Adjust weight arrays for the amount of folds
            av_loss += test_loss/n_folds
            self.y_pred_weights += model.predict(X_test).clip(min=0)/n_folds
            self.y_submit_weights += model.predict(self.X_submit).clip(min=0)/n_folds
        return av_loss, self.y_pred_weights

    def calc_bce(self, y_true, y_pred):
        """Calculates Binary Crossentropy for predicted and true matrices"""
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(y_true, y_pred).numpy()

    def plot_targets_to_zero(self, bottom_n_cols):
        lis_num = []
        lis_val = []

        for i in range(1, bottom_n_cols):
            
            #Create df with label counts per column
            count_target_df = pd.DataFrame(self.y.sum(axis=0), columns=["target count"]).sort_values(ascending=False, by="target count")
            
            #Grab bottom X target names (X least target occurances)
            bottom_X_labels = count_target_df["target count"][-i:].index
            
            #Turn best matrix back into a pandas dataframe
            best_mat = pd.DataFrame(self.best_mat, columns=self.y.columns)

            #Set bottom X target occurances to zero
            best_mat[bottom_X_labels] = 0
            
            #Calculate binary cross-entropy for the 
            bce = modelbuilder.calc_bce(y_true=np.array(self.y_test).astype(float), y_pred=best_mat)
            
            #Append amount of dropped target columns and respective binary cross entropies for each drop
            lis_num.append(i)
            lis_val.append(bce)

        #Plot number of target columns dropped to zero, compare to bce
        fig, axs = plt.subplots(1,1, figsize=(7,5))
        drop_target_plot = sns.lineplot(x=lis_num, y=lis_val)
        plt.xlabel('Amount of target predictions set to zero', fontsize=18)
        plt.ylabel("Binary crossentropy value", fontsize=18)
        drop_target_plot.set_title('Binary cross-entropy for least represented classes to zero probability')
        plt.tight_layout()
        fig.savefig(output_folder + "dropping_targets.jpg") 
   
    def best_matrix_to_csv(self, submit_id_col, y_cols):
        """Writes the submit prediction matrix to a csv file"""
        submit_df = np.concatenate((np.array(submit_id_col).reshape(-1,1), self.best_submit_mat), axis=1)
        pd.DataFrame(submit_df).to_csv(path_or_buf=output_folder + "submission.csv", index=False, header=y_cols)


#%%
#------------------------ Model settings ------------------------#
print_versions = False #Print versions of packages that were used
is_kaggle = False #Set to true if making upload to kaggle

test_scalers = False #Set to true to ONLY compute baseline model with various scalers.
test_pca = False #Sets various PCA component settings and their corresponding loss
test_upsampling = False #Compares an upsampled dataframe to non-upsampled dataframe on baseline model with k-fold

plot_graps = True #Set to true if plots should be created
N_FOLDS = 2 #Determines how many folds are used for K-fold

create_random_param_models = False #Create extra softmax target probability prediction models and save them to pickle object 
create_random_row_models = False
N_RAND_MODELS = 1 #Number of extra models to add to pickle object for both neural nets.

save_best_models = False #Saves the best models from all randomly created objects (hardcoded)

ENC_TYPE = "map" #Encoding type --> "map"
SC_TYPE = "quantile_normal" #Scaling type --> "standardize", "normalize", "quantile_normal", "quantile_uniform", "power", "robust"
RANDOM_STATE = 0 #Set random seed

print("Printing model settings....")
print("Print versions: ", print_versions)

print("Test scalers: ", test_scalers)
print("Test PCA: ", test_pca)
print("Test upsampling: ", test_upsampling)

print("Plot graphs: ", plot_graps)
print("N-folds for k-fold: ", N_FOLDS)

print("Create prediction matrix models (random params): ", create_random_param_models)
print("Create prediction row weight models (random params): ", create_random_row_models)
print("Number of random models to create if true", N_RAND_MODELS)
print("Creating and saving best models from random search: ", save_best_models)

print("Encoding type: ", ENC_TYPE)
print("Scaling type: ", SC_TYPE)
print("Random state: ", RANDOM_STATE)
#%%
#------------------------ Package versions ------------------------#
#Print versions for "Language and Package section of report"
if print_versions == True:
    print("numpy version: ", np.__version__)
    print("pandas version: ", pd.__version__)
    print("tensorflow version: ", tf.__version__)
    print("tensorflow addons version: ", tfa.__version__)
    print("seaborn version: ", sns.__version__)
    print("matplotlib version: ", m.__version__)
    print("sklearn version: ", sk.__version__)

#------------------------ Loading data ------------------------#
if is_kaggle == True:
    data_folder = "/kaggle/input/lish-moa/"
    output_folder = "/kaggle/working/"
    if os.path.exists("/kaggle/working/submission.csv"):
        os.remove("/kaggle/working/submission.csv")
else:
    data_folder = "data/"
    output_folder = "output/"

#------------------------ Test various scalers ------------------------#
if test_scalers == True:
    scaling_results = []

    #Loop over all scaler types
    for scale_type in ["None", "standardize", "normalize", "quantile_normal", "quantile_uniform", "power", "robust"]:
        
        #Load data
        X = pd.read_csv(data_folder + "train_features.csv")
        y = pd.read_csv(data_folder + "train_targets_scored.csv")
        X_submit = pd.read_csv(data_folder + "test_features.csv")

        #Init preprocessor and drop Id column
        pre = Preprocessor(X=X, X_submit=X_submit, y=y, encode_cols=["cp_type", "cp_time", "cp_dose"])
        pre.drop_id()

        #Encode and scale X data
        pre.encode_df(encoder_type="map")
        pre.scale_df(scaler_type=scale_type)

        #Init model builder
        modelbuilder = ModelBuilder(X=pre.X, y=pre.y, X_submit=pre.X_submit, 
                            random_state=RANDOM_STATE, is_kaggle=is_kaggle)

        #Create baseline with each dataframe scaling type
        scaling_results.append([scale_type, modelbuilder.create_baseline(random_state=RANDOM_STATE, n_folds=N_FOLDS)])

    #Sort scaled results from low to high binary cross-entropy
    scaling_results.sort(key=lambda x:x[1])
    print(scaling_results)
    sys.exit()
#['quantile_uniform', 0.017554578371345997] 
#['power', 0.018034077249467373]
#['standardize', 0.018125144764780998]
#['None', 0.01817983016371727]
#['quantile_normal', 0.018644552677869797]
#['robust', 0.018999390304088593]
#['normalize', 0.019516831263899803]]
#%%
#Tests binary cross-entropy with various column sets
if test_pca == True:
    pca_results = []

    #Lists how many cell viability and gene expression columns will be used in test
    cell_components = [5, 10, 15, 35, 50, 100] 
    gene_components = [25, 50, 75, 150, 200, 772]

    #For each set of columns perform test
    for i in range(len(cell_components)):
        
        #Loads data
        X = pd.read_csv(data_folder + "train_features.csv")
        y = pd.read_csv(data_folder + "train_targets_scored.csv")
        X_submit = pd.read_csv(data_folder + "test_features.csv")

        #Inits preprocessor and drops Id column
        pre = Preprocessor(X=X, X_submit=X_submit, y=y, encode_cols=["cp_type", "cp_time", "cp_dose"])
        pre.drop_id()

        #Does PCA for given cell and gene columns
        pre.pca(c_req=cell_components[i], g_req=gene_components[i])

        #Encodes few categorical/numercial columns and scales dataframe
        pre.encode_df(encoder_type="map")
        pre.scale_df(scaler_type="None")
        
        #Inits modelbuilder
        modelbuilder = ModelBuilder(X=pre.X, y=pre.y, X_submit=pre.X_submit, 
                            random_state=RANDOM_STATE, is_kaggle=is_kaggle)

        #Creates baseline model with each PCA component set
        test_loss = modelbuilder.create_baseline(random_state=RANDOM_STATE, n_folds=N_FOLDS)
        pca_results.append([cell_components[i], gene_components[i], test_loss])

    #Returns dataframe with pca cell, gene column counts and test loss
    pca_results = pd.DataFrame(data=pca_results, columns=["cell_cols", "gene_cols", "test loss"])

    print(pca_results)
    sys.exit()

#cell_cols  gene_cols  test loss
#5          25         0.017841
#10         50         0.017775
#15         75         0.017724
#35         150        0.017826
#50         200        0.017889
#100        772        0.017695
#%%
#Tests upsampling all targets with less than 6 instances up to 100
if test_upsampling == True:
    upsampling_results = []

    #Compares upsampling to doing nothing
    for upsampling in [None, True]:
        
        #Loads data
        X = pd.read_csv(data_folder + "train_features.csv")
        y = pd.read_csv(data_folder + "train_targets_scored.csv")
        X_submit = pd.read_csv(data_folder + "test_features.csv")

        #Inits preprocessor and drops Id column
        pre = Preprocessor(X=X, X_submit=X_submit, y=y, encode_cols=["cp_type", "cp_time", "cp_dose"])
        pre.drop_id()

        #Encode and scale X data
        pre.encode_df(encoder_type="map")
        pre.scale_df(scaler_type="None")

        #Init modelbuilder
        modelbuilder = ModelBuilder(X=pre.X, y=pre.y, X_submit=pre.X_submit, 
                            random_state=RANDOM_STATE, is_kaggle=is_kaggle)

        #Creates baseline with or without upsampled data
        upsampling_results.append([upsampling, modelbuilder.create_baseline(random_state=RANDOM_STATE, n_folds=N_FOLDS, use_upsampling=upsampling)])

    upsampling_results.sort(key=lambda x:x[1])
    print(upsampling_results)
    sys.exit()
#[None, 0.01817983016371727] 
#[True, 0.023640152998268604]
#%%
#------------------------ Main Code ------------------------#
X = pd.read_csv(data_folder + "train_features.csv")
y = pd.read_csv(data_folder + "train_targets_scored.csv")
X_submit = pd.read_csv(data_folder + "test_features.csv")

#Init preprocessor and drop Id columns
pre = Preprocessor(X=X, X_submit=X_submit, y=y, encode_cols=["cp_type", "cp_time", "cp_dose"])
pre.drop_id()

#Init plot object and plot histrograms, skew, kurtosis, target frequencies and row target count plots
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

#%%
#Use mapping and scaling to preprocess data
pre.encode_df(encoder_type=ENC_TYPE)
pre.scale_df(scaler_type=SC_TYPE)

#Init model builder
modelbuilder = ModelBuilder(X=pre.X, y=pre.y, X_submit=pre.X_submit, 
                            random_state=RANDOM_STATE, is_kaggle=is_kaggle)

#Create random search generated models through model builder (output: prediction matrix)
if create_random_param_models == True:
    modelbuilder.create_more_random_models(n_folds=N_FOLDS, n_rand_models=N_RAND_MODELS, random_state=RANDOM_STATE, output_folder=output_folder, weight_or_matrix="matrix")

#Create random search generated models through model builder (output: row weight vector)
if create_random_row_models == True:
    modelbuilder.create_more_random_models(n_folds=N_FOLDS, n_rand_models=N_RAND_MODELS, random_state=RANDOM_STATE, output_folder=output_folder, weight_or_matrix="weight")


#%%
#Loads model parameters and test-loss for prediction matrices
model_list = pickle.load(open(output_folder + "random_models.pickle", 'rb'))
print("Random model list model amount: ", len(model_list), " models: ", model_list)

#Loads model parameters and test-loss for row weight vectors
weight_list = pickle.load(open(output_folder + "random_weights.pickle", 'rb'))
print("Random weights list model amount: ", len(weight_list), " models: ", weight_list)

#Prints all parameter sets in weight and matrix objects (from random model creation)
for params in model_list:
    print(params[0], params[1])
for params in weight_list:
    print(params[0], params[1])
#%%
if save_best_models == True:
    #Hardcode the top 5 best parameters sets that were found for prediction matrix modeling
    #Model param performance (Binary cross-entropy for prediction matrix versus y_test)
    #1: 0.0168133,  2: 0.0168480    3: 0.0168571    4: 0.0168735    5: 0.0169396
    model_params =[{'lay': 2, 'acti_hid': 'elu', 'neur': 96, 'drop': 0.25, 'opti': AdamW(weight_decay=0.0001)}, 
                    {'lay': 2, 'acti_hid': 'elu', 'neur': 64, 'drop': 0.2, 'opti': AdamW(weight_decay=0.0001)}, 
                    {'lay': 3, 'acti_hid': 'relu', 'neur': 128, 'drop': 0.2, 'opti': AdamW(weight_decay=0.0001)},
                    {'lay': 2, 'acti_hid': 'softplus', 'neur': 64, 'drop': 0.3, 'opti': 'adam'},
                    {'lay': 2, 'acti_hid': 'sigmoid', 'neur': 160, 'drop': 0.15, 'opti': AdamW(weight_decay=0.0001)}] 

    #Create K-fold prediction matrix for each model
    pred_list = []

    for model in model_params:
        loss, average_pred = modelbuilder.k_fold_model(n_folds=N_FOLDS, model_params=model)
        pred_list.append(average_pred)

    #Dump prediction matrices to pickle
    pickle.dump(pred_list, open(output_folder + "pred_matrices.pickle", 'wb'))

    #Hardcode the top 5 best parameters sets that were found for row weight vector modeling 
    #Model param performance (Mean absolute error of amount of targets predicted versus y_actual targets)
    #1: 0.3953630,   2: 0.3993239,  3: 0.3998349,   4: 0.4014825,   5: 0.4017003   
    row_weight_params = [{'lay': 2, 'acti_hid': 'sigmoid', 'neur': 192, 'drop': 0.3, 'opti': 'nadam'}, 
                        {'lay': 2, 'acti_hid': 'softsign', 'neur': 64, 'drop': 0.15, 'opti': 'adam'}, 
                        {'lay': 2, 'acti_hid': 'sigmoid', 'neur': 96, 'drop': 0.15, 'opti': AdamW(weight_decay=0.0001)},
                        {'lay': 2, 'acti_hid': 'sigmoid', 'neur': 96, 'drop': 0.2, 'opti': 'adam'}, 
                        {'lay': 2, 'acti_hid': 'sigmoid', 'neur': 128, 'drop': 0.25, 'opti': AdamW(weight_decay=0.0001)}]

    #Create K-fold prediction matrix for each model
    pred_weight_list = []

    #Run k_fold to return weight vectors
    for model in row_weight_params:
        loss, average_weight = modelbuilder.k_fold_weights(n_folds=N_FOLDS, row_weight_params=model)
        pred_weight_list.append(average_weight)

    #Dump prediction row vector weights to pickle
    pickle.dump(pred_weight_list, open(output_folder + "pred_weight.pickle", 'wb'))

#%%
#Loads the pickled prediction matrices and row weight vectors for 5 best models
pred_weights = pickle.load(open(output_folder + "pred_weight.pickle", 'rb'))
pred_matrices = pickle.load(open(output_folder + "pred_matrices.pickle", 'rb'))
ensemble_list = []
print("Binary cross-entropy of amount of matrices in ensemble * amount weight vectors in ensemble")

#Loops over all prediction matrices
#BCE of baseline model with no scaling
#BCE of best matrix 0.01624726504087448
#BCE of best ensemble matrix 0.01606087014079094
#BCE of best enseble matrix with best ensemble weigth 0.015337980352342129

for matrix in range(1, len(pred_matrices)+1):

    #Average 1 to 5 prediction matrices, moving from best to worst
    av_matrix = sum(pred_matrices[:matrix])/(matrix)

    bce_without_weight = modelbuilder.calc_bce(y_true=np.array(modelbuilder.y_test).astype(float), y_pred=av_matrix)
    #print("Binary crossentropy for just the matrix", matrix, " is ", bce_without_weight)
    #Loops over all row weight vectors
    for weight in range(1, len(pred_weights) + 1):

        #Average 1 to 5 weight vectors, moving from best to worst
        av_weights = sum(pred_weights[:weight])/(weight)

        #Create perfect weight array based on actual y_test
        perfect_weights = np.array(modelbuilder.y_test.sum(axis=1)).reshape(-1,1)

        #Computes the ensemble that can be made with X average prediction matrices and Y average row vectors
        current_ensemble = av_matrix * av_weights
        perfect_weight_ensemble = av_matrix * perfect_weights
        
        #Compute and log binary cross-entropy for each combination of ensembles
        bce_perfect_row = modelbuilder.calc_bce(y_true=np.array(modelbuilder.y_test).astype(float), y_pred=perfect_weight_ensemble)
        bce_ensemble = modelbuilder.calc_bce(y_true=np.array(modelbuilder.y_test).astype(float), y_pred=current_ensemble)
        ensemble_list.append([weight, matrix, bce_ensemble, bce_perfect_row, av_weights, av_matrix])
        #print("matrix: ", matrix, "weight vector: ", weight, " is ", bce_ensemble, " and perfect weight BCE is ", bce_perfect_row)
        print(matrix," & ",weight," & ",round(bce_ensemble,6), " & ",round(bce_perfect_row,6))

#Sort ensemble results with lowest binary cross-entropy first
ensemble_list.sort(key=lambda x:x[2])

#Round the best row weight vector
best_row_weight_rounded = ensemble_list[0][4].round()

#Define empty prediction matrix and predicted matrix
e_matrix = np.zeros(modelbuilder.y_test.shape)
best_pred_mat = np.array(modelbuilder.y_test)

#Get row weight rounded amount of highest probablity indices per row
print(ensemble_list[0][5])

#Loops over all rows in best prediction matrix
for i, amount_probs in enumerate(best_row_weight_rounded):
    
    #Loop over the amount of indices to grab from each row
    for j in range(0, int(amount_probs[0]+1)):
        
        #Get argmax of row
        arg_max = np.argmax(best_pred_mat[i])
        #print("Index: ", i, "value: ", j, " with argmax: ", arg_max)
        
        #Set index found by argmax to 0
        best_pred_mat[i][arg_max] = 0

        #Set argmax indices to 1 in e_matrix
        e_matrix[i][arg_max] = 1

row_weight_bce_improve = 100-((ensemble_list[0][3]/ensemble_list[0][2])*100)
prob_matrix_bce_improve = 100-row_weight_bce_improve
print(row_weight_bce_improve)
print(prob_matrix_bce_improve)
#%%
#Print the best, worst and delta between best and worst ensemble
print("Best ensemble has ", ensemble_list[0][0], " matrices ", ensemble_list[0][1], " weights and Binary cross-entropy of", ensemble_list[0][2])
print("Perfect row weight would yield ", ensemble_list[0][3], " Which represents ", row_weight_bce_improve , 
"% of possible improvement while probablity prediction improvements could yield ", prob_matrix_bce_improve, "%")

print("Worst ensemble has ", ensemble_list[-1][0], " matrices ", ensemble_list[-1][1], " weights and Binary cross-entropy of", ensemble_list[-1][2])
print("Gain due to ensemble model: ", ensemble_list[-1][2] - ensemble_list[0][2])
#%%
#Save the type of drug target across the columns
drug_target_counts = []
for col in y.columns:
    drug_target_counts.append([col.split("_")[-1]])
#%%
#Print a table of drug target types and their column counts
print(pd.DataFrame(drug_target_counts).value_counts()[:4])
print("Other", 205-pd.DataFrame(drug_target_counts).value_counts()[:4].sum())

#Percentage of targets to be predicted.
print(modelbuilder.y_test.sum().sum()/(modelbuilder.y_test.shape[0]*modelbuilder.y_test.shape[1])*100)
#%%