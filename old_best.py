#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import seaborn as sns
import matplotlib.pyplot as plt
import kerastuner as kt
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout, Activation, ActivityRegularization, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2, L1L2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K

#------------------------ Loading data ------------------------#
is_kaggle = False
plot_graps = True

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

#Print few columns for report before scaling
print(X.loc[[0,1,2,23810,23811],["sig_id", "cp_type", "cp_time", "cp_dose", "g-0", "c-0"]])

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
#Show distribution of amount of labels per ROW
def plot_sum_per_target_count(y):
    
    #Count amount of targets per row and sum by target count
    label_per_row = y.sum(axis=1).value_counts().sort_index(axis=0)
    print(100-((303+55+13+6)/len(y)*100), " percent has 0,1 or 2 labels")

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
    row_plot.figure.savefig("figs/sum_per_target_count.jpg")

#Show distribution of amount of labels per COLUMN
def plot_sum_per_target(y):

    #Create df with label counts per column
    count_target_df = pd.DataFrame(y.sum(axis=0).sort_values(ascending=False), columns=["target count"])
    
    #print top 50 targets as pecentage of total targets
    tot_label = count_target_df["target count"].sum()
    top_50_label = count_target_df["target count"][:50].sum()
    print("Top 50 targets have " + str((top_50_label/tot_label)*100) + " percent of all labels")

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
    col_plot.figure.savefig("figs/sum_per_target.jpg")

#Plots cell and gene distributions
def plot_gene_cell_dist(df, scaler_type=None):

    # Get 4 random cell viability and 4 random gene expression cols
    cols = ["g-0","g-175","g-363","g-599", "c-4", "c-33", "c-65", "c-84"]

    # Create four polar axes and access them through the returned array
    fig, axs = plt.subplots(2,4, figsize=(20,10))

    #Adjust plot title based on pre/post data set scaling
    if scaler_type != None:
        fig.suptitle("Columns scaled with " + SC_TYPE, fontsize=22)
    else:
        fig.suptitle("Gene expression and cell viability distributions", fontsize=22)

    #Loop over plot grid
    count = 0
    for i in range(0, 2):
        for j in range(0, 4):

            #Color first and last four plots differently (seperate cell and gene by color)
            if count >= 4:
                axs[i, j].hist(x=X.loc[:, cols[count]], bins=50, color="#E1812B")
            else:
                axs[i, j].hist(x=X.loc[:, cols[count]], bins=50, color="#3174A1")

            axs[i, j].set_title("Distribution " + cols[count])
            count += 1

    #Adjust format of plot and save
    plt.tight_layout()
    fig.savefig("figs/genes_cells-dist.jpg")

#Plots skew and kurtotis for gene and cell data
def skew_kurtosis(df):

    #Calculate skewness and kurtosis
    kurtosis = X.loc[:,"g-0":].kurtosis(axis=0)
    skew = X.loc[:,"g-0":].skew(axis=0)

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
    gene_df = skew_kurt_df.loc["g-0":"g-771", :]
    cell_df = skew_kurt_df.loc["c-0":"c-99",:]

    return gene_df, cell_df

#Plot skew and kurtosis for gene/cell cols
def plot_skew_kurtosis(df, g_c, s_k, color):

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
    img_path = "figs/"+ g_c + "_" + s_k + ".jpg"
    fig.savefig(img_path)
    return img_path


#%%
def combine_graphs(images):
    #https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    import sys
    from PIL import Image

    figs = [Image.open(x) for x in images]
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
        
    new_im.save('figs/total_skew_kurt.jpg')    
    #fig.savefig('figs/total_skew_kurt.jpg')

    fig = plt.figure(figsize=(11,5))
    imgplot = plt.imshow(new_im)
    fig.suptitle("Skewness and Kurtosis values for gene expression and cell viability columns" , fontsize=12.8)
    plt.axis('off')
    plt.show()

combine_graphs(images=[path1, path2, path3, path4])
#%%

if plot_graps == True:
    #Create histogram of 8 columns pre scaling
    plot_gene_cell_dist(df=X)

    #Plot target distributions across columns and with rows summed by target counts
    plot_sum_per_target_count(y=y)
    plot_sum_per_target(y=y)

    #Get skew and kurtosis values for cell viability and gene expression
    gene_df, cell_df = skew_kurtosis(df=X)

    #Plot skew and kurtosis for gene expression and cell viability
    path1 = plot_skew_kurtosis(df=gene_df, g_c="gene", s_k="skewness", color="#3174A1")
    path2 = plot_skew_kurtosis(df=cell_df, g_c="cell", s_k="skewness", color="#E1812B")
    path3 = plot_skew_kurtosis(df=gene_df, g_c="gene", s_k="kurtosis", color="#3174A1")
    path4 = plot_skew_kurtosis(df=cell_df, g_c="cell", s_k="kurtosis", color="#E1812B")
    combine_graphs(images=[path1, path2, path3, path4])

#%%
#------------------------ Parameters ------------------------#
#Encoding type --> "map", "dummy"
ENC_TYPE = "dummy"

#Scaling type --> "standardize", "normalize", "quantile_normal", "quantile_uniform", "power", "robust"
SC_TYPE = "quantile_uniform"

#------------------------ Encoding and scaling dataframe columns ------------------------#
def scale_df(df, scaler_type):
    df_other = df.iloc[:, :3]
    df = df.iloc[:, 3:]
    
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
    print("Scaler used: ", sc)
    return pd.concat([df_other, pd.DataFrame(sc.fit_transform(df), columns=df.columns)], axis=1)

def encode_df(df, encoder_type):
    cols = ["cp_type", "cp_time", "cp_dose"]

    #Encode columns as dummy variables
    if encoder_type == "dummy":
        df = pd.concat([pd.get_dummies(df[cols], columns=cols), df],axis=1)
        df = df.drop(cols, axis=1)

    #Map values to encodable columns
    elif encoder_type == "map":           
        df['cp_type'] = df['cp_type'].map({"ctl_vehicle": 0, "trt_cp": 1})
        df['cp_time'] = df['cp_time'].map({24: 0, 48: 0.5, 72: 1})
        df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})

    print("Encoder used: ", encoder_type)
    return df

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
 
