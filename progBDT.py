import os
import numpy as np
import pandas as pd
import requests
from pandas.plotting import scatter_matrix
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from keras.regularizers import l2
from keras.regularizers import l1
from keras.callbacks import ModelCheckpoint
import xgboost as xgb
from xgboost import plot_importance
import graphviz
pd.options.mode.chained_assignment = None

seed = 12345
np.random.seed(seed)
path = "https://www.dropbox.com/s/dr64r7hb0fmy76p/atlas-higgs-challenge-2014-v2.csv?dl=1"

def get_data(datapath):
    '''
    Downloads the dataset, saves it on disk.

    Parameters
    ----------
    datapath : String
        path of data in csv format.

    Raises
    ------
    SystemExit
        prints an error and calls sys.exit .

    Returns
    -------
    dataset : pandas.dataframe
        Dataframe containing data readed from CSV file.

    '''
    if("http" in datapath):
        print("Downloading Dataset")
        try:
            # Download
            dataset = requests.get(datapath)
            dataset.raise_for_status()
        except requests.exceptions.RequestException as e:
            print("Error: Could not download file")
            raise SystemExit(e)
        response = requests.get(datapath)
        print("Writing dataset on disk") 
        with open('data.csv', 'wb') as f:
            f.write(response.content)
        datapath = "data.csv"
    
    # Reading dataset and creating pandas.DataFrame.
    dataset = pd.read_csv(datapath,header=0)
    print("Entries ", len(dataset))        
    
    return dataset
df = get_data(path)
#df = get_data(path='atlas-higgs-challenge-2014-v2.csv')
df.drop('EventId', axis=1, inplace=True)

Epoch_Value = 40

feature_list = df.columns.tolist()
feature_list.remove('Weight')
feature_list.remove('Label')
feature_list.remove('KaggleSet')
feature_list.remove('KaggleWeight')
#feature_list.remove('PRI_jet_num')

def Adding_Feature_Category(df):
    '''
    Denotes to which category the event belongs and adds a new feature to the dataset.
    It can assume three possible values: 
        2 for the boosted category,
        1 for the VBF category,
        0 for those events that do not belong to a defined category.

    Parameters
    ----------
    df : pandas.dataframe
        Dataframe containing data readed from CSV file.
    Returns
    -------
    df : pandas.dataframe
        Dataframe with a new column added for the "Category" feature.

    '''
    #Category 0 = Non , 1 = VBF , 2 = Boosted
    df_Non= pd.DataFrame(columns= df.columns)
    df_VBF = pd.DataFrame(columns= df.columns)
    df_Boosted = pd.DataFrame(columns= df.columns)

    df_01_Jets = df[df['PRI_jet_num'] == 0]
    df_01_Jets = df_01_Jets.append(df[df['PRI_jet_num'] == 1])

    df_Boosted_final = df_01_Jets[df_01_Jets['DER_pt_h'] > 100]
    df_Non = df_Non.append(df_01_Jets[df_01_Jets['DER_pt_h'] <= 100])

    df_23_Jets = df[df['PRI_jet_num'] > 1]

    df_lead_up = df_23_Jets[df_23_Jets['PRI_jet_leading_pt'] > 50]
    df_lead_down = df_23_Jets[df_23_Jets['PRI_jet_leading_pt'] <= 50]

    df_Boosted = df_Boosted.append(df_lead_down)

    df_sublead_up = df_lead_up[df_lead_up['PRI_jet_subleading_pt'] > 30]
    df_sublead_down = df_lead_up[df_lead_up['PRI_jet_subleading_pt'] <= 30]

    df_Boosted = df_Boosted.append(df_sublead_down)

    df_deltaeta_up = df_sublead_up[df_sublead_up['DER_deltaeta_jet_jet'] > 3.0]
    df_deltaeta_down = df_sublead_up[df_sublead_up['DER_deltaeta_jet_jet'] <= 3.0]

    df_Boosted = df_Boosted.append(df_deltaeta_down)

    df_VBF = df_deltaeta_up[df_deltaeta_up['DER_mass_vis'] > 40]
    df_massvis_down = df_deltaeta_up[df_deltaeta_up['DER_mass_vis'] <= 40]

    df_Boosted = df_Boosted.append(df_massvis_down)

    df_Boosted_final = df_Boosted_final.append(df_Boosted[df_Boosted['DER_pt_h'] > 100])
    df_Non = df_Non.append(df_Boosted[df_Boosted['DER_pt_h'] <= 100])

    df_Non['Category'] = 0
    df_VBF['Category'] = 1
    df_Boosted_final['Category'] = 2

    df = pd.concat([df_Non, df_VBF, df_Boosted_final])

    return df

#Adding Category Feature
df = Adding_Feature_Category(df)
df = df.sort_index(axis=0)
#possospostarlo più giu?
#df['PRI_jet_num'] = df['PRI_jet_num'].astype(str).astype(int)


def Label_to_Binary(Label):

    Label[Label == 'b'] = 0
    Label[Label == 's'] = 1
    return(Label)

df['Label'] = Label_to_Binary(df['Label'])

'''
#Scaling

#MinMaxScaling
#scaler = MinMaxScaler()

#StandardScaling
scaler = StandardScaler()
df[feature_list] = scaler.fit_transform(df[feature_list])
'''
print('PRI_jet_num', df['PRI_jet_num'].dtype)

def Train_Valid_Test(df):

    TrainingSet = df[df['KaggleSet'] == 't']
    ValidationSet = df[df['KaggleSet'] == 'b']
    TestSet = df[df['KaggleSet'] == 'v']
    Unused = df[df['KaggleSet'] == 'u']
    return TrainingSet, ValidationSet, TestSet, Unused

def Separate_data_label(df):

    Label = df['Label']
    Weight = df['Weight']
    KaggleWeight = df['KaggleWeight']
    df.drop('Weight', axis=1 ,inplace=True)
    df.drop('Label', axis=1 ,inplace=True)
    df.drop('KaggleSet', axis=1 ,inplace=True)
    df.drop('KaggleWeight', axis=1 ,inplace=True)
    return df, Label, Weight, KaggleWeight

df['PRI_jet_num'] = df['PRI_jet_num'].astype(str).astype(int)

TrainingSet, ValidationSet, TestSet, Unused = Train_Valid_Test(df)

TrainingSet, Tr_Label, Tr_Weight, Tr_KaggleWeight = Separate_data_label(TrainingSet)
ValidationSet, V_Label, V_Weight, V_KaggleWeight = Separate_data_label(ValidationSet)
TestSet, Te_Label, Te_Weight, Te_KaggleWeight = Separate_data_label(TestSet)

dtrain = xgb.DMatrix(data = TrainingSet, label = Tr_Label, weight = Tr_KaggleWeight, missing = -999.0)
dvalid = xgb.DMatrix(data = ValidationSet, label = V_Label, weight = V_KaggleWeight, missing = -999.0)
dtest = xgb.DMatrix(data = TestSet, label = Te_Label, weight = Te_KaggleWeight, missing = -999.0)

def cross_validation(seed):

    # Parameters to tune with cross validation
    CVparams = {'objective' : 'binary:logistic',
                'bst:max_depth' : 9,
                'min_child_weight' : 7, 
                'gamma' : 0, 
                'sub_sample' : 0.5,
                'colsample_bytree' : 0.5, 
                'bst:eta' : 0.1,
                'eval_metric' : ['ams@0.15', 'auc'],
                'silent' : 1,
                'nthread' : 16}
        
    res = xgb.cv(CVparams, dtrain, nfold= 5, num_boost_round = 999, seed = seed, early_stopping_rounds=25)
    return res

def train_BDT():

    #These are the parameters used by the train method of xgb
    # so they have been already updated to their best values
    #founded with hyperparameters tuning with cross validation
    params = {'objective' : 'binary:logistic',
              'bst:max_depth' : 9,
              'min_child_weight' : 7, 
              'gamma' : 0, 
              'sub_sample' : 0.9,
              'colsample_bytree' : 0.9, 
              'bst:eta' : 0.1,
              'eval_metric' : ['ams@0.15', 'auc'],
              'silent' : 1,
              'nthread' : 16}
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    #training
    num_round = 3000
    bst = xgb.train(params, dtrain, num_round , evals = evallist, early_stopping_rounds=25)
    bst.save_model('BDT.model')
    #print('best_score ', bst.best_score, "\n" 'best_iteration ', bst.best_iteration, "\n" 'best_ntree_limit ', bst.best_ntree_limit)
    score = bst.best_score
    iteration = bst.best_iteration
    ntree_lim = bst.best_ntree_limit
    return score, iteration, ntree_lim
res = cross_validation(seed)
print('CV results'"\n", res)


score, iteration, ntree_lim = train_BDT()
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model('BDT.model')  # load data

'''
It will give error because: 'Booster' object has no attribute 'best_score'
xgboost.train() will return a model from the last iteration, not the best one
I must find a way to save also early_stopping
'''
print('best_score ', score, "\n" 'best_iteration ', iteration, "\n" 'best_ntree_limit ', ntree_lim)

