#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import requests
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import xgboost as xgb
import argparse
pd.options.mode.chained_assignment = None

seed = 12345
np.random.seed(seed)
Epoch_Value = 40

datapath = "https://www.dropbox.com/s/dr64r7hb0fmy76p/atlas-higgs-challenge-2014-v2.csv?dl=1"

def get_data(datapath):
    '''
    Function to download the dataset and to save it on disk.

    Parameters
    ----------
    datapath : String
        path of data in csv format that should be downloaded.

    Raises
    ------
    SystemExit
        prints an error and calls sys.exit .

    Returns
    -------
    datapath : String
        path of data in csv format written on disk.

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
    return datapath

def Adding_Feature_Category(df):
    '''
    Denotes to which category the event belongs and adds a new feature to the dataset.
    The new feature can assume three possible values: 
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


def Clean_Missing_Data(df):
    '''
    Replaces the default values used for missing data (-999.0) with the average of the defined values of the corresponding feature.
    
    Parameters
    ----------
    df : pandas.dataframe

    Returns
    -------
    df : pandas.dataframe
        Dataframe with mean values in place of the missing data values.
    feature_list : list
        list of column names that could contain missing values.

    '''
    feature_list = df.columns.tolist()
    feature_list.remove('Weight')
    feature_list.remove('Label')
    feature_list.remove('KaggleSet')
    feature_list.remove('KaggleWeight')
    feature_list.remove('PRI_jet_num')    
    for col in feature_list:
        df[col].replace({-999 : np.nan}, inplace= True)
        m = df[col].mean()
        df[col].replace({np.nan : m}, inplace= True)
    return df, feature_list


def Label_to_Binary(Label):
    '''
    Encode target labels with value 0 for background and 1 for signal.

    Parameters
    ----------
    Label : String
        label of the event (b or s).

    Returns
    -------
    Label : Int
        numerical label (0 or 1).

    '''

    Label[Label == 'b'] = 0
    Label[Label == 's'] = 1
    return(Label)


def Train_Valid_Test(df):
    '''
    Splits the dataset into training set, validation set and test set with respect to the KaggleSet variable.

    Parameters
    ----------
    df : pandas.dataframe
        dataset containing all the events.

    Returns
    -------
    TrainingSet : pandas.dataframe
        events used for training.
    ValidationSet : pandas.dataframe
        events used for validation.
    TestSet : pandas.dataframe
        events used for test.
    Unused : pandas.dataframe
        unused events.

    '''
    TrainingSet = df[df['KaggleSet'] == 't']
    ValidationSet = df[df['KaggleSet'] == 'b']
    TestSet = df[df['KaggleSet'] == 'v']
    Unused = df[df['KaggleSet'] == 'u']
    return TrainingSet, ValidationSet, TestSet, Unused


def Split_Jets(df):
    '''
    Distinguishes the events with respect to the number of jets ('PRI_jet_num' variable) into three different subsets:
    events with zero jets, events with one jet and events with two or more jets.

    Parameters
    ----------
    df : pandas.dataframe

    Returns
    -------
    df0 : pandas.dataframe
        events with zero jets.
    df1 : pandas.dataframe
        events with one jet.
    df2 : pandas.dataframe
        events with two or more jets.

    '''
    df0 = df[df['PRI_jet_num'] == 0]
    df1 = df[df['PRI_jet_num'] == 1]
    df2 = df[df['PRI_jet_num'] >= 2]
    return df0, df1, df2

def Separate_data_label(df):
    '''
    Function to have the Label, Weight and KaggleWeight variables in different sets, separated by the dataset.

    Parameters
    ----------
    df : pandas.dataframe
        dataset containing all the variables together.

    Returns
    -------
    df : pandas.dataframe
        dataset without Label, Weight and KaggleWeight variables.
    Label : pandas.dataframe
        set containing the Label variables.
    Weight : pandas.dataframe
        set containing the Weight variables.
    KaggleWeight : pandas.dataframe
        set containing the KaggleWeight variables.

    '''
    Label = df['Label']
    Weight = df['Weight']
    KaggleWeight = df['KaggleWeight']
    df.drop('Weight', axis=1 ,inplace=True)
    df.drop('Label', axis=1 ,inplace=True)
    df.drop('KaggleSet', axis=1 ,inplace=True)
    df.drop('KaggleWeight', axis=1 ,inplace=True)
    return df, Label, Weight, KaggleWeight


def Clean_0_Jet(Train, Val, Test):
    '''
    Removes variable columns that are meaningless for events with 0 jets.

    Parameters
    ----------
    Train : pandas.dataframe
        dataset with also meaningless variables.
    Val : pandas.dataframe
        dataset with also meaningless variables.
    Test : pandas.dataframe
        dataset with also meaningless variables.

    Returns
    -------
    Train : pandas.dataframe
        dataset cleaned of meaningless variables ready for the NN training.
    Val : pandas.dataframe
        dataset cleaned of meaningless variables ready for the NN validation.
    Test : pandas.dataframe
        dataset cleaned of meaningless variables ready for the NN test.

    '''
    sets = [Train, Val, Test]
    for df in sets:
        df.drop('PRI_jet_leading_pt', axis=1, inplace=True)
        df.drop('PRI_jet_leading_eta', axis=1, inplace=True)
        df.drop('PRI_jet_leading_phi', axis=1, inplace=True)
        df.drop('PRI_jet_subleading_pt', axis=1, inplace=True)
        df.drop('PRI_jet_subleading_eta', axis=1, inplace=True)
        df.drop('PRI_jet_subleading_phi', axis=1, inplace=True)
        df.drop('PRI_jet_all_pt', axis=1, inplace=True)
    return Train, Val, Test

def Clean_1_Jet(Train, Val, Test):
    '''
    Removes variable columns that are meaningless for events with 1 jet.

    Parameters
    ----------
    Train : pandas.dataframe
        dataset with also meaningless variables.
    Val : pandas.dataframe
        dataset with also meaningless variables.
    Test : pandas.dataframe
        dataset with also meaningless variables.

    Returns
    -------
    Train : pandas.dataframe
        dataset cleaned of meaningless variables ready for the NN training.
    Val : pandas.dataframe
        dataset cleaned of meaningless variables ready for the NN validation.
    Test : pandas.dataframe
        dataset cleaned of meaningless variables ready for the NN test.

    '''
    sets = [Train, Val, Test]
    for df in sets:
        df.drop('PRI_jet_subleading_pt', axis=1, inplace=True)
        df.drop('PRI_jet_subleading_eta', axis=1, inplace=True)
        df.drop('PRI_jet_subleading_phi', axis=1, inplace=True)
        df.drop('PRI_jet_all_pt', axis=1, inplace=True)
    return Train, Val, Test


def compiling_model(n_jet):
    '''
    Defines the Neural Network structure, then its compilation.

    Parameters
    ----------
    n_jet : Int
        denotes on which subset the NN will be trained.
        The distinction is between events with zero jets (n_jet = 0), events with one jet (n_jet =1) and events with two or more jets (n_jet =2).

    Returns
    -------
    model : keras.engine.sequential.Sequential
        Sequntial NN object.
    
    '''
    model = Sequential()
    if n_jet == 0:
        model.add(Dense(units=64, activation='relu', input_dim=24, kernel_regularizer=l2(0.001)))
    elif n_jet == 1:
        model.add(Dense(units=64, activation='relu', input_dim=27, kernel_regularizer=l2(0.001)))
    elif n_jet == 2:
        model.add(Dense(units=64, activation='relu', input_dim=31, kernel_regularizer=l2(0.001)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def training_model(n_jet, TrainingSet, T_Label, ValidationSet, V_Label, epoch):
    '''    
    Neural Network training and validation.
    
    Parameters
    ----------
    n_jet : Int
        denotes on which subset the NN is trained.
        The distinction is between events with zero jets (n_jet = 0), events with one jet (n_jet =1) and events with two or more jets (n_jet =2).
    TrainingSet : pandas.dataframe
        events used for the training.
    T_Label : pandas.dataframe
        labels corresponding to training data.
    ValidationSet : pandas.dataframe
        events used for validation.
    V_Label : pandas.dataframe
        labels corresponding to validation data.
    epoch : Int
        number of epochs.

    Returns
    -------
    model : keras.engine.sequential.Sequential
        Sequntial NN object trained on a specific subset, defined by n_jet.
    history : History.history
        Record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.

    '''
    checkpoint = ModelCheckpoint( 'model_check_point.h5',
                                 monitor='val_accuracy',
                                 mode='max',
                                 save_best_only=True,
                                 verbose=1)
    model = compiling_model(n_jet)
    history = model.fit(np.asarray(TrainingSet, dtype = 'float32'), np.asarray(T_Label, dtype = 'float32'), validation_split=0, validation_data=(np.asarray(ValidationSet, dtype = 'float32') , np.asarray(V_Label, dtype = 'float32') ), epochs=epoch, batch_size=700, callbacks=[checkpoint])
    save =['KerasNN', n_jet, 'Model']
    NN_name ='_'.join([str(elem) for elem in save])
    model.save(NN_name)
    return model, history
    
def plot_NN(n_jet, history, model, TestSet, Te_Label):
    '''
    Produces three different plots:
        a) loss function evaluated at each epoch for the training and the validation sets
        b) accuracy evaluated at each epoch for the training and the validation sets
        c) output of the Deep Neural Network: value of the prediction obtained on the events of the test sets.

    Parameters
    ----------
    n_jet : Int
        denotes which NN performances to plot.
    history : History.history
        Record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.
    model : keras.engine.sequential.Sequential
        Sequntial NN object trained on a specific subset, defined by n_jet.
    TestSet : pandas.dataframe
        events used for testing.        
    Te_Label : pandas.dataframe
        labels corresponding to test data.

    Returns
    -------
    None.

    '''
    lt = ['Model loss for ', n_jet,' Jets DataSet']
    at = ['Accuracy for ', n_jet,' Jets DataSet']
    ot = ['TestSet Distribution for ', n_jet, ' Jets DataSet']
    l_save = ['Loss', n_jet, 'Jet.pdf']
    a_save = ['Acc', n_jet, 'Jet.pdf']
    o_save =['DNN_Output', n_jet, 'Jet.pdf']
    
    l_title =' '.join([str(elem) for elem in lt])
    a_title =' '.join([str(elem) for elem in at])
    o_title =' '.join([str(elem) for elem in ot])
    loss_name_fig ='_'.join([str(elem) for elem in l_save])
    acc_name_fig ='_'.join([str(elem) for elem in a_save])
    output_name_fig ='_'.join([str(elem) for elem in o_save])
    
    
    plt.figure()
    plt.plot(history.history['loss'], Label = 'Loss')
    plt.plot(history.history['val_loss'])
    plt.title(l_title)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(loss_name_fig)
    plt.clf()

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(a_title)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='lower right')        
    plt.savefig(acc_name_fig)
    plt.clf()
    
    plt.figure()
    plt.title(o_title)
    plt.xlabel('DNN Output')
    plt.ylabel('Counts')
    plt.yscale('log')
    DNN_Output = model.predict(np.asarray(TestSet, dtype = 'float32'))[:,0]
    plt.hist([DNN_Output[Te_Label==0], DNN_Output[Te_Label==1]], color=['red', 'blue'], bins= 100, histtype = 'barstacked')
    plt.savefig(output_name_fig)
    plt.clf()

def Predict_NN(model0, model1, model2, TestSet0, TestSet1, TestSet2):
    '''
    Function to compute the inferences and the class inferences of the three NN combined together.

    Parameters
    ----------
    model0 : keras.engine.sequential.Sequential
        Sequntial NN object trained on the subset with zero jets events.
    model1 : keras.engine.sequential.Sequential
        Sequntial NN object trained on the subset with one jet events.
    model2 : keras.engine.sequential.Sequential
        Sequntial NN object trained on the subset with two jets events.
    TestSet0 : pandas.dataframe
        events to test the NN trained on zero jets data.
    TestSet1 : pandas.dataframe
        events to test the NN trained on one jet data.
    TestSet2 : pandas.dataframe
        events to test the NN trained on two jets data.

    Returns
    -------
    Output : numpy.array
        inferences done by the three Neural Networks over the corresponding test sets.
        They are joined together to have predictions for each event of the whole test set.
    Label_Predict : numpy.array
        class inferences done by the three Neural Networks over the corresponding test sets.
        They are joined together to have predictions for each event of the whole test set.

    '''
    DNN_Output0 = model0.predict(np.asarray(TestSet0, dtype = 'float32'))[:,0]
    DNN_Output1 = model1.predict(np.asarray(TestSet1, dtype = 'float32'))[:,0]
    DNN_Output2 = model2.predict(np.asarray(TestSet2, dtype = 'float32'))[:,0]
    
    Label_Predict0 = model0.predict_classes(np.asarray(TestSet0, dtype = 'float32'))[:,0]
    Label_Predict1 = model1.predict_classes(np.asarray(TestSet1, dtype = 'float32'))[:,0]
    Label_Predict2 = model2.predict_classes(np.asarray(TestSet2, dtype = 'float32'))[:,0]
    
    Output = np.concatenate((DNN_Output0, DNN_Output1, DNN_Output2))
    Label_Predict = np.concatenate((Label_Predict0, Label_Predict1, Label_Predict2))
    return Output, Label_Predict
# BDT functions
def cross_validation(seed, dtrain, CVparams):
    '''
    Function to perform cross validation before training.

    Parameters
    ----------
    seed : Int
        Random seed.
    dtrain : DMatrix 
        The training DMatrix.
    CVparams : dict
        Parameters to tune with cross validation.        

    Returns
    -------
    res : list(string)
        Evaluation history.

    '''        
    res = xgb.cv(CVparams, dtrain, nfold= 5, num_boost_round = 500, seed = seed, verbose_eval = 1, early_stopping_rounds=25)
    return res

def train_BDT(dvalid, dtrain, BDT_params):
    '''
    Trains a BDT with given parameters (values finded by using cross validation).

    Parameters
    ----------
    dvalid : DMatrix 
        The validating DMatrix.
    dtrain : DMatrix 
        The training DMatrix.
    BDT_params : dict
        Parameters for training.  

    Returns
    -------
    score : Int
        Evaluation score at the best iteration.
    iteration : Int
        Iteration at which boosting iteration the best score has occurred.
    ntree_lim : Int
        Variable used to get predictions from the best iteration during BDT training.
    bst : trained booster model
        Booster.

    '''
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    #training
    num_round = 3000
    bst = xgb.train(BDT_params, dtrain, num_round, evals = evallist, early_stopping_rounds=25)
    bst.save_model('BDT.model')
    score = bst.best_score
    iteration = bst.best_iteration
    ntree_lim = bst.best_ntree_limit
    return score, iteration, ntree_lim, bst

def plot_BDT(bst):
    '''
    Produces two different plots:
        a) Plot specified tree
        b) Plot importance based on fitted trees.

    Parameters
    ----------
    bst : trained booster model
        Booster.

    Returns
    -------
    None.

    '''
    xgb.plot_importance(bst)
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    fig.savefig('plotImportanceHiggs.pdf') 
    plt.clf()
    
    xgb.plot_tree(bst, num_trees=4)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    fig.savefig('plotTreeHiggs.pdf', dpi = 300)
    plt.clf()


#Plotting AMS
def AMS(Cut, Label, Label_Predict, KaggleWeight, Output):
    '''
    Function to compute the Approximate Median Significance(AMS) of the total classificator.

    Parameters
    ----------
    Cut : float
        It must assume values in the range [0,1[
    Label : pandas.dataframe
        True labels of the whole test set.
    Label_Predict : numpy.array
        Class inferences done by the model over the corresponding test set.
    KaggleWeight : pandas.dataframe
        KaggleWeight variables of the whole test set.
    Output : numpy.array
        Inferences done by the model over the corresponding test set.
            
    Returns
    -------
    float
        Value of the AMS function for a specific value of Cut.

    '''
    
    #Cut = Accepting only Classified Data which is classified with Cut% safety
    Label_Cut = Label[Output > Cut]
    KaggleWeight_Cut = KaggleWeight[Output > Cut]
    Label_Predict_Cut = Label_Predict[Output > Cut]
    Label_Predict_Cut = np.concatenate((Label_Predict[Output > Cut], Label_Predict[Output < (1-Cut)]))
    Label_Cut = pd.concat([Label_Cut, Label[Output < (1-Cut)]], ignore_index=True)
    KaggleWeight_Cut = pd.concat([KaggleWeight_Cut, KaggleWeight[Output < (1-Cut)]], ignore_index=True)
    s = np.sum(KaggleWeight_Cut[(Label_Predict_Cut == 1) & (Label_Cut == 1)])
    KaggleWeight_Cut = KaggleWeight_Cut[Label_Predict_Cut == 1]
    Label_Cut = Label_Cut[Label_Predict_Cut == 1]
    b = np.sum(KaggleWeight_Cut[Label_Cut == 0])
    breg = 10
    return (np.sqrt(2*((s+b+breg)*np.log(1+(s/(b+breg)))-s)))


def Plot_AMS_NN(x, Te_Label, Label_Predict, Te_KaggleWeight, Output):
    '''
    Computes and plots the AMS value for different values of the x sequence.
    The maximum AMS score is printed.
    Function used for the NN.

    Parameters
    ----------
    x : numpy.linspace
        Sequence of numbers between 0.5 and 1.
    Te_Label : pandas.dataframe
        testset containing the Label variables.
    Label_Predict : numpy.array
        class inferences done by the three Neural Networks over the corresponding test sets.
        They are joined together to have predictions for each event of the whole test set.
    Te_KaggleWeight : pandas.dataframe
        testset containing the KaggleWeight variables.
    Output : numpy.array
        inferences done by the three Neural Networks over the corresponding test sets.
        They are joined together to have predictions for each event of the whole test set.

    Returns
    -------
    None.

    '''

    AMS_values = np.zeros(np.size(x))
    i = 0
    while i < np.size(x):
        AMS_values[i] = AMS(x[i], Te_Label, Label_Predict, Te_KaggleWeight, Output)
        i += 1
    MaxAMS = np.amax(AMS_values)
    print('Maximum AMS for TestSet:', MaxAMS)
    plt.plot(x, AMS_values)
    plt.xlabel('Cut')
    plt.ylabel('AMS Score')
    plt.savefig('AMS_Score_NN.pdf')
    plt.clf()
    
def Plot_AMS_BDT(x, dtest, Te_Label, Te_KaggleWeight, bst, ntree_lim):
    '''
    Computes and plots the AMS value for different values of the x sequence.
    The maximum AMS score is printed.
    Function used for the BDT.

    Parameters
    ----------
    x : numpy.linspace
        Sequence of numbers between 0.5 and 1.
    dtest : DMatrix 
        The testinging DMatrix.
    Te_Label : pandas.dataframe
        Testset containing the Label variables.
    Te_KaggleWeight : pandas.dataframe
        Testset containing the KaggleWeight variables.
    bst : trained booster model
        Booster
    ntree_lim : Int
        Variable used to get predictions from the best iteration during BDT training.

    Returns
    -------
    None.

    '''    
    Output = bst.predict(dtest, ntree_limit=ntree_lim)
    Output = np.asarray(Output)
    Label_Predict = np.asarray([np.round(line) for line in Output])
    print('best_preds',Label_Predict)
    test_labels = np.asarray([line for line in Te_Label])
    accuracy_test = accuracy_score(test_labels,Label_Predict)
    print('accuracy test', accuracy_test)
    
    AMS_values = np.zeros(np.size(x))
    i = 0
    while i < np.size(x):
        AMS_values[i] = AMS(x[i], Te_Label, Label_Predict, Te_KaggleWeight, Output)
        i += 1
    MaxAMS = np.amax(AMS_values)
    print('Maximum AMS for TestSet:', MaxAMS)
    plt.plot(x, AMS_values)
    plt.xlabel('Cut')
    plt.ylabel('AMS Score')
    plt.savefig('AMS_Score_BDT.pdf')
    plt.clf()

def play(Model, datapath):
    '''
    Function called for the execution of the program.

    Parameters
    ----------
    Model : String
        Defines which model will be used for the classification of the dataset: Boosted Decision Tree('BDT') or Neural Network ('NN').
        If not defined by the user 'NN' is passed as default value.
    datapath : String
        Path of data in csv format that should be downloaded.

    Returns
    -------
    None.

    '''
    data_file = get_data(datapath)
    #data_file = get_data('atlas-higgs-challenge-2014-v2.csv')
    
    # Reading dataset and creating pandas.DataFrame.
    df = pd.read_csv(data_file,header=0)
    
    #EventId column is useless because pandas.dataframe has a default index 
    df.drop('EventId', axis=1, inplace=True)
    
    #Feature engeneering: adding Category Feature to data
    df = Adding_Feature_Category(df)
    df = df.sort_index(axis=0)
    
    #Label Encoding
    df['Label'] = Label_to_Binary(df['Label'])
    
    if (Model == 'NN'):
        #CLASSIFICATION WITH NEURAL NETWORKS
        df, feature_list =  Clean_Missing_Data(df)
        #Scaling of features that could span wide ranges
        scaler = StandardScaler()
        df[feature_list] = scaler.fit_transform(df[feature_list])
    
        #Splitting into trainig, validation, test sets
        TrainingSet, ValidationSet, TestSet, Unused = Train_Valid_Test(df)
    
        #Splitting the sets with respect to the value of PRI_jet_num' variable
        TrainingSet0, TrainingSet1, TrainingSet2 = Split_Jets(TrainingSet)
        ValidationSet0, ValidationSet1, ValidationSet2 = Split_Jets(ValidationSet)
        TestSet0, TestSet1, TestSet2 = Split_Jets(TestSet)
        
        
        TrainingSet0, Tr_Label0, Tr_Weight0, Tr_KaggleWeight0 = Separate_data_label(TrainingSet0)
        TrainingSet1, Tr_Label1, Tr_Weight1, Tr_KaggleWeight1 = Separate_data_label(TrainingSet1)
        TrainingSet2, Tr_Label2, Tr_Weight2, Tr_KaggleWeight2 = Separate_data_label(TrainingSet2)
        ValidationSet0, V_Label0, V_Weight0, V_KaggleWeight0 = Separate_data_label(ValidationSet0)
        ValidationSet1, V_Label1, V_Weight1, V_KaggleWeight1 = Separate_data_label(ValidationSet1)
        ValidationSet2, V_Label2, V_Weight2, V_KaggleWeight2 = Separate_data_label(ValidationSet2)
        TestSet0, Te_Label0, Te_Weight0, Te_KaggleWeight0 = Separate_data_label(TestSet0)
        TestSet1, Te_Label1, Te_Weight1, Te_KaggleWeight1 = Separate_data_label(TestSet1)
        TestSet2, Te_Label2, Te_Weight2, Te_KaggleWeight2 = Separate_data_label(TestSet2)
    
        #Removal of meaningless features
        TrainingSet0, ValidationSet0, TestSet0 = Clean_0_Jet(TrainingSet0, ValidationSet0, TestSet0)
        TrainingSet1, ValidationSet1, TestSet1 = Clean_1_Jet(TrainingSet1, ValidationSet1, TestSet1)
        
        #training of the three NN
        model0, history0 = training_model(0, TrainingSet0, Tr_Label0, ValidationSet0 ,V_Label0, Epoch_Value)
        model1, history1 = training_model(1, TrainingSet1, Tr_Label1, ValidationSet1 ,V_Label1, Epoch_Value)
        model2, history2 = training_model(2, TrainingSet2, Tr_Label2, ValidationSet2 ,V_Label2, Epoch_Value)
        
        
        #Predictions, plotting and AMS computation
        Output ,Label_Predict = Predict_NN(model0, model1, model2, TestSet0, TestSet1, TestSet2)
        Te_Label = pd.concat([Te_Label0, Te_Label1, Te_Label2])
        Te_KaggleWeight = pd.concat([Te_KaggleWeight0, Te_KaggleWeight1, Te_KaggleWeight2])
        
        plot_NN(0, history0, model0, TestSet0, Te_Label0)
        plot_NN(1, history1, model1, TestSet1, Te_Label1)
        plot_NN(2, history2, model2, TestSet2, Te_Label2)
        
        Cut = np.linspace(0.5, 1, num=200)
        AMS_values = Plot_AMS_NN(Cut, Te_Label, Label_Predict, Te_KaggleWeight, Output)
   
    elif (Model == 'BDT'):
        df['PRI_jet_num'] = df['PRI_jet_num'].astype(str).astype(int)
    
        TrainingSet, ValidationSet, TestSet, Unused = Train_Valid_Test(df)
        
        TrainingSet, Tr_Label, Tr_Weight, Tr_KaggleWeight = Separate_data_label(TrainingSet)
        ValidationSet, V_Label, V_Weight, V_KaggleWeight = Separate_data_label(ValidationSet)
        TestSet, Te_Label, Te_Weight, Te_KaggleWeight = Separate_data_label(TestSet)
        
        dtrain = xgb.DMatrix(data = TrainingSet, label = Tr_Label, weight = Tr_KaggleWeight, missing = -999.0)
        dvalid = xgb.DMatrix(data = ValidationSet, label = V_Label, weight = V_KaggleWeight, missing = -999.0)
        dtest = xgb.DMatrix(data = TestSet, label = Te_Label, weight = Te_KaggleWeight, missing = -999.0)
        
        '''
        #Cross Validation
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
        res = cross_validation(seed, dtrain, CVparams)
        print('CV results'"\n", res)
        '''
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
        score, iteration, ntree_lim, bst = train_BDT(dvalid, dtrain, params)

        #print('best_score ', score, "\n" 'best_iteration ', iteration, "\n" 'best_ntree_limit ', ntree_lim)
        plot_BDT(bst)
        Cut = np.linspace(0.5, 1, num=200)
        AMS_values = Plot_AMS_BDT(Cut, dtest, Te_Label, Te_KaggleWeight, bst, ntree_lim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default= "NN", choices = ["NN", "BDT"], help="Choose the model you want to use")
    parser.add_argument("-d", "--datapath", default= datapath, help="Data path of csv file")
    try:
        parser.parse_args()
        options = parser.parse_args()

        play(Model = options.model, datapath = options.datapath)

    except argparse.ArgumentError:
        print('Catching an argumentError')

    