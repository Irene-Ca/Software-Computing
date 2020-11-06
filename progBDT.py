import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
pd.options.mode.chained_assignment = None

seed = 12345
np.random.seed(seed)
Epoch_Value = 40
datapath = "https://www.dropbox.com/s/dr64r7hb0fmy76p/atlas-higgs-challenge-2014-v2.csv?dl=1"

def get_data(datapath):
    '''
    Downloads the dataset, saves it on disk.

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

def cross_validation(seed, dtrain):
    '''
    function to perform cross validation before training.

    Parameters
    ----------
    seed : Int
        Random seed. It's important to properly compare the scores with different parameters.

    Returns
    -------
    res : list(string)
        Evaluation history.

    '''
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

def train_BDT(dvalid, dtrain):
    '''
    Trains a BDT with given parameters, values founded using cross validation.

    Returns
    -------
    score : Int
        evaluation score at the best iteration.
    iteration : Int
        at which boosting iteration the best score has occurred.
    ntree_lim : Int
        variable used to get predictions from the best iteration during BDT training.

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
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    #training
    num_round = 2000
    bst = xgb.train(params, dtrain, num_round , evals = evallist, early_stopping_rounds=25)
    bst.save_model('BDT.model')
    #print('best_score ', bst.best_score, "\n" 'best_iteration ', bst.best_iteration, "\n" 'best_ntree_limit ', bst.best_ntree_limit)
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
    bst : XGBModel
        XGBModel instance.

    Returns
    -------
    None.

    '''
    xgb.plot_tree(bst, num_trees=4)
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    fig.savefig('plotTreeHiggs.pdf')
    plt.clf()
    plt.show()

    xgb.plot_importance(bst)
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    fig.savefig('plotImportanceHiggs.pdf') 
    plt.clf()
    plt.show()


def AMS(Model, Cut, Label, Label_Predict, KaggleWeight, Output):

    '''
    Function to compute the Approximate Median Significance(AMS) of the total classificator.

    Parameters
    ----------
    Model : Int
        it determines which the model is used for the classification.
        if Model == 1 : NNs 
        if Model == 2 : BDT.
    Cut : float
        it must assume values in the range [0,1[
    Label : numpy.array
        true labels of the whole test set.
    Label_Predict : numpy.array
        class inferences done by the three Neural Networks over the corresponding test sets.
        They are joined together to have predictions for each event of the whole test set.
    KaggleWeight : numpy.array
        KaggleWeight variables of the whole test set.
    Output : numpy.array
        if Model == 1 : 
            inferences done by the three Neural Networks over the corresponding test sets.
            They are joined together to have predictions for each event of the whole test set.
        if Model == 2 :
            inferences done by the BDT over the test set.

    Returns
    -------
    float
        Value of the AMS function.

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




def Plot_AMS_BDT(x, dtest, Te_Label, Te_KaggleWeight, bst, ntree_lim):
    
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
        AMS_values[i] = AMS(2, x[i], Te_Label, Label_Predict, Te_KaggleWeight, Output)
        i += 1
    MaxAMS = np.amax(AMS_values)
    print('Maximum AMS for TestSet:', MaxAMS)
    plt.plot(x, AMS_values)
    plt.xlabel('Cut')
    plt.ylabel('AMS Score')
    plt.savefig('AMS_Score.pdf')
    plt.clf()
    return

def play(args):
    #data_file = get_data(datapath)
    data_file = get_data('atlas-higgs-challenge-2014-v2.csv')

    # Reading dataset and creating pandas.DataFrame.
    df = pd.read_csv(data_file,header=0)
    #EventId column is useless because pandas.dataframe has a default index 
    df.drop('EventId', axis=1, inplace=True)
    
    #Adding Category Feature
    df = Adding_Feature_Category(df)
    df = df.sort_index(axis=0)
    #possospostarlo più giu?
    #df['PRI_jet_num'] = df['PRI_jet_num'].astype(str).astype(int)
    
    df['Label'] = Label_to_Binary(df['Label'])
    #print('PRI_jet_num', df['PRI_jet_num'].dtype)
    
    df['PRI_jet_num'] = df['PRI_jet_num'].astype(str).astype(int)
    
    TrainingSet, ValidationSet, TestSet, Unused = Train_Valid_Test(df)
    
    TrainingSet, Tr_Label, Tr_Weight, Tr_KaggleWeight = Separate_data_label(TrainingSet)
    ValidationSet, V_Label, V_Weight, V_KaggleWeight = Separate_data_label(ValidationSet)
    TestSet, Te_Label, Te_Weight, Te_KaggleWeight = Separate_data_label(TestSet)
    
    dtrain = xgb.DMatrix(data = TrainingSet, label = Tr_Label, weight = Tr_KaggleWeight, missing = -999.0)
    dvalid = xgb.DMatrix(data = ValidationSet, label = V_Label, weight = V_KaggleWeight, missing = -999.0)
    dtest = xgb.DMatrix(data = TestSet, label = Te_Label, weight = Te_KaggleWeight, missing = -999.0)
    
    #Cross Validation
    #res = cross_validation(seed, dtrain)
    #print('CV results'"\n", res)
    
    
    score, iteration, ntree_lim, bst = train_BDT(dvalid, dtrain)
    #bst = xgb.Booster({'nthread': 4})  # init model
    #bst.load_model('BDT.model')  # load data

    print('best_score ', score, "\n" 'best_iteration ', iteration, "\n" 'best_ntree_limit ', ntree_lim)
    
    Cut = np.linspace(0.5, 1, num=200)
    AMS_values = Plot_AMS_BDT(Cut, dtest, Te_Label, Te_KaggleWeight, bst, ntree_lim)
    