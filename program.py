import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
pd.options.mode.chained_assignment = None

seed = 12345
np.random.seed(seed)

df = pd.read_csv('atlas-higgs-challenge-2014-v2.csv')

#EventId column is useless because pandas.dataframe has a default index 
df.drop('EventId', axis=1, inplace=True)

Epoch_Value = 40

feature_list = df.columns.tolist()
feature_list.remove('Weight')
feature_list.remove('Label')
feature_list.remove('KaggleSet')
feature_list.remove('KaggleWeight')
feature_list.remove('PRI_jet_num')

def Adding_Feature_Category(df):
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

    for col in feature_list:
        df[col].replace({-999 : np.nan}, inplace= True)
        m = df[col].mean()
        df[col].replace({np.nan : m}, inplace= True)
    return df

df =  Clean_Missing_Data(df)

#Adding Category Feature to data
df = Adding_Feature_Category(df)
df = df.sort_index(axis=0)

def Label_to_Binary(Label):

    Label[Label == 'b'] = 0
    Label[Label == 's'] = 1
    return(Label)

df['Label'] = Label_to_Binary(df['Label'])


#StandardScaling
scaler = StandardScaler()
df[feature_list] = scaler.fit_transform(df[feature_list])

def Train_Valid_Test(df):

    TrainingSet = df[df['KaggleSet'] == 't']
    ValidationSet = df[df['KaggleSet'] == 'b']
    TestSet = df[df['KaggleSet'] == 'v']
    Unused = df[df['KaggleSet'] == 'u']
    return TrainingSet, ValidationSet, TestSet, Unused


def Split_Jets(df):

    df0 = df[df['PRI_jet_num'] == 0]
    df1 = df[df['PRI_jet_num'] == 1]
    df2 = df[df['PRI_jet_num'] >= 2]
    return df0, df1, df2

def Separate_data_label(df):

    Label = df['Label']
    Weight = df['Weight']
    KaggleWeight = df['KaggleWeight']
    df.drop('Weight', axis=1 ,inplace=True)
    df.drop('Label', axis=1 ,inplace=True)
    df.drop('KaggleSet', axis=1 ,inplace=True)
    df.drop('KaggleWeight', axis=1 ,inplace=True)
    return df, Label, Weight, KaggleWeight

TrainingSet, ValidationSet, TestSet, Unused = Train_Valid_Test(df)

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

def Clean_0_Jet(Train, Val, Test):

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

    sets = [Train, Val, Test]
    for df in sets:
        df.drop('PRI_jet_subleading_pt', axis=1, inplace=True)
        df.drop('PRI_jet_subleading_eta', axis=1, inplace=True)
        df.drop('PRI_jet_subleading_phi', axis=1, inplace=True)
        df.drop('PRI_jet_all_pt', axis=1, inplace=True)
    return Train, Val, Test


TrainingSet0, ValidationSet0, TestSet0 = Clean_0_Jet(TrainingSet0, ValidationSet0, TestSet0)
TrainingSet1, ValidationSet1, TestSet1 = Clean_1_Jet(TrainingSet1, ValidationSet1, TestSet1)

