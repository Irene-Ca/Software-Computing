# Reference

## get_data(datapath)
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

## Adding_Feature_Category(df)
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

## Clean_Missing_Data(df)
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

## Label_to_Binary(Label)
Encode target labels with value 0 for background and 1 for signal.

    Parameters
    ----------
    Label : String
        label of the event (b or s).

    Returns
    -------
    Label : Int
        numerical label (0 or 1).

## Train_Valid_Test(df)
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

## Split_Jets(df)
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

## Separate_data_label(df)
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

## Clean_0_Jet(Train, Val, Test)
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

## Clean_1_Jet(Train, Val, Test)
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

## compiling_model(n_jet)
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
 
## training_model(n_jet, TrainingSet, T_Label, ValidationSet, V_Label, epoch)
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

## plot_NN(n_jet, history, model, TestSet, Te_Label)
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

## Predict_NN(model0, model1, model2, TestSet0, TestSet1, TestSet2)
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

## cross_validation(seed, dtrain, CVparams)
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

## train_BDT(dvalid, dtrain, BDT_params)
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

## plot_BDT(bst)
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

## AMS(Cut, Label, Label_Predict, KaggleWeight, Output)
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


## Plot_AMS_NN(x, Te_Label, Label_Predict, Te_KaggleWeight, Output)
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

## Plot_AMS_BDT(x, dtest, Te_Label, Te_KaggleWeight, bst, ntree_lim)
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
        Booster.
    ntree_lim : Int
        Variable used to get predictions from the best iteration during BDT training.

    Returns
    -------
    None.

## play(Model, datapath)
Function called for the execution of the program.

    Parameters
    ----------
    Model : String
        Defines which model will be used for the classification of the dataset: Boosted Decision Tree('BDT') or Neural Network ('NN').
        If not defined by the user 'NN' is passsed as default value.
    datapath : String
        Path of data in csv format that should be downloaded.

    Returns
    -------
    None.

  
    
