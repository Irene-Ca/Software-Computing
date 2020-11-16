# Software-Computing exam
Code for the exam of the programming course of the physics department of the university of Bologna.

Here are proposed two different approaches for the classification of events taken from a dataset provided by the ATLAS experiment at CERN for 
the Kaggle Higgs Challenge.

## Dataset
The dataset was simulated by the official ATLAS full detector simulator. It reproduced the random proton-proton collisions and the measures 
of the resulting particles by a virtual model of the detector.

The signal sample corresponds to the events in which the Higgs boson decays into two taus (H → τ τ ). 

The background sample was generated by 
the three dominant background processes:
- Z → τ τ production
- t t̄  events with a hadronic τ and a leptonic τ
- W boson decays.

## Classification algorithms
The goal of this project is to get an introduction to ML algorithms with the classification of the events into signal or background.
Two different approaches were developed to classify the events.
- A model based on three Deep Neural Networks.
- A boosted decision tree provided by the Xgboost library.


The performance of the two algorithms is evaluated by computing the Approximate Median Significance (AMS) score, which is defined as
 	
![alt tag](https://github.com/IreneCa-gh/Software-Computing/blob/master/Images/AMSfunc.png)

Where 
-  s represents the true positive rate
- b indicates the false negative rate 
- b_reg is a constant regularisation term equal to 10.
 	
## Usage
To run the script program.py you have to type on the Command Line:
    
    $ python program.py

By default, the dataset will be downloaded from an internet link, saved on disk and the Neaural Networks algorithm will be trained.

If you have already downloaded the dataset as csv file and you want to use it, type on Command Line:
    
    $ python program.py --datapath <data_path>
    
    or
    
    $ python program.py -d <data_path>
    
Where <data_path> is the data path of your csv file.

Sice two different algorithms can be used for the classification, you can state on the Command Line which algorithm should be used.
    
    To train the model based on three Deep Neural Networks:
        
        $ python program.py --model NN
        
        or
        
        $ python program.py -m NN
        
    To train boosted decision tree:
    
        $ python program.py --model BDT
        
        or
        
        $ python program.py -m BDT
    
## Notes
- **Splitting into training set, validation set and test set:** 
this splitting was done by looking the KaggleSet variable, which allows to recover the original Kaggle training, public and private data sets provided for the challenge. 
In this way it has been guaranteed that the three subsets would have been good representatives of the data set as a whole.
    - The training set has 250000 events, 
    - the validation set has 100000 events 
    - the test set has 450000 events.

- **Category feature:**
a new feature, called "Category", is added to the data set before the training of the algorithms. It has been added to improve the classification and also to let the models to learn faster and easier. 
The events are distinguished into two analysis categories: the boosted and the VBF category. So this new feature can assume three possible values: 
    - 2 for the boosted category,
    - 1 for the VBF category,
    - 0 for those events that do not belong to a defined category.

The requirements applied to determine the Category value for each event are the same that
were used by [The ATLAS collaboration](https://link.springer.com/article/10.1007/JHEP04(2015)117) for the event selection and are summarized in the following table


| Category  | Selection criteria |
| --------- | ------------------ |
|   VBF     | At least two jets with pT1 > 50 GeV and pT2 > 30 GeV |
|           | ∆η(j 1 , j 2 ) > 3.0 |
|           | Mvis _ττ > 40 GeV |
| Boosted   | Failing the VBF selection |
|           | pT_H > 100 GeV |

pT1 is the transverse momentum of the leading jet with the largest transverse momentum. 
pT2 is the transverse momentum of the subleading jet with the second largest transverse momentum. 
∆η(j 1 , j 2 ) is the absolute value of the pseudorapidity separation between the two jets. 
Mvis_ττ is the invariant mass of the visible tau decay products.
pT_H is the transverse momentum of the Higgs boson candidate.

- **Splitting with respect to the number of jets (only NN):**
the data set has been split into events with zero jets, events with one jet and events with two or more jets.
In this way, three different NN have been trained, one for each subset.
In general only one NN is enough if it is trained on all the training data. 
The NN should be able to learn to generalize the data. 
Consequently splitting the data sample into three subsets and then train three separated NN would have a worse performance.
It is possible that the limited data set could affect the performance of only one NN classifier. 
Using this strategy it has been observed that a three NN classifier has a better performance than only one NN on the whole data set.

- The idea of developing a method that could be applied for the Higgs Boson Machine Learning Challenge was derived from the collaboration
between the universities of Dortmund and of Bologna. In that preceding project we only worked on the algorithm based on the Neural Networks.
Afterwards the original code have been reorganized (to improve the readability) and the classification has been a bit improved. 
Moreover the BDT analysis have been added and the documentation have been written.

## External links
Higgs Kaggle Challenge on [Kaggle platform](https://www.kaggle.com/c/higgs-boson).

Open data: http://opendata.cern.ch/record/328

Keras: [Keras website](https://keras.io/about/)

XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

## Software and libraries required
| Package            Version|
|------------------ ---------|
| Keras               2.3.1 |
| matplotlib          3.1.3 |
| numpy               1.18.5 |
| pandas              1.0.3 |
| requests            2.25.0 |
| scikit-learn        0.21.2 |
| tensorflow          2.3.1 |
| xgboost             1.0.2 |
