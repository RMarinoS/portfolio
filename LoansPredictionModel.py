"""
Project Creditworthiness.

A small bank office has received a great increase in the number of loan petitions. Usually, they manage 200 per week, but they are expecting to receive 500 next week. 
The manager sees a great opportunity to start automating the process and wants to create a model capable of differentiating between those that will be approved and those that will be rejected.
Therefore, a model will be trained using previous loan classification data, meaning that we will use a predictive model. 
It will be binary as it will decide if a petition will be admitted or dismissed.
"""
"""
The steps are:
    1. Explore the data
    2. Look for column correlations
    3. Create the model
    4. Validate the model
    5. Iterate if it's needed
"""
#First we load the packages need
import os
import pandas as pd
from sklearn import tree
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 12})

def calculate_model_params(model, x_test, y_test):
    y_predicted = model.predict(x_test)
    y_pred = list(map(round, y_predicted))
    model_confusionMatrix = confusion_matrix(y_test, y_pred)
    model_AccuracyScore= accuracy_score(y_test, y_pred)
    display_ConfussionMatrix = ConfusionMatrixDisplay(confusion_matrix=model_confusionMatrix, display_labels=['Non Creditworthy','Creditworthy'])
    errors = abs(y_pred  - y_test)

    return display_ConfussionMatrix, model_AccuracyScore, errors

def create_ROC_curves(model, x_test, y_test, title):
    #define metrics
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Logistic Regression Model: ROC Curve')
    plt.legend(loc=4)
    plt.show()

os.chdir('/Users/ramon.marino/Documents/Predictive Analysis Course/Project 3 -  Bank Loans/Creditworthiness/')
df_credit = pd.read_excel("credit-data-training.xlsx")
#Dataframe exploration
df_credit.describe()
"""
We have identified a few columns that are categorical, even when the data type is "int64". For find them, we just use nunique() function.
These columns will be removed when are not providing extra information; in other words, those which has always the same value.
These columns are:
    - Occupation
    - Concurrent-Credits
    - Duration-in-Current-address


In addition, we have detected columns with just 2 data values. We will analyse if they are balanced with an histogram.
Those columns found were:
    - Foreign-Worker
    - No-of-Credits-at-this-Bank
    - Guarantors

Other columns that could impact on our model are those with empty values. We found some of them using describe() function.
However, exploring column 'Value-Savings-Stocks' we so that majority of the rows had value 'None'. Either way, this field is not valuable for us.
Columns with many empty values:
    - Value-Savings-Stocks

All these columns are removed.
"""
df_credit.drop(columns=(['Occupation', 'Foreign-Worker', 'Concurrent-Credits', 'Telephone', 'No-of-Credits-at-this-Bank', 'Guarantors', 'Value-Savings-Stocks', 'Duration-in-Current-address']), inplace=True)
#We want to select all columns with numetic type as with them can be calculated the correlation
credit_numeric_columns = []
for column in df_credit.columns:
     if df_credit[column].dtype == "int64":
             credit_numeric_columns.append(column)
     else:
             continue
            
#Next, we calculate the correlation among all the columns
df_corr = pd.DataFrame(columns=(['Parameter', 'Corr-parameter', 'Correlation']))
index = 0
for parameter in credit_numeric_columns:
    for column in credit_numeric_columns:
        if parameter == column:
            print(f'{parameter} is equal to {column}')
            continue
        else:
            corr = df_credit[f'{parameter}'].corr(df_credit[f'{column}'])
            df_corr.at[index, 'Parameter'] = parameter
            df_corr.at[index, 'Corr-parameter'] = column
            df_corr.at[index, 'Correlation'] = corr
            index += 1
"""
Now we have all calculated all correlations. For working with this new dataframe, first we drop any combination which has produced a non valid result.
We only consider as a relevant correlation that which is equals to 0.7 or higher; in absolute values.
Thus, we will filter for only seeing those relation with those parameters.
"""
df_corr.dropna(inplace=True)
#We look for correlation among the variables
df_corr.query('abs(Correlation) >= 0.7')
"""
None of the variables has shown a strong correlation with other, so there are no reasons for dropping any of them
For following all process, we have to handle missing values from 'Age-years' column. We do not want to drop it as it's a relevant one, but we cannot have empty rows.
So, we decided to fill those spaces with the median. The main reason is that the median is an statistic parameter which is not being so much disturbed by outliers, as happens with the average.
In this case, this field has a median value of 33, so we will add it in those rows which have nothing in the field.
"""
df_credit['Age-years'].fillna(33, inplace=True)

#We also adapt our dependent variable to be 1 and 0s. For optimize the process, we will use a dictionary
change_dict = {'Creditworthy':1, 'Non-Creditworthy':0}
df_credit = df_credit.replace({'Credit-Application-Result':change_dict})

"""
Our dataset looks great now, so we will move to create our models.

"""
dataset_dummies = pd.get_dummies(df_credit.drop(columns='Credit-Application-Result'))
dataset_dummies = dataset_dummies.astype('int')
x = dataset_dummies.drop(columns=['Payment-Status-of-Previous-Credit_No Problems (in this bank)','Purpose_Other','Length-of-current-employment_< 1yr', 'Account-Balance_No Account'])
y = df_credit['Credit-Application-Result'].astype('int64')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

"""
For logistic regression we will use statsmodel as sklearn has no funcionality for reviewing p-values.
In addition, statsmodels provides a really good summary with model main indicator
"""

logisticModel = sm.Logit(y_train, x_train).fit()
print(logisticModel.summary())
#Predicting test data for calculating the accuracy
logisticModel_confusionMatrix, logisticModel_accuractyScore, errors = calculate_model_params(logisticModel,x_test, y_test)

#Second interation
x2 = dataset_dummies[[
    'Account-Balance_Some Balance'
        , 'Payment-Status-of-Previous-Credit_Paid Up'
        , 'Payment-Status-of-Previous-Credit_Some Problems'
            , 'Purpose_Home Related'
            , 'Purpose_New car'
            , 'Purpose_Other'
            , 'Purpose_Used car'
    ]]

x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y,test_size=0.3)
logisticModel2 = sm.Logit(y2_train, x2_train).fit()
print(logisticModel2.summary())
#Predicting test data for calculating the accuracy
logisticModel2_confusionMatrix, logisticModel2_accuracyScore, errors_2 = calculate_model_params(logisticModel2, x2_test, y2_test)

"""
Tree models:
Taking advantage from have done 3 iterations reshaping the datasetfrom sklearn.metrics import confusion_matrix, ConfusionMatrixDisplays, we just use those but with the Tree model.
"""
#Tree model
treeModel = tree.DecisionTreeClassifier(random_state=100, max_depth=3, min_samples_leaf=5)
treeModel.fit(x_train, y_train)
tM_confusionMatrix, tM_accuracyScore, tM_errors = calculate_model_params(treeModel, x_test, y_test)
#Random Forest model
rF = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rF.fit(x_train, y_train)
rF_confusionMatrix, rF_accuracyScore, rF_errors = calculate_model_params(rF, x_test, y_test)
#Boosted model
boostedModel = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
boostedModel.fit(x_train, y_train)
bM_confusionMatrix, bM_accuracyScore, bM_errors = calculate_model_params(boostedModel, x_test, y_test)

print(f'LogisticRegression accuracy: {logisticModel_accuractyScore}\n Tree Model Accuracy: {tM_accuracyScore}\n Random Forest Accuracy: {rF_accuracyScore}\n Boosted Model Accuracy: {bM_accuracyScore}')

model_features = x_test.columns
for model in ['logisticModel', 'treeModel', 'randomForest', 'boostedModel']:
    if model == 'logisticModel':
        print(logisticModel.pvalues())
    elif  model == 'treeModel':
        plt.barh(model_features, treeModel.feature_importances_)
        plt.xlabel(f'Feature Relevance: {model}')
    elif  model == 'randomForest':
        plt.barh(model_features, rF.feature_importances_)
        plt.xlabel(f'Feature Relevance: {model}')
    elif  model == 'boostedModel':
        plt.barh(model_features, boostedModel.feature_importances_)
        plt.xlabel(f'Feature Relevance: {model}')
    else:
        continue
    plt.show()
"""
The model with the best accuracy is the Random Forest, with a 77,33%. 
In addition, we have considered that in the confusion matrix it is really relevant to consider the number of predictions done correctly also the number of false positives; 
these ones are extremely relevant as those specially affecting the bank.
Based on the confusion matrix, also the Random Forest model is having the best results; 
it’s true that logistic regression model makes less false positives, but only one less while the false negatives are much higher and the accuracy is several points lower.
"""
#Fore predicting the new users, we have to load the data and transform it to have our defined parameters
df_new = pd.read_excel('customers-to-score.xlsx')
#From our previous dataset, we select only the relevant columns and we do the dummies also
dataset_columns = df_credit.drop(columns='Credit-Application-Result').columns
dataset_new = pd.get_dummies(df_new[dataset_columns])
#Finally, we drop the dummy variables we have used as baseline and transform everything into int datatype
dataset_new_final = dataset_new.drop(columns=['Payment-Status-of-Previous-Credit_No Problems (in this bank)','Length-of-current-employment_< 1yr', 'Account-Balance_No Account'])
dataset_new_dummies = dataset_new_final.astype('int')
#We will use our model to do predictions about the loans!
new_users_prediction = logisticModel.predict(dataset_new_final).round()
#We store the prediction score in the dataframe
df_new['Score Prediction Result'] = new_users_prediction

