import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


data_audit = pd.read_csv('data/audit_risk.csv')
data_trial = pd.read_csv('data/trial.csv')

data_audit.head()

data_trial.head()

print('Shape of Audit Dataset {}'.format(data_audit.shape))
print('\nShape of Trial Dataset {}'.format(data_trial.shape))

print('\nColumns in Audit Dataset\n {}'.format(data_audit.columns.values))
print('\nColumns in Trial Dataset\n {}'.format(data_trial.columns.values))

repeated_columns = ['Sector_score', 'LOCATION_ID', 'PARA_A', 'PARA_B', 'TOTAL', 'numbers','Money_Value', 'History', 'Score', 'Risk']

for i in repeated_columns:
    a=sum(data_audit[i]==data_trial[i])
    if a==776:
        print('{} column has same values in both dataframes\n'.format(i))


a=sum(data_audit['Score_A']==data_trial['SCORE_A']/10)
b=sum(data_audit['Score_B']==data_trial['SCORE_B']/10)
if a==776:
    print('Score_A column has same values in both dataframes\n')
if b==776:
    print('Score_B column has same values in both dataframes')



print(sum(data_audit['Money_Value']==data_trial['Money_Value']))
print(sum(data_audit['Risk']==data_trial['Risk']))
print(sum(data_audit['District_Loss']==data_trial['District']))


data_trial['LOCATION_ID'].unique()

data_trial=data_trial[data_trial['LOCATION_ID']!='LOHARU']
data_trial=data_trial[data_trial['LOCATION_ID']!='NUH']
data_trial=data_trial[data_trial['LOCATION_ID']!='SAFIDON']

# Dropping repeated columns from trial dataset in order to prepare it to merge with audit dataset
repeated_columns = ['Sector_score', 'District','LOCATION_ID', 'PARA_A', 'SCORE_A','PARA_B','SCORE_B',
                     'TOTAL', 'numbers','Money_Value','History','Score','Risk']
data_trial.drop(columns=repeated_columns, inplace=True, axis=1)

dataset=pd.concat([data_audit,data_trial], axis=1)

dataset.isnull().any()
dataset.dropna(inplace=True)

plt.figure(figsize=(35,25)) 
sns.heatmap(dataset.corr(), annot=True) 

dataset.drop(columns=['Detection_Risk','LOCATION_ID'],inplace=True,axis=1)

print(sum(dataset['Score_B.1']*10==dataset['Marks']))
print(sum(dataset['History_score']/10==dataset['Prob']))
print(sum(dataset['LOSS_SCORE']/10==dataset['PROB']))
print(sum(dataset['Score_MV']*10==dataset['MONEY_Marks']))
print(sum(dataset['Loss']==dataset['LOSS_SCORE']-2))

dataset.drop(columns=['Score_B.1','Prob', 'PROB', 'Score_MV','LOSS_SCORE'],inplace=True,axis=1)


a=dataset['Risk_A']+dataset['Risk_B']+dataset['Risk_C']+dataset['Risk_D']+dataset['RiSk_E']+dataset['Risk_F']
df=pd.concat([a,dataset['Inherent_Risk']],axis=1)
df=df.round(3)
sum(df[0]==df['Inherent_Risk'])


# for 773 rows out of 775 Sum of Risk A to RIsk F equals Inherent RIsk

dataset.drop(columns=['Risk_A','Risk_B','Risk_C','Risk_D','RiSk_E','Risk_F'],axis=1, inplace=True)


dataset.columns


dataset.info()

dataset.describe()


dataset.shape

plt.figure(figsize=(30,20)) 
sns.heatmap(dataset.corr(), annot=True,  cmap="YlGnBu")

dataset[dataset.dtypes[(dataset.dtypes=="float")].index.values].hist(figsize=[20,20])
dataset[dataset.dtypes[(dataset.dtypes=="int64")].index.values].hist(figsize=[11,11])

sns.pairplot(dataset, hue = 'Risk', vars = ['Sector_score','History','Score', 'TOTAL', 'Money_Value'])

dataset['Risk'] = dataset['Risk'].astype(str) 

custom_colors = {"0": "blue", "1": "orange"}

sns.countplot(x='Risk', data=dataset, palette=custom_colors)

plt.title("Count of Risk Levels")
plt.xlabel("Risk")
plt.ylabel("Count")

plt.show(block=False)

plt.hist(dataset['Audit_Risk'],bins=100)
plt.show(block=False)


dataset=dataset[dataset['Audit_Risk']<100]
X=dataset.drop(columns=['Audit_Risk','Risk'],axis=1)
y=dataset['Audit_Risk']

from pandas.plotting import scatter_matrix
attributes = X.columns.values[:5]
scatter_matrix(X[attributes], figsize = (15,15), c = y, alpha = 0.8, marker = 'O')
plt.show(block=False)


X=dataset.drop(columns=['Audit_Risk','Risk'],axis=1)
y=dataset['Audit_Risk']


X.head()

y.head()

X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train_unscaled.head()


X_train_unscaled.shape

y_train.shape


standard_scaler=StandardScaler()
X_train=standard_scaler.fit_transform(X_train_unscaled)
X_test=standard_scaler.transform(X_test_unscaled)

x2=pd.DataFrame(X_train)
plt.figure(figsize=(15,10))
sns.boxplot(data = x2)
plt.ylim(-1,2.5)


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=0)
bag_reg = BaggingRegressor(dt, n_estimators=500, max_samples=200, bootstrap=True, random_state=0,oob_score=True)

# 500 patches out of the dataset
# max_samples =100, number of samples in a classifier
# bootstrap = bagging/pasting ; =True is bagging i.e. with replacement select the samples
# oob: the left out sample score
# On an average only 63% of the data is selected in total

bag_reg.fit(X_train, y_train)
dt.fit(X_train,y_train)
y_pred = bag_reg.predict(X_test)

print('Model-01 using Decision Tree for Bagging\n')
print('Train score with bagging: {:.2f}'.format(bag_reg.score(X_train, y_train)))
print('Test score with bagging: {:.2f}\n'.format(bag_reg.score(X_test, y_test)))
print('Train score decision Tree: {:.2f}'.format(dt.score(X_train, y_train)))
print('Test score decision Tree: {:.2f}\n'.format(dt.score(X_test, y_test)))
print('Out of Bag score: {:.2f}'.format(bag_reg.oob_score_))


# Hyperparameter Tuning - Bagging (GridSearchCV)


from sklearn.model_selection import GridSearchCV

param_dist = { 'n_estimators': [50, 100,300,500,800], 'max_samples':[50,100,200,300,400]}

gs = GridSearchCV(BaggingRegressor(DecisionTreeRegressor(random_state=0),bootstrap=True, oob_score=True, random_state=0), 
                  param_grid = param_dist, cv=5, n_jobs=-1)

gs.fit(X_train, y_train)
print('Best parameters {}'.format(gs.best_params_))

import seaborn as sns
import matplotlib.pyplot as plt

# Extract cross-validation results from GridSearchCV
results = gs.cv_results_

# Reshape mean test scores into a 2D array for heatmap visualization
scores = np.array(results["mean_test_score"]).reshape(len(param_dist['max_samples']), len(param_dist['n_estimators']))

# Create a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(scores, annot=True, xticklabels=param_dist['n_estimators'], yticklabels=param_dist['max_samples'], cmap="YlGnBu")

plt.xlabel('# of estimators')
plt.ylabel('Max Samples')
plt.title('Hyperparameter Tuning - Bagging : Validation Score across different parameters')
plt.show(block=False)


print('Train score: {}'.format(gs.score(X_train,y_train)))
print('Test score: {}'.format(gs.score(X_test,y_test)))


X_b = X_train[:100,10].reshape(-1,1)
y_b = y_train[:100]

bag_reg = BaggingRegressor(DecisionTreeRegressor(random_state=0), n_estimators=300, bootstrap=True, 
                           random_state=0,oob_score=True)
bag_reg.fit(X_b, y_b)

X_new=np.linspace(X_b.min(), X_b.max(), 100).reshape(100, 1)
y_predict = bag_reg.predict(X_new)

plt.figure(figsize=(10,6))
plt.plot(X_new, y_predict, c = 'r',label='Prediction')
plt.scatter(X_b, y_b,label='Actual Data Points')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Audit Risk')
plt.title('Hyperparameter Tuning - Bagging : Score Vs. Audit Risk')
plt.show(block=False)






# ---------------------------------------------- Bagging with KNN -------------------------------------------------------------

# Training on a single KNN and then using Aggregate Bootstraping


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(3)
bag_reg = BaggingRegressor(knn, n_estimators=500, bootstrap=True, random_state=0,oob_score=True)

bag_reg.fit(X_train, y_train)
knn.fit(X_train,y_train)

print('Model-02 using KNN for Bagging\n')
print('Train score with bagging: {:.2f}'.format(bag_reg.score(X_train, y_train)))
print('Test score with bagging: {:.2f}\n'.format(bag_reg.score(X_test, y_test)))
print('Train score KNN: {:.2f}'.format(knn.score(X_train, y_train)))
print('Test score KNN: {:.2f}\n'.format(knn.score(X_test, y_test)))
print('Out of Bag score: {:.2f}'.format(bag_reg.oob_score_))

# Finding best parameteters for Bagging of KNN Regressor using GridSearch Cross Validation

from sklearn.model_selection import GridSearchCV

param_dist = { 'n_estimators': [50, 100,300,500,800], 'max_samples':[50,100,200,300,400]}

gs = GridSearchCV(BaggingRegressor(KNeighborsRegressor(3),bootstrap=True, oob_score=True, random_state=0), 
                  param_grid = param_dist, cv=5, n_jobs=-1)

gs.fit(X_train, y_train)
print('Best parameters {}'.format(gs.best_params_))

# Visualizing the Cross validation Results
# Extract cross-validation results from GridSearchCV
results = gs.cv_results_

# Reshape mean test scores into a 2D array for heatmap visualization
scores = np.array(results["mean_test_score"]).reshape(len(param_dist['max_samples']), len(param_dist['n_estimators']))

# Create a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(scores, annot=True, xticklabels=param_dist['n_estimators'], yticklabels=param_dist['max_samples'], cmap="viridis")

plt.xlabel('# of estimators')
plt.ylabel('Max Samples')
plt.title('Bagging (Decision Tree) - Validation Score Heatmap')
plt.show(block=False)


# Fitting the model with best parameters and Visualizing how it fits the data

print('Train score: {}'.format(gs.score(X_train,y_train)))
print('Test score: {}'.format(gs.score(X_test,y_test)))

X_b = X_train[:100,10].reshape(-1,1)
y_b = y_train[:100]

bag_reg = BaggingRegressor(KNeighborsRegressor(3), n_estimators=800, bootstrap=True, 
                           random_state=0,oob_score=True)
bag_reg.fit(X_b, y_b)

X_new=np.linspace(X_b.min(), X_b.max(), 100).reshape(100, 1)
y_predict = bag_reg.predict(X_new)

plt.figure(figsize=(10,6))
plt.plot(X_new, y_predict, c = 'r',label='Prediction')
plt.scatter(X_b, y_b,label='Actual Data Points')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Audit Risk')
plt.title('Bagging with KNN : Score Vs. Audit Risk')
plt.show(block=False)




# ii)-----------------------------------------------------------  Pasting Pasting with Decision Tree ---------------------------------------------

# Training on a single Decision Tree and then using Aggregate Bagging

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=0)
bag_reg = BaggingRegressor(dt, n_estimators=500, max_samples=200, bootstrap=False, random_state=0)

bag_reg.fit(X_train, y_train)
dt.fit(X_train,y_train)
y_pred = bag_reg.predict(X_test)

print('Model-03 using Decision Tree for Pasting\n')
print('Train score with pasting: {:.2f}'.format(bag_reg.score(X_train, y_train)))
print('Test score with pasting: {:.2f}\n'.format(bag_reg.score(X_test, y_test)))
print('Train score decision Tree: {:.2f}'.format(dt.score(X_train, y_train)))
print('Test score decision Tree: {:.2f}\n'.format(dt.score(X_test, y_test)))

# Finding best parameteters for Pasting of Decision Tree Regressor using GridSearch Cross Validation

from sklearn.model_selection import GridSearchCV

param_dist = { 'n_estimators': [50, 100,300,500,800], 'max_samples':[50,100,200,300,400]}

gs = GridSearchCV(BaggingRegressor(DecisionTreeRegressor(random_state=0),bootstrap=False,random_state=0), 
                  param_grid = param_dist, cv=5, n_jobs=-1)

gs.fit(X_train, y_train)
print('Best parameters {}'.format(gs.best_params_))

# Visualizing the Cross validation Results

# Extracting cross-validation results from GridSearchCV
results = gs.cv_results_

# Convert mean test scores into a 2D array for heatmap
scores = np.array(results["mean_test_score"]).reshape(len(param_dist['max_samples']), len(param_dist['n_estimators']))

# Create a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(scores, annot=True, xticklabels=param_dist['n_estimators'], yticklabels=param_dist['max_samples'], cmap="YlGnBu")

plt.xlabel('# of estimators')
plt.ylabel('Max Samples')
plt.title('Pasting with Decision Tree : Validation Score across different parameters')
plt.show(block=False)

# Fitting the model with best parameters and Visualizing how it fits the data

print('Train score: {}'.format(gs.score(X_train,y_train)))
print('Test score: {}'.format(gs.score(X_test,y_test)))

X_b = X_train[:100,10].reshape(-1,1)
y_b = y_train[:100]

bag_reg = BaggingRegressor(DecisionTreeRegressor(random_state=0), n_estimators=100, bootstrap=False, random_state=0)
bag_reg.fit(X_b, y_b)

X_new=np.linspace(X_b.min(), X_b.max(), 100).reshape(100, 1)
y_predict = bag_reg.predict(X_new)

plt.figure(figsize=(10,6))
plt.plot(X_new, y_predict, c = 'r',label='Prediction')
plt.scatter(X_b, y_b,label='Actual Data Points')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Audit Risk')
plt.title('Pasting Pasting with Decision Tree : Score Vs. Audit Risk')
plt.show(block=False)







# ----------------------------------------------    Pasting with KNN ---------------------------------------------------------

# Training on a single KNN and then using Pasting


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(3)
bag_reg = BaggingRegressor(knn, n_estimators=500, bootstrap=False, random_state=0)

bag_reg.fit(X_train, y_train)
knn.fit(X_train,y_train)

print('Model-04 using KNN for Pasting\n')
print('Train score with pasting: {:.2f}'.format(bag_reg.score(X_train, y_train)))
print('Test score with pasting: {:.2f}\n'.format(bag_reg.score(X_test, y_test)))
print('Train score KNN: {:.2f}'.format(knn.score(X_train, y_train)))
print('Test score KNN: {:.2f}\n'.format(knn.score(X_test, y_test)))


# Finding best parameteters for Pasting of KNN Regressor using GridSearch Cross Validation

from sklearn.model_selection import GridSearchCV

param_dist = { 'n_estimators': [50, 100,300,500,800], 'max_samples':[50,100,200,300,400]}

gs = GridSearchCV(BaggingRegressor(KNeighborsRegressor(3),bootstrap=False, random_state=0), 
                  param_grid = param_dist, cv=5, n_jobs=-1)

gs.fit(X_train, y_train)
print('Best parameters {}'.format(gs.best_params_))


# Visualizing the Cross validation Results

# Extract GridSearchCV results
results = gs.cv_results_

# Reshape scores array for heatmap
scores = np.array(results["mean_test_score"]).reshape(len(param_dist['max_samples']), len(param_dist['n_estimators']))

# Plot heatmap
plt.figure(figsize=(10,10))
sns.heatmap(scores, annot=True, xticklabels=param_dist['n_estimators'], yticklabels=param_dist['max_samples'], cmap="viridis")

plt.xlabel('# of estimators')
plt.ylabel('Max Samples')
plt.title('Pasting with KNN : Validation Score across different parameters')
plt.show(block=False)


# Fitting the model with best parameters and Visualizing how it fits the data

print('Train score: {}'.format(gs.score(X_train,y_train)))
print('Test score: {}'.format(gs.score(X_test,y_test)))

X_b = X_train[:100,10].reshape(-1,1)
y_b = y_train[:100]

bag_reg = BaggingRegressor(KNeighborsRegressor(3), n_estimators=300, bootstrap=False, random_state=0)
bag_reg.fit(X_b, y_b)

X_new=np.linspace(X_b.min(), X_b.max(), 100).reshape(100, 1)
y_predict = bag_reg.predict(X_new)

plt.figure(figsize=(10,6))
plt.plot(X_new, y_predict, c = 'r',label='Prediction')
plt.scatter(X_b, y_b,label='Actual Data Points')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Audit Risk')
plt.title(' Pasting with KNN  : Score Vs. Audit Risk')
plt.show(block=False)











# iii) ------------------------------------------ Adaboost with Decision Tree ---------------------------------------------------------

# Training on a single Decision Tree and then using Adaboost

from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=100, learning_rate=0.3, random_state=0)
ada_reg.fit(X_train, y_train)

dt=DecisionTreeRegressor(max_depth=3)
dt.fit(X_train, y_train)
print('Train score with Adaboost: {:.2f}'.format(ada_reg.score(X_train, y_train)))
print('Test score with Adaboost: {:.2f}\n'.format(ada_reg.score(X_test, y_test)))
print('Train score with Decision Tree: {:.2f}'.format(dt.score(X_train, y_train)))
print('Test score with Decision Tree: {:.2f}'.format(dt.score(X_test, y_test)))

# Finding best parameteters Adaboost using GridSearch Cross Validation

from sklearn.model_selection import GridSearchCV

param_dist = { 'n_estimators': [50, 100,200,300,400], 'learning_rate' : [0.01,0.05,0.1,0.3,1]}

gs = GridSearchCV(AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), random_state=0), 
                  param_grid = param_dist, cv=5, n_jobs=-1)

gs.fit(X_train, y_train)
print('Best parameters {}'.format(gs.best_params_))

# Visualizing the Cross validation
# List item
# Results

# Extract GridSearchCV results
results = gs.cv_results_

# Reshape scores array for heatmap
scores = np.array(results["mean_test_score"]).reshape(len(param_dist['learning_rate']), len(param_dist['n_estimators']))

# Plot heatmap
plt.figure(figsize=(10,10))
sns.heatmap(scores, annot=True, xticklabels=param_dist['n_estimators'], 
            yticklabels=param_dist['learning_rate'], cmap="viridis")

plt.xlabel('# of Estimators')
plt.ylabel('Learning Rate')
plt.title('AdaBoost -  with Decision Tree:  Validation Score Heatmap')
plt.show(block=False)

# Fitting the model with best parameters and Visualizing how it fits the data

print('Train score: {}'.format(gs.score(X_train,y_train)))
print('Test score: {}'.format(gs.score(X_test,y_test)))

X_b = X_train[:100,10].reshape(-1,1)
y_b = y_train[:100]

ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=50, learning_rate=0.3, random_state=0)
ada_reg.fit(X_b, y_b)

X_new=np.linspace(X_b.min(), X_b.max(), 100).reshape(100, 1)
y_predict = ada_reg.predict(X_new)

plt.figure(figsize=(10,6))
plt.plot(X_new, y_predict, c = 'r',label='Prediction')
plt.scatter(X_b, y_b,label='Actual Data Points')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Audit Risk')
plt.title('Adaboost with Decision Tree: Score Vs. Audit Risk')
plt.show(block=False)





# -----------------------------------  Adaboost with Linear SVR --------------------------------------------------------------

# Training on Linear SVR and then using Adaboost

from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR

ada_reg = AdaBoostRegressor(LinearSVR(), n_estimators=100, learning_rate=0.1, random_state=0)
ada_reg.fit(X_train, y_train)

svr=LinearSVR(max_iter=10000, random_state=0)
svr.fit(X_train, y_train)
print('Train score with Adaboost: {:.2f}'.format(ada_reg.score(X_train, y_train)))
print('Test score with Adaboost: {:.2f}\n'.format(ada_reg.score(X_test, y_test)))
print('Train score with SVR: {:.2f}'.format(svr.score(X_train, y_train)))
print('Test score with SVR: {:.2f}'.format(svr.score(X_test, y_test)))

# Finding best parameteters Adaboost using GridSearch Cross Validation

from sklearn.model_selection import GridSearchCV

param_dist = { 'n_estimators': [50, 100,200,300,400], 'learning_rate' : [0.01,0.05,0.1,0.3,1]}

gs = GridSearchCV(AdaBoostRegressor(LinearSVR(), random_state=0),param_grid = param_dist, cv=5, n_jobs=-1)

gs.fit(X_train, y_train)
print('Best parameters {}'.format(gs.best_params_))

# Visualizing the Cross validation Results

if 'gs' in locals():
    # Extract mean test scores from GridSearchCV results
    results = gs.cv_results_
    scores = np.array(results["mean_test_score"]).reshape(
        len(param_dist['learning_rate']), len(param_dist['n_estimators'])
    )

    # Plot heatmap
    plt.figure(figsize=(10,10))
    sns.heatmap(scores, annot=True, fmt=".2f", xticklabels=param_dist['n_estimators'], 
                yticklabels=param_dist['learning_rate'], cmap="viridis")

    plt.xlabel('# of Estimators')
    plt.ylabel('Learning Rate')
    plt.title('Adaboost with Linear SVR : Validation Score across Different Parameters')
    plt.show(block=False)
else:
    print("Error: GridSearchCV has not been run. Run gs.fit(X_train, y_train) first.")

# Fitting the model with best parameters and Visualizing how it fits the data

print('Train score: {}'.format(gs.score(X_train,y_train)))
print('Test score: {}'.format(gs.score(X_test,y_test)))

X_b = X_train[:100,10].reshape(-1,1)
y_b = y_train[:100]

ada_reg = AdaBoostRegressor(LinearSVR(), n_estimators=100, learning_rate=0.3, random_state=0)
ada_reg.fit(X_b, y_b)

X_new=np.linspace(X_b.min(), X_b.max(), 100).reshape(100, 1)
y_predict = ada_reg.predict(X_new)

plt.figure(figsize=(10,6))
plt.plot(X_new, y_predict, c = 'r',label='Prediction')
plt.scatter(X_b, y_b,label='Actual Data Points')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Audit Risk')
plt.title('Adaboost with Linear SVR: Score Vs. Audit Risk')
plt.show(block=False)







# iv)----------------------------------------------------- Gradient Boosting  -----------------------------------------------------------

# Training on Gradient Boosting algorithm

from  sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X_train, y_train)

print('Train score with Adaboost: {:.2f}'.format(gbrt.score(X_train, y_train)))
print('Test score with Adaboost: {:.2f}\n'.format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingRegressor(max_depth=3, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X_train, y_train)

print('Train score with Adaboost: {:.2f}'.format(gbrt.score(X_train, y_train)))
print('Test score with Adaboost: {:.2f}\n'.format(gbrt.score(X_test, y_test)))

# Finding best parameteters using GridSearch Cross Validation

from sklearn.model_selection import GridSearchCV

param_dist = { 'n_estimators': [3,10,100,200,400], 'learning_rate' : [0.01,0.05,0.1,0.3,1]}

gs = GridSearchCV(GradientBoostingRegressor(random_state=42,max_depth=3),param_grid = param_dist, cv=5, n_jobs=-1)

gs.fit(X_train, y_train)
print('Best parameters {}'.format(gs.best_params_))


# Visualizing the Cross validation Results

if 'gs' in locals():
    # Extract mean test scores from GridSearchCV results
    results = gs.cv_results_
    scores = np.array(results["mean_test_score"]).reshape(
        len(param_dist['learning_rate']), len(param_dist['n_estimators'])
    )

    # Plot heatmap
    plt.figure(figsize=(10,10))
    sns.heatmap(scores, annot=True, fmt=".2f", xticklabels=param_dist['n_estimators'], 
                yticklabels=param_dist['learning_rate'], cmap="viridis")

    plt.xlabel('# of Estimators')
    plt.ylabel('Learning Rate')
    plt.title('Gradient Boosting - Validation Score Heatmap')
    plt.show(block=False)
else:
    print("Error: GridSearchCV has not been run. Run gs.fit(X_train, y_train) first.")

# Fitting the model with best parameters and Visualizing how it fits the data


print('Train score: {}'.format(gs.score(X_train,y_train)))
print('Test score: {}'.format(gs.score(X_test,y_test)))

X_b = X_train[:100,10].reshape(-1,1)
y_b = y_train[:100]

gb = GradientBoostingRegressor(max_depth=3, n_estimators=400, learning_rate=0.1, random_state=42)
gb.fit(X_b, y_b)

X_new=np.linspace(X_b.min(), X_b.max(), 100).reshape(100, 1)
y_predict = gb.predict(X_new)

plt.figure(figsize=(10,6))
plt.plot(X_new, y_predict, c = 'r',label='Prediction')
plt.scatter(X_b, y_b,label='Actual Data Points')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Audit Risk')
plt.title('Gradient Boosting : Score Vs. Audit Risk')
plt.show(block=False)





# v) ---------------------------------------------------------   XG BOOST ----------------------------------------------------------

from xgboost import XGBRegressor
from sklearn.datasets import make_regression

# Generate Sample Data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1️⃣ Train XGBoost Model with Different Parameters
xgb1 = XGBRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
xgb1.fit(X_train, y_train)

print('Train score (XGBoost, Depth=2, Estimators=3): {:.2f}'.format(xgb1.score(X_train, y_train)))
print('Test score (XGBoost, Depth=2, Estimators=3): {:.2f}\n'.format(xgb1.score(X_test, y_test)))


xgb2 = XGBRegressor(max_depth=3, n_estimators=3, learning_rate=1.0, random_state=42)
xgb2.fit(X_train, y_train)

print('Train score (XGBoost, Depth=3, Estimators=3): {:.2f}'.format(xgb2.score(X_train, y_train)))
print('Test score (XGBoost, Depth=3, Estimators=3): {:.2f}\n'.format(xgb2.score(X_test, y_test)))


# Hyperparameter Tuning using GridSearchCV


param_dist = {
    'n_estimators': [3, 10, 100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
}

gs = GridSearchCV(XGBRegressor(random_state=42, max_depth=3), param_grid=param_dist, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)

print('Best Parameters:', gs.best_params_)

# Visualizing Cross-Validation Results

# Ensure GridSearchCV has been run before accessing results
if 'gs' in locals() and hasattr(gs, "cv_results_"):
    # Extracting scores from GridSearchCV results
    results = gs.cv_results_
    
    # Ensure 'mean_test_score' exists in results
    if "mean_test_score" in results:
        scores = np.array(results["mean_test_score"]).reshape(
            len(param_dist['learning_rate']), len(param_dist['n_estimators'])
        )

        # Plot heatmap
        plt.figure(figsize=(10,10))
        sns.heatmap(scores, annot=True, fmt=".2f", xticklabels=param_dist['n_estimators'], 
                    yticklabels=param_dist['learning_rate'], cmap="viridis")

        plt.xlabel('# of Estimators')
        plt.ylabel('Learning Rate')
        plt.title('Validation Score across Different Parameters')
        plt.show(block=False)
    else:
        print("Error: 'mean_test_score' not found in GridSearchCV results. Check `gs.cv_results_`.")
else:
    print("Error: GridSearchCV object `gs` not found or not fitted. Run `gs.fit(X_train, y_train)` first.")

# Fitting Model with Best Parameters

print('Train Score with Best Params:', gs.score(X_train, y_train))
print('Test Score with Best Params:', gs.score(X_test, y_test))

# Visualizing Predictions

X_b = X_train[:100, 10].reshape(-1, 1)
y_b = y_train[:100]

best_xgb = XGBRegressor(max_depth=3, n_estimators=gs.best_params_['n_estimators'],
                        learning_rate=gs.best_params_['learning_rate'], random_state=42)
best_xgb.fit(X_b, y_b)

X_new = np.linspace(X_b.min(), X_b.max(), 100).reshape(-1, 1)
y_predict = best_xgb.predict(X_new)

plt.figure(figsize=(10,6))
plt.plot(X_new, y_predict, c='r', label='Prediction')
plt.scatter(X_b, y_b, label='Actual Data Points')
plt.legend()
plt.xlabel('Score')
plt.ylabel('Audit Risk')
plt.title('XGBoost - GridSearch Heatmap: Estimators vs Learning Rate')
plt.show(block=False)

import os
import joblib

# Get the absolute path of the backend folder
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Ensure 'backend' directory exists
os.makedirs(backend_path, exist_ok=True)

# Save the model directly in the main backend folder
model_path = os.path.join(backend_path, "xgb_model.pkl")
joblib.dump(gs.best_estimator_, model_path)

print(f"Model saved successfully at '{model_path}'")




# --------------------------------------------------------- SHAP   ------------------------------------------------------------

import shap
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes  # Use a valid dataset

 # Load dataset (use any dataset suitable for regression)
data = load_diabetes()
X, y = data.data, data.target
feature_names = data.feature_names

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

print("Model training completed.")

# Create SHAP Explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Convert SHAP values to DataFrame
shap_df = pd.DataFrame(shap_values, columns=feature_names)

print("SHAP values computed.")

# Global Feature Importance (Summary Plot)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)
plt.title("SHAP Global Feature Importance - XGBoost")

# Feature Importance as Bar Chart

shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names)
plt.title("SHAP Feature Importance as Bar Chart - XGBoost")

# Feature Interaction (Dependence Plot)

shap.dependence_plot("bmi", shap_values, X_test, feature_names=feature_names)
plt.title("SHAP Feature Interaction - XGBoost")

# Force Plot (Explaining a Single Prediction)

i = 10  # Choose a specific observation
shap.force_plot(explainer.expected_value, shap_values[i, :], X_test[i, :], feature_names=feature_names, matplotlib=True)
plt.title("SHAP Force Plot - XGBoost")

# Waterfall Plot (Breakdown of Prediction)

shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test[0], feature_names=feature_names))
plt.title("SHAP Waterfall Plot - XGBoost")

# Decision Plot

shap.decision_plot(explainer.expected_value, shap_values, feature_names=feature_names)
plt.title("SHAP Decision Plot - XGBoost")


