#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Linzy Programming Revision - Python
#import modules needed to execute functions
#standard modules
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import sys
import os
from pandas import DataFrame
from ranges import RangeDict, Range
#%matplotlib inline
plt.rcParams['figure.figsize']=(15,6)

#data preprocessing modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

#model modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

#metric modules
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import altair as alt
import streamlit as st


# In[2]:


#import data from DAT650
def dataset(path_filename):
	#insert import error if file does not exist or type is not CSV
	try:
		data=pd.read_csv(path_filename, encoding='latin1')
		return data
	except IOError:
		print ('File does not exist or file type is not CSV.')
		sys.exit(1)

#list path and file name for data to import
original_data=dataset('Credit_Data.csv')
#original_data

#make copy of data
od=original_data


# In[3]:


#make groups for continuous data (Amount, Duration, Age)
AMT_DICT=RangeDict({Range(0,1360):1,Range(1360,2300):2,Range(2300,3900):3,Range(3900,max(od.AMOUNT)+1):4})
od['AMOUNT_GROUP']=[AMT_DICT[i] for i in od.AMOUNT]

DUR_DICT=RangeDict({Range(0,12):1,Range(12,18):2,Range(18,24):3,Range(24,max(od.DURATION)+1):4})
od['DURATION_GROUP']=[DUR_DICT[i] for i in od.DURATION]

AGE_DICT=RangeDict({Range(0,27):1,Range(27,33):2,Range(33,42):3,Range(42,max(od.AGE)+1):4})
od['AGE_GROUP']=[AGE_DICT[i] for i in od.AGE]


# In[4]:


#create dummies for categorical columnns; including prefix to reference original feature
def dummies(feature):
	dummy_feature=pd.get_dummies(feature,
	prefix=feature.name, dtype=int)
	return dummy_feature

dummy_variables=pd.concat([dummies(od["SAV_ACCT"]),
	dummies(od["EMPLOYMENT"]),
	dummies(od["PRESENT_RESIDENT"]),
	dummies(od["JOB"]),
	dummies(od["HISTORY"]),
	dummies(od["AMOUNT_GROUP"]),
	dummies(od["DURATION_GROUP"]),
	dummies(od["AGE_GROUP"]),
	dummies(od["INSTALL_RATE"]),
	dummies(od["CHK_ACCT"]),
	dummies(od["NUM_CREDITS"]),
	dummies(od["NUM_DEPENDENTS"])], axis=1)


# In[5]:


"""
concatenate dummy variables and variable for data without dropped columns to create new dataset
print columns names and inspect. Output will show a list of new columns with format
("Old_Column_1", "Old_Column_2"), etc...
"""
def new_dataset(original_dataset,dummies):
	new=pd.concat([original_dataset,dummies], axis=1)
	return new

new_data=new_dataset(od, dummy_variables)
#new_data.columns


# In[6]:


"""
drop the columns for the original features transformed by dummy variables
and create a data variable for remaining columns
"""
data2=new_data.drop(["DURATION","NUM_DEPENDENTS", "NUM_CREDITS","INSTALL_RATE", "AMOUNT",'AGE','OBS#', "AMOUNT_GROUP", "AGE_GROUP", "DURATION_GROUP", "CHK_ACCT", "SAV_ACCT", "EMPLOYMENT", "PRESENT_RESIDENT", "JOB", "HISTORY"], axis=1)
#data2.columns
#data2.to_csv('credit_data_new2.csv')


# In[7]:


#create data variables for independent features. Ouput variable count
def independent_variables(data, target_column):
	ind_var = (data.drop(target_column, axis=1))
	return ind_var

independent_variable=independent_variables(data2, "DEFAULT")
#len(independent_variable.columns)


# In[8]:


#create a list of feature names for independent variables to be used for labeling later
ind_var_list=list(independent_variable.columns)

#create data variables for depentent features
def dependent_variables(target_column):
	dep_var = (target_column)
	return dep_var

dependent_variable=dependent_variables(data2[["DEFAULT"]])
#dependent_variable.columns


# In[9]:


#split independent and depentent features into a training set(70%) and a testing set(30%)
train_features, test_features, train_labels, test_labels = train_test_split(independent_variable,
dependent_variable,test_size = 0.30,random_state = 42)


"""
print the shape of each of the 4 split data sets. Output is
("Training Features Shape: (700, 50), Training Labels Shape: (700, 1)"), etc...
"""

#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)


# In[10]:


"""
Each of the 3 top model were ran against 3 techniques which were:
1. run without optimization
2. run on cross validation and grid search with best parameter selection
3. run with feature reduction by importance
"""
def random_forest_cross_validation_gridsearch(train_feature, train_label):
	#random forest model. seed set at 42
	rf_exp = RandomForestClassifier(random_state=42)
	"""
	The best performing model for random forest was the cross validation
	and grid search with best parameter selection
	"""
	"""
	create selection pool for random forest randomized grid search parameters.
	Randomized search will try all parameter options
	"""
	#Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 10)]
	#Number of features to consider at every split
	max_features = ['auto', 'sqrt', 'log2']
	#Maximum number of levels
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	#Minimum number of samples
	min_samples_split = [2, 5, 10]
	#Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	#Method of selecting samples for training each tree
	bootstrap = [True, False]
	#function to measure the quality of a split
	criterion = ['gini', 'entropy']
	"""
	minimum weighted fraction of the sum total
	of weights (of all the input samples) required to be at a leaf node
	"""
	class_weight = ['balanced', 'balanced_subsample']
	#provide error exception that specifically states the problem is "missing parameter
	try:
		#build parameter grid for the randomized grid search
		random_grid = {'n_estimators': n_estimators,
		'max_features': max_features,
		'max_depth': max_depth,
		'min_samples_split': min_samples_split,
		'min_samples_leaf': min_samples_leaf,
		'bootstrap': bootstrap,
		'criterion': criterion,
		'class_weight': class_weight}
	except NameError:
		print("Missing parameter from grid parameter definition!")
	else:
		pass
	"""
	create cross validation randomized grid search for random forest incorporating random forest model
	parameter grid, number of iteration set to 100, 5 folds. seed set at 42
	"""
	rf_exp_random = RandomizedSearchCV(estimator = rf_exp,
	param_distributions = random_grid,
	n_iter = 100, cv = 5, verbose=2,
	random_state=42, n_jobs = -1)
	#create and fit the cross validation model
	fit=rf_exp_random.fit(train_feature, train_label)
	return fit


# In[11]:


rf_best_parameters=random_forest_cross_validation_gridsearch(train_features, train_labels)
rf_best_parameters


# In[12]:


"""
capture best paramerter from the resulted randomized grid search model.
Output is a list of best parameters from the search
"""
best_parameters = rf_best_parameters.best_params_
print(best_parameters)


# In[13]:


"""
update random forest with best parameters results and fit.
"""
#create parameter value list to be used in classifier
parameter_list=list(best_parameters.values())


# In[36]:


parameter_list


# In[14]:


#set classifier parameters to best_parameter.values to ensure they are updated to output
rf_exp_par = RandomForestClassifier(n_estimators= parameter_list[0], min_samples_split = parameter_list[1],
		min_samples_leaf= parameter_list[2],max_features=parameter_list[3],
		max_depth=parameter_list[4],criterion=parameter_list[5],
		class_weight=parameter_list[6], bootstrap=parameter_list[7],
		random_state=42)
rf_exp_par.fit(train_features, train_labels)


# In[15]:


def predct (model):
	predictions = model.predict(test_features)
	predictions = predictions.reshape((predictions.shape[0], 1))
	return predictions

rf_prdt=predct(rf_exp_par)


# In[16]:


#incorporate prediction results into the confusion matrix and generate metric reports
rf_exp_par_confusion=confusion_matrix(test_labels, rf_prdt)

print("RF Confusion Matrix:\n\n {}".format(rf_exp_par_confusion))
rf_exp_par_class_report=classification_report(test_labels, rf_prdt)
print("\n\n RF Classification Report:\n\n {}".format(rf_exp_par_class_report))


# In[17]:


"""
create metric function that calculates accuracy, missclassifications, precision,
sensitivity based on the results of the confusion matrix. Output is a list with
values for each metric in the format ('Precision: [0.33]', 'Sensitivity: [0.714]'), etc...
"""
def metric (confusion_matrix):
	#add exception to state that if error here, then confusion matrix format changed
	try:
		cm1=np.split(confusion_matrix[0],2)
		cm2=np.split(confusion_matrix[1],2)
		tn=cm1[0]
		fn=cm1[1]
		fp=cm2[0]
		tp=cm2[1]
		accuracy=(tn+tp)/(tn+fn+fp+tp)
		missclass=(fn+fp)/(tn+fn+fp+tp)
		precision=tp/(fp+tp)
		sensitivity=tp/(fn+tp)
	except TypeError:
		print("Problem with confusion matrix. Fix format!")
	else:
		pass

	return ["Accuracy: {}".format(np.round(accuracy,3)),
	"Missclassification: {}".format(np.round(missclass,3)),
	"Precision: {}".format(np.round(precision, 3)),
	"Sensitivity: {}".format(np.round(sensitivity, 3))]
rf_metric=metric(rf_exp_par_confusion)
print(metric(rf_exp_par_confusion))


# In[18]:


"""
Create a function to get a list of the top 5 importance features based on model results, and
chart importance features. Output is a list of important features with their
importance values. Importance value calculate with built-in model feature_importance_.
The chart is a bar chart.
"""
# Get numerical feature importances
rf_importances = list(rf_exp_par.feature_importances_)

def important_list (var_list, importances):
	# List of tuples with variable and importance
	feature_importances = [(feature, round(importance, 2)) for feature,
	importance in zip(var_list, importances)]
	# Sort the feature importances by most important first
	feature_importances = sorted(feature_importances,
	key = lambda x: x[1], reverse = True)
	# list of x locations for plotting
	x_values = list(range(len(importances)))
	return[ # Make a bar chart
	plt.bar(x_values, importances, orientation = 'vertical',
	color = 'r', edgecolor = 'k', linewidth = 1.2),
	# Tick labels for x axis
	plt.xticks(x_values, var_list, rotation='vertical'),
	# Axis labels and title
	plt.ylabel('Importance'), plt.xlabel('Variable'), plt.title('Variable Importances'),
	feature_importances[0:5]]

rf_importance_list=important_list(ind_var_list, rf_importances)
important_list(ind_var_list, rf_importances)


# In[19]:


rf_metric=metric(rf_exp_par_confusion)
rf_top_five=rf_importance_list[-1:]
rf_top_five


# In[20]:


r_imf=[]
r_irt=[]
for i in rf_top_five[0]:
    r_imf.append(list(i)[0])
    r_irt.append(list(i)[1])


# In[21]:


#create variables for prediction probabilities for each model
rf_predict=rf_exp_par.predict_proba(test_features)[:,1]
#gb_predict=gbrt.predict_proba(test_features)[:,1]
#y_pred_rt = rt_lm.predict_proba(test_features)[:,1]
#Create variables for precision/recall curve using predicted probability variables with test labels
#fpr_rt_lm2, tpr_rt_lm2, _ = precision_recall_curve(test_labels, y_pred_rt)
fpr_rf2, tpr_rf2, _ = precision_recall_curve(test_labels, rf_predict)
#fpr_grd2, tpr_grd2, _ = precision_recall_curve(test_labels, gb_predict)



"""
Calculate average precision/recall score for each model and plot
and plot the precision/recall curve for each model using
precision/recall score for labeling. Out put for scores are
("RF Average Precision/Recall Score 0.6331",
"LG Average Precision/Recall Score 0.5289"), etc...
"""
print("RF Average Precision/Recall Score {}".format(round(average_precision_score(test_labels, rf_predict),4
)))
#print("LG Average Precision/Recall Score {}".format(round(average_precision_score(test_labels, y_pred_rt),4)))
#print("GB Average Precision/Recall Score {}".format(round(average_precision_score(test_labels, gb_predict),4
#)))


# In[22]:


plt.figure(1)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_rt_lm2, tpr_rt_lm2, label="LG Average Precision/Recall Score {}".format(round(average_precision_score(test_labels, y_pred_rt),4)))
plt.plot(fpr_rf2, tpr_rf2, label="RF Average Precision/Recall Score {}".format(round(average_precision_score(test_labels, rf_predict),4)))
#plt.plot(fpr_grd2, tpr_grd2, label="GB Average Precision/Recall Score {}".format(round(average_precision_score
#(test_labels, gb_predict),4)))
plt.xlabel('Precision')
plt.ylabel('Sensitivity')
plt.title('Random Forest')#, Gradient Boost & Logistic Regression Recall/Sensitity ROC curve
plt.legend(loc='best')
plt.show()


# In[23]:


#create variables for ROC curve for each model
#fpr_rt_lm, tpr_rt_lm, _ = roc_curve(test_labels, y_pred_rt)
fpr_rf, tpr_rf, _ = roc_curve(test_labels, rf_predict)
#fpr_grd, tpr_grd, _ = roc_curve(test_labels, gb_predict)
"""
Plot the precision/recall curve for each model
"""
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_rt_lm, tpr_rt_lm, label='LR')#.format(round(roc_auc_score(test_labels, y_pred_rt), 4)))
plt.plot(fpr_rf, tpr_rf, label='RF')#.format(round(roc_auc_score(test_labels2, rf_predict),4)))
#plt.plot(fpr_grd, tpr_grd, label='GB')#.format(round(roc_auc_score(test_labels, gb_predict),4)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Random Forest')#, Gradient Boost & Logistic Regression TP Rate/Probability ROC curve
plt.legend(loc='best')
plt.show()


# In[24]:


"""
print ROC_AUC scores for each model. Output are ('LG ROC Score
RF ROC Score 0.7984'), etc...
"""
#print("LG ROC Score {}".format(round(roc_auc_score(test_labels,y_pred_rt), 4)))
print("RF ROC Score {}".format(round(roc_auc_score(test_labels,rf_predict), 4)))
#print("GB ROC Score {}".format(round(roc_auc_score(test_labels,gb_predict), 4)))

"""
References:
Koehrsen, W. (2017). Random forest in python. Towards Data Science.
Retrieved from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
Muller, A and Guido, S. (2017). Introduction to machine learning with python:
a guide for data scientists. Sabastopol, CA: O'Reilly Medial, Inc.
Pedregosa et al.(2011). Feature transformations with ensembles of trees.
Retrieved from https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html
"""



