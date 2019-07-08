#!/usr/bin/env python
# coding: utf-8

# # Evaluating Regression Models and Random Forests

# ##  **Data Ingestion**

# In[56]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor  # machine learning tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # ensemble method
from math import sqrt  # for root mean-squared error calculation

#set working directory
os.chdir('C:\\Users\R\Desktop\\MSDS 422\\Boston Housing Study\\jump-start-boston-housing-v001\\jump-start-boston-housing-v001')

# read data for the Boston Housing Study
# creating data frame restdata
boston_input = pd.read_csv('boston.csv')


# ## **Data Transformation & EDA**

# In[18]:


# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())


# In[19]:


print('\nGeneral description of the boston_input DataFrame:')

print(boston_input.info())


# In[20]:


# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())


# In[21]:


print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())


# In[22]:


# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,    boston.crim,    boston.zn,    boston.indus,    boston.chas,    boston.nox,    boston.rooms,    boston.age,    boston.dis,    boston.rad,    boston.tax,    boston.ptratio,    boston.lstat]).T


# In[23]:


# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)


# In[24]:


# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)


# In[25]:


# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)


# In[42]:


#create Pandas data frame of model data
boston_input = pd.DataFrame(data=prelim_model_data)

#create column names
boston_input.columns = ['Median_value','Crime_Rate','Land_Zoned', 'Industrial', 'Charles_River','Pollution', 'Avg_Rooms', 'Pre1940', 'Distance_center', 'Highway_access', 'Avg_tax', 'Teacher_ratio','Low_income']

#examine first five values of data frame
boston_input.head()


# In[43]:


list = ['Median_value','Crime_Rate','Land_Zoned', 'Industrial', 'Charles_River','Pollution', 'Avg_Rooms', 'Pre1940', 'Distance_center', 'Highway_access', 'Avg_tax', 'Teacher_ratio','Low_income']

#loop through list of continuous variables and fill NA values with mean value
for i in list:
    boston_input[i] = boston_input[i].fillna((boston_input[i].mean()))
    
boston_input.describe()


# In[132]:


#save data information to txt file
def data_info_save_to_file(data, dataname):
    print('\n---------{} Data Info----------\n'.format(dataname))
    print('\n{} Data shape: {}'.format(dataname, data.shape))
    print('\n{} Data types: {}'.format(dataname, data.dtypes))
    print('\n{} Data column values: {}'.format(dataname, data.columns.values)) 
    print('\n{} Data head: {}'.format(dataname, data.head())) 
    print('\n{} Data tail: {}'.format(dataname, data.tail()))
    print('\n{} Data descriptive statistics: {}'.format(dataname, data.describe()))
    print('\n{} Data correlations: {}'.format(dataname, data.corr(method='pearson')))
    with open("{}_Data_Info.txt".format(dataname), "w") as text_file:
        text_file.write('\n---------{} Data Info----------\n'.format(dataname)+
                        '\n{} Data shape: {}'.format(dataname, str(data.shape)) +
                        '\n{} Data types: {}'.format(dataname, str(data.dtypes)) +
                        '\n{} Data column values: {}'.format(dataname, str(data.columns.values)) + 
                        '\n{} Data head: {}'.format(dataname, str(data.head()))+ 
                        '\n{} Data tail: {}'.format(dataname, str(data.tail()))+
                        '\n{} Data descriptive statistics: {}'.format(dataname, str(data.describe()))+ 
                        '\n{} Data Info: {}'.format(dataname, str(data.info()))+
                        '\n{} Data correlations: {}'.format(dataname, str(boston_input.corr(method='pearson'))))


# In[133]:


#examine data information txt file
data_info_save_to_file(boston_input, 'boston_input')


# ## **Data Visualization**

# In[30]:


# Overal distributions of the variables
boston_input.hist( bins = 50, figsize = (30, 20)); plt.show()


# In[53]:


#plot histogram/density plots for data frame
f, axes = plt.subplots(2, 3, figsize=(12, 12), sharex=True)
sns.set_style('darkgrid')
sns.distplot(boston_input['Crime_Rate'], color= 'black', ax=axes[0,0])
sns.distplot(boston_input['Land_Zoned'], color = 'black', ax=axes[0,1])
sns.distplot(boston_input['Industrial'], color = 'black', ax=axes[0,2])
sns.distplot(boston_input['Charles_River'], color = 'black', ax=axes[1,0])
sns.distplot(boston_input['Pollution'], color = 'black', ax=axes[1,1])
sns.distplot(boston_input['Avg_Rooms'], color = 'black', ax=axes[1,2])


# In[52]:


#plot histogram/density plots for data frame
f, axes = plt.subplots(2, 3, figsize=(12, 12), sharex=True)
sns.set_style('darkgrid')
sns.distplot(boston_input['Pre1940'], color= 'black', ax=axes[0,0])
sns.distplot(boston_input['Distance_center'], color = 'black', ax=axes[0,1])
sns.distplot(boston_input['Highway_access'], color = 'black', ax=axes[0,2])
sns.distplot(boston_input['Avg_tax'], color = 'black', ax=axes[1,0])
sns.distplot(boston_input['Teacher_ratio'], color = 'black', ax=axes[1,1])
sns.distplot(boston_input['Low_income'], color = 'black', ax=axes[1,2])


# In[54]:


#plot histogram/density plot of response variable (Mean_value)
Density_plot_median_value = sns.distplot(boston_input['Median_value'] , color="black")
fig = Density_plot_median_value.get_figure()
fig.savefig("Median Value Density Plot.png", bbox_inches='tight')


# In[50]:


# plot linear relationships with response variable 'response_mv' together in one figure
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
f.suptitle('Linear relationships to response variable Median_value (% variables)', size = 16, y=.9)
sns.regplot(x='Crime_Rate', y='Median_value', data=boston_input, x_jitter=.1, color = "black", ax=axes[0, 0])
sns.regplot(x='Land_Zoned', y='Median_value', data=boston_input , x_jitter=.1, color = "black", ax=axes[0, 1])
sns.regplot(x='Industrial', y='Median_value', data=boston_input , x_jitter=.1, color = "black", ax=axes[1, 0])
sns.regplot(x='Charles_River', y='Median_value', data=boston_input , x_jitter=.1, color = "black", ax=axes[1, 1])
f.savefig('Linear relationships to response variable Median_value (% variables)' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)


# In[51]:


# plot linear relationships with response variable 'response_mv' together in one figure
f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
f.suptitle('Linear relationships to response variable response_mv (low range variables)', size = 16, y=.9)
sns.regplot(x='Avg_Rooms', y='Median_value', data=boston_input, x_jitter=.1, color = "black", ax=axes[0, 0])
sns.regplot(x='Distance_center', y='Median_value', data=boston_input, x_jitter=.1, color = "black", ax=axes[0, 1])
sns.regplot(x='Highway_access', y='Median_value', data=boston_input, x_jitter=.1, color = "black", ax=axes[1, 0])
sns.regplot(x='Teacher_ratio', y="Median_value", data=boston_input, x_jitter=.1, color = "black", ax=axes[1, 1])
f.savefig('Linear relationships to response variable Median_value(low range variables)' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)


# In[55]:


# Compute the correlation matrix
corr = boston_input.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})

#save png file of heatmap
fig = heatmap.get_figure()
fig.savefig("Correlation Heatmap.png")


# ## **Data Preperation and Model Building**

# ### Regression Models

# In[71]:


RANDOM_SEED = 1
SET_FIT_INTERCEPT = True
TEST_SIZE = .1


# In[72]:


#identify regression models to be evaluated
names = ['Linear_Regression','Lasso_Regression', 'ElasticNet_Regression', 'Ridge_Regression'] 

#specify paramaters to be used on models
regressors = [LinearRegression(fit_intercept = SET_FIT_INTERCEPT),
              Ridge(alpha = 1, solver = 'cholesky', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED), 
               Lasso(alpha = 0.1, max_iter=10000, tol=0.01, 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     random_state = RANDOM_SEED),                                         
               ElasticNet(alpha = 0.1, l1_ratio = 0.5, 
                          max_iter=10000, tol=0.01, 
                          fit_intercept = SET_FIT_INTERCEPT, 
                          normalize = False, 
                          random_state = RANDOM_SEED)]


# In[73]:


#use 10 fold cross validation
N_FOLDS = 10


# In[74]:


# create numpy array to store results
cv_results = np.zeros((N_FOLDS, len(names)))


# In[75]:


kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)


# In[76]:


# set index for fold to 0
index_for_fold = 0  

for train_index, test_index in kf.split(model_data):
    print('\nFold Index:', index_for_fold,
          '------------------------------------------')
#create variables to train and test data
#response variable (mv) is first column so X_train and X_test will be 1:
    X_train = model_data[train_index, 1:model_data.shape[1]]
    X_test = model_data[test_index, 1:model_data.shape[1]]
    y_train = model_data[train_index, 0]
    y_test = model_data[test_index, 0]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

   #train data on models
    index_for_method = 0 
    for name, reg_model in zip(names, regressors):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', reg_model)
        reg_model.fit(X_train, y_train)  
        print('Fitted regression intercept:', reg_model.intercept_)
        print('Fitted regression coefficients:', reg_model.coef_)
 
        # Evaluate on test sets
        y_test_predict = reg_model.predict(X_test)
        print('Coefficient of determination (R-squared):',
              r2_score(y_test, y_test_predict))
        fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
        print(reg_model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_result,
              '\n--------------------------------------------------------\n')
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
    
    index_for_fold += 1   


# In[77]:


#convert numpy array of results into dataframe
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

#save results of cross-validation to txt file
with open("CV Results.txt", "w") as text_file:
    text_file.write('\nCross-validation results:\n'+
                    str(cv_results_df)+
                    '\nCross validation results column names:\n'+
                    str(names))


# In[78]:


#save average results of cross-validation to txt file
pd.set_option('precision', 5)
print('\n----------------------------------------------')
print('Average results from 10-fold cross-validation\n',
      '\nModel            Root mean-squared error (RMSE)', sep = '') 
print(cv_results_df.mean())   
with open("CV Average Results.txt", "w") as text_file:
    text_file.write('\nAverage results from 10-fold cross-validation\n'+
                    '\nModel            Root mean-squared error (RMSE)'+ 
                     str(cv_results_df.mean()))


# ### Decision Tree Models

# In[80]:


#identify regression models to be evaluated
tree_names = ['DecisionTreeRegressor','RandomForestRegressor', 'GradientBoostingRegressor'] 

#specify paramaters to be used on models
tree_regressors = [DecisionTreeRegressor(random_state = 9999, max_depth = 5), 
              RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100), 
              GradientBoostingRegressor(max_depth=2, n_estimators=100, random_state=42)]


# In[88]:


#use 10 fold cross validation
N_FOLDS = 10

# create numpy array to store results
cv_tree_results = np.zeros((N_FOLDS, len(tree_names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)


# In[89]:


# set index for fold to 0
index_for_fold = 0  

for train_index, test_index in kf.split(model_data):
    print('\nFold Index:', index_for_fold,
          '------------------------------------------')
#create variables to train and test data
#response variable (mv) is first column so X_train and X_test will be 1:
    X_train = model_data[train_index, 1:model_data.shape[1]]
    X_test = model_data[test_index, 1:model_data.shape[1]]
    y_train = model_data[train_index, 0]
    y_test = model_data[test_index, 0]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

   #train data on models
    index_for_method = 0 
    for name, reg_model in zip(tree_names, tree_regressors):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', reg_model)
        reg_model.fit(X_train, y_train)  
 
        # Evaluate on test sets
        y_test_predict = reg_model.predict(X_test)
        print('Coefficient of determination (R-squared):',
              r2_score(y_test, y_test_predict))
        fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
        print(reg_model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_result,
              '\n--------------------------------------------------------\n')
        cv_tree_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
    
    index_for_fold += 1   


# In[90]:


#convert numpy array of results into dataframe
cv_tree_results_df = pd.DataFrame(cv_tree_results)
cv_tree_results_df.columns = tree_names

#save results of cross-validation to txt file
with open("CV Decision Tree Results.txt", "w") as text_file:
    text_file.write('\nCross-validation results:\n'+
                    str(cv_tree_results_df)+
                    '\nCross validation results column names:\n'+
                    str(tree_names))


# In[91]:


#save average results of cross-validation to txt file
pd.set_option('precision', 5)
print('\n----------------------------------------------')
print('Average results from 10-fold cross-validation\n',
      '\nModel            Root mean-squared error (RMSE)', sep = '') 
print(cv_tree_results_df.mean())   
with open("CV Decision Tree Average Results.txt", "w") as text_file:
    text_file.write('\nAverage results from 10-fold cross-validation\n'+
                    '\nModel            Root mean-squared error (RMSE)'+ 
                     str(cv_tree_results_df.mean()))


# ## **Conclusion**

# #### It appears that the Gradient Boosting Regressor (max_depth =2 and n_estimators = 100) is the most suitable method for this project. Each method was evaluated using RMSE (root mean-squared error) as an index for prediction error. Using 10-fold cross-validation, Gradient Boosting achieved an RMSE of .42166, superior to the RandomForestRegressor (.47454) and DecisionTreeRegressor (.60855)

# In[ ]:




