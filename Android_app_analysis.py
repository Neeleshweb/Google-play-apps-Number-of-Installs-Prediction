#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px


# # READING THE DATASET
# ### The first step in any machine learning problem is reading the data from a given file format, in this case we have a csv file from where we will read the data.

# In[2]:


Application_data=pd.read_csv("googleplaystore.csv")


# # CLEANING THE DATASET
# ### The dataset will have redundant values like NaN, or some columns will not have any value at all, some columns will have unrelated values, some will be having some special charcters which cannot be feeded to our machine learning model. So these inconsistencies will be resolved in this section using basic python skills and pandas tricks.

# In[3]:


Application_data.head()


# In[4]:


Application_data.columns


# In[5]:


Application_data.describe()


# ## Lets move from left to right in the columns of the dataset, we start from "RATING" column, and move till "PRICE" column, since these are the numeric columns and are neccessary features for our model. We will do following process on each of these columns:
# #### 1)- Checking all the unique values in the column.
# #### 2)- If there are some unrelated unique values which are not significant, these will be replaced.
# #### 3)- The null check is performed on each numerical column, if null entries are found, they are replaced with the mean values.
# #### 4)- The values in the columns contain some special characters, that needs to be removed in order to perform aggregations, so those will be removed like "+" and "," in Installs column, "M", "Varies with Device" and "k" from Size column etc.
# #### 5)- The columns that are in object type will be converted to their numerical counterparts for analysis and trends.
# #### 6)- Final filtration will be done to make sure there is no inconsistency in any column that may affect the performance of our model.

# # "RATING" column cleaning by following the 6 steps

# In[6]:


#Checking whether there are null values in the Ratings column
nullcheck_ratings=pd.isnull(Application_data["Rating"])
Application_data[nullcheck_ratings]


# In[7]:


#Replacing the NaN values with the mean rating value
Application_data["Rating"].fillna(value=Application_data["Rating"].mean(),inplace=True)
Application_data["Rating"]


# In[8]:


# Checking the unique values in the Rating column,we find there is an inconsistent value of 19.
Application_data["Rating"].unique()


# In[9]:


# Replacing the inconsistent value with the mean value of ratings
Application_data["Rating"].replace(19.,4.1,inplace=True)


# #### There was no special character found in the ratings column, so step 4 is not required. Moreover, the datatype of ratings column is already float, so no need for the conversion. So now our Rating column is ready for analysis.

# # "REVIEWS" column cleaning by following the 6 steps

# In[10]:


# Checking the unique values of the number of reviews column, we find there are no unrelated values.
len(Application_data["Reviews"].unique())


# In[11]:


# Checking the Null values of the number of reviews column, we find there are no null values.
nullcheck_reviews=pd.isnull(Application_data["Reviews"])
Application_data[nullcheck_reviews]


# In[12]:


# Checking for any special character that might prevent numeric conversion, 3.0M is replaced with its real value to make the data consistent.
Application_data["Reviews"].replace("3.0M","3000000",inplace=True)


# In[13]:


# Finally converting the datatype of Reviews column from Object type(String) to Numeric type(float or int)
Application_data["Reviews"]=pd.to_numeric(Application_data["Reviews"])


# #### All the steps have been completed for the reviews column and it is also ready for the analysis.

# #  "SIZE" column cleaning by following the 6 steps

# In[14]:


# Checking for the unique values of the Size column, it is observed it has values appended with M,k and "Varies with device"
Application_data["Size"].unique()


# In[15]:


# Replacing the "Varies with device" field with NaN entry, so that later on these can be replaced with mean values.
Application_data['Size'].replace('Varies with device', np.nan, inplace = True )
Application_data['Size'].replace('1,000+', np.nan, inplace = True )


# In[16]:


# Checking for null values which we will find, since in the above line we have added few null values.
nullcheck_size=pd.isnull(Application_data["Size"])
Application_data[nullcheck_size]


# #### Now we need to replace the NaN values with the mean size of all the applications, but we cannot calculate mean since our column is of Object type String, so we need to convert it into a numeric type. Moreover, we need to remove "M", "k" from the values of the column, since we cannot convert them to numeric without handling these special symbols.

# In[17]:


Application_data.Size = (Application_data.Size.replace(r'[kM]+$','', regex=True).astype(float) *
                         Application_data.Size.str.extract(r'[\d\.]+([kM]+)', expand=False).fillna(1).replace(['k','M'], [10**3, 10**6]).astype(int))


# In[18]:


# Finally replacing the NaN values with the mean value.
Application_data["Size"].fillna(value="21516530",inplace=True)


# In[19]:


# After removing the special characters, lets convert it to numeric data type for finding the mean value.
Application_data["Size"]=pd.to_numeric(Application_data["Size"])


# #### Here we have completed the cleaning of the Size column by following all the 6 steps which were required, since this column was very uncleaned.

# # "INSTALL" column cleaning by following the 6 steps

# In[20]:


# Checking the unique values of the column Installs, we observe that there is a type called "free", which is inconsistent and non numeric, so it should be replaced.
Application_data["Installs"].unique()


# #### We need to remove the "free" with the average number of installs for the applications, but for calculating the average, we need to remove the "+" and "," from the values. After removing them, we will have to convert these into numeric type and then we can calculate the mean and finally substitute the mean value in place of "Free".

# In[21]:


# Removing the "+" symbol to make the column numeric.
Application_data["Installs"]=Application_data["Installs"].map(lambda x: x.rstrip('+'))


# In[22]:


# Removing the "," from the digits to make it easier.
Application_data["Installs"]=Application_data["Installs"].str.replace(",","")


# In[23]:


# There was no null entries found in this column
nullcheck_installs=pd.isnull(Application_data["Installs"])
Application_data[nullcheck_installs]


# In[24]:


# Replacing the inconsistent label value with the mean value of the column.
Application_data["Installs"].replace("Free","15462910",inplace=True)


# In[25]:


# Converting the Datatype to the numeric type for analysis
Application_data["Installs"]=pd.to_numeric(Application_data["Installs"])


# #### In this way, we have made our Installs column ready for the analysis by following all the 6 steps again.

# # "TYPE" column cleaning by following the 6 steps

# In[26]:


# Checking for the unique values, we found nan and 0 which should be replaced with Free.
Application_data["Type"].unique()


# In[27]:


# Replacing 0 with Free
Application_data["Type"].replace("0","Free",inplace=True)


# In[28]:


# Filling the missing values with Free, since most of the applications are free on Google play.
Application_data["Type"].fillna(value="Free",inplace=True)


# In[29]:


# Addding the dummy columns for this, so that it can contribute to our model.
dummy_type=pd.get_dummies(Application_data["Type"])


# In[30]:


#Concatenating the dummy columns with the main dataframe.
Application_data=pd.concat([Application_data,dummy_type],axis=1)


# In[31]:


# Finally dropping the type column.
Application_data.drop(["Type"],axis=1,inplace=True)


# In[32]:


Application_data.head()


# #### In this way we have removed the Type categorical column, used dummy columns to make our feature space more accurate.

# # "PRICE" column cleaning by following the 6 steps

# In[33]:


# By checking the unique values we observe that "Everyone" is an inconsistent value that should be removed.
Application_data["Price"].unique()


# #### Here to get the mean of the values, the datatype of the column should be numeric and for that to happen we need to remove the dollar symbol from the values and drop the everyone row, since it contains redundant data that will compromise the performance of our model.

# In[34]:


# Removing the dollar symbol
Application_data["Price"]=Application_data["Price"].map(lambda x: x.lstrip('$'))


# In[35]:


# Removing the non essential row value.
Application_data.drop(Application_data[Application_data["Price"] == "Everyone"].index, inplace=True)


# In[36]:


# By checking there were no null values found
nullcheck_Prices=pd.isnull(Application_data["Price"])
Application_data[nullcheck_Prices]


# In[37]:


# Finally converting to numeric type for analysis
Application_data["Price"]=pd.to_numeric(Application_data["Price"])


# #### We have cleaned the Price column by following all the 6 steps as per the requirement, now this column is ready for the analysis.

# # "CATEGORY" column cleaning by following the 6 steps

# In[38]:


# Checking the unique values, we found 
Application_data["Category"].unique()


# In[39]:


Application_data["Category"].replace("1.9","MISCELLANEOUS",inplace=True)


# In[40]:


# Checking for null values, there were no null values found for this column
nullcheck=pd.isnull(Application_data["Category"])
Application_data[nullcheck]


# #### For this column, we will perform the label encoding and not the dummies, since by making dummies there will be too many extra columns added to our feature matrix that is not required, so label encoding is done by providing numerical values to each and every category of application.

# In[41]:


# Importing the required library
from sklearn.preprocessing import LabelEncoder


# In[42]:


# Instantiating the encoder
labelencoder2 = LabelEncoder()


# In[44]:


#Encoding the Ctegory column using scikit learn
Application_data['Categories_encoded'] = labelencoder2.fit_transform(Application_data['Category'])


# In[45]:


# finally dropping the type column, since it is already splitted.
Application_data.drop(["Category"],axis=1,inplace=True)


# In[46]:


Application_data.head()


# # "CONTENT RATING" Column cleaning by 6 Steps

# #### For this categorical column also, we are doing the label encoding similarly we did for the Category column.

# In[47]:


# Checking for unique values
Application_data["Content Rating"].unique()


# In[48]:


# Null check for Content Rating
nullcheck_contentrating=pd.isnull(Application_data["Content Rating"])
Application_data[nullcheck_contentrating]


# In[50]:


# importing the required package
from sklearn.preprocessing import LabelEncoder


# In[51]:


#instantiating the encoder
labelencoder = LabelEncoder()


# In[52]:


# encoding the column
Application_data['Content_Rating_encoded'] = labelencoder.fit_transform(Application_data['Content Rating'])


# In[53]:


# finally removing the content ratig column after encoding
Application_data.drop(["Content Rating"],axis=1,inplace=True)


# In[54]:


Application_data.head()


# In[55]:


# Checking the datatypes of the columns to ensure that we have successfully gathered all the numerical columns.
Application_data.dtypes


# In[56]:


# Finding the mean of all the numerical columns
Application_data.mean()


# # EXPLORATORY DATA ANALYSIS
# ## Below there is a complete analysis of various relationships between the features of our data. This is required so that we can understand what all features will play a significant role when predicting the number of installs for any application.

# In[57]:


sns.pairplot(Application_data)


# #### Here a pairplot is shown between all the numerical columns of the data. This gives a high level of intuition between the relationships between the various features. Now firstly, histograms will be drawn for all the numerical columns just to know thier counts and distribution. Plotly is used here for graphical representations.

# In[58]:


colorassigned=Application_data["Rating"]
fig = px.histogram(Application_data, x="Rating", marginal="rug",
                   hover_data=Application_data.columns,nbins=30,color=colorassigned)
fig.show()


# #### The above Graph is an Histogram, that shows the distribution of ratings of various android applications. The histogram is divided into colors based on the values of the rating. The color scale is given on the right side. The count of rating 4.1 is maximum(1474) as can be found by hovering on the graph. Moreover, the count of rating uniformly increases from 3.4(128) to going to maximum of 4.1(1474), and then again going up and down. This means most of the applications on google play have their ratings between 4 to 4.5.

# In[59]:


fig = px.histogram(Application_data, x="Reviews", marginal="rug",
                   hover_data=Application_data.columns,nbins=30)
fig.show()


# #### This ia an histogram to show the distribution of number of reviews for each application. It is quite clearly visible that, 90% of the applications on Google play store have reviews less than 5 million. 138 applications have reviews between 5 Million to 10 Million. Only 47 android applications have reviews between 10 Million to 15 Million. So majority of the applications have less then 5 Million reviews.

# In[60]:


colorassigned=Application_data["Size"]
fig = px.histogram(Application_data, x="Size", marginal="rug",
                   hover_data=Application_data.columns,nbins=30,color=colorassigned)
fig.show()


# #### The above Graph is an Histogram, that shows the distribution of size of various android applications. It can be observed that most of the applications have lesser size, since as the size increases on the x-axis, the bars are getting shorter and shorter, which means the count of such types of apps is decreasing. So we have more applications on Google playstore that are smaller in size than the larger ones. Most of the applications have the size of around 21.5 MB.

# In[61]:


colorassigned=Application_data["Installs"]
fig = px.histogram(Application_data, x="Installs", marginal="rug",
                   hover_data=Application_data.columns,nbins=30,color=colorassigned)
fig.show()


# #### The above graph shows the number of installs of the android applications. It can be observed that majority of the applications have less than 10 Million installs. Moreover, there are only 58 applications that have more than 1 billion installs on Google play.

# In[62]:


colorassigned=Application_data["Price"]
fig = px.histogram(Application_data, x="Price", marginal="rug",
                   hover_data=Application_data.columns,nbins=30,color=colorassigned)
fig.show()


# #### This histogram shows the price distribution of various android applications on Google play. Majority of the applications are free of cost. There are 12 android applications that are the most expensive costing 400 bucks. 

# ## With this we have completed the individual analysis of all the numerical columns of our dataset. Now we will find the relation between each column to analyse deeply. The step that is followed below is:
# ### 1)- Calculate the correlation value and draw a heatmap to know the correlation between different columns.
# ### 2)- Once we find the correlation, then we know which columns are affecting one another, then we start plotting columns in pair of two based on thier correlation values. If the correlation is negative or very less, there is no point in plotting those columns.
# ### 3)- After plotting we will fit a linear regression line to our data points. The more the correlation value, better line of fit we get.
# 

# In[63]:


# Calculating the Correlation and plotting the heatmap to know the relations.
cors=Application_data.corr()
fig = px.imshow(cors,labels=dict(color="Pearson Correlation"), x=['Rating', 'Reviews', 'Size', 'Installs', 'Price','Paid','Free','Content_Rating_encoded','Categories_encoded'],
                y=['Rating', 'Reviews', 'Size','Installs','Price','Paid','Free','Content_Rating_encoded','Categories_encoded'])
fig.show()


# ## Following inferences can be drawn from this heatmap:
# ### CORRELATION VALUE                                FEATURES INVOLVED                                              VERDICT
# 
# ###             -0.020                                                      Price vs Rating                                                    No Correlation
# 
# ###             -0.009                                                      Price vs Reviews                                                 No Correlation
# 
# ###             -0.022                                                      Price vs Size                                                        No Correlation
# 
# ###             -0.011                                                      Price vs Installs                                                   No Correlation
# 
# ###              0.051                                                      Installs vs Rating                                                 No Correlation
# 
# ###              0.643                                                      Installs vs Reviews                                              Great Correlation
# 
# ###              0.082                                                      Installs vs Size                                                     No Correlation
# 
# ###             -0.011                                                      Installs vs Price                                                   No Correlation
# 
# ###              0.074                                                      Size vs Rating                                                      No Correlation
# 
# ###              0.128                                                      Size vs Reviews                                                 Very less Correlation
# 
# ###              0.082                                                      Size vs Installs                                                    No Correlation
# 
# ###             -0.022                                                      Size vs Price                                                        No Correlation
# 
# ###              0.067                                                      Reviews vs Rating                                              No Correlation
# 
# 
# 
# ## We will be plotting only those relations whose correlation value is greater than 0.1, rest all do not have any correlation, so plotting will not be fruitful. 

# In[64]:


# Plotting scatter plot with a line of fit between Installs and Reviews, these two have the highest Correlation between them.
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Application_data["Installs"],Application_data["Reviews"])
colorassigned=Application_data["Reviews"]
fig = px.scatter(Application_data, x="Installs", y="Reviews",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)


# ## It is observed that we have a good fit to the data points, since the correlation between these 2 columns is significant. As it is visible, as the number of installs increases, the number of reviews are also increasing which makes sense, since if the user has installed the application, then only they can give feedback to it. Without using an application, reviews cannot be given. If we get a new data point, we can predict its number of installs based on the number of reviews. By hovering on the red line, the equation of the straight line can be seen. Hovering on each data point gives its installs and reviews at that point.

# In[65]:


# Plotting scatter plot with a line of fit between Rating and Reviews, these two have very less correlation between them. 
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Application_data["Rating"],Application_data["Reviews"])
colorassigned=Application_data["Reviews"]
fig = px.scatter(Application_data, x="Rating", y="Reviews",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)


# ## As can be observed from this graph, we can see that the applications that have ratings between 4 to 4.7 have maximum number of reviews. However, we cannot say that as the ratings increases the reviews increases, this happens just for a particular range of 4 to 4.7 where Reviews increase as the ratings increase but before 4 and after 4.7 there is different trend. It is observed that after rating 4.7, the count of number of reviews have reduced, that is the applications to which review is given is reduced. The apps having 5 star rating have only 4 reviews. However, the apps having rating less then 4 have been rated by many users.

# In[66]:


# Plotting scatter plot with a line of fit between Size and Reviews, these two have very less correlation between them. 
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Application_data["Size"],Application_data["Reviews"])
colorassigned=Application_data["Reviews"]
fig = px.scatter(Application_data, x="Size", y="Reviews",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)


# ## There is no general trend observed in this graph, since there is very less correlation observed in these two columns. There are applications of size 21 MB getting 80 Million reviews, and there are applications with larger size like 98 MB, getting 45 million reviews. So there is no trend observed here.

# In[67]:


from scipy.stats import pearsonr 
corryu,_ =pearsonr(Application_data["Installs"],Application_data["Categories_encoded"])
colorassigned=Application_data["Categories_encoded"]
fig = px.scatter(Application_data, x="Installs", y="Categories_encoded",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)


# # MODEL BUILDING AND EVALUATION USING SKLEARN

# ### The final step is creating the model that will predict the number of installs for an android application on Google play. We will be using 3 regressors for this purpose, Linear, DecisonTree and RandomForest. Finally the performance for all the 3 will be compared in the graphical format.

# ## LINEAR REGRESSOR 

# In[68]:


# Splitting the target variable and the feature matrix
X=Application_data[["Reviews","Size","Rating","Price","Paid","Free","Categories_encoded","Content_Rating_encoded"]]
y=Application_data["Installs"]


# In[69]:


# importing train test set
from sklearn.model_selection import train_test_split


# In[70]:


# splitting the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[71]:


# importing linear regressor
from sklearn.linear_model import LinearRegression


# In[72]:


# Instantiating linear regressor
lm=LinearRegression()


# In[73]:


# Fitting the model
lm.fit(X_train,y_train)


# In[74]:


# making predictions on the test set
predictions=lm.predict(X_test)


# In[75]:


# displaying predictions
predictions


# In[76]:


# Accuracy score for Linear regressor
linearregressionscore=lm.score(X_test,y_test)
linearregressionscore


# In[77]:


# The coefficient for Linear regressor per feature.
lm.coef_


# #### Evaluating the metrics for linear regresssor, the mean absolute error, mean squared error and finally root mean square error.

# In[78]:


# Importing the metrics
from sklearn import metrics


# In[79]:


# Mean absolute error on test data
metrics.mean_absolute_error(y_test,predictions)


# In[80]:


# Mean squared error on test data
metrics.mean_squared_error(y_test,predictions)


# In[81]:


# Root mean squared error on test data
rmelinear=np.sqrt(metrics.mean_absolute_error(y_test,predictions))
rmelinear


# # DECISION TREE REGRESSOR

# In[82]:


# Defining the feature matrix and the target variable
X=Application_data[["Reviews","Size","Rating","Price","Paid","Free","Categories_encoded","Content_Rating_encoded"]]
y=Application_data["Installs"]


# In[83]:


# Importing the train test split
from sklearn.model_selection import train_test_split


# In[84]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[85]:


# Importing the regressor
from sklearn.tree import DecisionTreeRegressor


# In[86]:


# Instantiating the regressor
decisiontreereg=DecisionTreeRegressor()


# In[87]:


# Fitting the model
decisiontreereg.fit(X_train,y_train)


# In[88]:


# Gettting the predicted values
y_prediction=decisiontreereg.predict(X_test)


# In[89]:


# The accuracy score for decision tree regressor
decisiontreescore=decisiontreereg.score(X_test,y_test)
decisiontreescore


# In[90]:


from sklearn import metrics


# In[91]:


# Mean absolute error
metrics.mean_absolute_error(y_test,y_prediction)


# In[92]:


# Root mean square error
rmetree=np.sqrt(metrics.mean_absolute_error(y_test,y_prediction))
rmetree


# # RANDOM FOREST REGRESSOR

# In[122]:


# Separating the feature matrix and target variable
X=Application_data[["Reviews","Size","Rating","Price","Paid","Free","Categories_encoded","Content_Rating_encoded"]]
y=Application_data["Installs"]


# In[123]:


# Importing the train test split
from sklearn.model_selection import train_test_split


# In[124]:


# Splitting the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[125]:


# Importing the random forest regressor
from sklearn.ensemble import RandomForestRegressor


# In[126]:


# Instantiating with giving the value of number of sub trees to be created.
Randomforestreg=RandomForestRegressor(n_estimators = 100,n_jobs = -1,oob_score = True, bootstrap = True,random_state=42)


# In[127]:


# fitting the model
Randomforestreg.fit(X_train,y_train)


# In[128]:


# Predicting the number of installs
y_prediction_randomforest=Randomforestreg.predict(X_test)


# ## It is very fruitful if we know how much significant a feature is for predicting our target variable. Below is the plot showing the importance of various features in predicting the number of installs.

# In[129]:


# Using barplot from seaborn to show importance of features in sorted manner.
feature_imp=pd.DataFrame(sorted(zip(Randomforestreg.feature_importances_,Application_data[["Reviews","Size","Rating","Price","Paid","Free","Categories_encoded","Content_Rating_encoded"]])),columns=["Significance","Features"])
fig=plt.figure(figsize=(6,6))
sns.barplot(x="Significance",y="Features",data=feature_imp.sort_values(by="Significance",ascending=False),dodge=False)
plt.title("Important features for predicting the number of installs")
plt.tight_layout()
plt.show()


# In[130]:



from sklearn.metrics import r2_score,mean_squared_error


# In[131]:


# The performance of random forest.
print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(Randomforestreg.score(X_train, y_train), 
                                                                                             Randomforestreg.oob_score_,
                                                                                             Randomforestreg.score(X_test, y_test)))


# In[132]:


# Accuracy score for random forest
randomforestscore=Randomforestreg.score(X_test,y_test)
randomforestscore


# In[133]:


# Importing the performance metrics
from sklearn import metrics


# In[134]:


# Mean absolute error
metrics.mean_absolute_error(y_test,y_prediction_randomforest)


# In[135]:


# Root mean squared error
rmerandom=np.sqrt(metrics.mean_absolute_error(y_test,y_prediction_randomforest))
rmerandom


# ## The final step to compare our models accuracy for this task. We will be comparing the accuracy score and the root mean square error for all the 3 models. For that to achieve, we need to create a dataframe that consists of the accuracy score and root mean square error for each model and then we can plot that dataframe.

# In[136]:


# Creating the dataframe that has accuracy score and root mean squared error for all the 3 models.
dict={"Linear Regressor":[linearregressionscore,rmelinear],"DecisionTree Regressor":[decisiontreescore,rmetree],"RandomForest Regressor":[randomforestscore,rmerandom]}
df_comparison_models=pd.DataFrame(dict,["Score","Root Mean Square Error"])


# In[137]:


df_comparison_models.head()


# In[138]:


# Plotting the accuracy of all the 3 models
get_ipython().run_line_magic('matplotlib', 'inline')
model_accuracy = pd.Series(data=[linearregressionscore,decisiontreescore,randomforestscore], 
        index=['Linear_Regressor','DecisionTree Regressor','RandomForest Regressor'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accuracy')


# In[139]:


# Plotting the Root Mean Squared Error comaparison
get_ipython().run_line_magic('matplotlib', 'inline')
model_accuracy = pd.Series(data=[rmelinear,rmetree,rmerandom], 
        index=['Linear_Regressor','DecisionTree Regressor','RandomForest Regressor'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values().plot.barh()
plt.title('Model Root Mean Squared Error')


# # FINAL THOUGHTS

# ### This was all about this dataset, where we performed all the process from the scratch. We loaded the data, cleaned the features, did a thorough exploratory data analysis to understand which will be the key features that will be vital for predicting our target variable, finally created the models and made some predictions. At the last compared the accuracy of all the models on which the analysis was performed. 
