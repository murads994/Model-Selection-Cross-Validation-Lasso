# Model Selection: Cross-Validation and Lasso Regression
In this project, we will be exploring the concept of Model Selection by predicting the probability of a heart attack on the heart.csv dataset. We will be talking about different Model Selection and Out of Sample Experimentation methods such as Cross Validation, Lasso Regression, and AIC/BIC and discussing the pros and cons associated with each one of them. 

## In sample vs Out of Sample R^2
We will start by cleaning the heart.csv dataset and dividing it into random train and test splits(80%/20%). To do that, we will first preprocess the data by getting rid of the columns that are filled with null values, and while dividing the dataset into train and test splits we will make sure that the distribution of the target variable(heart attack) is the same in both of the splits by using the stratified sampling. 
Then, we will Fit a simple linear regression model (full model) to predict the heart attack probability and test the model against the test set.  We will explain our model and obtain the R^2 for the predictions of the test data set. 

In General, for Linear Regression the model is always in the form of: 

$$\mathbb{E}[y|x]=f(x\beta)$$

The 2 main metrics we use the assess the predictive performance of the model are **Likelihood** and **Deviance**:

* $$LHD = p(y_{1}|x_{1}) * p(y_{2}|x_{2})...*p(y_{n}|x_{n})$$

* Deviance is proportional to $-log(LHD)$.
