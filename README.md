# Model Selection: Cross-Validation and Lasso Regression
In this project, we will be exploring the concept of Model Selection by predicting the probability of a heart attack on the heart.csv dataset. We will be talking about different Model Selection and Out of Sample Experimentation methods such as Cross Validation, Lasso Regression, and AIC/BIC and discussing the pros and cons associated with each one of them. 

## In sample vs Out of Sample R^2
We will start by cleaning the heart.csv dataset and dividing it into random train and test splits(80%/20%). To do that, we will first preprocess the data by getting rid of the columns that are filled with null values, and while dividing the dataset into train and test splits we will make sure that the distribution of the target variable(heart attack) is the same in both of the splits by using the stratified sampling. 
Then, we will Fit a simple linear regression model (full model) to predict the heart attack probability and test the model against the test set.  We will explain our model and obtain the R^2 for the predictions of the test data set. 

In General, for Linear Regression the model is always in the form of: 

$$\mathbb{E}[y|x]=f(x\beta)$$

The 2 main metrics we use the assess the predictive performance of the model are **Likelihood** and **Deviance**:

* $LHD = p(y_{1}|x_{1}) * p(y_{2}|x_{2})...*p(y_{n}|x_{n})$
* Deviance is proportional to $-log(LHD)$.
* $\beta$ is commonly fit to maximize LHD or minimize Deviance.
* Fit is summarized by $R^{2} = 1 - \frac{dev(\beta)}{dev(\beta = 0)}$

In real-life settings, we never care about the in-sample $R^2$ value of the model for predictive analytics purposes. The only thing we care about is the Out of Sample $R^{2}$ value. 

The main difference between out-of-sample and in-sample $R^2$ values is what data is used to fit $\beta$'s and what data deviances are calculated on. 
* For in sample $R^2$ we use the same data to fit $\beta$'s and also to calculate Deviance.
* For out-of-sample $R^2$ we use the training data to fit $\beta$'s, and the deviances are now calculated on new observations (test sample).

## Cross Validation
Next, we will use only the training set from above and estimate an 8-fold cross-validation to estimate the R^2 of the full model. i.e., we will use cross-validation to train (on 7/8 of the training set) and evaluate (on 1/8 of the training set) our model. We will then calculate the mean R^2 from the 8-fold cross-validation and compare it with the R^2 from the first part, and explain our observations. 

In general, The process of using Out of Sample Experiments to do model selection is called cross-validation. Cross Validation is a technique used in Machine Learning to benchmark the performance of different models we build. 

In CV we divide our dataset into multiple train and test subsets.  We train the model using our train subsets and evaluate it on the test subset. In k-fold cross-validation, the process is repeated k times, with each subset serving as the test set once. At the end, we take the final performance measure as the average performance across all iterations.

One of the main problems associated with CV is that it is computationally very expensive. Especially if we are dealing with big datasets. Another bottleneck is that CV is sensitive to the way data is partitioned.  If the data is not randomly partitioned, the results we find from the CV may not be a good representative of the model's performance on unforeseen data. Last but not least, if the data contains a large number of outliers or if the data is imbalanced then traditional k-fold CV may not be an appropriate approach.


