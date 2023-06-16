library(caret)

# In sample vs Out of Sample R^2
### For choosing a sample subset from this dataset I took into consideration
### Several factors that I will state below.
### Firstly, I got rid off the 3 columns that were consisted off all NA or missing values
### I did it because those columns were not going to be useful for training our model.
### Next, I made sure that the distribution of the target variable is going to be 
### the same for both my train and test subsets.I did it because I wanted to make sure
### the distribution of that target variable is a correct representative of the population
### for both train and test subsets. 
### I achieved that by using stratified sampling over the target variable to make sure
### distribution is matching for both train and test subsets.

heart<-read.csv('heart.csv')
summary(heart)

#### Dropping columns of only NA values
heart_smpl<-heart[,!(names(heart) %in% c("family_record","past_record","wrist_dim"))]
summary(heart_smpl)

### Dropping all the rows that contains NA Values
heart_smpl <- na.omit(heart_smpl) 

### Dividing the cleaned data set into stratified test and train subsets
set.seed(1)

heart_dt <- createDataPartition(heart_smpl$heart_attack, p = .8,
                                  list = FALSE,
                                  times = 1)
heart_train <- heart_smpl[ heart_dt,]
heart_test <- heart_smpl[-heart_dt,]

### Running a Simple Regression Model on the train dataset
heart_regress1<-lm(heart_attack ~ ., data=heart_train)
summary(heart_regress1)

### Testing the model against the test set
heart_pred1<-predict(heart_regress1,heart_test)

heart_test_pred1<-cbind(heart_test,heart_pred1)

### Calculating OOS R^2 value
D<-sum((heart_test_pred1$heart_attack - heart_test_pred1$heart_pred1)**2)
D0<-sum((heart_test_pred1$heart_attack - mean(heart_train$heart_attack))**2)
OOS_R2<-1-(D/D0)
OOS_R2

### As a result of our regression model, we get 11 significant variables at 5%
### significance level. Overall In-Sample R^2 of the model is 94.6%. The
### reason why our model has very high r^2 value might be due to the fact that
### we have too many covariates in the model, and that the model is overfitting.
### After testing the model against the test set, we get 83% OOS r^2 value, which is
### very high. This means our model is doing a good job explaining the variation 
### in chance of getting heart attack, even for the datasets that it hasn't seen before.

# Cross Validation

### The process of using Out of Sample Experiments to do model selection is called cross validation.
### Cross Validation is technique used in Machine Learning to benchmark the performance of 
### different models we build. In CV we divide our dataset into multiple train and test subsets. 
### We train the model using our train subsets and evaluate it on the test subset. 
### In k-fold cross validation, the process is repeated k times, with each subset serving as the test set once.
### At the end we take the final performance measure as the average performance across all iterations.

### One of the main problems associated with CV is that it is computationally very expensive. 
### Especially if we are dealing with big datasets. 
### Another bottleneck is that CV is sensitive to the way data is partitioned. 
### if the data is not randomly partitioned, the results we find from CV
### may not be a good representative of the model's performance on unforeseen data. 
### Last but not the least,if the data contains a large number of outliers or 
### if the data is imbalanced then traditional k-fold CV may not be an appropriate approach.


set.seed(2)

# defining training control
# as cross-validation and 
# value of K equal to 10
train_control <- trainControl(method = "cv",
                              number = 8)

# training the model by assigning heart_attack column
# as target variable and other columns
# as independent variable
heart_kcv <- train(heart_attack ~., data = heart_train, 
               method = "lm",
               trControl = train_control)
### Show the results of the model
print(heart_kcv)

### Show how the final model looks like
heart_kcv$finalModel

### Show Predictions for each fold
heart_kcv$resample

kcv_R2 <- mean(heart_kcv$resample$Rsquared)
kcv_R2

### As a result of k-fold cross validation we get an average R2 value of 87.6%
### This is somehow aligned with our findings from the OOS experiment in Q1.2
### Where we got an OOS R2 value of 83%. Again, this means that our model is 
### doing a good job in explaining the variation in heart_attack variable,
### even for the datasets it hasn't seen before. 
### If we look at the R2 value of each fold from our k-fold CV, we can actually
### see that in most folds our R2 values came out to be more than 90%. 
### Only for the first fold we get an R2 value of 48% which might be due to 
### Sampling Variation. Overall, this model is doing a good job predicting 
### heart_attack variable even for the data it hasn't seen before. 


# Lasso Regression
### Lasso regression is type of regression that adds a regularization constraint to the model.
### The regularization term is the sum of the absolute values of the coefficients, 
## and it is multiplied by a scalar value lambda, which controls the strength of the regularization.
### The main goal of the lasso regression is to decrease the coefficients of less importance to 0.

### Some of the pros of using Lasso Regression are:
### By default it performs feature selection, by shrinking coefficients of less importance to 0.
### It is very useful for cases when we have more features than the number of observations.
### The regularization term is robust to outliers in the dataset and to multicollinearity between coefficients.
### However, multicolliniearity can still cause problems for choosing features which I will adress in details very soon. 

### Some of the Cons of using Lasso Regression are:
### If there is high multicollinearity between some features, Lasso regression may not be able to select true important features.
### Lasso Regression does not perform well with categorical variables.
### Depending on the Lambda value, sometimes it can overpenalize some features resulting in underfitted model.

library(glmnet)

x <- data.matrix(heart_train[, 1:16])
y <- heart_train$heart_attack
heart_regress_lasso_kcv <- cv.glmnet(x, y, alpha = 1, nfolds = 8, family = "gaussian", standardize = TRUE)
summary(heart_regress_lasso_kcv)

### Get the lambda min value from Lasso Regression
lmbd_min <- heart_regress_lasso_kcv$lambda.min
lmbd_min

### Get the lambda 1se value from Lasso Regression
lmbd_1se <- heart_regress_lasso_kcv$lambda.1se
lmbd_1se

### Plot the model
plot(heart_regress_lasso_kcv)
### In general, lambda min gives us the lambda value with minimum average OOS Deviance.
### Lambda 1se gives us the biggest lambda value with average OOS deviance no more
### than 1 SD away from the minimum.

### Looking at the plot we can see that both model with lambda min and lambda 1se 
### perform similarly in terms of MSE values.
### We will chose the model with lambda 1se as the model to predict the values of test dataset, as
### looking at the graph we can clearly see that the MSE values does not differ much between those two,
### and the model with Lambda 1se is a simpler model with only 5 variables compared to 6 in Lambda min model.

# Rerun the model with lambda 1se
heart_regress_lasso_kcv2 <- glmnet(x, y,alpha=1, lambda = lmbd_1se)

# Predict the test dataset using the model
heart_test_x <- data.matrix(heart_test[,1:16])
heart_test_y <- heart_test$heart_attack
heart_test_pred2 <- predict(heart_regress_lasso_kcv2, s=lmbd_1se, newx = heart_test_x)

#obtain OOS R^2
D_lasso <- sum((heart_test_y - heart_test_pred2)^2)
D0_lasso <- sum((heart_test_y - mean(y))^2)

R2_lasso <- 1-(D_lasso/D0_lasso)
R2_lasso
### As a result, we get OOS R2 value of 79% from 8-fold Cross Validated Lasso regression,
### tested on test dataset

######## Question 5.2 ##########
q1_res <- heart_regress1$coefficients
q3_res <- heart_kcv$finalModel$coefficients
q5_res <- coef(heart_regress_lasso_kcv, select = "1se")

result_r2 <- cbind(OOS_R2, kcv_R2, R2_lasso)
colnames(result_r2) <- c("q1-OOS_R2", "q3-kcv_R2", "q5-lasso_R2")
result_r2

model_covariates<-cbind(q1_res, q3_res, q5_res)
model_covariates

### As we can see from the results printed above, we achieved the highest OOS R2
### value, as a result of 8 fold cross validation from question 3(88%).
### The OOS experiment we did in Question 1 also provided similar R2 value of 83%
### Last but not the least, 8 fold Cross Validated Lasso Regression with lambda 1se got rid of the 11 variables,
### and returned an R2 value of 79% with only 5 variables which is again great OOS predictive power. 
### If I were to choose only 1 model among these 3, I would go with the last one from question 5
### as it is the simplest model with 5 independent variables compared to 16 in other models.
### On top of that it also does not sacrifice much of its predictive power, even after dropping 11 variables.



#### AIC #####
### AIC(Akaike Information Criterion) is a way to measure relative quality of regression models. 
### It is very commonly used to compare different regression models and to select the best model among them. 
### AIC = Deviance + 2df
### Where df is degrees of freedom used in our model fit. 
### AIC is trying to estimate the OOS deviance i.e what your deviance would be in another sample of size n.
### Usually the lower the AIC value, the better the model is.
### It is commonly believed that AIC does not perform well when the sample size is small. 
### In fact, AIC is only good for big n/df
### In big data, number of parameters can be huge. Often df=n.
### In those cases, AIC tends to overfit. 
### AICc (corrected AIC) is an adjusted version of AIC that is used when the sample size is small or df is too big.
### It corrects for the bias that can occur when the sample size is small or df is too big. AICc is calculated as:
### AICc = Deviance + 2df(n/(n-df-1))
### AICc is a more robust measure of model fit than AIC in many cases,
### as it adjusts for the bias that can occur when the sample size is small or df is too big.
