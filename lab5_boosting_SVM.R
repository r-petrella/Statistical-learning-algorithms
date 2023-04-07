## note lab 5 - 15/03/23    ####
## boosting ####
setwd("C:/Users/ricky/Desktop/primo_anno/Statistical Learning/Lab5")
load("SAheart.RData")
x <- SAheart[,-ncol(SAheart)]
y <- SAheart[, ncol(SAheart)]

heart <- data.frame(chd = y, x)

misc <- function(y.hat, y){
  mean(y.hat != y)
}

n <- nrow(x)
p <- ncol(x)

set.seed(1234)
train <- sample(1:n, ceiling(n/2))
heart.train <- heart[train,]
heart.test <- heart[-train,]

## Boosting via gbm package ####
# Generalized Boosted Regression Modeling (GBM)

# GBM) is a popular machine learning algorithm used for regression and
# classification tasks. It is an ensemble learning method that combines 
# multiple weak models (simple models that perform only slightly better 
# than random guessing) to create a strong model.

# The GBM algorithm iteratively fits a sequence of regression trees to the 
# data, with each subsequent tree trying to correct the errors made by the 
# previous tree. At each step, the algorithm calculates the residuals 
# (the differences between the predicted and actual values) of the previous 
# tree and fits a new tree to those residuals. The final model is the sum of 
# all the individual trees.

# GBM has several advantages over other machine learning algorithms, including 
# its ability to handle both continuous and categorical predictors, its 
# robustness to outliers, and its ability to handle large datasets. It is 
# also highly flexible, as it allows users to specify a wide range of 
# hyperparameters, such as the number of trees, the depth of each tree, and
# the learning rate.

#install.packages("gbm")
library(gbm)

# if the response variable is of the form (1,0), then it is distributed
# as a bernoulli
boost.out <- gbm(chd ~ ., data = heart[train,], distribution = "bernoulli",
                 n.trees = 100, bag.fraction = 1,
                 interaction.depth = 1)

# "interaction.depth": integer specifying the maximum depth of each tree
# (i.e., the highest level of variable interactions allowed). A value of 1 
# implies an additive model, a value of 2 implies a model with up to 2-way 
# interactions, etc. Default is 1


# "bag.fraction": the fraction of the training set observations randomly
# selected to propose the next tree in the expansion. This introduces 
# randomnesses into the model fit. If bag.fraction is set to 0.5, then 
# each subsample will contain approximately 50% of the original training 
# data. The remaining 50% of the data (known as the out-of-bag, or OOB, 
# data) is used to evaluate the performance of the model during training 
# and to estimate the optimal number of trees to use in the final model.

# Setting bag.fraction to a value less than 1 can help reduce overfitting 
# in the GBM model by introducing more randomness into the fitting process.
boost.out

ntr <- length(train)

K= 5
set.seed(1234)
folds <- sample(1:K, ntr, replace = T)
table(folds)

# choose the best number of trees among:
B <- c(25,50,100,150)

# cross-validation on the training set
err.cv <- matrix(NA, K, length(B))

for(i in 1:K){
  test <- heart.train[folds == i,]
  train <- heart.train[folds != i,]
  
  for (j in 1:length(B)){
    out.gbm <- gbm(chd ~ ., train, distribution = "bernoulli",
                   n.trees = B[j], interaction.depth = 1,
                   bag.fraction = 1)
  
    p.hat <- predict(out.gbm, newdata = test, n.trees = B[j],
                     type="response") 
    #type = "response" gives the predicted probabilities.
    #                 if p.hat>0.5 then y.hat=1
    y.hat <- ifelse(p.hat > 0.5, 1, 0)
    
    err.cv[i,j] <- misc(y.hat, test$chd)
  }
}
colMeans(err.cv)

# choose the number of trees whose cross-validation error is the smallest
b_best <- B[which.min(colMeans(err.cv))]


# now we recreate the model with the best b
## Fit the boosting on the whole training set with B=25 and do the prediction on
# the validation set.
boost.heart <- gbm(chd ~ ., data = heart.train,
                 distribution = "bernoulli",
                 n.trees = b_best,
                 interaction.depth = 1, bag.fraction = 1)

phat <- predict(boost.heart, newdata = heart.test,
              n.trees = b_best, type = "response")

yhat <- ifelse(phat > 0.5, 1, 0)

misc(yhat, heart.test$chd)

## Support Vector Machines ####
library(e1071)
# SVM can be implemented only for regression unless we put as.factor(y)
set.seed(1234)

heart <- data.frame(chd = as.factor(y), x)

train <- sample(n, ntr, replace = F )

heart.train <- heart[train,]
heart.test <- heart[-train,]
# to answer my question:
# both in regression and classification y and x must belong to the same dataset
# i.e. they must not be a separate vector and matrix
# in facts, "chd~., heart" means "chd ~ all the remaining variables of heart"

# to sum up: "y.train ~ x.train, data.train,(subset is not required if we use .train)
# alternatively: "y ~ x, data, subset=train"
svm.out <- svm(chd~., heart, subset=train, kernel="linear", cost=10)

svm.out
head(svm.out$index)
summary(svm.out)

# example: reduce the cost
svm.out <- svm(chd~., heart, subset=train, kernel="linear", cost=0.1) 
# smaller cost-> more points will fall within the two margins
summary(svm.out)

# Cross-validation
set.seed(1234)
best.mod <- tune(METHOD=svm, chd~., data=heart[train,], kernel="linear",
                  ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)))   
#This generic function tunes hyperparameters of statistical methods 
# using a grid search over supplied parameter ranges.

summary(best.mod)
best.cost <- best.mod$best.model
summary(best.cost)

# prediction in the test set
yhat <- predict(best.cost, heart[-train,])
table(yhat, heart.test$chd)
misc(yhat, heart.test$chd)

#### Support Vector Machines with Radial Kernel ####
set.seed(1234)
best.mod <- tune(METHOD=svm, chd~., data=heart[train,], kernel="radial",
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                             gamma=c(1:5)))
summary(best.mod)
# the best model is refit
yhat.rad <- predict(best.mod$best.model, newdata=heart[-train,])
table(yhat.rad, heart.test$chd)
misc(yhat.rad, heart.test$chd)


#### Support Vector Machines with Polynomial Kernel ####
set.seed(1234)
best.mod <- tune(METHOD=svm, chd~., data=heart[train,], kernel="polynomial",
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                             d=c(1:5)))
summary(best.mod)

yhat.pol <- predict(best.mod$best.model, newdata=heart[-train,])
table(yhat.pol, heart.test$chd)
misc(yhat.pol, heart.test$chd)

### Find the overall best SVM model ####
set.seed(1234)
best.mod <- tune(METHOD=svm, chd~., data=heart[train,], 
                 kernel= c("linear","radial", "polynomial"),
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5,10,100),
                 d=c(1:5),
                 gamma=1:2))

svm.out <- svm(chd~., data=heart[train,], kernel="linear", cost=0.1)
yhat <- predict(svm.out, newdata=heart[-train,])

table(yhat, heart.test$chd)
misc(yhat, heart.test$chd)

