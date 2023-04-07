# Regularization and Dimension Reduction - March 10th 2023 #####

## Ridge regression ####
#library(ElemStatLearn)
setwd("C:/Users/ricky/Desktop/primo_anno/Statistical Learning/Lab3")
load("prostate.RData")
summary(prostate)

# let us get rid of train because we want to choose ourselves the
# training set
x<-prostate[,-ncol(prostate)] # we remove train (lpsa is still in)
y<-prostate$lpsa
p<-ncol(x)-1 # no of variables

#install.packages('glmnet')
library(glmnet)
# model.matrix turns qualitative variables into dummy variables
# if we have strings that are not considered as factors, we have
# to transform them.
# in this case we only have quantitative data so it is not necessary
x <- model.matrix(lpsa~.,x)[,-1] # to remove the intercept (1st column)

head(x)
grid <- 10^seq(10,-2,length=100)

# with the same function we can perform both ridge and lasso regression
ridge.mod <- glmnet(x, y, alpha=0, lambda = grid)  #alpha 0 means ridge regr
# x = input matrix
# y = response variable
# alpha = 1 is the lasso penalty, and alpha=0 the ridge penalty
# lambda = decreasing sequence

dim(coef(ridge.mod))
coef(ridge.mod)[,50]
ridge.mod$lambda[50]

# euclidean norm of the regularized vector regression coefficients
sqrt(sum(coef(ridge.mod)[-1,50]^2))

coef(ridge.mod)[,60]
ridge.mod$lambda[60]

sqrt(sum(coef(ridge.mod)[-1,60]^2))

# in an alternative way:
predict(ridge.mod,s=50, type ="coefficients")[1:9,]
# in this case lambda is equal to s

## Training + Validation set ####
set.seed(1234)
train <- sample(1:nrow(x), ceiling(nrow(x)/2))
y.test <- y[-train]

ridge.mod <- glmnet(x[train,],y[train],alpha=0,
                  lambda=grid)
ridge.pred <- predict(ridge.mod, s=4,newx = x[-train,])

mean((y.test-ridge.pred)^2)

# if we had instead simply fit a model with just an intercept, we would have
# predicted each test observation using the mean of the training
# observations. In that case, we could compute
# the test set MSE like this:
mean((mean(y[train]) - y.test)^2)  # mse in case of null model
# We could also get the same result by fitting a ridge regression
# model with a very large value of lambda.


# Compare vector of regularized regression coefficients with
# those of OLS (s=0 means ols)
ridge.pred <- predict(ridge.mod, s = 0, exact=T,
                    x = x[train,], y = y[train], newx = x[-train,])

mean((ridge.pred - y.test)^2)

lm(y ~ x,subset = train)

predict(ridge.mod, s=0, exact =T,
        x=x[train,],y=y[train],type="coefficients")[1:9,]

## Choice of lambda via 10-fold CV
set.seed(1234)
# cross validation on the generalized linear model
cv.out <- cv.glmnet(x[train,], y[train], alpha=0)  
plot(cv.out)
# lambda which minimizes the MSE 
best.lambda <- cv.out$lambda.min
best.lambda

ridge.pred <- predict(ridge.mod, s=best.lambda, newx = x[-train,],
                    exact = T, x = x[train,], y = y[train])
mean((y.test - ridge.pred)^2)

predict(ridge.mod,s=best.lambda,newx = x[-train,],
        exact=T, x=x[train,],y=y[train],type="coefficients")[1:9,]

# ridge has better accuracy
# lasso may be easier to interpret


## Lasso Regression ####
lasso.mod <- glmnet(x[train,],y[train],alpha=1, lambda = grid)
plot(lasso.mod)
# We can see from the coefficient plot that depending on the choice
# of tuning parameter, some of the coefficients will be exactly equal to zero.


set.seed(1234)
cv.out <- cv.glmnet(x[train,], y[train], alpha=1)
plot(cv.out)

best.lambda <- cv.out$lambda.min

lasso.pred <- predict(lasso.mod, s=best.lambda, newx=x[-train,],
                    exact=T, x=x[train,],y=y[train])

mean((y.test-lasso.pred)^2)

out <- glmnet(x, y, alpha=1, lambda=grid)
lasso.coef=predict(out, type ="coefficients",s=best.lambda)[1:9,]
lasso.coef
# remeber exact=T avoids interpolation and approximation

# Dimension reduction via PCR####

## PCR ####
#install.packages('pls')
library(pls)

set.seed(1234)
pcr.fit <- pcr(lpsa~., data = prostate[,-ncol(prostate)],scale=T, validation="CV")
summary(pcr.fit)

validationplot(pcr.fit, val.type = "MSEP", legendpos = "top")
# val.type: validation statistics 

#MSEP stands for "Mean Squared Error of Prediction" and is a statistical 
#metric used to evaluate the accuracy of a predictive model. It is a 
#measure of the average squared difference between the predicted values and 
#the actual values.
#A lower MSEP indicates a better fit between the predicted and actual 
#values, while a higher MSEP indicates a poorer fit

ncomp.1sigma <- selectNcomp(pcr.fit, method = "onesigma", plot = T)
# The "onesigma" method implemented in "SelectNcomp" uses the one
# standard deviation rule to determine the optimal number of principal
# components to retain. Specifically, it retains all components with
# eigenvalues greater than the mean eigenvalue minus the standard
# deviation. This rule is based on the idea that components with
# eigenvalues greater than this threshold explain more variance in the
# data than would be expected by chance, and therefore are likely to
# contain meaningful patterns or information.

ncomp.random <- selectNcomp(pcr.fit, method = "randomization", plot = T)
# The randomization test approach checks whether the squared prediction 
#errors of models with fewer components are significantly larger than in 
#the reference model. This leads for each model considered to a p value; the 
#smallest model not significantly worse than the reference model is returned 
#as the selected one.

# Test error estimate ####
set.seed(1234)
train <- sample(1:nrow(x), nrow(x)/2)
y.test <- prostate[-train,9]

pcr.fit <- pcr(lpsa~.,data = prostate[,-ncol(prostate)],
             subset = train, scale=T, validation="CV")

validationplot(pcr.fit)
summary(pcr.fit)

pcr.pred <- predict(pcr.fit, prostate[-train,], ncomp=5)
mean((y.test-pcr.pred)^2)  #test error estimate with 5 components

# PCR via eigen()  ####
# Computes eigenvalues and eigenvectors of numeric (double, integer, logical) 
# or complex matrices.

x.pcs <- eigen(cor(prostate[train,1:8]))$vectors #extract eigenvectors


test.x <- prostate[-train,1:8]

for(i in 1:8){ #standardization of the test set with the training set
  test.x[,i] <- (test.x[,i] - mean(prostate[train,i]))/sd(prostate[train,i])
}

# principal components = x*eigenvectors
x.train <- scale(prostate[train,1:8],T,T) %*% x.pcs[,1:5]
x.test <- as.matrix(test.x) %*% x.pcs[,1:5]

# In pca, the response variable remains the same
y.train <- prostate[train,]$lpsa
y.test <- prostate[-train,]$lpsa

# create a new dataset with princ components (both train and test)
data.pcs <- data.frame(y = c(y.train, y.test), rbind(x.train, x.test))

# remember that subset requires only the observations' index and 
# not the dataset
out.pcs <- lm(y ~ ., data.pcs, subset = 1:length(train))

#                                                 49          :  97            -y
y.hat <- predict(out.pcs, 
                 newdata = data.pcs[(length(train)+1):nrow(data.pcs),])
mean((y.hat-y.test)^2)

# PCR via svd()
xx <- scale(prostate[train,1:8],T,T)
#The SVD provides a way to express a matrix in terms of its underlying 
#structure, and is useful for data compression, dimensionality reduction, 
#and data analysis. In particular, the SVD can be used to compute the principal 
#components of a dataset, which are the directions of maximum variance in the data.

#SVD stands for Singular Value Decomposition, which is a technique used in 
#linear algebra to decompose a matrix into three separate matrices. In  
#statistics, SVD is a commonly used tool in data analysis and machine learning.

svd.xx <- svd(xx)

# svd has 3 components: d, u, v
# U - the left singular vectors: an m x m orthogonal matrix whose columns 
#     are the eigenvectors of AA^T if m < n
#
# d - the singular values: an m x n diagonal matrix whose entries are the 
#     square roots of the eigenvalues of AA^T if m < n
#
# v - the right singular vectors: an n x n orthogonal matrix whose columns 
#     are the eigenvectors of A^TA if m > n
xx.pcs <- svd.xx$v[,1:5]

test.xx <- prostate[-train, 1:8]

for(i in 1:8){
  test.xx[,i] <- (test.xx[,i] - mean(prostate[train,i]))/sd(prostate[train, i])
}
#           x*eigenvectors
xx.train <- xx %*% xx.pcs
xx.test <- as.matrix(test.xx) %*% xx.pcs[,1:5]
# do not transform y
y.train <- prostate[train,]$lpsa
y.test <- prostate[-train,]$lpsa

#create the whole new dataframe
data.svd <- data.frame(y = c(y.train,y.test), rbind(xx.train, xx.test))
out.pcs <- lm(y ~., data.svd, subset = 1:length(train))

yy.hat <- predict(out.pcs, newdata = data.svd[(length(train)+1):nrow(data.svd),-1])
mean((yy.hat-y.test)^2)


# final fit of the model with the whole dataset
pcr.fit <- pcr(y ~ x, scale=TRUE, ncomp = 5)
summary(pcr.fit)
