# Random Forest - 15th of March
setwd("C:/Users/ricky/Desktop/primo_anno/Statistical Learning/Lab4")
library(randomForest)

## Bagging and Random Forest Regression ####
# In random forest regression, multiple decision trees are trained on randomly
# sampled subsets of the training data. Each tree is trained to predict the 
# target variable based on a different subset of the input features. During 
# prediction, the output of each tree is averaged to produce the final prediction.

load("prostate.RData")
names(prostate)


x <- prostate[,-ncol(prostate)]
p <- ncol(x) - 1
n <- nrow(x)

# split the dataset into train+validation set
set.seed(1234)
train <- sample(1:n, ceiling(n/2))
x.test <- x[-train,]

################ Bagging regression ####
out.bag <- randomForest(lpsa ~ ., data = x, subset=train, mtry=p, importance=TRUE)
# "mtry": Number of variables randomly sampled as candidates at each split
out.bag

# "importance": if it is true, it considers the importance of features in 
#determining the accuracy of the prediction of the model.
# As default it measures the %increase of MSE (i.e the %decrease of accuracy for
# classification trees)
# and the increase in the node purity if a variable is moved in 
# the "out-of-bag" sample 
importance(out.bag)
# "IncNodePurity": it measures how much the node purity is increased on
#  average by splitting on a given feature.

varImpPlot(out.bag)

## Estimate test error ####
yhat.bag <- predict(out.bag, newdata = x[-train,])

# same as decision trees, you use x.test$lpsa = y.test 
mean((x.test$lpsa - yhat.bag)^2)
# begging often works better in terms of accuracy than decision trees

########## Random Forests regression####
# bagging -> mtry=p
# random forest -> no mtry
out.rf <- randomForest(lpsa ~., data = x, subset = train, importance = T)

out.rf
obj <- importance(out.rf)

varImpPlot(out.rf)  
# remember that the variables at the top are the most important ones
# that means that if you remove a certain variable, you will experience
# a certain %increase in the MSE

# example: extract the top 5 variables in terms of %IncMSE from importance
rownames(obj)
rownames(obj)[order(obj[,1], decreasing=T)[1:5]]  # also [1:5] can be out of []

# example
a <- c(2,6,3,5)
sort(a)
a[order(a)]   #order(a) returns the position and not the sorted vector

## Estimate the test error ####
yhat.rf <- predict(out.rf, newdata = x[-train,])
mean((x.test$lpsa - yhat.rf)^2)

## Bagging and Random Forest Classification ####
# REMARK: regression always on prostate
#         classification always on heart
load("SAheart.RData")
x <- SAheart[,-ncol(SAheart)]
y <- SAheart[, ncol(SAheart)]
# we create an entire dataframe because with trees we don't want the division 
# of x and y
heart <- data.frame(chd = as.factor(y), x)

misc <- function(y.hat, y){
  mean(y.hat != y)
}

n <- nrow(x)
p <- ncol(x)

set.seed(1234)
train <- sample(1:n, ceiling(n/2))
heart.test <- heart[-train,]

## Bagging ####
out.bag <- randomForest(chd ~ ., heart, subset=train, mtry=p, importance=T)

out.bag

importance(out.bag)
varImpPlot(out.bag)

# Test error estimate ####
yhat.bag <- predict(out.bag, newdata= heart[-train,])
table(yhat.bag, heart.test$chd)
misc(yhat.bag, heart.test$chd)

## Random forests ####
out.rf <- randomForest(chd ~., data=heart, subset=train, importance=T)

out.rf
obj <- importance(out.rf)

varImpPlot(out.rf)  

## Test Error estimate ####
yhat.rf <- predict(out.rf, newdata = heart[-train,])
table(yhat.rf, heart.test$chd)
misc(yhat.rf, heart.test$chd)
