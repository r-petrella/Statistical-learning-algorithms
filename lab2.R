##### LDA #####

# library(ElemStatLearn)
setwd("C:/Users/ricky/Desktop/primo_anno/Statistical Learning/Lab2")
load("SAheart.RData")
summary(SAheart)

str(SAheart)  # 10 predictors     chd binary response variable

n <- nrow(SAheart)  # number of rows: 462
set.seed(1234)

# randomly pick n/2 integers from 1 to n without replacement,
# so to generate two equally sized random subsets
index <- sample(1:n, ceiling(n/2), replace = F)

library(MASS) # contains linear discriminant analysis

# it yields the group means and predictors means
out.lda <- lda(chd ~ ., data = SAheart[index,])
# the data are just 1/2 of the observations randomly taken from index 

out.lda

# this is the proportion of "successes" in this random subset
sum(SAheart$chd[index])/length(index)    #75/231

library(dplyr)

# how to retrieve the numeric predictor means in the "success" group
SAheart[index,] %>%    
# "%>%" is used to concatenate commands or functions
  filter(chd == 1) %>%  
#The filter() function is used to subset a data frame, 
# retaining all rows that satisfy your conditions. To be retained, 
# the row must produce a value of TRUE for all conditions.
# Note that when a condition evaluates to NA the row will be dropped
  summarize_if(is.numeric, mean)

# Now we do the prediction of chd on the other half of the dataset
# we get decimal numbers between 0 and 1
class.pred <- predict(out.lda, newdata = SAheart[-index,])$class

# pick for instance the first individual in the validation set
SAheart[-index,][1,]

# this person was assigned to the "success" class by LDA
class.pred[1]

# the score for this individual will be the dot product of
# the linear discriminants and the individual's predictors' values

# "t"= matrix transpose
# "$scaling" retrieves the coefficients of linear discriminants
# "%*%" product between matrices or vectors
# 9x1 * 1x9 
t(out.lda$scaling) %*% as.numeric(SAheart[-index,][1,-10]) #-10 to remove chd
# "as.numeric" removes the labels of the retrieved values 
# and converts categorical values into numerical values


SAheart[-index,][2,]
class.pred[2]
# the score for this individual, who was assigned to the
# "failure" class, is lower
t(out.lda$scaling) %*% as.numeric(SAheart[-index,][2,-10])


# What's the relationship between LDA scores and posterior
# probabilities? They are monotonically related and, since
# we have two classes, LDA assigns to the "success" group
# the individuals whose predicted posterior probability
# is larger than 50%.
lda.scores <- c()

              # 231
for(i in 1:(n -length(index))){
  lda.scores[i] <- c(
                  t(out.lda$scaling) %*% as.numeric(
                    SAheart[-index,][i,-10]))
}

lda.scores

# without "$posterior" predict yields predicted classes, post.probab and scores
posterior.pred <- predict(out.lda, newdata=SAheart[-index,])$posterior

point_colors <- c("coral", "cornflowerblue")

# posterior.pred[,2] are post prob of people with chd
# cex a numerical vector giving the amount by which plotting characters 
# and symbols should be scaled relative to the default. 
plot(y = posterior.pred[,2], x = lda.scores, cex=2,
     ylim = c(0, 1), col = point_colors[SAheart[-index, 10] + 1],
# by using point_colors[SAheart[-index, 10] + 1], we create a list of
# 231 indexes (1 and 2) which retrieve the two colors and apply them to the points

     pch = as.integer(class.pred),  # symbol to plot(ex. circles and squares) 
     xlab = "LDA individual scores",
     ylab = "Posterior probability of disease presence")

abline(h = .5, lty = 2, col = "olivedrab")


# "pch": shape of the symbols
legend("left", inset=.02, col = point_colors, pch = 15,
       legend = c("Actual ABSENCE of chd",
                "Actual PRESENCE of chd"),
       pt.cex = 2)

legend("topleft", inset=.02, col=1, pch = 1:2,
       legend = c("Predicted ABSENCE of chd",
                "Predicted PRESENCE of chd"),
       pt.cex = 2)

# max score with associated posterior probability below .5


#posterior.pred[,2]<.5) labels the observations with predicted class 0 as TRUE
# "which" yields the observations where the condition inside the function is true

# the observations where the predicted class is 0 are used to extract their scores

# the max scor is the vertical threshold
abline(v = max(lda.scores[which(posterior.pred[,2]<.5)]),
       col="red")


table(class.pred)   # "table" counts the predicted 1s and the 0s (y.hat)
table(SAheart[-index,]$chd)   # the observed 1s and 0s (y.test)

# confusion matrix
table(pred_class = class.pred,
      true_class = SAheart[-index,]$chd)
# so we notice:
# - 46 false negatives
# - 15 false positives

# error rate
(46+15)/(n-length(index))

# function defined by Prof automatically computes the error rate
misc <- function(yhat, y){
  if (length(table(yhat)) != length(table(y)))
    stop("The levels of the two vectors do not match")
  1-sum(diag(table(yhat,y)))/length(y)
}

misc(yhat = class.pred, y = SAheart[-index,]$chd)


# logistic regression

out.log <- glm(chd ~ . , data = SAheart[index,],
               family = "binomial")
summary(out.log)

# predicted probabilities of chd with log reg
p.hat <- predict(out.log, newdata = SAheart[-index,],
                 type="response")

# labeling with 1 and 0 to the predicted probabilities
y.hat <- ifelse(p.hat > .5, 1, 0)

# confusion matrix
table(pred_class = y.hat,
      true_class = SAheart[-index,]$chd)

# error rate
misc(y.hat, SAheart[-index,]$chd)

# the two methods are almost indistinguishable from the
# point of view of the misclassification rate


##### LINEAR MODEL SELECTION #####

load("prostate.RData")
summary(prostate)

# there's an additional variable we won't use  (it is called train)
dim(prostate)
x <- prostate[,-10]

# the package that helps us perform model selection is
library(leaps)


# response: lpsa (decimal values between -0.5 and + 5.4)
# "regsubsets":This function computes all possible models by selecting a subset 
#of predictor variables and fits a linear regression model for each subset Model
regfit.full <- regsubsets(lpsa ~ ., x) #method: "exhaustive" (default)
                                      # best subset selection
summary(regfit.full)

# Don't be misled by the fact that each subsequent model
# contains the previous one (nesting): this is not forward selection!

# the exhaustive selection algorithm refers to the "best
# subset" strategy

reg.summary <- summary(regfit.full)

# "names" yields either attributes or options of its argument
names(reg.summary)

# it outputs the scores of the 8 models with different selected features
reg.summary$rsq
# R^2 represents the proportion of the total variance in the dependent variable
# that is explained by all of the independent variables together.
#It is a popular measure of goodness of fit for regression models, 
#indicating how well the model fits the data.

reg.summary$bic
#The BIC is a goodness-of-fit measure that takes into account both the fit of
#the model to the data and the complexity of the model. 
#It is based on the principle of Bayesian inference, which states that 
#a model should be selected based on its ability to explain the data 
#while also being as simple as possible.

#The BIC is calculated as:
  
 # BIC = n * ln(SSR/n) + k * ln(n)

#where n is the sample size, SSR is the sum of squared residuals from the
# regression model, and k is the number of parameters in the model
# (including the intercept term).

#The BIC penalizes models with more parameters, and therefore favors
# simpler models over more complex ones. A lower BIC value indicates a 
# better model fit relative to competing models.


reg.summary$cp
#Mallow's C_p is a statistical measure that is used to evaluate the fit of 
#a linear regression model, and to compare different regression models in 
#terms of their predictive accuracy. It is a goodness-of-fit measure that 
#takes into account the number of predictors used in the model, and is based on
# the principle of minimizing the mean squared error (MSE) of the model.

# to understand formula of Mallow's C_p, check p. 233 and 69
# of the reference book

# C_p = (RSS / MSE) - n + 2(m + 1)
# The value of C_p ranges from 0 to infinity, with smaller values indicating 
# better model fit. In practice, a C_p value close to p is often used as 
# a threshold for model selection, since this indicates that the fitted model 
# is almost as good as the full model in terms of predictive accuracy, while 
# using fewer predictors.

tmplm <- lm(lpsa ~ ., x)
summary(tmplm)

# S**2/df is the residual standard error of the full model
# 88 is the degrees of freedom from summary(tmplm)
sqrt(sum(tmplm$residuals**2)/88)

# model with lcavol and lweight, i.e. best model with two predictors
tmplm_lcavol <- lm(lpsa ~ lcavol + lweight, x)
# RSS from this model with 2 predictors
sum(tmplm_lcavol$residuals**2)

#C_p with two predictors(2 alternative methods to compute it:
# the 1st is cp formula)                                    df    n   2(2+1)
c(sum(tmplm_lcavol$residuals**2) / (sum(tmplm$residuals**2)/88) - 97 + 6, 
  reg.summary$cp[2]) #this is the 2nd


c(which.max(reg.summary$adjr2),    
#which.max yields the position of the list with the max value 
#(7th, model with 7 predictors)
  which.min(reg.summary$bic),  # 3rd: model with 3 variables
  which.min(reg.summary$cp))   # 5th: model with 5 variables

# Let us visualize more indicators at once
par(mfrow=c(2,2))  # "par" creates (2 rows and 2 columns of subplots) 

plot(reg.summary$rss,xlab="Number of Variables",
     ylab="RSS", type="l", lwd=2)

plot(reg.summary$adjr2,xlab ="Number of Variables",
     ylab="Adjusted RSq", type="l", lwd=2)

plot(reg.summary$bic,xlab ="Number of Variables",
     ylab="BIC", type="l", lwd=2)

plot(reg.summary$cp,xlab ="Number of Variables",
     ylab="C_p",type="l", lwd=2)


# So we can get the estimates of the regression coefficients
# from the model with seven predictors (best according to adjR2)
coef(regfit.full, id = 7)


# Let us try forward, backward and hybrid selection

regfit.fwd <- regsubsets(lpsa ~ ., data = x, method = "forward")
summary(regfit.fwd)

regfit.bwd <- regsubsets(lpsa ~ ., data = x, method = "backward")
summary(regfit.bwd)

regfit.hyb <- regsubsets(lpsa ~ ., data = x, method = "seqrep")
summary(regfit.hyb)

# All identical but the hybrid version, which has a tiny difference


##### VALIDATION SET APPROACH AND CROSS-VALIDATION #####

set.seed(1234)
train <- sample(c(TRUE, FALSE), nrow(x), rep=T) # 97 random T/F
test <- !train   # the opposite of test

# best subset selection on the train dataset (this is the correct way !!!)
# (do not use the full dataset or the estime will not be correct)
regfit.best <- regsubsets(lpsa ~ ., data=x[train,])  
#only the TRUE rows from x are extracted (41)

# we'll need a model matrix from the test data
# The model.matrix() function is used in many regression packages
# for building an X matrix from data. 
test.mat <- model.matrix(lpsa ~ ., data = x[test,])  #(56 rows)

submodels.MSE <- c()
for(M in 1:(ncol(x)-1)){
  coef_M <- coef(regfit.best, id=M)  # beta0 + b1+...+b8  dim= 9x1
  
  #            56 rows x 9 columns     *   9x1 = 56x1
  pred_M <- test.mat[, names(coef_M)] %*% coef_M
  submodels.MSE[M] <- mean((x$lpsa[test] - pred_M)**2)
}

which.min(submodels.MSE)# MSE is the lowest with the 5th model (5 predictors)

# coefficient of the linear model with 5 predictors
coef(regfit.best, 5)

# There is no predict() function for regsubsets
predict(regfit.best, x[test,]) # gives error

#This was a little tedious, partly because there is no predict() method 
#for regsubsets(). Since we will be using this function again, 
#we can capture our steps above and write our own predict method.

# this function basically yields y.hat
predict.regsubsets <- function(object, newdata, id){
  form <- as.formula(object$call[[2]]) # extracts formula
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id) # get the id estimated coefficients 
  # what variables are included in the id^th best model?
  xvars <- names(coefi)
  # return the predictions
  mat[, xvars] %*% coefi
}

predict.regsubsets(regfit.best, x[test,], 5)

# So the cycle would become
submodels.MSE_2 <- c()
for(M in 1:(ncol(x)-1)){
  pred_M <- predict.regsubsets(regfit.best, x[test,], M)
  submodels.MSE_2[M] <- mean((x$lpsa[test] - pred_M)**2)
}
all(submodels.MSE == submodels.MSE_2)  #check all the values
# the output is TRUE, all the values are the same


# Rerun selection on full dataset to improve estimates accuracy
# and possibly pick another subset of variables
regfit.best<-regsubsets(lpsa~., data=x)

# coefficients of best model on train data
coef(regsubsets(lpsa ~ ., data=x[train,]), 5)

# coefficients of model with 5 variables on full data set
coef(regfit.best, 5)


# CROSS VALIDATION

k <- 5
set.seed(1234)

# five disjoint groups:
folds <- sample(1:k, nrow(x), replace = TRUE)

# we initalize the matrix that will contain the MSE of each submodel
# for each fold

# matrix with k rows and 8 columns. the elements are NA 
cv.errors <- matrix(NA, k, 8,
                  dimnames = list(NULL, paste(1:8)))

# from 1 to k, each fold represents the test set
# we exploit our predict.regsubsets function here
for (j in 1:k) { # for each fold
  # exhaustive (default) selection strategy on train data
  best.fit <- regsubsets(lpsa~., data = x[folds != j,])
  for (M in 1:8) {
    pred <- predict.regsubsets(best.fit, x[folds == j,], id = M)
    cv.errors[j, M] <- mean((x$lpsa[folds == j] - pred)**2)
  }
}

mean.cv.errors <- apply(cv.errors, 2, mean) # 2: by column
mean.cv.errors

# Let us plot these values to have a chart similar to the ones
# generated above
par(mfrow=c(1,1))
plot(mean.cv.errors,type='b')

# So the best model seems to be the one with three variables.
# Let us thus apply the best subset strategy on the whole data set
# in order to obtain more precise estimates of the coefficients:
reg.best <- regsubsets(lpsa~., data = x)
coef(reg.best, 3)
