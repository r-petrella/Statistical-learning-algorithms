# Lab I ####

## Logistic Regression ####
#library(ElemStatLearn)
setwd("C:/Users/ricky/Desktop/primo_anno/Statistical Learning/lab1")
load("SAheart.RData")
summary(SAheart)


# chd: binary response variable: 1 is coronary disease

pairs(SAheart[,-ncol(SAheart)],   # equal to -10: we remove chd from the columns
      col=ifelse(SAheart$chd==1,"coral","blue"), #if chd is 1, then the color
                                                  # is coral; else is blue
      lwd=1.5)

### Full model ####
out.log <- glm(chd ~., data=SAheart,
             family = "binomial")
# family = "binomial" is used to indicate the linear models where the
# response variable is binary (so it indicates logistic regression)

summary(out.log)

### Reduced model ####
out.log.red<-glm(chd ~ tobacco + ldl + famhist +  # we select only the significative predictors
               typea + age, data = SAheart,
             family = "binomial")

summary(out.log.red)

### k-fold CV for test error estimate

K <- 5
n <- nrow(SAheart)
set.seed(1234)
folds <- sample(1:K,n,replace = T) #extract n numbers from [1,5]
table(folds)  # frequencies of the extracted numbers

# Function for the misclassification error rate
# it computes the difference between the predicted response (y.hat)
# and y.test
misc <- function(yhat, y){
  mean(yhat != y)
}

# Classification - 5-fold CV
x <- subset(SAheart,
          select = c("chd","tobacco","ldl", 
            # we select only the significative predictors from above
                  "famhist","typea","age"))

# x is a subset of SAheart that includes a smaller
# set of variables
yy.test <- rep(NA,n)
yy.hat <- rep(NA,n)
err.cv <- NULL

for (i in 1:K){
  x.test <- x[folds == i,] 
# returns only the rows of x where folds is equal to i for all the columns
  x.train <- x[folds != i,]
  y.test <- x$chd[folds == i] 
# returns only the rows of x where folds is equal to i for y=chd
  
  out.cv <- glm(chd ~ ., data = x.train, family = "binomial") 
  #remark: we use x.train as data which is filtered with FS
  
  p.hat <- predict(out.cv, newdata=x.test, 
# the prediction is done on x.test and is based on the model 
# applied to x.train  that is why we have to specify the newdata
                 type="response")    
 
# "Type": The default is on the scale of the linear predictors; 
# the alternative "response" is on the scale of the response variable. 
# Thus for a default binomial model the default predictions are of log-odds
# (probabilities on logit scale) and 
# type = "response" gives the predicted probabilities.
# The "terms" option returns a matrix giving the fitted values 
# of each term in the model formula on the linear predictor scale.
  
# "p.hat" yields estimated probabilities:  
# if p.hat > 0.5 then the patient has chd
# else, the patient does not have chd
  y.hat <- ifelse(p.hat > 0.5, 1, 0)
  
  err.cv[i] <- misc(y.hat, y.test) # for each CV we compute the average diff 
  # between y.hat and y.test
  yy.hat[folds==i] <- y.hat
  yy.test[folds==i] <- y.test
}

err.cv # the errors for all the five CV
mean(err.cv)   #the mean of these trials is the aimed result
table(yy.hat, x$chd)  # confusion matrix
table(yy.hat, yy.test)     #x$chd == yy.test
misc(yy.hat, yy.test)

# y-axis: actual classes
# x-axis: predicted classes
# 252 = TN
# 50 = FP
# 74 = FN
# 86 = TP

## Naive Bayes Classifier ####   (gaussian "g" and kernel "k)
library(klaR)

err.cv.nb.g <- err.cv.nb.k <- NULL
yy.hat.g <- yy.hat.k <- rep(NA,n)

for (i in 1:K){
  x.test <- x[folds == i,-1]# we remove the first column because it is chd
  x.train <- x[folds!=i,-1]
  
  y.train<- x$chd[folds != i]   
  y.test <- x$chd[folds == i]
  
  out.nb.g <- NaiveBayes(x = x.train, #x = a numeric matrix, or a data frame 
                         # of categorical and/or numeric variables.
                grouping = as.factor(y.train),#= class vector (a factor)
                # as.factor turns the input into n classes for n 
                # different observed values. In our case the factors are(0,1)
                usekernel = FALSE) 
# if TRUE a kernel density estimate (density) is used for density estimation. 
# If FALSE a normal density is estimated
  
  yhat.g <- predict(out.nb.g,
                  newdata = x.test)$class
# predict applied to Naive Bayes has two lists: 
  # one with the observations and their predicted class
  # the other with the observations and their estimated probability
  out.nb.k <- NaiveBayes(x = x.train,
                       grouping = as.factor(y.train),
                       usekernel = TRUE)
  yhat.k <- predict(out.nb.k,
                  newdata = x.test)$class
  
  yy.hat.g[folds == i] <- yhat.g
  yy.hat.k[folds == i] <- yhat.k
  
  err.cv.nb.g[i] <- misc(yhat.g, y.test)
  err.cv.nb.k[i] <- misc(yhat.k, y.test)
}

mean(err.cv.nb.g)
mean(err.cv.nb.k)
mean(err.cv) #comparision with the test error rate from logistic regression


## k-NN classifier ####
library(class)
x <- SAheart[,-c(5,10)]   # to remove the categorical and response variable
# we remove also the categorical variable because we use the euclidean distance
y <- SAheart$chd

#### Divide into training + validation set ####
set.seed(1234)
# this time we sample size random numbers from the number of rows 
# without replacement. half of the rows are randomly extracted

index <- sample(1:n, size = ceiling(n/2), replace = F)

# ceiling approximates a decimal number to the first bigger integer
train <- x[index,] 
y.train <- y[index]
test <- x[-index,]
y.test <- y[-index]

# we standardize the train (not the test)
train.std <- scale(train, T, T)  #scale with centering
test.std <- test
# we standardize the test
for (j in 1:ncol(test)){  #we take the values of half of the rows from 
#  the test and we subtract the values from the other half of the rows from 
# the train and we divide by the st deviation of the train
 
  test.std[,j] <- (test[,j] - mean(train[,j])) / sd(train[,j])
}

colMeans(train.std)  # perfect 0s with scale
colMeans(test.std)   # approximatively 0s with the manual standardization

## CV
K <- 5
ntrain <- nrow(train.std)
set.seed(1234)
folds <- sample(1:K, ntrain, replace=T)

k <- c(1,3,5,11,15,25,45,105) # no. of neighbors to test (take it as given)
err.cv <- matrix(NA, K, length(k))
# K= number of rows of the matrix and number of folds
# k= number of columns of the matrix and number of knn

# 1) general cross-validation
for (i in 1:K){# for each CV from 1 to 5
  x.test <- train.std[folds == i,]
  x.train <- train.std[folds != i,]
  y.test <- y.train[folds == i]
  y_train <- y.train[folds != i]
  #2) cv for each model
  for (j in 1:length(k)){ # from 1 to 8
    
    y.hat <- knn(train = x.train, test = x.test,
               cl = y_train, k = k[j])  
# cl = factor of true classifications of training set
# k = number of neighbours considered
    
    err.cv[i,j] <- misc(y.hat, y.test)
# matrix with 5 rows and 8 columns
  }
}
err.cv
colMeans(err.cv)

best_k <- k[which.min(colMeans(err.cv))]
best_k

# Estimating the test error on the validation set (the one with best k)

y.hat.valid <- knn(train = train.std, test = test.std,
    cl = y.train, k = best_k)

table(y.hat.valid, y[-index])

err.valid <- misc(y.hat.valid, y[-index])
err.valid
