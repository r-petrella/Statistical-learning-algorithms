# Decision trees - 14/03/2023 #####
setwd("C:/Users/ricky/Desktop/primo_anno/Statistical Learning/Lab4")
## Regression trees ####
#library(ElemStatLearn)
load("prostate.RData")

dim(prostate)
x <- prostate[,-ncol(prostate)] # remove the column "train"
p <- ncol(x)-1
n <- nrow(x)

### Training + Validation set ####
set.seed(1234)
train <- sample(1:n, ceiling(n/2))
x.test <- x[-train,]

library(tree)
tree.prostate <- tree(lpsa ~ ., x, subset = train)
summary(tree.prostate)
tree.prostate

plot(tree.prostate)
# pretty: regulates the space of the notation
# digits: regulates the amount of figures for the observations in the nodes
text(tree.prostate, pretty=0, digits=3)
# remember: the model fit uses x.train
#             "predict" uses x.test         
y.hat <- predict(tree.prostate, x.test)
mean((x.test$lpsa - y.hat)^2)  #REMARK: in decision trees we don't split the dataset
# in x and y, just in x.train and x.test. 
## Cross-validation ####
set.seed(1234)
# Cross-validation for Choosing Tree Complexity
# Runs a K-fold cross-validation experiment to find the deviance or number 
# of misclassifications as a function of the cost-complexity parameter k.
# FUN = the function to do the pruning
cvtree.prostate <- cv.tree(tree.prostate, K=5, FUN = prune.tree)
cvtree.prostate
# we want to extract the tree with the minimum deviance and we call it best.terminal
which.min(cvtree.prostate$dev)  # the tree with 4 terminal nodes

best.terminal <- cvtree.prostate$size[which.min(cvtree.prostate$dev)]
best.terminal

# best: integer requesting the size (i.e. number of terminal nodes) of a 
# specific subtree 
prune.prostate <- prune.tree(tree.prostate, best = best.terminal)
plot(prune.prostate)
text(prune.prostate)

y.pruned <- predict(prune.prostate, x.test)
mean((x.test$lpsa - y.pruned)^2)


## Classification trees ####
load("SAheart.RData")
n <- nrow(SAheart)
p <- ncol(SAheart)-1
x <- SAheart[,-ncol(SAheart)]
y <- SAheart[,ncol(SAheart)]
heart <- data.frame(chd = as.factor(y),x)

misc <- function(y,yhat){
  mean(y != yhat)
}

## Training + Validation set ####
set.seed(1234)
train <- sample(1:n, ceiling(n/2))
heart.test <- heart[-train,]

tree.heart <- tree(chd ~ ., heart, subset = train)
summary(tree.heart)
tree.heart

plot(tree.heart)
text(tree.heart, pretty = 0)

# Compute the test error estimate
tree.pred <- predict(tree.heart, heart.test, type = "class")
table(tree.pred, heart.test$chd)
#error rate
misc(tree.pred, heart.test$chd)

## Pruning the tree ####
set.seed(1234)
cv.heart <- cv.tree(tree.heart, FUN = prune.misclass)
cv.heart
# even with classif trees we use the deviance as criterion to prune
best.size <- cv.heart$size[which.min(cv.heart$dev)]
best.size  #the tree with 5 terminal nodes

prune.heart <- prune.misclass(tree.heart, best = best.size)
plot(prune.heart)
text(prune.heart, pretty = 0)

pruned.y <- predict(tree.heart, heart.test, type = "class")
table(pruned.y, heart.test$chd)
# smaller error rate 
misc(pruned.y, heart.test$chd)

