library(glmnet)

calLogisFunc <- function(x,par)
{
  #f(x) = 1/(1+ e^(- (x[1]*par[1] + x[2]*par[2] + ... + x[n]*par[n] + par[n+1]) )
  nfeatures <- length(x)
  #print(length(x))
  #print(length(par))
  # print(length(x)-length(par))
  #   print(paste0("x is :",x))
  #   print(paste0("par is",par))
  #   print(paste0("sum is:", sum(x*par[1:nfeatures])))
  #   print(paste0("last param:", par[nfeatures + 1]))
  f <- as.numeric(sum(x*par[1:nfeatures])) + as.numeric(par[nfeatures + 1])
  # print(f)
  f <- as.numeric(1/(1+exp(-f)))
  
  return(f)
}  

calFirstLogisDerivative <- function(x, y, par, k)
{
  nfeatures <- length(x)
  f <- calLogisFunc(x, par)
  if (k == (nfeatures + 1)) {
    #intercept
    return( (f-y)*f*(1-f) )
  } else {
    return( (f-y) * f * (1-f) * x[k] )
  }
}
#theta <- theta - learningRate*(1/m*sum((f-y)*y(1-y)*theta) + lambda*((1-alpha)theta + alpha ))
updateParam <- function(data, par, learningRate, lambda, alpha)
{
  nfeatures <- ncol(data) - 1
  numObs <- nrow(data)
  newPar <- vector(mode="numeric", length=length(par))
  xMatrix <- data[,1:nfeatures] #features matrix
  yVec <- data[,ncol(data)] #label vector
  for (i in 1:length(par))
  {
    gradientLogis <- 0
    for (m in 1:numObs) {
      x <- as.numeric(xMatrix[m,])
      y <- yVec[m]
      gradientLogis <- gradientLogis + calFirstLogisDerivative(x, y, par, i)
    }
    if (i == length(par)) {
      #intercept
      newPar[i] <- par[i] - learningRate*(  1/numObs * gradientLogis )
    } else {
      newPar[i] <- par[i] - learningRate*(  1/numObs * gradientLogis + lambda * ((1-alpha)*par[i] + alpha*sign(par[i]) ) )  
    }
    
  }
  return (as.vector(newPar))
}
costFun <- function(data, par, lambda, alpha)
{
  nfeatures <- ncol(data) - 1
  xMatrix <- data[,1:nfeatures] #features matrix
  yVec <- data[,ncol(data)] #label vector
  numObs <- nrow(data)
  total <- 0
  
  for (m in 1:numObs) {
    x <- as.numeric(xMatrix[m,])
    y <- yVec[m]
    total <- total + (as.numeric(calLogisFunc(x, par)) - y)^2
  }
  
  l1reg <- sum(abs(par[1:(length(par)-1)])) # don't count intercept
  l2reg <- 1/2 * norm(par[1:(length(par)-1)], type = "2")^2   # don't count intercept
  
  cost <- total/(2*numObs) + lambda*((1-alpha)* l2reg + alpha * l1reg)
  
  return(cost)
}

learningModel <- function(data)
{
  #init parameters
  nfeatures <- ncol(data) - 1 #last column is label
  par <- vector(mode="numeric", length = nfeatures + 1) # intercept is par[nfeatures + 1]
  for (i in 1:length(par)) {
    par[i] <- 1
  }
  
  learningRate <- 0.01
  lambda <- 0.01
  alpha <- 0.5
  
  epsilon <- 10^-5 #stop condition when difference between two consecutive cost functions is less than epsilon
  maxit <- 1000
  
  cost <- costFun(data, par, lambda, alpha)
  
  iter <- 0
  while(TRUE) {
    iter <- iter + 1
    if (iter %% 10 == 0)
      print(paste0("iter:", iter, "  cost: ", cost))
    learningRate <- 10/(10 + iter)
    par <- updateParam(data, par, learningRate, lambda, alpha)
    costTemp <- costFun(data, par, lambda, alpha)
    if (abs(costTemp-cost) < epsilon || iter > maxit) {
      break
    } else {
      if (costTemp < cost) {
        cost <- costTemp
      }  
    }
  }
  
  return(par)  
}
testingModel <- function(data, par)
{
  X <- data[, 1: (ncol(data)-1)]
  Y <- data[, ncol(data)]
  predictedY <- c()
  numObs <- nrow(data)
  for (i in 1:numObs) {
    y <- as.numeric(Y[numObs])
    x <- as.numeric(X[numObs,])
    
    proby <- calLogisFunc(x, par)
    if (proby < 0.5) {
      predictedY <- c(predictedY, 0)
    } else {
      predictedY <- c(predictedY, 1)
    }
  }
  #return accuracy
  accuracy <- confusionMatrix(predictedY, Y)$overall["Accuracy"]
  print(paste0("my model: ", accuracy))
  return (accuracy)
}


runAll <- function(dataSet) {
  KFOLD <- 10
  
  #split to 10 folds:
  set.seed(1)
  folds <- createFolds(factor(dataSet$y), k=KFOLD, list = FALSE)
  dataSet$fold <- folds
  
  accuracy <- c()
  accuracyGlmnet <- c()
  
  for (i in 1:KFOLD) {
    
    dataTrain <- subset(dataSet, fold != i)
    dataTest <- subset(dataSet, fold == i)
    
    dataTrain <- subset(dataTrain, select = -fold)
    dataTest <- subset(dataTest, select = -fold)
    
    XTrain <- as.matrix( dataTrain[,1:(ncol(dataTrain) - 1)] )
    YTrain <- as.matrix(dataTrain[,ncol(dataTrain)])
    XTest <- as.matrix(dataTest[,1:(ncol(dataTest) - 1)])
    YTest <- as.matrix(dataTest[,ncol(dataTest)])
    
    par <- learningModel(dataTrain)
    accuracy <- c(accuracy, testingModel(dataTest, par))
    
    
    #same model but implemented in glmnet
    alpha <- 0.5
    lambda <- 0.01
    model <- glmnet(XTrain, YTrain, family = "binomial", type.logistic="modified.Newton", alpha = alpha, lambda=lambda)
    predictedY <- predict(model, XTest, s=c(0.001), type="response", alpha= alpha, lambda= lambda)
    predictedY <- replace(predictedY, predictedY < 0.5, 0)
    predictedY <- replace(predictedY, predictedY > 0.5, 1)
    accuracyGlmnet <- c(accuracyGlmnet, confusionMatrix(predictedY, YTest)$overall["Accuracy"])
    
    
    
  }
  print("************************************************")
  print(paste0("Overall my accuracy:", mean(accuracy)))
  print(paste0("Overall GLMNET accuracy:", mean(accuracyGlmnet)))
  
}

dataSet <- read.csv('test.csv', header = T, sep=',')
dataSet <- dataSet[1:50,]
runAll(dataSet)

