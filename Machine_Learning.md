Course project for Pactical Machine Learning
---

**Loading and preprocessing the data**

For the following R script, seed was set to 333 for reproducibility. First, "pml-training.csv" and "pml-testing.csv" was imported using read.csv() and "pml-testing.csv" will be treated as the validation set for final model to predict. "pml-training.csv" needed to be further divided into training set (70%) and testing set (30%) so that model trained from training set can be used to estimate in sample and out of sample accuracy. However, before doing that, data was first explored.

```r
set.seed(333)
library(caret);
#you may also need these packages
library(e1071);library(randomForest)
#reading data and defining missing values
buildData1 <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"), na.strings=c("NA",""," "))
validation <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"), na.strings=c("NA",""," "))
dim(buildData1);dim(validation)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

First, buildData1 was explored using summary(), from the 160 variables, many variables have large amount of missing values. They were dropped before model training. The first column was also dropped since it was only the observation identifier and not relevant to the prediction. Note that even though the 20 cases to be predicted were based on the same 6 users, user_name was decided to be dropped since the final model was aimmed to be trained to predict the outcome to any users instead of the only 6 users in the current dataset.

```r
summary(buildData1)
#first make a subset of the buildData1 as buildData2 with the first column and user_name dropped
buildData2<-buildData1[,c(-1,-2)]
#then a subset of buildData2 was made, only columns with less than 50% missing were subset out
buildData<-buildData2[,(colSums(is.na(buildData2))/19622)<0.5]
```

Remaining data has 58 columns for model training. 

```r
dim(buildData)
```

```
## [1] 19622    58
```

With data cleaned, buildData were partitioned into training set(70%) and testing set(30%).

```r
inTrain<-createDataPartition(y=buildData$classe,p=0.7,list=FALSE)
training<-buildData[inTrain,]
testing<-buildData[-inTrain,]
dim(training);dim(testing);dim(validation)
```

```
## [1] 13737    58
```

```
## [1] 5885   58
```

```
## [1]  20 160
```

**Model training and cross-validation**

Due to random forest ("rf") being one of the best performing methods when there is no know model assumption, this method was chosen to train the model. K-fold cross-validation was also used during the training with a 3-fold setting which aimed to improve the model performance.

```r
modRF <- train(classe~ .,data = training,method="rf")
modRFcv <- train(classe~ .,data = training,method="rf",trControl=trainControl(method="cv",number=3))
```

**In sample accuracy/error**

To obtain the estimate of model accuracy/error, predictions were made on training set and were compared with truth values from training set for in sample accuracy.

```r
#[Random forest only] predict on training set
predRF_train <- predict(modRF, training)
#[Random forest only] confusion matrix on training and in sample accuracy with 95% CI
confusionMatrix(predRF_train, training$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
```

```r
confusionMatrix(predRF_train, training$classe)$overall[c(1,3:4)]
```

```
##      Accuracy AccuracyLower AccuracyUpper 
##     1.0000000     0.9997315     1.0000000
```

```r
#[Random forest with 3-fold cv] predict on training set
predRFcv_train <- predict(modRFcv, training)
#[Random forest with 3-fold cv] confusion matrix on training and in sample accuracy with 95% CI
confusionMatrix(predRFcv_train, training$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
```

```r
confusionMatrix(predRFcv_train, training$classe)$overall[c(1,3:4)]
```

```
##      Accuracy AccuracyLower AccuracyUpper 
##     1.0000000     0.9997315     1.0000000
```

From above, the in sample accuracy is 100% for both random forest alone and random forest with 3-fold cross validation, even though out of sample accuracy is always smaller, it was expected to still be over 95%.

**Out of sample accuracy/error**

Similarly, these were done again on testing set for out of sample accuracy.

```r
#[Random forest only] predict on testing set
predRF_test <- predict(modRF, testing)
#[Random forest only] confusion matrix on testing and out of sample accurarcy with 95% CI
confusionMatrix(predRF_test, testing$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    1    0    0
##          C    0    0 1025    2    0
##          D    0    0    0  962    0
##          E    0    0    0    0 1082
```

```r
confusionMatrix(predRF_test, testing$classe)$overall[c(1,3:4)]
```

```
##      Accuracy AccuracyLower AccuracyUpper 
##     0.9994902     0.9985110     0.9998949
```

```r
#[Random forest with 3-fold cv] predict on testing set
predRFcv_test <- predict(modRFcv, testing)
#[Random forest with 3-fold cv] confusion matrix on testing and in sample accuracy with 95% CI
confusionMatrix(predRFcv_test, testing$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    1    0    0
##          C    0    0 1025    2    0
##          D    0    0    0  962    0
##          E    0    0    0    0 1082
```

```r
confusionMatrix(predRFcv_test, testing$classe)$overall[c(1,3:4)]
```

```
##      Accuracy AccuracyLower AccuracyUpper 
##     0.9994902     0.9985110     0.9998949
```

**Predicting on validation set**

Based on the results, both models have the same performance with over 99% estimated out of sample accuracy on the testing set, so either method should be good enough for the predictions and other training methods were not further considered. Random forest with 3-fold cross validation was chosen for prediction on the validation set. Predictions are as followed:

```r
predRFcv_V<-predict(modRFcv ,validation)
predRFcv_V
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
