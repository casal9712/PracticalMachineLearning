Course project for Pactical Machine Learning
---

**Loading and preprocessing the data**

For the following R script, seed was set to 333 for reproducibility. First, "pml-training.csv" and "pml-testing.csv" was imported using read.csv() and "pml-testing.csv" will be treated as the validation set for final model to predict. "pml-training.csv" needed to be further divided into training set (70%) and testing set (30%) so that model trained from training set can be used to estimate in sample and out of sample accuracy. However, before doing that, data was first explored.

```r
set.seed(333)
library(ggplot2);library(caret);
#you may also need these packages
library(e1071);library(randomForest)
buildData1 <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
validation <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))
```

First, buildData1 was explored using summary(), from the 160 variables, many variables were defined as factor type while they should have been numeric according to the their variable name. However, due to large amount of blank values, they were defined as factor after import. To clean the data up, these variables with large amount of missing values and blank values will be droped before model training.

```r
summary(buildData1)
#first make a copy of the buildData1 as buildData2
buildData2<-buildData1
#then for all columns, except 'window'(column 6) and 'classe'(column 160),
#were converted to first character then numeric,
#this way the blank values from those factor columns will be converted into 'NA',
#while those numeric columns with many 'NA' will remain the same
#this will also covert all values from user_name column into 'NA'
for (i in c(1:5,7:(length(names(buildData1))-1))){
	buildData2[,i]=as.numeric(as.character(buildData1[,i]))
}
#then a subset of buildData2 was made, only columns with less than 50% missing were subset out
#this will remove 'user_name','cvtd_timestamp' and columns that originally have many blanks and 'NA'
buildData<-buildData2[,(colSums(is.na(buildData2))/19622)<0.5]
```

Remaining data has 58 columns for model training. Note that even though the 20 cases to be predicted were based on the same 6 user, user_name was decided to be excluded since the final model was aimmed to be trained to predict the outcome to any users instead of the only 6 users in the current dataset.

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
modRF <- train(classe~ .,data = training,method="rf",trControl=trainControl(method="cv"),number=3)
```

**In sample and out of sample accuracy/error**

To obtain the estimate of model accuracy/error, predictions were made on training set and were compared with truth values from training set for in sample accuracy and then similarly on testing set for out of sample accuracy.

```r
#predict on training set
predRF_train <- predict(modRF, training)
#Confusion matrix on training
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
#In sample accuracy and 95% CI
confusionMatrix(predRF_train, training$classe)$overall[c(1,3:4)]
```

```
##      Accuracy AccuracyLower AccuracyUpper 
##     1.0000000     0.9997315     1.0000000
```

From above, the in sample accuracy is 100%, even though out of sample accuracy is always smaller, it was expected to still be over 95%.

```r
#predict on testing set
predRF_test <- predict(modRF, testing)
#Confusion matrix on testing
confusionMatrix(predRF_test, testing$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    1    0    0
##          C    0    0 1025    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
```

```r
#Out of sample accuracy and 95% CI
confusionMatrix(predRF_test, testing$classe)$overall[c(1,3:4)]
```

```
##      Accuracy AccuracyLower AccuracyUpper 
##     0.9998301     0.9990536     0.9999957
```

**Predicting on validation set**

Based on the results, the random forest model was already performing very well and it turned out to have over 99% estimated out of sample accuracy on the testing set, so other training methods were not further considered and this model was chosen for prediction on the validation set. Predictions are as followed:

```r
predRF_V<-predict(modRF ,validation)
predRF_V
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```
