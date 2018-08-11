setwd("C:/Users/piyush.gupta/Desktop/Machine Learning/Analytics vidhya/Loan prediction- Ensemble")
library("caret")
library("RANN")

set.seed(1)
data <- read.csv("train.csv")
test_final <- read.csv("test.csv")
str(data)

sum(is.na(data))

preProcValue <- preProcess(data,method=c("medianImpute","center","scale"))

data_processed <- predict(preProcValue,data)

sum(is.na(data_processed))

index <- createDataPartition(data_processed$Loan_Status,p=0.75,list = F)
train_data <- data_processed[index,]
test_data <- data_processed[-index,]

firControl <- trainControl(method = "cv",
                           number=10,
                           savePredictions = 'final',
                           classProbs = T)
predictors <- c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome","CoapplicantIncome")
outcomeName<-'Loan_Status'

model_rf <- train(train_data[,predictors],train_data[,outcomeName],method='rf',trControl = firControl,tuneLength = 3)
test_data$pred_rf <- predict(object=model_rf,test_data[,predictors])
confusionMatrix(test_data$Loan_Status,test_data$pred_rf)

model_knn <- train(train_data[,predictors],train_data[,outcomeName],method='knn',trControl = firControl,tuneLength = 3)
test_data$pred_knn <- predict(object=model_knn,test_data[,predictors])
confusionMatrix(test_data$Loan_Status,test_data$pred_knn)

model_lr <- train(train_data[,predictors],train_data[,outcomeName],method='glm',trControl = firControl,tuneLength = 3)
test_data$pred_lr <- predict(object=model_lr,test_data[,predictors])
confusionMatrix(test_data$Loan_Status,test_data$pred_lr)


#Averaging of ensemble
test_data$pred_rf_prob<-predict(object = model_rf,test_data[,predictors],type='prob')
test_data$pred_knn_prob<-predict(object = model_knn,test_data[,predictors],type='prob')
test_data$pred_lr_prob<-predict(object = model_lr,test_data[,predictors],type='prob')


test_data$pred_avg<-(test_data$pred_rf_prob$Y+test_data$pred_knn_prob$Y+test_data$pred_lr_prob$Y)/3
test_data$pred_avg<-as.factor(ifelse(test_data$pred_avg>0.5,'Y','N'))

#majority voting
test_data$pred_majority<-as.factor(ifelse(test_data$pred_rf=='Y' & test_data$pred_knn=='Y','Y',ifelse(test_data$pred_rf=='Y' & test_data$pred_lr=='Y','Y',ifelse(test_data$pred_knn=='Y' & test_data$pred_lr=='Y','Y','N'))))

#Weighted average
test_data$pred_weighted_avg<-(test_data$pred_rf_prob$Y*0.25)+(test_data$pred_knn_prob$Y*0.25)+(test_data$pred_lr_prob$Y*0.5)
test_data$pred_weighted_avg<-as.factor(ifelse(test_data$pred_weighted_avg>0.5,'Y','N'))






train_data$pred_rf<-model_rf$pred$Y[order(model_rf$pred$rowIndex)]
train_data$pred_knn<-model_knn$pred$Y[order(model_knn$pred$rowIndex)]
train_data$pred_lr<-model_lr$pred$Y[order(model_lr$pred$rowIndex)]

test_data$pred_rf<-predict(model_rf,test_data[predictors],type='prob')$Y
test_data$pred_knn<-predict(model_knn,test_data[predictors],type='prob')$Y
test_data$pred_lr<-predict(model_lr,test_data[predictors],type='prob')$Y


predictors_top <- c('pred_rf','pred_knn','pred_lr')

model_gbm <- train(train_data[,predictors_top],train_data[,outcomeName],method='gbm',trControl=firControl,tuneLength=3)
model_glm<-train(train_data[,predictors_top],train_data[,outcomeName],method='glm',trControl=firControl,tuneLength=3)


test_data$gbm_stacked<-predict(model_gbm,test_data[,predictors_top])
confusionMatrix(test_data$Loan_Status,test_data$gbm_stacked)

test_data$glm_stacked<-predict(model_glm,test_data[,predictors_top])
confusionMatrix(test_data$Loan_Status,test_data$glm_stacked)





