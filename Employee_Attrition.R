install.packages("corrplot")
install.packages("ggplot2")
install.packages("caret")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("e1071")
install.packages("caTools")
install.packages("rattle")
install.packages("gridExtra")
install.packages("ROCR")
install.packages("randomForest")
install.packages("randomForestSRC")
install.packages("reshape2")
install.packages("RColorBrewer")
install.packages("pROC")
install.packages("reshape")
library(reshape)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(caTools)
library(ROCR)
library(ipred)
library(gbm)
library(caret)
library(C50)
library(rpart)
library(tree)
library(RColorBrewer)
library(rpart.plot)
library(rattle)
library(randomForest)
library(e1071)
library(C50)
data<- read.csv("Employeeattrition.csv")
head(data)

str(data)
sum(is.na(data))
data$EmployeeNumber=data$Over18=data$EmployeeCount=data$StandardHours = NULL

p1= qplot(BusinessTravel,data = data,geom="auto")
p2 = qplot(Gender, data=data,geom="auto")
grid.arrange(p1,p2,nrow=1,ncol=2)

plottable1=table(data$Attrition,data$JobLevel)
plottable2=table(data$Attrition,data$Education)
plottable3=table(data$Attrition,data$EnvironmentSatisfaction)
plottable4=table(data$Attrition,data$JobInvolvement)
plottable5=table(data$Attrition,data$PercentSalaryHike)
plottable6=table(data$Attrition,data$PerformanceRating)
plottable7=table(data$Attrition,data$StockOptionLevel)
plottable8=table(data$Attrition,data$YearsAtCompany)
plottable9=table(data$Attrition,data$YearsInCurrentRole)
plottable10=table(data$Attrition,data$OverTime)
plottable11=table(data$Attrition,data$TrainingTime)


barplot(plottable1, main="Employees left vs Job Level", xlab="JobLevel",col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)
barplot(plottable2, main="Employees left vs Education", xlab="Education",col=c("Blue","Yellow"),legend=rownames(plottable2),beside = TRUE)
barplot(plottable3, main="Employees left vs Environment Satisfaction", xlab="JobLevel", col=c("Blue","Yellow"),beside = TRUE)
barplot(plottable4, main="Employees left vs Job Involvement", xlab="Job Involvement", col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)
barplot(plottable5, main="Employees left vs salary hike", xlab="salary hike in %", col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)
barplot(plottable6, main="Employees left vs Performance Rating", xlab="PerformanceRating",col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)
barplot(plottable7, main="Employees left vs stock option level", xlab="Stock Option Level", col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)
barplot(plottable8, main="Employees left vs Num of Years at Company", xlab="Num of Years", col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)
barplot(plottable9, main="Employees left vs Years in current Role", xlab="Years In Current Role ", col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)
barplot(plottable10, main="Employees left vs OverTime", xlab="overtime ", col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)
barplot(plottable11, main="Employees left vs TrainingTime", xlab="Trainingtime ", col=c("Blue","Yellow"),legend=rownames(plottable1),beside = TRUE)


dummy = data
dummy$Attrition=as.numeric(dummy$Attrition)
dummy$BusinessTravel=as.numeric(dummy$BusinessTravel)
dummy$Department=as.numeric(dummy$Department)
dummy$EducationField=as.numeric(dummy$EducationField)
dummy$Gender=as.numeric(dummy$Gender)
dummy$JobRole=as.numeric(dummy$JobRole)
dummy$MaritalStatus=as.numeric(dummy$MaritalStatus)
dummy$OverTime=as.numeric(dummy$OverTime)

corTable=cor(dummy)
corr=melt(corTable)
corTable
corr

corrplot( cor(as.matrix(dummy), method = "pearson", use = "complete.obs") ,is.corr = FALSE, type = "lower", order = "hclust", tl.col = "black", tl.srt = 360)

data$MaritalStatus=data$MonthlyIncome=data$PerformanceRating= NULL

set.seed(3000)
split=sample.split(data$Attrition,SplitRatio = .7)
train=subset(data,split==T)
test=subset(data,split==F)

attLog=glm(Attrition~.,data=train,family = binomial)
predGlm=predict(attLog,type="response",newdata=test)
table(test$Attrition,predGlm>.5)


decisionTreeModel= rpart(Attrition~.,data=train,method="class",minbucket =20 )
fancyRpartPlot(decisionTreeModel)
predDT=predict(decisionTreeModel,newdata = test,type = "class")
table(test$Attrition,predDT)

dt_ROC=predict(decisionTreeModel,test)
pred_dt=prediction(dt_ROC[,2],test$Attrition)
perf_dt=performance(pred_dt,"tpr","fpr")

auc_dt <- performance(pred_dt,"auc")
auc_dt <- round(as.numeric(auc_dt@y.values),3)

print(paste('AUC of Bagged Tree:',auc_glm))
print(paste('AUC of Decision Tree:',auc_dt))
print(paste('AUC of Extreme Boosting:',auc_RF))


baggingmodel = bagging(Attrition~.,
                       data=train,control=rpart.control(cp=.00001),nbagg=100, coob= TRUE)
predprobs = predict(baggingmodel,newdata = test,type="prob")
testwithprobs = cbind(test,predprobs[,2])
print(testwithprobs)
print(baggingmodel)
predprobs
bagpred = prediction(predprobs[,2], test$Attrition)
bagperf = performance(bagpred, "tpr", "fpr")
 
plot(bagperf, col=2, add=TRUE,col ='blue')
 auc.curve = performance(bagpred, "auc")
 auc.curve
 
 plot(perf_dt,add=TRUE, col='red')
 plot(perf_RF, add=TRUE, col='green3')
 legend('bottom', c("Bagged Tree", "Decision Tree", "Extreme Boosting"), fill = c('blue','red','green3'), bty='n')
#Extreme gradient Boosting
library(xgboost)
library(caret)
library(Matrix)
library(readr)
library(plyr)
library(ROCR)
names(getModelInfo())
#getModelInfo()$xgbTree$type
library(readxl)
library(tidyr)
Employeeattrition <- read.csv("Employeeattrition.csv")
View(Employeeattrition)
#making it to dataframe
df=data.frame(Employeeattrition)
#removing the unwanted columns form the dataset
df1=subset(df,select = -c(EmployeeCount,Over18,StandardHours,WorkLifeBalance))
tempAttrition=df1$Attrition
#df1$JobInvolveent
#reorderign the columns
df2=df1[c('Attrition','Age','DailyRate','Education','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','MonthlyRate','PercentSalaryHike','StockOptionLevel','YearsSinceLastPromotion','Department','EducationField','Gender','JobLevel','MaritalStatus','NumCompaniesWorked','PerformanceRating','TotalWorkingYears','YearsAtCompany','YearsWithCurrManager','BusinessTravel','DistanceFromHome','EmployeeNumber','HourlyRate','JobRole','MonthlyIncome','OverTime','RelationshipSatisfaction','TrainingTimesLastYear','YearsInCurrentRole')]
#making all the cloumns to factors
df2 = as.data.frame(unclass(df2))
str(df2)
df2$Attrition=ifelse(df2$Attrition=="Yes", 1,0)
df2$Attrition=as.numeric(df2$Attrition)
outcomename='Attrition'
predictornames=names(df2)[names(df2)!=outcomename]
#df1Dummy=dummyVars("~.",data=df2,fullRank = F)
set.seed(233)
#test_rows <- sample(nrow(df1), nrow(df1)/3)
test_rows=createDataPartition(df2$Attrition,p=0.3,list=FALSE,times=1)

test=df2[test_rows,]
train=df2[-test_rows,]
train_sp=sparse.model.matrix(Attrition ~ .-1,data = train)
train_sp
head(train_sp)
train_label <- train[,"Attrition"]
train_matrix <- xgb.DMatrix(data = as.matrix(train_sp), label = train_label)
test_sp <- sparse.model.matrix(Attrition~.-1, data = test)
test_label <- test[,"Attrition"]
test_matrix <- xgb.DMatrix(data = as.matrix(test_sp), label = test_label)
##Parameters
nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 6749,
                       watchlist = watchlist,
                       eta = 0.001,
                       max.depth = 3,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
                       seed = 333)

# Training & test error plot #6749
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')

min(e$test_mlogloss)
e[e$test_mlogloss == 0.350296,]
# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)
# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)
pred <- matrix(p, nrow = nc, ncol = length(p)/nc) %>%
         t() %>%
         data.frame() %>%
         mutate(label = test_label, max_prob = max.col(., "last")-1)
table(Prediction = pred$max_prob, Actual = pred$label)
conmat=confusionMatrix(pred$max_prob,pred$label)
conmat
auc=plot(roc(pred$max_prob,pred$label))
auc
