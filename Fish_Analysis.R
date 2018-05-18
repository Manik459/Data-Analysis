library(plyr)
library(survival)
library(ggplot2)
library(plotly)
library(tree)
library(corrplot)
library(caTools)
library(caret)
library(lattice)
library(rpart)
library(rpart.plot)
library(e1071)
eco <- read.csv("economic.csv", sep = ",")
Euro <- "\u20AC"
str(eco)
eco$Capacity..GT. <-  sub("," ,"",as.character(eco$Capacity..GT.)) 
eco$Capacity..GT. <- as.numeric(as.character(eco$Capacity..GT.))
eco$Total.Number.of.vessels.in.Segment <-  sub("," ,"",as.character(eco$Total.Number.of.vessels.in.Segment)) 
eco$Total.Number.of.vessels.in.Segment <- as.numeric(as.character(eco$Total.Number.of.vessels.in.Segment))
eco$Total.Engine.Power..kW..for.Segment.Size <-  sub("," ,"",as.character(eco$Total.Engine.Power..kW..for.Segment.Size)) 
eco$Total.Engine.Power..kW..for.Segment.Size <- as.numeric(eco$Total.Engine.Power..kW..for.Segment.Size)
eco$Total.Capacity..GT...for.Segment.Size <-  sub("," ,"",as.character(eco$Total.Capacity..GT...for.Segment.Size)) 
eco$Total.Capacity..GT...for.Segment.Size <- as.numeric(eco$Total.Capacity..GT...for.Segment.Size)
eco[,17:39] <- apply(eco[,17:39] ,2 ,function(x){(gsub("\u20AC","",x))})
eco[,17:39] <- apply(eco[,17:39] ,2 ,function(x){(gsub(",","",x))})
eco[,17:39] <- apply(eco[,17:39] ,2 ,function(x){(gsub(",","",x))})
eco[,17:39] <- apply(eco[,17:39] ,2 ,as.numeric)
is.na(eco)
eco[is.na(eco)] <- 0
set.seed(101)
split <- sample.split(eco$Net.Profit....Loss.,SplitRatio = 0.7)
train <- subset(eco, split == TRUE)
test <- subset(eco, split == FALSE)

###############################
## Exploration of Data  ##
###############################
str(eco)
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  numPlots = length(plots)
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  if (numPlots==1) {
    print(plots[[1]])
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))    
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
y1 <- ggplot(eco, aes(Fishing.Income, Net.Profit....Loss.))+
  geom_point(aes(color=Segment), alpha = 0.6) + geom_vline(xintercept = 0) + geom_hline(yintercept = 0)
y2 <- ggplot(eco, aes(Size.Category, Fishing.Income))+
  geom_point(aes(color=Segment), alpha = 0.6) + geom_hline(yintercept = 0)
y3 <- ggplot(eco, aes(Size.Category))+ geom_bar(aes(fill=Size.Category), alpha = 0.6)+ scale_y_continuous(breaks = seq(0,700, by=50))
multiplot(y1,y2,y3, cols=2)
groupColumns = c("Year")
dataColumns = c("Fishing.Income", "Net.Profit....Loss.")
res = ddply(eco, groupColumns, function(x) colSums(x[dataColumns]))
head(res)
y4 <- ggplot(res, aes(factor(Fishing.Income), fill=factor(Year)))+
  geom_bar(position = "fill", alpha=0.7)+ facet_grid(Year~.)
print(y4)
#############################
###### SINGLE TREE ##########
#############################
tree <- rpart(train$Net.Profit....Loss. ~ . ,method = "anova", data= train, control = rpart.control(minsplit = 30, cp=0.001))
tree.pred <- predict(object = tree, newdata= test)
plot(tree, uniform=TRUE, 
     main="Regression Tree for Net Profit/Loss ")
text(tree, use.n=TRUE, all=TRUE, cex=.8)
tree_preds_df <- data.frame(cbind(actuals=test$Net.Profit....Loss., predicteds=tree.pred))
tree_rmse <- (mean((test$Net.Profit....Loss. - tree.pred)^2))**0.5
tree_sse <- sum( (tree_preds_df$predicted - tree_preds_df$actuals)^2 )
tree_sst <- sum( (mean(eco$Net.Profit....Loss.) - tree_preds_df$actuals)^2) 
tree_r2 <- 1- tree_sse/tree_sst
cat('FOR SINGLE TREE \n')
cat('Squar Root of MSE',tree_rmse,'\n')
cat('R squared value:', tree_r2*100, '%\n' )
head(tree_preds_df)
printcp(tree)
plotcp(tree)
rsq.rpart(tree)
#############################
###### RANDOM FOREST ########
#############################
forest.tree <- train(Net.Profit....Loss. ~ ., method = "rf", 
                     data = train, importance = T,ntree=250, 
                     trControl = trainControl(method = "cv", number = 3))
forest.pred <- predict(forest.tree, test)
rf_preds_df <- data.frame(cbind(actuals=test$Net.Profit....Loss., predicteds=forest.pred))
rf_rmse <- (mean((test$Net.Profit....Loss. - forest.pred)^2))**0.5
rf_sse <- sum( (rf_preds_df$predicteds - rf_preds_df$actuals)^2 )
rf_sst <- sum( (mean(eco$Net.Profit....Loss.) - rf_preds_df$actuals)^2) 
rf_r2 <- 1- rf_sse/rf_sst
cat('FOR RANDOM FOREST \n')
cat('Squar Root of MSE',rf_rmse,'\n')
cat('R squared value:', rf_r2*100, '%\n' )
head(rf_preds_df)
print(forest.tree)
predictors(forest.tree)
plot(forest.tree, type=c("g","o"))
.##########
# BOOSTING
##########
boost.model <- train(Net.Profit....Loss. ~ ., method = "gbm",data = train, verbose = F, 
              trControl = trainControl(method = "cv", number = 3))
boost.pred <- predict(boost.model, test)
boost_preds_df <- data.frame(cbind(actuals=test$Net.Profit....Loss., predicteds=boost.pred))
boost_rmse <- (mean((test$Net.Profit....Loss. - boost.pred)^2))**0.5
boost_sse <- sum( (boost_preds_df$predicteds - boost_preds_df$actuals)^2 )
boost_sst <- sum( (mean(eco$Net.Profit....Loss.) - boost_preds_df$actuals)^2) 
boost_r2 <- 1- boost_sse/boost_sst
cat('FOR GBM \n')
cat('Squar Root of MSE',boost_rmse,'\n')
cat('R squared value:', boost_r2*100, '%\n' )
##################################
head(preds_df)
head(tree_preds_df)
head(rf_preds_df)
head(boost_preds_df)
x1<- ggplot(preds_df, aes(predicteds, actuals))+geom_smooth(method = "loess", se=FALSE)+
  geom_abline(color='#E41A1C')+ geom_point(alpha=0.3) + geom_vline(xintercept = 0) + geom_hline(yintercept = 0) + ggtitle('LINEAR REGRESSION')
print(x1)
x2<- ggplot(tree_preds_df, aes(predicteds, actuals))+ geom_abline(color='#E41A1C')+ geom_point(alpha=0.3) + 
  geom_vline(xintercept = 0) + geom_hline(yintercept = 0) +ggtitle('SINGLE TREE')
print(x2)
x3<- ggplot(rf_preds_df, aes(predicteds, actuals))+ geom_abline(color='#E41A1C')+ geom_point(alpha=0.3) + 
  geom_vline(xintercept = 0) + geom_hline(yintercept = 0)+geom_smooth(method = "loess", se=FALSE)+ggtitle('RANDOM FOREST')
print(x3)
x4<- ggplot(boost_preds_df, aes(predicteds, actuals))+ geom_abline(color='#E41A1C')+ geom_point(alpha=0.3) + 
  geom_vline(xintercept = 0) + geom_hline(yintercept = 0)+geom_smooth(method = "loess", se=FALSE)+ggtitle('BOOSTING')
print(x4)
cat('R squared value for Single Tree:', tree_r2*100, '%\n' )
cat('R squared value for Random Forest:', rf_r2*100, '%\n' )
cat('R squared value for GBM:', boost_r2*100, '%\n' )
################################################################
svm_model <- svm(Net.Profit....Loss. ~ . ,data=train)
summary(svm_model)
pred.values <- predict(svm_model,test)
svm_preds_df <- data.frame(cbind(actuals=test$Net.Profit....Loss., predicteds=pred.values))
head(svm_preds_df)
tune.result <- tune.svm(Net.Profit....Loss.~. , data=train,
                        kernal='radial', cost = c(1:10), gamma=c(0.5:3))
summary(tune.result)
new.model <- svm(Net.Profit....Loss. ~ . ,data=train,cost=6,gamma=0.5)
pred.values <- predict(new.model,test)
svm_preds_df <- data.frame(cbind(actuals=test$Net.Profit....Loss., predicteds=pred.values))
svm_sse <- sum( (svm_preds_df$predicteds - svm_preds_df$actuals)^2 )
svm_sst <- sum( (mean(eco$Net.Profit....Loss.) - svm_preds_df$actuals)^2) 
svm_r2 <- 1- rf_sse/rf_sst
svm_r2*100
