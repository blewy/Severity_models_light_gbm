#data wrangling
library(data.table)
library(tidyverse)
library(magrittr)
library(lazyeval)
library(readr)

#overall data analysis
library(Amelia)
library(corrplot)
library(gridExtra)
library(Hmisc)

#Ml packages
library(caret)
library(Matrix)
library(lightgbm)
library(methods)

#bayes
library(rBayesianOptimization)

#aux_ function
source("./scripts/aux_functions.R")

#get data

load("./data/datasets.RData") %>% print()
#attach(train.dataset)

dtrain <- lgb.Dataset(as.matrix(train.dataset), label = train.target)
dvalid <- lgb.Dataset(as.matrix(validation.dataset), label = validation.target)
dtest <- lgb.Dataset(as.matrix(test.dataset), label = test.target)

dtrainvalid <- lgb.Dataset(as.matrix(cbind(train.dataset,validation.dataset)), 
                           label = cbind(train.target,validation.target))

r.Grid <- expand.grid(
  metric = "huber"
  , min_data_in_leaf = seq(1,4,by=1)
  , feature_fraction = seq(0.5,0.8,by=0.05)
  , bagging_fraction =  seq(0.5,0.8,by=0.05)
  , bagging_freq = c(0)
  , lambda_l1 =  seq(0.5,1,by=0.1)
  , lambda_l2 = seq(0.5,1,by=0.1)
  , num_leaves = seq(5,10,by=1)
  , learning_rate = c(0.05,0.1,0.2,0.5)
  , max_depth = seq(4,10,by=2)
)

dim(r.Grid)
tune.length=50
index<-sample(1:nrow(r.Grid),tune.length)

Grid <- r.Grid[index,]

results<- data.frame(error=rep(0,nrow(Grid)),best.iter=rep(0,nrow(Grid)))
for(i in 1:nrow(Grid)) {
  params = list(
    metric = Grid[i,"metric"]
    , min_data_in_leaf = Grid[i,"min_data_in_leaf"]
    , feature_fraction = Grid[i,"feature_fraction"]
    , bagging_fraction = Grid[i,"bagging_fraction"]
    , bagging_freq = Grid[i,"bagging_freq"]
    , lambda_l1 = Grid[i,"lambda_l1"]
    , lambda_l2 = Grid[i,"lambda_l2"]
    , num_leaves = Grid[i,"num_leaves"]
    , learning_rate = Grid[i,"learning_rate"]
    , max_depth = Grid[i,"max_depth"]
  )
  cat("=== Iteration: ",i,"==== \n")
  
  lgb.model <- lgb.train(
    params = params
    , objective ="regression"
    , data = dtrain
    , valids = list(test = dvalid)
    , num_threads = 2
    , nrounds = 300
    , early_stopping_rounds = 20
    , eval_freq = 100
    , verbose= 0
    , boosting = "gbdt"
  )
  
  #Results
  bst.iter <- if_else(lgb.model$best_iter==-1,300,lgb.model$best_iter)
  results[i,"best.iter"] <- bst.iter
  results[i,"error"] <- lgb.model$record_evals$test$huber$eval[bst.iter]
  
}

View(as.data.frame(cbind(Grid, results)))

#Best parameters
bst<-which.min(results$error)

Grid[bst,]
bst.iteration<-results$best.iter[bst]
bst.params = list(
  metric = Grid[bst,"metric"]
  , min_data_in_leaf = Grid[bst,"min_data_in_leaf"]
  , feature_fraction = Grid[bst,"feature_fraction"]
  , bagging_fraction = Grid[bst,"bagging_fraction"]
  , bagging_freq = Grid[bst,"bagging_freq"]
  , lambda_l1 = Grid[bst,"lambda_l1"]
  , lambda_l2 = Grid[bst,"lambda_l2"]
  , num_leaves = Grid[bst,"num_leaves"]
  , learning_rate = Grid[bst,"learning_rate"]
  , max_depth = Grid[bst,"max_depth"]
)

# Best Model
bst.model <- lgb.train(
  params = bst.params
  , objective ="regression"
  , data = dtrain
  , nrounds = bst.iteration
  , num_threads = 2
  , eval_freq = 300
  , verbose= 1
  , boosting = "gbdt"
)



var.imp<-lgb.importance(bst.model, percentage = TRUE)
str(var.imp)
var.imp_top<-var.imp %>% top_n(20,wt=Gain)
ggplot(var.imp_top , aes(x=reorder(Feature, Gain),y=Gain)) + geom_col(fill ="red" , colour ="white",alpha=0.7 ) + coord_flip() + labs(title = "Feature Importance", subtitle = "Top 20", x = "Feature")+ theme_light()


plot_learning<- data.frame(iter=1:lgb.model$best_iter,weight_loss=as.numeric(lgb.model$record_evals$test$l2$eval[1:lgb.model$best_iter]))
ggplot(plot_learning, aes(x=iter,y=weight_loss)) + geom_line(color="red") + labs(title = "Learning Rate", subtitle = "L2 loss", x = "Iterations", y="L2 loss") 


pred <- predict(bst.model, as.matrix(test.dataset))

cat(".Correlation.")
cor(pred,test.target)
cat(".MAE.")
mean(abs(pred-test.target))
cat(".RMSE.")
sqrt(mean(abs(pred-test.target)^2))

#saves model and predictions
save(pred,bst.model,file="./data/randomModel.RData")


plot.data <- data.frame(obs=test.target,preds=pred)
p0<-ggplot(data=plot.data, aes(x=obs,y=preds)) + geom_point() +geom_smooth()

plot.data.p<-plot.data  %>%
  gather(key,value, obs, preds) 

p1<-ggplot(plot.data.p,aes(x=log(value), fill=key)) + geom_density(alpha=0.25,colour = "white")
p2<-ggplot(plot.data.p,aes(x=log(value), fill=key)) + geom_histogram(alpha=0.25,colour = "white")
p3<-ggplot(plot.data.p,aes(x=key, y=log(value), fill=key)) + geom_boxplot() + coord_flip() +theme_minimal()

grid.arrange(p0, p1, p2, p3, ncol=2)

plot.data$cut <- cut(test.target,breaks=quantile(test.target, probs=seq(0,1, by=0.05), na.rm=TRUE),include.lowest=TRUE)
plot.data.long<- plot.data %>% group_by(cut) %>% summarise(obs=sum(obs),preds=sum(preds)) %>%  gather(key,value, obs, preds) 
ggplot(plot.data.long , aes(x=cut,y=value)) + geom_col(alpha=0.7,aes(fill = key,color=key),position = "dodge") + labs(title = "Quantile Plot", subtitle = "Comparing observed vs predicted values", x = "Quantile", y="Sum")+ theme_light() +theme(axis.text.x  = element_text(angle=45, vjust=0.5, size=6)) 



