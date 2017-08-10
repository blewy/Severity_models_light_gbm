#data wrangling
library(tidyverse)
library(magrittr)
library(lazyeval)

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

#dtrainvalid <- lgb.Dataset(as.matrix(cbind(train.dataset,validation.dataset)), 
#                           label = cbind(train.target,validation.target))

lgbm_bayes <- function( 
                          min_data_in_leaf 
                        , feature_fraction 
                        , bagging_fraction 
                        , bagging_freq 
                        , lambda_l1 
                        , lambda_l2
                        , num_leaves
                        , learning_rate
                        , max_depth ) 
{
#Reload the dataset every time we call this fuction  
dtrain <- lgb.Dataset(as.matrix(train.dataset), label = train.target)
dvalid <- lgb.Dataset(as.matrix(validation.dataset), label = validation.target)  
nrounds=300
lgb.model <- lgb.train(
      min_data_in_leaf = min_data_in_leaf
    , feature_fraction = feature_fraction
    , bagging_fraction = bagging_fraction
    , bagging_freq = bagging_freq
    , num_leaves = num_leaves
    , learning_rate = learning_rate
    , max_depth = max_depth
    , metric ="huber"
    , objective ="regression"
    , data = dtrain
    , valids = list(test = dvalid)
    , num_threads = 2
    , nrounds = nrounds
    , early_stopping_rounds = 20
    , verbose= 0
    , boosting = "gbdt"
  )
  #best Iteration
  bst.iter <- if_else(lgb.model$best_iter==-1,nrounds,lgb.model$best_iter)
  #Score RMSE
  Score <- unlist(lgb.model$record_evals$test$huber$eval[bst.iter])
  # Predictions
  Pred <- predict(lgb.model,as.matrix(validation.dataset))
  #we need to unload lgbm or rBayes will crash and burn
  suppressMessages(lgb.unloader(wipe = TRUE))
  # return the negative Score, becouse BayesianOptimization does a maximization
  return(list(Score =-Score, Pred = Pred))
}

#
#----------------  Baysisn hyper-parameter optimization using initila valies form grid search------------------
#

load("./data/gridResults.RData") %>% print()
View(grid.results)
names(grid.results)
best.iter <-grid.results$best.iter
init_grid_dt <- grid.results  %>% mutate(Value=-error) %>% select(-metric,-best.iter,-error)
View(init_grid_dt)
names(init_grid_dt)
OPT_Res <- BayesianOptimization(lgbm_bayes,bounds = list(
                                            min_data_in_leaf = c(2L,4L)
                                          , feature_fraction = c(0.5,0.8)
                                          , bagging_fraction =  c(0.5,0.8)
                                          , bagging_freq = c(0L,10L)
                                          , lambda_l1 =  c(0.5,1)
                                          , lambda_l2 = c(0.5,1)
                                          , num_leaves = c(5L,10L)
                                          , learning_rate = c(0.05,0.5)
                                          , max_depth = c(4L,10L)
                                          )
                                , init_grid_dt = init_grid_dt
                                , init_points = 10
                                , n_iter = 20
                                , acq = "ucb"
                                , kappa = 2.576
                                , eps = 0.0
                                , verbose = TRUE)
OPT_Res$Best_Par
OPT_Res$Best_Value
history <- as.data.frame(OPT_Res$History)
library(knitr)
history %>% kable()


params = list(
  min_data_in_leaf = OPT_Res$Best_Par["min_data_in_leaf"]
  ,feature_fraction = OPT_Res$Best_Par["feature_fraction"]
  ,bagging_fraction = OPT_Res$Best_Par["bagging_fraction"]
  ,bagging_freq = OPT_Res$Best_Par["bagging_freq"]
  ,lambda_l1 = OPT_Res$Best_Par["lambda_l1"]
  ,lambda_l2 = OPT_Res$Best_Par["lambda_l2"]
  ,num_leaves = OPT_Res$Best_Par["num_leaves"]
  ,learning_rate = OPT_Res$Best_Par["learning_rate"]
  ,max_depth = OPT_Res$Best_Par["max_depth"]
)

#Reload the dataset every time we call this fuction  
dtrain <- lgb.Dataset(as.matrix(train.dataset), label = train.target)
dvalid <- lgb.Dataset(as.matrix(validation.dataset), label = validation.target)  
nrounds=300
lgb.model <- lgb.train(
    params = params
  , metric ="huber"
  , objective ="regression"
  , data = dtrain
  , valids = list(test = dvalid)
  , num_threads = 2
  , nrounds = nrounds
  , early_stopping_rounds = 20
  , verbose= 1
  , boosting = "gbdt"
)

bst.iter <- if_else(lgb.model$best_iter==-1,nrounds,lgb.model$best_iter)

# Best Model
bst.model <- lgb.train(
  params = params
  , objective ="regression"
  , data = dtrain
  , nrounds = bst.iter
  , num_threads = 2
  , eval_freq = 100
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
save(pred,bst.model,file="./data/bayesModel.RData")


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
