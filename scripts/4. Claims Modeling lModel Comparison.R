#data wrangling
library(tidyverse)
library(magrittr)
library(lazyeval)

#Ml packages
library(caret)
library(Matrix)
library(lightgbm)
library(methods)

#get data

load("./data/datasets.RData") %>% print()

#grid models
load("./data/gridModel.RData") %>% print()
grid.preditions <- pred
grid.model<-bst.model

#random search models
load("./data/randomModel.RData") %>% print()
random.preditions <- pred
random.model<-bst.model


#Bayes models
load("./data/bayesModel.RData") %>% print()
bayes.preditions <- pred
bayes.model<-bst.model


plot.data <- data.frame(obs=test.target,grid.preditions=grid.preditions,random.preditions=random.preditions,bayes.preditions=bayes.preditions)

plot.data.p<-plot.data  %>%
  gather(key,value, obs, grid.preditions,random.preditions,bayes.preditions) 

p1<-ggplot(plot.data.p,aes(x=log(value), fill=key)) + geom_density(alpha=0.25,colour = "white")
p2<-ggplot(plot.data.p,aes(x=log(value), fill=key)) + geom_histogram(alpha=0.25,colour = "white")
p3<-ggplot(plot.data.p,aes(x=key, y=log(value), fill=key)) + geom_boxplot() + coord_flip() +theme_minimal()

grid.arrange(p0, p1, p2, p3, ncol=2)

plot.data$cut <- cut(test.target,breaks=quantile(test.target, probs=seq(0,1, by=0.05), na.rm=TRUE),include.lowest=TRUE)
plot.data.long<- plot.data %>% group_by(cut) %>% 
  summarise(obs=sum(obs),grid.preditions=sum(grid.preditions),
            random.preditions=sum(random.preditions),
            bayes.preditions=sum(bayes.preditions)) %>%  
  gather(key,value, obs, grid.preditions,random.preditions,bayes.preditions) 

ggplot(plot.data.long , aes(x=cut,y=value)) + geom_col(alpha=0.7,aes(fill = key,color=key),position = "dodge") + labs(title = "Quantile Plot", subtitle = "Comparing observed vs predicted values", x = "Quantile", y="Sum")+ theme_light() +theme(axis.text.x  = element_text(angle=45, vjust=0.5, size=6)) 

loss.functions <- function(obs,preds){
  cor <-cor(preds,obs)
  mae <- mean(abs(preds-obs))
  rmse <-sqrt(mean(abs(preds-obs)^2))
  return(c(cor,mae,rmse))
  }

temp.loss<- rbind(loss.functions(test.target,grid.preditions),
      loss.functions(test.target,random.preditions),
      loss.functions(test.target,bayes.preditions))

plot.loss<- data.frame(model= c("grid","random","bayes"),correlation=temp.loss[,1],mae=temp.loss[,2],rmse=temp.loss[,3])
ggplot(data=plot.loss, aes(x=mae,y=rmse) )+ geom_point(aes(size=correlation,color=correlation))+ scale_colour_gradient(low = "blue") +geom_text(aes(label=model), size=3,hjust=1.5, vjust=0.5) + xlim(1215,1250)+ labs(title = "Comparing Models", subtitle = "Multiple loss metrics")+ theme_light() 
