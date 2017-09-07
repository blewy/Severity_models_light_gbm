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

#aux_ function
source("./scripts/aux_functions.R")

#Unzip the zip file into movies directly of the current working directory.
#unzip("./data/train.csv.zip", exdir = "./data/train")
#dir("./data/train")

#unzip("./data/test.csv.zip", exdir = "./data/test")
#dir("./data/test")


#User Dataset
train <- read.table("./data/train/train.csv", sep=",", header=TRUE)
head(train)
dim(train)
str(train, list.len=ncol(train))

test <- read.table("./data/test/test.csv", sep=",", header=TRUE)

#Response Variable
response_name<-setdiff(names(train),names(test))
train_target <- train[,response_name]
#train[,response_name]<-NULL
response_name


names(train) <- make.names(names(train))

train_id <- train$id
train$id <- NULL

# Heuristic for indentifiers to possibly ignore.
train  %>%
  sapply(function(x) x %>% unique() %>% length()) %>%
  equals(nrow(train))%>%
  which() %>%
  names() %>%
  print() ->
  ids


# Review a random sample of observations.
View(sample_n(train, size=6))


# Overall View of the dataset
describe(train) 

cat("\nDistribuition of Claims: ")

p1<-ggplot(train, aes(x=(loss),fill = cut(loss, 100))) + geom_histogram(binwidth=10,show.legend = FALSE) + xlim(0,40000)+ labs(title = "Loss Distribution", subtitle = "Raw Data")+ ylab("")+theme_light() 

library(plotly)

ggplotly(ggplot(train, aes(x=(loss))) + geom_histogram(binwidth=10) + xlim(0,50000))

#log loss
ggplotly(ggplot(train, aes(x=log(loss))) + geom_histogram(binwidth=0.1))
p2<-ggplot(train, aes(x=log(loss),fill = cut(loss, 100))) + geom_histogram(binwidth=0.1,show.legend = FALSE)+ xlim(3,12)+ labs(title = "", subtitle = "Log Loss")+ ylab("")+ theme_light() 

grid.arrange(p1, p2, ncol=2)


# Check for duplicated rows.
cat("The number of duplicated rows are", nrow(train) - nrow(unique(train)))


# ----------------- Missings ------------------

# Identify variables with only missing values.
train %>%
  sapply(function(x) x %>% is.na %>% sum) %>%
  equals(nrow(train)) %>%
  which() %>%
  names() %>%
  print() ->
  missing

#Identify a threshold above which proportion missing is fatal.
missing.threshold <- 0.7

# Identify variables that are mostly missing.
train %>%
  sapply(function(x) x %>% is.na() %>% sum()) %>%
  is_weakly_greater_than(missing.threshold*nrow(train)) %>%
  which() %>%
  names() %>%
  print() ->
  mostly

#library(Amelia)
#missmap(train, legend = TRUE, col = c("wheat","darkred"), main="Missing Plot",
#        y.cex = 0.8, x.cex = 0.8)

# Identify variables that have a single value.
train %>%
  sapply(function(x) all(x == x[1L])) %>%
  which() %>%
  names() %>%
  print() ->
  constants 


# Identify a threshold above which we have too many levels.
levels.threshold <- 20

# Identify variables that have too many levels.
train %>%
  sapply(is.factor) %>%
  which() %>%
  names() %>%
  sapply(function(x) train %>% extract2(x) %>% levels() %>% length()) %>%
  is_weakly_greater_than(levels.threshold) %>%
  which() %>%
  names() %>%
  print() ->
  too.many.levels


# Getting factor vars
factor_var <- names(train)[which(sapply(train, is.factor))]
cleanVars.factor.Train<-setdiff(factor_var, too.many.levels)

# Identify the categoric variables by index.
train %>%
  sapply(is.factor) %>%
  which() %>%
  print() ->
  cat.index

train_fact <-train[,cleanVars.factor.Train]
train_fact$response <- train_target


## Box-plots & Barplots for the categorical features
doPlots(train_fact, fun = plotBox,lab=log(train_fact$response), ii = 1:4, ncol = 2)

doPlots(train_fact, fun = plotHist, ii = 1:4, ncol = 2)

#--- Feature engineering for categoriacal vars ------
categoricals<- names(train_fact %>% select(-response))
all_categorical.data <- rbind(train_fact %>% select(categoricals),test %>% select(categoricals))

# this is a trick to create the same levels between test and train data set
# ehere the factor variables/features are parsed to integer
for (factor in categoricals) {
  if (class(all_categorical.data[[factor]])=="character"| class(all_categorical.data[[factor]])=="factor" ) {
    cat("VARIABLE : ",factor,"\n")
    levels <- unique(all_categorical.data[[factor]])
    train_fact[[factor]] <- as.integer(factor(train_fact[[factor]], levels=levels))
  }
}

# Calculate averages by this variable
var_value<- c("response") 

#limpar a tabela final
features_factor<-NULL
features_factor<-train_fact
for(var.factor in cleanVars.factor.Train){
  #Temp. Table
  for(var.value in var_value){
    temp.data<-NULL
    cat("==",var.factor,"by",var.value, " : \n")
      x = aggregate_by_var(train_fact,var.factor,var.value)
      if ( length(temp.data) == 0)  {temp.data <- x}
      else {temp.data <-rbind(temp.data,x)}
      cat("...","\n")
      #saves statistics  
      temp.data %>% as.data.frame()
      #write.table(temp.data, file = paste0("", var.value, "_by_", paste(var.factor, collapse = "_"), ".csv"), 
      #            row.names = F, col.names = T, sep = ";", dec = ",")
    }
  cat("== Join ==")
  features_factor <- features_factor %>% left_join(temp.data)
}
#View(features_factor)
names(features_factor)

features.factor.train<-setdiff(names(features_factor), names(train_fact))

#Categorical vars usign feature engeneering
train_categorical <-features_factor[,features.factor.train]
#View(train_categorical)
#Categorical vars usign to.int trick to be used on xgboost or lgbm
train_categorical.int <-features_factor[,categoricals]


#-----------  Continous variables  analysis  -------------
numeric_var <- names(train)[which(sapply(train, is.numeric))]
train_numeric <-train[,numeric_var]

## Density plots for numeric variables.
doPlots(train_numeric, fun = plotDen,lab=(train_numeric$loss), ii = 1:4, ncol = 2)

## Scatter plot.
doPlots(train_numeric, fun = plotScatter,lab=log(train_numeric$loss), ii = 1:4, ncol = 2)


# input missings
pp.numeric <- preProcess(train[,numeric_var],method=c("medianImpute"))
train_num.inputed <- predict(pp.numeric,train_numeric)

correlations <- cor(train_num.inputed)
corrplot(correlations, method="square", order="hclust")

nzv <- nearZeroVar(train_num.inputed, saveMetrics= FALSE)
to.remove<-names(train_num.inputed)[nzv]


#Find high correlated variables
summary(correlations[upper.tri(correlations)])
highlyCorDescr <- findCorrelation(correlations, cutoff = .85)
cat("\nHighly correlated variables above 0.85 : ", names(train[,numeric_var])[highlyCorDescr])

to.remove<- c(names(train_num.inputed)[highlyCorDescr])
calc.num.Train<-setdiff(names(train_numeric), to.remove)

train_numeric <-train_num.inputed[,calc.num.Train]
names(train_numeric)
names(train_fact)
final_train_data <-cbind(train_categorical,train_numeric)
final_train_data.int <-cbind(train_categorical.int,train_numeric)

# Identify the categoric variables by index.
final_train_data.int %>%
  sapply(is.integer) %>%
  which() %>%
  print() ->
  cat.index

#Remove target from data.set
final_train_data$loss<-NULL
final_train_data.int$loss<-NULL

View(final_train_data.int)
names(final_train_data)


# prep data set for test/training
set.seed(1234)
train.test.split <- sample(3
                           , nrow(final_train_data)
                           , replace = TRUE
                           , prob = c(0.7,0.15, 0.15))

sum(prop.table(table(train.test.split)))

train.dataset <- final_train_data[train.test.split == 1, ]
train.target <- train_target[train.test.split == 1]
validation.dataset <- final_train_data[train.test.split == 2, ]
validation.target <- train_target[train.test.split == 2]
test.dataset <- final_train_data[train.test.split == 3, ]
test.target <- train_target[train.test.split == 3]

save(train.dataset,train.target,validation.dataset,validation.target,test.dataset,test.target,pp.numeric,file="./data/datasets.RData")

# save dataset with int features for categorical vars
train.dataset.int <- final_train_data.int[train.test.split == 1, ]
validation.dataset.int <- final_train_data.int[train.test.split == 2, ]
test.dataset.int <- final_train_data.int[train.test.split == 3, ]

save(train.dataset.int,validation.dataset.int,test.dataset.int,categoricals,cat.index,file="./data/datasets.int.RData")

#train.dataset.int[!duplicated(lapply(train.dataset.int, summary))]
#duplicated(t(train.dataset.int))


