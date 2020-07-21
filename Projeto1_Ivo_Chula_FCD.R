# Projeto com Feedback - Formação Cientista de Dados
# Curso de Big Data Analytics com R e Microsoft Azure Machine Learning

"ip: ip address of click.
app: app id for marketing.
device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
os: os version id of user mobile phone
channel: channel id of mobile ad publisher
click_time: timestamp of click (UTC)
attributed_time: if user download the app for after clicking an ad, this is the time of the app download
is_attributed: the target that is to be predicted, indicating the app was downloaded
Note that ip, app, device, os, and channel are encoded.

The test data is similar, with the following differences:
  
click_id: reference for making predictions
is_attributed: not included"

setwd("C:/BigDataRAzure/Projetos_feedback/Projeto1")

library(dplyr)
library(tidyr)
library(data.table)
library(ROSE)
library(corrplot)
library(randomForest)
library(caret)
library(C50)

# Carregar arquivo "train_sample"
# Arquivo Principal de treino +\- 7Gb  

df = fread("train_sample.csv")
View(df)

# Análise Exploratória

View(df)
class(df$attributed_time) # "POSIXct" "POSIXt" / Formato de data
prop.table(table(df$is_attributed)) # 99,8 % de valores 0  
str(df)
summary(df)
sum(is.na(df$attributed_time)) # 99773 Valores NA 
dim(df) # 100.000 Valores / Linhas 
sum(df$is_attributed) # 227 linhas com valores 1 

# Retirar coluna attributed time pois essa informação não será útil devido a já sabermos que o download foi feito
# na variavel is_attributed

df <- select(df, -attributed_time)

# Separar Valores das datas

df <- df %>% 
  separate(click_time, sep = "-" , into = c( "Click_Year", "Click_Month", "Click_Day" )) %>%
  separate(Click_Day, sep = " ", into = c("Click_Day", "hour")) %>%
  separate(hour, sep = ":", into = c("Hour", "Minutes", "Seconds")) %>%
  select(-Click_Year, -Click_Month, -Click_Day)

View(df)

# Converter tipos de dados

df <- as.data.frame(apply(df, 2, as.numeric)) # valores numéricos
df$is_attributed <- as.factor(df$is_attributed) # fator
str(df)

sum(is.na(df))

# Balancear dados, criar dados sintéticos

prop.table(table(df$is_attributed))
df_rose <- ROSE(is_attributed ~ ., data = df, seed = 1)$data
prop.table(table(df_rose$is_attributed))

# Feature Selection 

importancia = randomForest(df_rose$is_attributed ~ ., data = df_rose, ntree = 100, nodesize = 10, importance = T)
varImpPlot(importancia)
importancia

df_rose <- select(df_rose, -Seconds)
View(head(df_rose))

# Normalização dos dados

scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

numeric.vars <- c("ip", "app", "os", "device", "channel", "Hour", "Minutes")
df_norm <- scale.features(df_rose, numeric.vars)

View(head(df_norm))

# Modelo preditivo com randomForest      

modelo_RF = randomForest(df_norm$is_attributed ~ ., data = df_norm)

# TESTE 
# carregar dados de teste / Limpar dados de teste

df_teste = fread("test.csv")

df_teste <- df_teste %>% 
  separate(click_time, sep = "-" , into = c( "Click_Year", "Click_Month", "Click_Day" )) %>%
  separate(Click_Day, sep = " ", into = c("Click_Day", "hour")) %>%
  separate(hour, sep = ":", into = c("Hour", "Minutes", "Seconds")) %>%
  select(-Click_Year, -Click_Month, -Click_Day)

df_teste <- select(df_teste, -Seconds)
df_teste <- as.data.frame(apply(df_teste, 2, as.numeric)) # valores numéricos

str(df_teste)
summary(df_teste$Click_Day)

df_sub_teste <- df_teste[sample(1:nrow(df_teste), 5000,replace=FALSE),]


View(df_sub_teste)
str(df_sub_teste)

df_norm_teste <- scale.features(df_sub_teste, numeric.vars)

View(df_norm_teste)
str(df_norm_teste)

# Realizar predição com randomForest

previsao_RF = predict(modelo_RF, df_norm_teste)
View(previsao_RF)
prop.table(table(previsao_RF))

# Confusion Matrix

View(head(df_teste))
View(head(df_norm))

confusionMatrix(table(data = df_norm_teste, reference = previsao_RF), positive = '1')

?confusionMatrix
confusionMatrix(df_norm_teste, previsao_RF, positive = '1')

# AUC 

roc.curve(df_rose$is_attributed, modelo_RF, plotit = T, col = "red")

?roc.curve
