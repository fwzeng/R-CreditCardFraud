
# load packages

library(dplyr)
library(keras)

setwd("C:/Users/fwzen/Documents/Learning/R/Code/CreditCardFraud/")

# Read data

data <- read.csv('../../Input_data/creditcard.csv')
data <- select(data, -c('Time'))

# Separate the label and the features

class_true <- data$Class
class_true <- as.data.frame(class_true)
colnames(class_true)[1] <- 'class_true'
data_x <- select(data, -c('Class')) 

# Standardize the feature "Amount". All other features are already standardized. 
data_x$Amount <- (data_x$Amount-mean(data_x$Amount)) / sd(data_x$Amount)

# data_x_pca <- prcomp(data_x)
# summary(data_x_pca)

# Set hyperparameters for the model

optimizer_selected <- 'adam'
n_epochs <- 200
mini_batch_size <- 64
n_examples <- nrow(data_x)
iter_per_epoch <- ceiling(n_examples/mini_batch_size)

# Build the keras autoencoder model

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 5, activation = "tanh", input_shape = ncol(data_x)) %>%
  layer_batch_normalization() %>% 
  layer_dense(units = 5, activation = "tanh") %>%
  layer_dense(units = 5, activation = "tanh") %>%
  layer_dense(units = 5, activation = "tanh") %>%
  layer_dense(units = 5, activation = "tanh") %>%
  layer_dense(units = ncol(data_x))

print(summary(model))
