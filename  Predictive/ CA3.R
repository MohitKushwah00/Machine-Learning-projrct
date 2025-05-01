# penguins dataset

library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(class)
library(palmerpenguins)

data("penguins")

# Preprocessing  the data
penguins <- penguins %>%
  drop_na() %>%
  mutate(species = as.factor(species),
         island = as.factor(island),
         sex = as.factor(sex))

# Splitting  data into training and testing 
set.seed(123)
train_index <- createDataPartition(penguins$species, p = 0.8, list = FALSE)
train_data <- penguins[train_index, ]
test_data <- penguins[-train_index, ]

# KNN Model
knn_model <- train(species ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g,
                   data = train_data,
                   method = "knn",
                   tuneGrid = expand.grid(k = c(1, 3, 5, 7)),
                   trControl = trainControl(method = "cv", number = 10))

# Decision Tree Model
tree_model <- train(species ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g,
                    data = train_data,
                    method = "rpart",
                    trControl = trainControl(method = "cv", number = 10))

# Naive Bayes Model
nb_model <- train(species ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g,
                  data = train_data,
                  method = "naive_bayes",
                  trControl = trainControl(method = "cv", number = 10))

# SVM Model
svm_model <- train(species ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g,
                   data = train_data,
                   method = "svmRadial",
                   trControl = trainControl(method = "cv", number = 10))

# Evaluate Models
knn_pred <- predict(knn_model, test_data)
tree_pred <- predict(tree_model, test_data)
nb_pred <- predict(nb_model, test_data)
svm_pred <- predict(svm_model, test_data)

# Confusion Matrices
knn_cm <- confusionMatrix(knn_pred, test_data$species)
tree_cm <- confusionMatrix(tree_pred, test_data$species)
nb_cm <- confusionMatrix(nb_pred, test_data$species)
svm_cm <- confusionMatrix(svm_pred, test_data$species)

# Calculate and print accuracies
accuracies <- data.frame(
  Model = c("KNN", "Decision Tree", "Naive Bayes", "SVM"),
  Accuracy = c(knn_cm$overall['Accuracy'], 
               tree_cm$overall['Accuracy'], 
               nb_cm$overall['Accuracy'], 
               svm_cm$overall['Accuracy'])
)

print(accuracies)
# linear regression model
lm_model <- lm(body_mass_g ~ bill_length_mm + bill_depth_mm + flipper_length_mm + species, data = train_data)
lm_pred <- predict(lm_model, test_data)

# Evaluation
lm_r_squared <- summary(lm_model)$r.squared
print(paste("Linear Regression R-squared:", lm_r_squared))




# some plots

library(ggplot2)
library(caret)



model_accuracies <- data.frame(
  Model = c("KNN", "Decision Tree", "Naive Bayes", "SVM"),
  Accuracy = c(knn_model$results$Accuracy[which.max(knn_model$results$k)],
               max(tree_model$results$Accuracy),
               max(nb_model$results$Accuracy),
               max(svm_model$results$Accuracy))
)

ggplot(model_accuracies, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Accuracies", x = "Model", y = "Accuracy") +
  theme_minimal()

