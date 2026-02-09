install.packages("readr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("mice")
install.packages("corrplot")
install.packages("car")
install.packages("rpart")
install.packages("randomForest")
install.packages("e1071")
install.packages("caret")
library(readr)
library(dplyr)
library(ggplot2)
library(mice)
library(ggplot2)
library(corrplot)
library(car)
library(rpart)
library(randomForest)
library(e1071)
library(caret)


phones <- read_csv("used_device_data.csv")
head(phones)

str(phones)
summary(phones)
colnames(phones)

#checking frequency distribution
table(phones$device_brand)
table(phones$os)
table(phones$`4g`)
table(phones$`5g`)

#summary statistics
summary(phones[, c("screen_size", "rear_camera_mp", "front_camera_mp",
                   "internal_memory", "ram", "battery", "weight",
                   "release_year", "days_used", "normalized_used_price",
                   "normalized_new_price")])
#above or below
summary(phones[, sapply(phones, is.numeric)])

#Data Exploration
#categorical variable viz
#Bar plots for categorical variables
barplot(table(phones$os), main="OS Distribution", col="lightcoral")
barplot(table(phones$device_brand), main="Device Brand Distribution", las=2, cex.names=0.7)

#OS distribution with percentage labels
os_counts <- table(phones$os)
os_pct <- round(prop.table(os_counts) * 100, 1)  # percentage rounded to 1 decimal
bar_heights <- barplot(os_counts, main="OS Distribution (%)", col="lightcoral", ylab="Count")
text(x = bar_heights, y = os_counts, labels = paste0(os_pct, "%"), pos = 3)  # pos=3 puts label above bar

#Device brand distribution with percentage labels
brand_counts <- table(phones$device_brand)
brand_pct <- round(prop.table(brand_counts) * 100, 1)
bar_heights <- barplot(brand_counts, main="Device Brand Distribution (%)", las=2, cex.names=0.7, ylab="Count")
text(x = bar_heights, y = brand_counts, labels = paste0(brand_pct, "%"), pos = 3)

#numerical variable viz
#Histograms
hist(phones$ram, main="Distribution of RAM", xlab="RAM (GB)", col="lightblue")
hist(phones$rear_camera_mp, main="Distribution of Rear Camera", xlab="MP", col="lightgreen")

#Boxplots
boxplot(phones$screen_size, main="Screen Size", ylab="Inches")
boxplot(phones$internal_memory, main="Internal Memory", ylab="GB")

#Relationships Between Variables
#Scatter plot and correlation
plot(phones$ram, phones$normalized_used_price,
     main="RAM vs Normalized Used Price",
     xlab="RAM (GB)", ylab="Normalized Used Price",
     pch=19, col="violet")

#Scatter plot of Rear Camera vs Normalized Used Price
plot(phones$rear_camera_mp, phones$normalized_used_price,
     main="Rear Camera MP vs Normalized Used Price",
     xlab="Rear Camera (MP)",
     ylab="Normalized Used Price",
     pch=19, col="steelblue")


#Checking Missing values
#Count missing values per column
colSums(is.na(phones))
#Calculate total rows
total_rows <- nrow(phones)

#Calculate percentage of missing values per column
missing_percent <- colSums(is.na(phones)) / total_rows * 100
missing_percent


##install.packages("mice")
##library(mice)

#Missing data pattern
md.pattern(phones)

#handling missing values in numeric
#list of numeric columns to impute
cols_to_impute <- c("rear_camera_mp", "front_camera_mp", "internal_memory", 
                    "ram", "battery", "weight")
#Loop through each column and replace NA with median
for (col in cols_to_impute) {
  median_value <- median(phones[[col]], na.rm = TRUE)  # compute median ignoring NA
  phones[[col]][is.na(phones[[col]])] <- median_value  # replace NA with median
}
#Check if any NAs remain
sapply(phones[cols_to_impute], function(x) sum(is.na(x)))

#Checking ZERO values
#Count zeros in each column
zero_counts <- sapply(phones, function(x) sum(x == 0, na.rm = TRUE))

#view Zero counts
zero_counts


# Step 4: Predictor Analysis and Relevancy
#library(ggplot2)
#library(corrplot)
#1. Correlation Analysis
numeric_vars <- phones[, sapply(phones, is.numeric)]
cor_matrix <- cor(numeric_vars, use = "complete.obs")
print(cor_matrix)

#Visualize correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

#2. Scatter Plots for key predictors vs. target variable
key_predictors <- c("ram", "internal_memory", "rear_camera_mp", "battery")

for (var in key_predictors) {
  ggplot(phones, aes_string(x = var, y = "normalized_used_price")) +
    geom_point(color = "blue", alpha = 0.6) +       #points for context
    ggtitle(paste(var, "vs Normalized Used Price")) +
    xlab(var) + ylab("Normalized Used Price") +
    theme_minimal() -> p
  print(p)
}


#3. Check multicollinearity among predictors
predictor_matrix <- cor(phones[, key_predictors], use = "complete.obs")
print(predictor_matrix)

#Check multicollinearity with VIF
#install.packages("car")
#library(car)

#Fit a linear model with your target
fit <- lm(normalized_used_price ~ ram + internal_memory + rear_camera_mp + battery, data = phones)

#Compute VIF
vif(fit)


#Step5: Data Transformation
#Handle zero values in numeric predictors
#front_camera_mp has 39 zeros - replace them with median
median_front_camera <- median(phones$front_camera_mp[phones$front_camera_mp != 0], na.rm = TRUE)
phones$front_camera_mp[phones$front_camera_mp == 0] <- median_front_camera

#Verify zero values
zero_summary <- sapply(phones[, c("screen_size", "rear_camera_mp", "front_camera_mp",
                                  "internal_memory", "ram", "battery", "weight",
                                  "release_year", "days_used", "normalized_used_price",
                                  "normalized_new_price")], function(x) sum(x == 0, na.rm = TRUE))
print(zero_summary)

#Confirm no missing values remain
missing_summary <- data.frame(Column = colnames(phones),
                              Missing_Count = colSums(is.na(phones)),
                              Missing_Percent = round(colSums(is.na(phones)) / nrow(phones) * 100, 2))
print(missing_summary)

#6. Data Partitioning
set.seed(123)

#decide the proportion for the training set
train_ratio <- 0.7

#Generate random indices for the training set
train_indices <- sample(1:nrow(phones), size = train_ratio * nrow(phones))

#create training and testing datasets
train_data <- phones[train_indices, ]
test_data  <- phones[-train_indices, ]

#Verify the dimensions of the partitioned data
nrow(train_data)
nrow(test_data)

#Model Fitting, Validation, and Test Accuracy
#Goal A: Accurate Pricing Estimation
#Multiple Linear Regression
lm_model <- lm(normalized_used_price ~ ram + internal_memory + rear_camera_mp +
                 battery + screen_size + days_used + release_year, 
               data = train_data)
lm_model

#Predict on train and test data
lm_train_pred <- predict(lm_model, newdata = train_data)
lm_test_pred  <- predict(lm_model, newdata = test_data)

#Evaluate performance (RMSE and R-squared)
lm_rmse_train <- sqrt(mean((lm_train_pred - train_data$normalized_used_price)^2))
lm_rmse_test  <- sqrt(mean((lm_test_pred - test_data$normalized_used_price)^2))

lm_r2_train <- 1 - sum((lm_train_pred - train_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - train_data$normalized_used_price)^2)

lm_r2_test  <- 1 - sum((lm_test_pred - test_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - test_data$normalized_used_price)^2)

lm_r2_train
lm_r2_test

#Regression Tree (CART)
#library(rpart)

tree_model <- rpart(normalized_used_price ~ ram + internal_memory + rear_camera_mp +
                      battery + screen_size + days_used + release_year, 
                    data = train_data, method = "anova")

#Predict on train and test data
tree_train_pred <- predict(tree_model, newdata = train_data)
tree_test_pred  <- predict(tree_model, newdata = test_data)

#Evaluate performance
tree_rmse_train <- sqrt(mean((tree_train_pred - train_data$normalized_used_price)^2))
tree_rmse_test  <- sqrt(mean((tree_test_pred - test_data$normalized_used_price)^2))
tree_rmse_train
tree_rmse_test

tree_r2_train <- 1 - sum((tree_train_pred - train_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - train_data$normalized_used_price)^2)

tree_r2_test  <- 1 - sum((tree_test_pred - test_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - test_data$normalized_used_price)^2)

tree_r2_train
tree_r2_test

#Random Forest Regression
#library(randomForest)

rf_model <- randomForest(normalized_used_price ~ ram + internal_memory + rear_camera_mp +
                           battery + screen_size + days_used + release_year, 
                         data = train_data, ntree = 100, importance = TRUE)
rf_model

#Predict on train and test data
rf_train_pred <- predict(rf_model, newdata = train_data)
rf_test_pred  <- predict(rf_model, newdata = test_data)

#Evaluate performance
rf_rmse_train <- sqrt(mean((rf_train_pred - train_data$normalized_used_price)^2))
rf_rmse_test  <- sqrt(mean((rf_test_pred - test_data$normalized_used_price)^2))
rf_rmse_train
rf_rmse_test

rf_r2_train <- 1 - sum((rf_train_pred - train_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - train_data$normalized_used_price)^2)

rf_r2_test  <- 1 - sum((rf_test_pred - test_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - test_data$normalized_used_price)^2)

rf_r2_train
rf_r2_test


#Summarize Model Performance
model_performance <- data.frame(
  Model = c("Multiple Linear Regression", "Regression Tree (CART)", "Random Forest Regression"),
  RMSE_Train = c(lm_rmse_train, tree_rmse_train, rf_rmse_train),
  RMSE_Test  = c(lm_rmse_test, tree_rmse_test, rf_rmse_test),
  R2_Train   = c(lm_r2_train, tree_r2_train, rf_r2_train),
  R2_Test    = c(lm_r2_test, tree_r2_test, rf_r2_test)
)

#results
model_performance

#GoalA: Actual vs Predicted Plot
#library(ggplot2)

ggplot(data = test_data, aes(x = normalized_used_price, y = rf_test_pred)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Goal A: Random Forest - Actual vs Predicted Prices",
       x = "Actual Normalized Price",
       y = "Predicted Normalized Price") +
  theme_minimal()

#GoalA: Residual Plot
rf_residuals <- rf_test_pred - test_data$normalized_used_price

ggplot(data = test_data, aes(x = rf_test_pred, y = rf_residuals)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Goal A: Random Forest - Residual Plot",
       x = "Predicted Price",
       y = "Residuals (Predicted - Actual)") +
  theme_minimal()


## Goal B: Key Feature Identification
#We will fit Regression Tree (CART) and Random Forest Regression
#to confirm model reliability before interpreting feature importance

#library(rpart)
#library(randomForest)
#library(ggplot2)

#Defining predictors
predictors <- c("ram", "internal_memory", "rear_camera_mp", "battery",
                "screen_size", "days_used", "release_year")

#Regression Tree (CART)
tree_features <- rpart(
  formula = as.formula(paste("normalized_used_price ~", paste(predictors, collapse = "+"))),
  data = train_data,
  method = "anova"
)

#Predict on train and test data
tree_train_pred <- predict(tree_features, newdata = train_data)
tree_test_pred  <- predict(tree_features, newdata = test_data)

#Evaluate performance (RMSE and R²)
tree_rmse_train <- sqrt(mean((tree_train_pred - train_data$normalized_used_price)^2))
tree_rmse_test  <- sqrt(mean((tree_test_pred - test_data$normalized_used_price)^2))

tree_r2_train <- 1 - sum((tree_train_pred - train_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - train_data$normalized_used_price)^2)

tree_r2_test <- 1 - sum((tree_test_pred - test_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - train_data$normalized_used_price)^2)

#Random Forest Regression
rf_features <- randomForest(
  formula = as.formula(paste("normalized_used_price ~", paste(predictors, collapse = "+"))),
  data = train_data,
  ntree = 100,
  importance = TRUE
)

#Predict on train and test data
rf_train_pred <- predict(rf_features, newdata = train_data)
rf_test_pred  <- predict(rf_features, newdata = test_data)

#Evaluate performance (RMSE and R²)
rf_rmse_train <- sqrt(mean((rf_train_pred - train_data$normalized_used_price)^2))
rf_rmse_test  <- sqrt(mean((rf_test_pred - test_data$normalized_used_price)^2))

rf_r2_train <- 1 - sum((rf_train_pred - train_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - train_data$normalized_used_price)^2)

rf_r2_test <- 1 - sum((rf_test_pred - test_data$normalized_used_price)^2) /
  sum((mean(train_data$normalized_used_price) - train_data$normalized_used_price)^2)

#Summarize model performance
model_performance_B <- data.frame(
  Model = c("Regression Tree (CART)", "Random Forest Regression"),
  RMSE_Train = c(tree_rmse_train, rf_rmse_train),
  RMSE_Test  = c(tree_rmse_test, rf_rmse_test),
  R2_Train   = c(tree_r2_train, rf_r2_train),
  R2_Test    = c(tree_r2_test, rf_r2_test)
)

#Display performance metrics
model_performance_B

#Feature Importance from Random Forest
rf_feat_importance <- importance(rf_features)
rf_feat_df <- data.frame(
  Feature = rownames(rf_feat_importance),
  Importance = rf_feat_importance[,1]
)
rf_feat_df <- rf_feat_df[order(-rf_feat_df$Importance), ]  # descending order

#Display feature importance table
rf_feat_df
#visualize
ggplot(rf_feat_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  geom_text(aes(label = round(Importance, 2)), hjust = -0.1, size = 3.5) +   
  coord_flip() +
  xlab("Feature") + ylab("Importance") +
  ggtitle("Goal B: Feature Importance from Random Forest")




#Goal C: Price Tier Classification
#Classify phones into Basic, Pro, Premium for improving inventory and marketing strategies
#Create price tiers
#install.packages("e1071")
#install.packages("rpart")
#install.packages("randomForest")
#library(e1071)
#library(rpart)
#library(randomForest)
#library(caret)
q <- quantile(train_data$normalized_used_price, probs = c(1/3, 2/3), na.rm = TRUE)
cut_points <- c(-Inf, q[1], q[2], Inf)

train_data$tier <- cut(train_data$normalized_used_price,
                       breaks = cut_points,
                       labels = c("Basic", "Pro", "Premium"))

test_data$tier <- cut(test_data$normalized_used_price,
                      breaks = cut_points,
                      labels = c("Basic", "Pro", "Premium"))

#Formula
cls_formula <- tier ~ ram + internal_memory + rear_camera_mp +
  battery + screen_size + days_used + release_year

#Decision Tree
tree_cls <- rpart(cls_formula, data = train_data, method = "class")
tree_cls_pred <- predict(tree_cls, newdata = test_data, type = "class")
tree_cls_acc <- mean(tree_cls_pred == test_data$tier)

#Random Forest
rf_cls <- randomForest(cls_formula, data = train_data, ntree = 100)
rf_cls_pred <- predict(rf_cls, newdata = test_data)
rf_cls_acc <- mean(rf_cls_pred == test_data$tier)

#Naive Bayes
nb_cls <- naiveBayes(cls_formula, data = train_data)
nb_cls_pred <- predict(nb_cls, newdata = test_data)
nb_cls_acc <- mean(nb_cls_pred == test_data$tier)

#Summarize Classification Performance
cls_performance <- data.frame(
  Model = c("Decision Tree", "Random Forest", "Naive Bayes"),
  Accuracy = c(tree_cls_acc, rf_cls_acc, nb_cls_acc)
)

cls_performance

# Confusion Matrix Heatmap
rf_pred <- predict(rf_cls, newdata = test_data)
cm <- confusionMatrix(rf_pred, test_data$tier)

cm_df <- as.data.frame(cm$table)
ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Goal C: Random Forest - Confusion Matrix for Price Tiers",
       x = "Predicted Tier",
       y = "Actual Tier") +
  theme_minimal()

