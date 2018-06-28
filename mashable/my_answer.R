library(RCurl)
web_address <- getURL("https://raw.githubusercontent.com/benkeser/sibsml/master/mashable/mashable_data.csv")
web_address_test <- getURL("https://raw.githubusercontent.com/benkeser/sibsml/master/mashable/mashable_test_data.csv")
full_data <- read.csv(text = web_address, header = TRUE)
test_data <- read.csv(text = web_address_test, header = TRUE)

Y <- full_data$viral
X <- full_data[,-ncol(full_data)]

library(SuperLearner)

sl <- SuperLearner(
   Y = Y, X = X, newX = test_data[,-ncol(test_data)],
   SL.library = list(c("SL.glm","All"),
                     c("SL.glm","screen.corP"),
                     c("SL.earth","All"),
                     c("SL.earth","screen.corP"),
                     c("SL.glmnet","All"),
                     c("SL.dbarts","All"),
                     c("SL.randomForest","All"),
                     c("SL.randomForest","screen.corP"),
                     c("SL.rpart","All"),
                     c("SL.rpart","screen.corP"),
                     c("SL.rpartPrune","All"),
                     c("SL.rpartPrune","screen.corP")),
   cvControl = list(V = 5), 
   verbose = TRUE, 
   family = binomial()
   )

cvsl <- CV.SuperLearner(
   Y = Y, X = X, 
   SL.library = list(c("SL.glm","All"),
                     c("SL.glm","screen.corP"),
                     c("SL.earth","All"),
                     c("SL.earth","screen.corP"),
                     c("SL.glmnet","All"),
                     c("SL.dbarts","All"),
                     c("SL.randomForest","All"),
                     c("SL.randomForest","screen.corP"),
                     c("SL.rpart","All"),
                     c("SL.rpart","screen.corP"),
                     c("SL.rpartPrune","All"),
                     c("SL.rpartPrune","screen.corP")),
   innerCvControl = list(list(V = 5),list(V = 5),list(V = 5)), 
   cvControl = list(V = 3),
   verbose = TRUE, 
   family = binomial()
   )

Psi <- sl$fitLibrary$SL.glmnet_All
# "true" mse
true_mse <- mean((test_data$viral - sl$SL.predict)^2)

# estimated mse
mse_Psi <- summary(cvsl)$Table[7,2]

# mse as percentage of truth
(mse_Psi - true_mse)/true_mse
