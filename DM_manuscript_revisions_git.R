rm(list=ls()) # Clear workspace

# Comments for manuscript =============================================================

# Models were trained on 04/04/18 using caret_6.0-78 and the dependencies thereof.

# Section 1: Data preparation =========================================================

# for formatting
library(lattice)
library(latticeExtra)
library(extrafont)
mySettings <- trellis.par.get() # for trellis graphics


# load packages
library(readxl)
library(tidyverse)
library(caret)
library(car)
library(pROC)
library(RColorBrewer)
library(corrplot)
library(viridis)
library(gridExtra)
library(readr)
library(doParallel)

# read in data
# be sure working directory is set correctly

DM <- read_csv("DM_age_sort.csv") ### "_sort" to clarify according to sample number
View(DM)

# Convert gender, smoker and response to factors
# Note, important to have character string in original csv file..."gender", "never",
# "former" 

DM$response <- factor(DM$response, levels = c("disease", "control"))
DM$Gender <- factor(DM$Gender, levels = c("M", "F"))
DM$Smoker <- factor(DM$Smoker, levels = c("never", "former", "current"))

# Machine learning training prepartion

# summarize the target response; establish base rate

table(DM$response) / nrow(DM)

# Data split to allow for external testing

set.seed(543)
inTrain <- createDataPartition(y = DM$response, p = .75, list = FALSE)
training <- DM[ inTrain,]
testing <- DM[-inTrain,]

table(training$response)/nrow(training)

# Summary function used to evaluate the models
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))

# Script used to train individual models before and after RFE was modeled after
# Chap. 19 of Kuhn, M. & Johnson, K., Applied Predictive Modeling, Springer,
# New York, 2013.

# collect feats that aren't response

predVars <- names(DM)[!(names(DM) %in% "response")] 

# collect feats that are continuous (arent reponse and factors)

predVars_num <- names(DM)[!(names(DM) %in% c("Smoker", "Gender", "response"))] 

# create cross validation folds to use across all modeling

set.seed(543)
index <- createMultiFolds(training$response, times = 5)

# For varImp from random forest, consider the entire set of observations

set.seed(543)
index_allObs <- createMultiFolds(DM$response, times = 5)

## The candidate set of the number of predictors to evaluate

varSeq <- seq(1, length(predVars)-1)

# Use parallel processing to run each resampled RFE iteration
# (or resampled model with train()) using different workers.

registerDoParallel()

# How many workers are being put to use

getDoParWorkers() 

# Correlation matrix of continuous data
# factors are converted to dummy vars which is what the model uses and reports for
# varImp

predCor <- cor(model.matrix(~., data = DM[predVars])[,-1])
corrplot(predCor)

# The rfe() is used for recursive feature elimination
# We setup control functions for this and train() that use
# the same cross-validation folds. The 'ctrl' object is modifed several
# times as different models are tested

ctrl <- rfeControl(method = "repeatedcv", repeats = 5,
                   saveDetails = TRUE,
                   index = index,
                   returnResamp = "final")

FullCtrl <- trainControl(method = "repeatedcv",
                         repeats = 5,
                         summaryFunction = fiveStats,
                         savePredictions = TRUE,
                         classProbs = TRUE,
                         index = index)

FullCtrl_allObs <- trainControl(method = "repeatedcv",
                                repeats = 5,
                                summaryFunction = fiveStats,
                                savePredictions = TRUE,
                                classProbs = TRUE,
                                index = index_allObs)

# Section 2: ML model training =========================================================

# Fit a series of models with the Full set of features

set.seed(543)
rfFull <- train(response ~ ., training,
                method = "rf",
                metric = "ROC",
                tuneGrid = data.frame(mtry = floor(sqrt(length(predVars)))),
                ntree = 1000,
                trControl = FullCtrl)
rfFull

set.seed(543)
ldaFull <- train(response ~ ., training,
                 method = "lda",
                 metric = "ROC",
                 trControl = FullCtrl)
ldaFull

set.seed(543)
svmFull <- train(response ~ ., training,
                 method = "svmRadial",
                 metric = "ROC",
                 tuneLength = 10,
                 preProc = c("center", "scale"),
                 trControl = FullCtrl)
svmFull

set.seed(543)
nbFull <- train(y = training$response,
                # fails with formula interface
                x = training[,predVars],
                method = "nb",
                metric = "ROC",
                trControl = FullCtrl)
nbFull

set.seed(543)
lrFull <- train(response ~ ., training,
                method = "glm",
                metric = "ROC",
                preProc = c("center", "scale"),
                trControl = FullCtrl)
lrFull

set.seed(543)
lr_netFull <- train(response ~ ., training,
                    method = "glmnet",
                    metric = "ROC",
                    preProc = c("center", "scale"),
                    tuneLength = 10,
                    trControl = FullCtrl)
lr_netFull

set.seed(543)
knnFull <- train(response ~ ., training,
                 method = "knn",
                 metric = "ROC",
                 tuneLength = 10,
                 preProc = c("center", "scale"),
                 trControl = FullCtrl)
knnFull

# RFE versions are fit. To do this, the 'functions' argument of the rfe()
# object is modified to the approproate functions. Refer to:
# http://caret.r-forge.r-project.org/featureSelection.html


ctrl$functions <- rfFuncs
ctrl$functions$summary <- fiveStats

set.seed(543)
rfRFE <- rfe(response ~ ., training,
             sizes = varSeq,
             metric = "ROC",
             ntree = 1000,
             rfeControl = ctrl)
rfRFE

ctrl$functions <- ldaFuncs
ctrl$functions$summary <- fiveStats

set.seed(543)
ldaRFE <- rfe(response ~ ., training,
              sizes = varSeq,
              metric = "ROC",
              rfeControl = ctrl)
ldaRFE

ctrl$functions <- nbFuncs
ctrl$functions$summary <- fiveStats

set.seed(543)
nbRFE <- rfe(response ~ ., training,
             sizes = varSeq,
             metric = "ROC",
             rfeControl = ctrl)
nbRFE

# caretFuncs list allows for a model to be tuned at each iteration 
# of feature seleciton.

ctrl$functions <- caretFuncs
ctrl$functions$summary <- fiveStats

# train runs sequentially
# RFE runs parallel

cvCtrl <- trainControl(method = "cv",
                       verboseIter = FALSE,
                       classProbs = TRUE,
                       allowParallel = FALSE)

set.seed(543)
svmRFE <- rfe(response ~ ., training,
              sizes = varSeq,
              rfeControl = ctrl,
              metric = "ROC",
              # Arguments to train()
              method = "svmRadial",
              tuneLength = 10,
              preProc = c("center", "scale"),
              trControl = cvCtrl)
svmRFE

ctrl$functions <- lrFuncs
ctrl$functions$summary <- fiveStats

set.seed(543)
lrRFE <- rfe(response ~ ., training,
             sizes = varSeq,
             metric = "ROC",
             method = "glm",
             preProc = c("center", "scale"),
             rfeControl = ctrl)
lrRFE

ctrl$functions <- caretFuncs
ctrl$functions$summary <- fiveStats

set.seed(543)
knnRFE <- rfe(response ~ ., training,
              sizes = varSeq,
              metric = "ROC",
              method = "knn",
              tuneLength = 10,
              preProc = c("center", "scale"),
              trControl = cvCtrl,
              rfeControl = ctrl)
knnRFE

# Section 3: Individual model comparisons ===============================================

# Make list object

cvSummary_all <- resamples(list(knnFull = knnFull,
                                knnRFE = knnRFE,
                                ldaFull = ldaFull,
                                ldaRFE = ldaRFE,
                                lrFull = lrFull,
                                lrRFE = lrRFE,
                                lr_netFull = lr_netFull,
                                nbFull = nbFull,
                                nbRFE = nbRFE,
                                rfFull = rfFull,
                                rfRFE = rfRFE,
                                svmFull = svmFull,
                                svmRFE = svmRFE))

cor_allModels <- as.data.frame(modelCor(cvSummary_all,
                                        metric = cvSummary_all$metric[3]))

cor_allModels_num <- modelCor(cvSummary_all, metric = cvSummary_all$metric[3])

corrplot(cor_allModels_num)

# summary stats

tabSum_all <- summary(cvSummary_all, metric = "ROC")
tabSum_all

# visual of confidence intervals for models, metric is AUC of ROC curve,
# average of out of fold predictions
# ROC AUC

dotplot(cvSummary_all, metric = "ROC",
        par.settings = list(plot.line=list(lty = 1, col = "black"),
                            plot.symbol=list(col = "black")))
# Accuracy

dotplot(cvSummary_all, metric = "Accuracy",
        par.settings = list(plot.line=list(lty = 1, col = "black"),
                            plot.symbol=list(col = "black")))

# performance across feature subsets

plot(rfRFE, type = "l", main = "Random Forest")
plot(knnRFE, type = "l", main = "KNN")
plot(ldaRFE, type = "l", main = "LDA")
plot(nbRFE, type = "l", main = "Naive Bayes")
plot(svmRFE, type = "l", main = "SVM")
plot(lrRFE, type = "l", main = "GLM")

# Comment for paper: dotplot across resampling results are the averages of
# out of fold predictions. These lead to estimates regarding the performance
# of the model with outside data.

# ROC curves from individual models, we look at the profile for aucs generated
# considering the entire resampling set at once, not the average from each out of
# fold predictions. Hence, there are differences. In this case, lesser performance
# generally.

# "Full" models

# roc function takes levels according to control case, respectively.
# Need to reorder the levels of the response feature appropriately.

# With the indices, need to pay really close attention.
# Also check to see if the length of the _I objects are true to the number
# included in the resamples

# Considering all models

knnFull_I <- which(knnFull$pred$k == knnFull[["bestTune"]][["k"]])
RS_knnFull_obs <- knnFull$pred$obs[knnFull_I]

# for whatever reason, knnFull preds returns just one digit values for probs
RS_knnFull_preds <- knnFull$pred$disease[knnFull_I] 

RS_knnFull_roc <- roc(RS_knnFull_obs, RS_knnFull_preds, ci = TRUE,
                      levels = c("control", "disease"))
plot.roc(RS_knnFull_roc, main = "knnFull", print.auc = TRUE)
RS_knnFull_roc

rfFull_I <- which(rfFull$pred$mtry == rfFull[["bestTune"]][["mtry"]])
RS_rfFull_obs <- rfFull$pred$obs[rfFull_I]
RS_rfFull_preds <- rfFull$pred$disease[rfFull_I]
RS_rfFull_roc <- roc(RS_rfFull_obs, RS_rfFull_preds, ci = TRUE,
                     levels = c("control", "disease"))
plot.roc(RS_rfFull_roc, main = "rfFull", print.auc = TRUE)
RS_rfFull_roc

lr_netFull_alpha <- lr_netFull$pred$alpha == lr_netFull[["bestTune"]][["alpha"]]
lr_netFull_lambda <- lr_netFull$pred$lambda == lr_netFull[["bestTune"]][["lambda"]]
lr_netFull_I <- which(lr_netFull_alpha & lr_netFull_lambda)
RS_lr_netFull_obs <- lr_netFull$pred$obs[lr_netFull_I]
RS_lr_netFull_preds <- lr_netFull$pred$disease[lr_netFull_I]
RS_lr_netFull_roc <- roc(RS_lr_netFull_obs, RS_lr_netFull_preds, ci = TRUE,
                         levels = c("control", "disease"))
plot.roc(RS_lr_netFull_roc, main ="lr_netFull", print.auc = TRUE)
RS_lr_netFull_roc

svmFull_sigma <- svmFull$pred$sigma == svmFull[["bestTune"]][["sigma"]]
svmFull_C <- svmFull$pred$C == svmFull[["bestTune"]][["C"]]
svmFull_I <- which(svmFull_sigma & svmFull_C)
RS_svmFull_obs <- svmFull$pred$obs[svmFull_I]
RS_svmFull_preds <- svmFull$pred$disease[svmFull_I]
RS_svmFull_roc <- roc(RS_svmFull_obs, RS_svmFull_preds, ci = TRUE,
                      levels = c("control", "disease"))
plot.roc(RS_svmFull_roc, main = "svmFull", print.auc = TRUE)
RS_svmFull_roc

nbFull_fL <- nbFull$pred$fL == nbFull[["bestTune"]][["fL"]]
nbFull_useKernel <- nbFull$pred$usekernel == nbFull[["bestTune"]][["usekernel"]]
nbFull_adjust <- nbFull$pred$adjust == nbFull[["bestTune"]][["adjust"]]
nbFull_I <- which(nbFull_fL & nbFull_useKernel & nbFull_adjust)
RS_nbFull_obs <- nbFull$pred$obs[nbFull_I]
RS_nbFull_preds <- nbFull$pred$disease[nbFull_I]
RS_nbFull_roc <- roc(RS_nbFull_obs, RS_nbFull_preds, ci = TRUE,
                     levels = c("control", "disease"))
plot.roc(RS_nbFull_roc, main = "nbFull", print.auc = TRUE)
RS_nbFull_roc

# auc below 0.5
ldaFull_I <- ldaFull$pred$rowIndex
RS_ldaFull_obs <- ldaFull$pred$obs
RS_ldaFull_preds <- ldaFull$pred$disease
RS_ldaFull_roc <- roc(RS_ldaFull_obs, RS_ldaFull_preds, ci = TRUE,
                      levels = c("control", "disease"))
plot.roc(RS_ldaFull_roc, main = "ldaFull", print.auc = TRUE)
RS_ldaFull_roc

lrFull_I <- lrFull$pred$rowIndex
RS_lrFull_obs <- lrFull$pred$obs
RS_lrFull_preds <- lrFull$pred$disease
RS_lrFull_roc <- roc(RS_lrFull_obs, RS_lrFull_preds, ci = TRUE,
                     levels = c("control", "disease"))
plot.roc(RS_lrFull_roc, main = "lrFull", print.auc = TRUE)
RS_lrFull_roc

# Checking for apples to apples comparison
# consider each model and the optimal subset, see if indices match

rfFull_indOpt <- rfFull[["pred"]][["rowIndex"]][rfFull_I]
svmFull_indOpt <- svmFull[["pred"]][["rowIndex"]][svmFull_I]
lr_netFull_indOpt <- lr_netFull[["pred"]][["rowIndex"]][lr_netFull_I]
knnFull_indOpt <- knnFull[["pred"]][["rowIndex"]][knnFull_I]
ldaFull_indOpt <- ldaFull_I
nbFull_indOpt <- nbFull[["pred"]][["rowIndex"]][nbFull_I]
lrFull_indOpt <- lrFull[["pred"]][["rowIndex"]][lrFull_I]

# are the indices equal?
identical(rfFull_indOpt, svmFull_indOpt, lr_netFull_indOpt,
          knnFull_indOpt, ldaFull_indOpt, nbFull_indOpt, lrFull_indOpt)

# with the indices, need to pay  close attention. Also check to see if the length
# of the _I objects are true to the number included in the resamples

# RFE models

# roc function takes levels according to control case respectively,
# need to reorder responses appropriately

# these are the entire resampling profile which are 155 observations (from CV folds)

rfRFE_I <- which(rfRFE$pred$Variables == rfRFE[["bestSubset"]]) 
RS_rfRFE_obs <- rfRFE$pred$obs[rfRFE_I]
RS_rfRFE_preds <- rfRFE$pred$disease[rfRFE_I]
RS_rfRFE_roc <- roc(RS_rfRFE_obs, RS_rfRFE_preds, ci = TRUE,
                    levels = c("control", "disease"))
plot.roc(RS_rfRFE_roc, main = "rfRFE", print.auc = TRUE)
RS_rfRFE_roc

knnRFE_I <- which(knnRFE$pred$Variables == knnRFE[["bestSubset"]])
RS_knnRFE_obs <- knnRFE$pred$obs[knnRFE_I]
RS_knnRFE_preds <- knnRFE$pred$disease[knnRFE_I]
RS_knnRFE_roc <- roc(RS_knnRFE_obs, RS_knnRFE_preds, ci = TRUE,
                     levels = c("control", "disease"))
plot.roc(RS_knnRFE_roc, main = "knnRFE", print.auc = TRUE)
RS_knnRFE_roc

svmRFE_I <- which(svmRFE$pred$Variables == svmRFE[["bestSubset"]])
RS_svmRFE_obs <- svmRFE$pred$obs[svmRFE_I]
RS_svmRFE_preds <- svmRFE$pred$disease[svmRFE_I]
RS_svmRFE_roc <- roc(RS_svmRFE_obs, RS_svmRFE_preds, ci = TRUE,
                     levels = c("control", "disease"))
plot.roc(RS_svmRFE_roc, main = "svmRFE", print.auc = TRUE)
RS_svmRFE_roc

nbRFE_I <- which(nbRFE$pred$Variables == nbRFE[["bestSubset"]])
RS_nbRFE_obs <- nbRFE$pred$obs[nbRFE_I]
RS_nbRFE_preds <- nbRFE$pred$disease[nbRFE_I]
RS_nbRFE_roc <- roc(RS_nbRFE_obs, RS_nbRFE_preds, ci = TRUE,
                    levels = c("control", "disease"))
plot.roc(RS_nbRFE_roc, main = "nbRFE", print.auc = TRUE)
RS_nbRFE_roc

ldaRFE_I <- which(ldaRFE$pred$Variables == ldaRFE[["bestSubset"]])
RS_ldaRFE_obs <- ldaRFE$pred$obs[ldaRFE_I]
RS_ldaRFE_preds <- ldaRFE$pred$disease[ldaRFE_I]
RS_ldaRFE_roc <- roc(RS_ldaRFE_obs, RS_ldaRFE_preds, ci = TRUE,
                     levels = c("control", "disease"))
plot.roc(RS_ldaRFE_roc, main = "ldaRFE", print.auc = TRUE)
RS_ldaRFE_roc

lrRFE_I <- which(lrRFE$pred$Variables == lrRFE[["bestSubset"]])
RS_lrRFE_obs <- lrRFE$pred$obs[lrRFE_I]
RS_lrRFE_preds <- lrRFE$pred$disease[lrRFE_I]
RS_lrRFE_roc <- roc(RS_lrRFE_obs, RS_lrRFE_preds, ci = TRUE,
                    levels = c("control", "disease"))
plot.roc(RS_lrRFE_roc, main = "lrRFE", print.auc = TRUE)
RS_lrRFE_roc

rfRFE_indOpt <- rfRFE[["pred"]][["rowIndex"]][rfRFE_I]
svmRFE_indOpt <- svmRFE[["pred"]][["rowIndex"]][svmRFE_I]
knnRFE_indOpt <- knnRFE[["pred"]][["rowIndex"]][knnRFE_I]
ldaRFE_indOpt <- ldaRFE[["pred"]][["rowIndex"]][lrRFE_I]
nbRFE_indOpt <- nbRFE[["pred"]][["rowIndex"]][nbRFE_I]
lrRFE_indOpt <- lrRFE[["pred"]][["rowIndex"]][lrRFE_I]

identical(rfRFE_indOpt, svmRFE_indOpt, knnRFE_indOpt, ldaRFE_indOpt, nbRFE_indOpt,
          lrRFE_indOpt)

# Across both sets
identical(rfFull_indOpt, rfRFE_indOpt) 

# Section 4: Stacked ensembles ===================================================================================================================================

# Out of fold predictions used to develop stacked ensembles

# Gettings probs from Full objs for for out of fold predictions
# We take the average

rfFull_training <- rfFull[["pred"]][rfFull_I,]
rfFull_training_preds <- mean(as.data.frame(rfFull_training[which(rfFull_training$rowIndex == 1),][3])[,1])
rfFull_training_preds <- sapply(1:dim(training)[1],function(x) mean(as.data.frame(rfFull_training[which(rfFull_training$rowIndex == x),][3])[,1]))

knnFull_training <- knnFull[["pred"]][knnFull_I,]
knnFull_training_preds <- mean(as.data.frame(knnFull_training[which(knnFull_training$rowIndex == 1),][3])[,1])
knnFull_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(knnFull_training[which(knnFull_training$rowIndex == x),][3])[,1]))

svmFull_training <- svmFull[["pred"]][svmFull_I,]
svmFull_training_preds <- mean(as.data.frame(svmFull_training[which(svmFull_training$rowIndex == 1),][3])[,1])
svmFull_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(svmFull_training[which(svmFull_training$rowIndex == x),][3])[,1]))

lrFull_training <- lrFull[["pred"]][lrFull_I,]
lrFull_training_preds <- mean(as.data.frame(lrFull_training[which(lrFull_training$rowIndex == 1),][3])[,1])
lrFull_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(lrFull_training[which(lrFull_training$rowIndex == x),][3])[,1]))

ldaFull_training <- ldaFull[["pred"]][ldaFull_I,]
ldaFull_training_preds <- mean(as.data.frame(ldaFull_training[which(ldaFull_training$rowIndex == 1),][3])[,1])
ldaFull_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(ldaFull_training[which(ldaFull_training$rowIndex == x),][3])[,1]))

nbFull_training <- nbFull[["pred"]][nbFull_I,]
nbFull_training_preds <- mean(as.data.frame(nbFull_training[which(nbFull_training$rowIndex == 1),][3])[,1])
nbFull_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(nbFull_training[which(nbFull_training$rowIndex == x),][3])[,1]))

lr_netFull_training <- lr_netFull[["pred"]][lr_netFull_I,]
lr_netFull_training_preds <- mean(as.data.frame(lr_netFull_training[which(lr_netFull_training$rowIndex == 1),][4])[,1]) # disease probs in 4th column
lr_netFull_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(lr_netFull_training[which(lr_netFull_training$rowIndex == x),][4])[,1]))

models_resampPredsFull <- data.frame(cbind(rfFull_training_preds,
                                           svmFull_training_preds,
                                           knnFull_training_preds,
                                           nbFull_training_preds,
                                           lrFull_training_preds,
                                           lr_netFull_training_preds,
                                           ldaFull_training_preds))

colnames(models_resampPredsFull) <- c("rfFull",
                                      "svmFull",
                                      "knnFull",
                                      "nbFull",
                                      "lrFull",
                                      "lr_netFull",
                                      "ldaFull")

# Gettings probs from Full objs for for out of fold predictions
# We take the average

rfRFE_training <- rfRFE[["pred"]][rfRFE_I,]
rfRFE_training_preds <- mean(as.data.frame(rfRFE_training[which(rfRFE_training$rowIndex == 1),][2])[,1])
rfRFE_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(rfRFE_training[which(rfRFE_training$rowIndex == x),][2])[,1]))

knnRFE_training <- knnRFE[["pred"]][knnRFE_I,]
knnRFE_training_preds <- mean(as.data.frame(knnRFE_training[which(knnRFE_training$rowIndex == 1),][2])[,1])
knnRFE_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(knnRFE_training[which(knnRFE_training$rowIndex == x),][2])[,1]))

svmRFE_training <- svmRFE[["pred"]][svmRFE_I,]
svmRFE_training_preds <- mean(as.data.frame(svmRFE_training[which(svmRFE_training$rowIndex == 1),][2])[,1])
svmRFE_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(svmRFE_training[which(svmRFE_training$rowIndex == x),][2])[,1]))

lrRFE_training <- lrRFE[["pred"]][lrRFE_I,]
lrRFE_training_preds <- mean(as.data.frame(lrRFE_training[which(lrRFE_training$rowIndex == 1),][2])[,1])
lrRFE_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(lrRFE_training[which(lrRFE_training$rowIndex == x),][2])[,1]))

ldaRFE_training <- ldaRFE[["pred"]][ldaRFE_I,]
ldaRFE_training_preds <- mean(as.data.frame(ldaRFE_training[which(ldaRFE_training$rowIndex == 1),][2])[,1])
ldaRFE_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(ldaRFE_training[which(ldaRFE_training$rowIndex == x),][2])[,1]))

nbRFE_training <- nbRFE[["pred"]][nbRFE_I,]
nbRFE_training_preds <- mean(as.data.frame(nbRFE_training[which(nbRFE_training$rowIndex == 1),][2])[,1])
nbRFE_training_preds <- sapply(1:dim(training)[1], function(x) mean(as.data.frame(nbRFE_training[which(nbRFE_training$rowIndex == x),][2])[,1]))

models_resampPredsRFE <- data.frame(cbind(rfRFE_training_preds,
                                          svmRFE_training_preds,
                                          knnRFE_training_preds,
                                          nbRFE_training_preds,
                                          lrRFE_training_preds,
                                          ldaRFE_training_preds))

colnames(models_resampPredsRFE) <- c("rfRFE",
                                     "svmRFE",
                                     "knnRFE",
                                     "nbRFE",
                                     "lrRFE",
                                     "ldaRFE")

# collect all OOF preds

models_resampPredsAll <- cbind(models_resampPredsFull, models_resampPredsRFE) 

ensemble_train_OOF <- cbind(models_resampPredsAll, training$response)
colnames(ensemble_train_OOF)[14] <- "response"

predMods <- names(ensemble_train_OOF)[!(names(ensemble_train_OOF) %in% "response")]

# The candidate set of the number of predictors to evaluate
varSeq_Ens <- seq(1, length(predMods)-1)

# stack training of each models with and without RFE

Full_ensemble_ctrlOOF <- trainControl(method = "repeatedcv",
                                      number = 10,
                                      repeats = 5,
                                      summaryFunction = fiveStats,
                                      savePredictions = TRUE,
                                      classProbs = TRUE,
                                      index = index)

RFE_ensemble_ctrOOF <- rfeControl(method = "repeatedcv", repeats = 5,
                                  saveDetails = TRUE,
                                  index = index,
                                  returnResamp = "final")

# Fit a series of models with the Full set of features
# non-formula interface because ensemble_train_OOF is continuous

set.seed(543)
rfFull_ENS <- train(y = ensemble_train_OOF$response,
                    x = ensemble_train_OOF[,predMods],
                    method = "rf",
                    metric = "ROC",
                    tuneGrid = data.frame(mtry = floor(sqrt(length(predMods)))),
                    ntree = 1000,
                    trControl = Full_ensemble_ctrlOOF)
rfFull_ENS

set.seed(543)
ldaFull_ENS <- train(y = ensemble_train_OOF$response,
                     x = ensemble_train_OOF[,predMods],
                     method = "lda",
                     metric = "ROC",
                     trControl = Full_ensemble_ctrlOOF)
ldaFull_ENS

set.seed(543)
svmFull_ENS <- train(y = ensemble_train_OOF$response,
                     x = ensemble_train_OOF[,predMods],
                     method = "svmRadial",
                     metric = "ROC",
                     tuneLength = 10,
                     preProc = c("center", "scale"),
                     trControl = Full_ensemble_ctrlOOF)
svmFull_ENS

set.seed(543)
nbFull_ENS <- train(y = ensemble_train_OOF$response,
                    x = ensemble_train_OOF[,predMods],
                    method = "nb",
                    metric = "ROC",
                    trControl = Full_ensemble_ctrlOOF)
nbFull_ENS

set.seed(543)
lrFull_ENS <- train(y = ensemble_train_OOF$response,
                    x = ensemble_train_OOF[,predMods],
                    method = "glm",
                    metric = "ROC",
                    preProc = c("center", "scale"),
                    trControl = Full_ensemble_ctrlOOF)
lrFull_ENS

set.seed(543)
lr_netFull_ENS <- train(y = ensemble_train_OOF$response,
                        x = ensemble_train_OOF[,predMods],
                        method = "glmnet",
                        metric = "ROC",
                        preProc = c("center", "scale"),
                        tuneLength = 10,
                        trControl = Full_ensemble_ctrlOOF)
lr_netFull_ENS

set.seed(543)
knnFull_ENS <- train(y = ensemble_train_OOF$response,
                     x = ensemble_train_OOF[,predMods],
                     method = "knn",
                     metric = "ROC",
                     tuneLength = 10,
                     preProc = c("center", "scale"),
                     trControl = Full_ensemble_ctrlOOF)
knnFull_ENS

# RFE versions

RFE_ensemble_ctrOOF$functions <- rfFuncs
RFE_ensemble_ctrOOF$functions$summary <- fiveStats

set.seed(543)
rfRFE_ENS <- rfe(y = ensemble_train_OOF$response,
                 x = ensemble_train_OOF[,predMods],
                 sizes = varSeq_Ens,
                 metric = "ROC",
                 ntree = 1000,
                 rfeControl = RFE_ensemble_ctrOOF)
rfRFE_ENS

RFE_ensemble_ctrOOF$functions <- ldaFuncs
RFE_ensemble_ctrOOF$functions$summary <- fiveStats

set.seed(543)
ldaRFE_ENS <- rfe(y = ensemble_train_OOF$response,
                  x = ensemble_train_OOF[,predMods],
                  sizes = varSeq_Ens,
                  metric = "ROC",
                  rfeControl = RFE_ensemble_ctrOOF)
ldaRFE_ENS

RFE_ensemble_ctrOOF$functions <- nbFuncs
RFE_ensemble_ctrOOF$functions$summary <- fiveStats

set.seed(543)
nbRFE_ENS <- rfe(y = ensemble_train_OOF$response,
                 x = ensemble_train_OOF[,predMods],
                 sizes = varSeq_Ens,
                 metric = "ROC",
                 rfeControl = RFE_ensemble_ctrOOF)
nbRFE_ENS

# caretFuncs list allows for a model to be tuned at each iteration 
# of feature seleciton.

RFE_ensemble_ctrOOF$functions <- caretFuncs
RFE_ensemble_ctrOOF$functions$summary <- fiveStats

# train runs sequentially
# RFE runs parallel

cvCtrl <- trainControl(method = "cv",
                       verboseIter = FALSE,
                       classProbs = TRUE,
                       allowParallel = FALSE)

set.seed(543)
svmRFE_ENS <- rfe(y = ensemble_train_OOF$response,
                  x = ensemble_train_OOF[,predMods],
                  rfeControl = RFE_ensemble_ctrOOF,
                  sizes = varSeq_Ens,
                  metric = "ROC",
                  # Arguments to train()
                  method = "svmRadial",
                  tuneLength = 10,
                  preProc = c("center", "scale"),
                  trControl = cvCtrl)
svmRFE_ENS

RFE_ensemble_ctrOOF$functions <- lrFuncs
RFE_ensemble_ctrOOF$functions$summary <- fiveStats

set.seed(543)
lrRFE_ENS <- rfe(y = ensemble_train_OOF$response,
                 x = ensemble_train_OOF[,predMods],
                 sizes = varSeq_Ens,
                 metric = "ROC",
                 method = "glm",
                 preProc = c("center", "scale"),
                 rfeControl = RFE_ensemble_ctrOOF)
lrRFE_ENS

RFE_ensemble_ctrOOF$functions <- caretFuncs
RFE_ensemble_ctrOOF$functions$summary <- fiveStats

set.seed(543)
knnRFE_ENS <- rfe(y = ensemble_train_OOF$response,
                  x = ensemble_train_OOF[,predMods],
                  sizes = varSeq_Ens,
                  metric = "ROC",
                  method = "knn",
                  tuneLength = 10,
                  preProc = c("center", "scale"),
                  trControl = cvCtrl,
                  rfeControl = RFE_ensemble_ctrOOF)
knnRFE_ENS

# comparisons

stack_ensembleFull <- list(rfFull_ENS = rfFull_ENS,
                           svmFull_ENS = svmFull_ENS,
                           knnFull_ENS = knnFull_ENS,
                           nbFull_ENS = nbFull_ENS,
                           lrFull_ENS = lrFull_ENS,
                           lr_netFull_ENS = lr_netFull_ENS,
                           ldaFull_ENS = ldaFull_ENS)

stack_ensembleFull_resamp <- resamples(stack_ensembleFull)

stack_ensembleRFE <- list(rfRFE_ENS = rfRFE_ENS,
                          svmRFE_ENS = svmRFE_ENS,
                          knnRFE_ENS = knnRFE_ENS,
                          nbRFE_ENS = nbRFE_ENS,
                          lrRFE_ENS = lrRFE_ENS,
                          ldaRFE_ENS = ldaRFE_ENS)

stack_ensembleRFE_resamp <- resamples(stack_ensembleRFE)

stack_ensembleAll <- c(stack_ensembleFull, stack_ensembleRFE)

stack_ensembleAll_resamp <- resamples(stack_ensembleAll)

# Ensemble performance
# ROC

dotplot(stack_ensembleAll_resamp, metric = "ROC",
        par.settings = list(plot.line=list(lty = 1, col = "black"),
                            plot.symbol=list(col = "black")))
# Accuracy

dotplot(stack_ensembleAll_resamp, metric = "Accuracy",
        par.settings = list(plot.line=list(lty = 1, col = "black"),
                            plot.symbol=list(col = "black")))

# individual models 

models_Full_all <- list(rfFull = rfFull,
                        svmFull = svmFull,
                        knnFull = knnFull,
                        nbFull = nbFull,
                        lrFull = lrFull,
                        lr_netFull = lr_netFull,
                        ldaFull = ldaFull,
                        rfFull_ENS = rfFull_ENS,
                        svmFull_ENS = svmFull_ENS,
                        knnFull_ENS = knnFull_ENS,
                        nbFull_ENS = nbFull_ENS,
                        lrFull_ENS = lrFull_ENS,
                        lr_netFull_ENS = lr_netFull_ENS,
                        ldaFull_ENS = ldaFull_ENS)

models_Full_Ind <- list(rfFull = rfFull,
                        svmFull = svmFull,
                        knnFull = knnFull,
                        nbFull = nbFull,
                        lrFull = lrFull,
                        lr_netFull = lr_netFull,
                        ldaFull = ldaFull)

models_RFE_all <- list(rfRFE = rfRFE,
                       svmRFE = svmRFE,
                       knnRFE = knnRFE,
                       nbRFE = nbRFE,
                       lrRFE = lrRFE,
                       ldaRFE = ldaRFE,
                       rfRFE_ENS = rfRFE_ENS,
                       svmRFE_ENS = svmRFE_ENS,
                       knnRFE_ENS = knnRFE_ENS,
                       nbRFE_ENS = nbRFE_ENS,
                       lrRFE_ENS = lrRFE_ENS,
                       ldaRFE_ENS = ldaRFE_ENS)

models_RFE_Ind <- list(rfRFE = rfRFE,
                       svmRFE = svmRFE,
                       knnRFE = knnRFE,
                       nbRFE = nbRFE,
                       lrRFE = lrRFE,
                       ldaRFE = ldaRFE)

models_all <- c(models_Full_all, models_RFE_all)
models_all_Ind <-c(models_Full_Ind, models_RFE_Ind)

models_all_resamp <- resamples(models_all)

# ROC

dotplot(models_all_resamp, metric = "ROC",
        par.settings = list(plot.line=list(lty = 1, col = "black"),
                            plot.symbol=list(col = "black")))
# Accuracy

dotplot(models_all_resamp, metric = "Accuracy",
        par.settings = list(plot.line=list(lty = 1, col = "black"),
                            plot.symbol=list(col = "black")))

# Section 5: Averaged ensembles =========================================================

# Build ensemble groups where simple averages are taken across
# out of fold averages like above

# establishing groups

obs_train <- training$response

# ensembles by simple averaging

# k = 2

k2_I <- combn(1:(dim(models_resampPredsAll)[2]),2)

k2_roc <- function(preds_df, k2_index, obs) {
  roc_k2 <- list()
  for (i in 1:dim(k2_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k2_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k2[i] <- tmp
  }
  return(roc_k2)
}

roc2 <- k2_roc(models_resampPredsAll, k2_I, obs_train)
Rocs_k2 <- sapply(roc2, '[[', "auc") ### extract aucs
k2 <- max(Rocs_k2)
k2_best_I <- which(Rocs_k2 == k2)
k2_ensemble_names <- names(models_resampPredsAll)[k2_I[, k2_best_I]]

# two pairs, rfFull + knnFull and knnFull + nbRFE ###

# k = 3

k3_I <- combn(1:length(models_resampPredsAll),3)

k3_roc <- function(preds_df, k3_index, obs) {
  roc_k3 <- list()
  for (i in 1:dim(k3_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k3_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k3[i] <- tmp
  }
  return(roc_k3)
}

roc3 <- k3_roc(models_resampPredsAll, k3_I, obs_train)
Rocs_k3 <- sapply(roc3, '[[', "auc") ### extract aucs
k3 <- max(Rocs_k3)
k3_best_I <- which(Rocs_k3 == k3)
k3_ensemble_names <- names(models_resampPredsAll)[k3_I[, k3_best_I]]


# k = 4

k4_I <- combn(1:length(models_resampPredsAll),4)

k4_roc <- function(preds_df, k4_index, obs) {
  roc_k4 <- list()
  for (i in 1:dim(k4_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k4_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k4[i] <- tmp
  }
  return(roc_k4)
}

roc4 <- k4_roc(models_resampPredsAll, k4_I, obs_train)
Rocs_k4 <- sapply(roc4, '[[', "auc") ### extract aucs
k4 <- max(Rocs_k4)
k4_best_I <- which(Rocs_k4 == k4)
k4_ensemble_names <- names(models_resampPredsAll)[k4_I[, k4_best_I]]

# two groups: rfFull knnFull nbRFE lrRFE; knnFull rfRFE nbRFE lrRFE

# k = 5

k5_I <- combn(1:length(models_resampPredsAll),5)

k5_roc <- function(preds_df, k5_index, obs) {
  roc_k5 <- list()
  for (i in 1:dim(k5_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k5_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k5[i] <- tmp
  }
  return(roc_k5)
}

roc5 <- k5_roc(models_resampPredsAll, k5_I, obs_train)
Rocs_k5 <- sapply(roc5, '[[', "auc") ### extract aucs
k5 <- max(Rocs_k5)
k5_best_I <- which(Rocs_k5 == k5)
k5_ensemble_names <- names(models_resampPredsAll)[k5_I[, k5_best_I]]

# four groups of five give the best AUC:
#"rfFull"  "knnFull" "svmRFE"  "nbRFE"   "lrRFE"   "rfFull"  "knnFull" "knnRFE" 
# "nbRFE"   "lrRFE"   "knnFull" "rfRFE"   "svmRFE"  "nbRFE"   "lrRFE"   "knnFull"
# "rfRFE"   "knnRFE"  "nbRFE"   "lrRFE"

# k = 6

k6_I <- combn(1:length(models_resampPredsAll),6)

k6_roc <- function(preds_df, k6_index, obs) {
  roc_k6 <- list()
  for (i in 1:dim(k6_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k6_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k6[i] <- tmp
  }
  return(roc_k6)
}

roc6 <- k6_roc(models_resampPredsAll, k6_I, obs_train)
Rocs_k6 <- sapply(roc6, '[[', "auc") ### extract aucs
k6 <- max(Rocs_k6)
k6_best_I <- which(Rocs_k6 == k6)
k6_ensemble_names <- names(models_resampPredsAll)[k6_I[, k6_best_I]]

# k = 7

k7_I <- combn(1:length(models_resampPredsAll),7)

k7_roc <- function(preds_df, k7_index, obs) {
  roc_k7 <- list()
  for (i in 1:dim(k7_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k7_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k7[i] <- tmp
  }
  return(roc_k7)
}

roc7 <- k7_roc(models_resampPredsAll, k7_I, obs_train)
Rocs_k7 <- sapply(roc7, '[[', "auc") ### extract aucs
k7 <- max(Rocs_k7)
k7_best_I <- which(Rocs_k7 == k7)
k7_ensemble_names <- names(models_resampPredsAll)[k7_I[, k7_best_I]]

# three groups:

# "rfFull"  "svmFull" "knnFull" "rfRFE"   "svmRFE"  "nbRFE"   "lrRFE"   "rfFull" 
# "knnFull" "nbFull"  "rfRFE"   "svmRFE"  "nbRFE"   "lrRFE"   "rfFull"  "knnFull"
# "rfRFE"   "svmRFE"  "knnRFE"  "nbRFE"   "lrRFE"  

# k = 8

k8_I <- combn(1:length(models_resampPredsAll),8)

k8_roc <- function(preds_df, k8_index, obs) {
  roc_k8 <- list()
  for (i in 1:dim(k8_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k8_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k8[i] <- tmp
  }
  return(roc_k8)
}

roc8 <- k8_roc(models_resampPredsAll, k8_I, obs_train)
Rocs_k8 <- sapply(roc8, '[[', "auc") ### extract aucs
k8 <- max(Rocs_k8)
k8_best_I <- which(Rocs_k8 == k8)
k8_ensemble_names <- names(models_resampPredsAll)[k8_I[, k8_best_I]]

# k = 9

k9_I <- combn(1:length(models_resampPredsAll),9)

k9_roc <- function(preds_df, k9_index, obs) {
  roc_k9 <- list()
  for (i in 1:dim(k9_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k9_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k9[i] <- tmp
  }
  return(roc_k9)
}

roc9 <- k9_roc(models_resampPredsAll, k9_I, obs_train)
Rocs_k9 <- sapply(roc9, '[[', "auc") ### extract aucs
k9 <- max(Rocs_k9)
k9_best_I <- which(Rocs_k9 == k9)
k9_ensemble_names <- names(models_resampPredsAll)[k9_I[, k9_best_I]]

# k = 10

k10_I <- combn(1:length(models_resampPredsAll),10)

k10_roc <- function(preds_df, k10_index, obs) {
  roc_k10 <- list()
  for (i in 1:dim(k10_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k10_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k10[i] <- tmp
  }
  return(roc_k10)
}

roc10 <- k10_roc(models_resampPredsAll, k10_I, obs_train)
Rocs_k10 <- sapply(roc10, '[[', "auc") ### extract aucs
k10 <- max(Rocs_k10)
k10_best_I <- which(Rocs_k10 == k10)
k10_ensemble_names <- names(models_resampPredsAll)[k10_I[, k10_best_I]]

# two groups:
# "rfFull"     "svmFull"    "knnFull"    "nbFull"     "lr_netFull" "rfRFE"     
# "svmRFE"     "knnRFE"     "nbRFE"      "lrRFE"      "rfFull"     "svmFull"   
# "knnFull"    "nbFull"     "rfRFE"      "svmRFE"     "knnRFE"     "nbRFE"     
# "lrRFE"      "ldaRFE" 

# k = 11

k11_I <- combn(1:length(models_resampPredsAll),11)

k11_roc <- function(preds_df, k11_index, obs) {
  roc_k11 <- list()
  for (i in 1:dim(k11_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k11_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k11[i] <- tmp
  }
  return(roc_k11)
}

roc11 <- k11_roc(models_resampPredsAll, k11_I, obs_train)
Rocs_k11 <- sapply(roc11, '[[', "auc") ### extract aucs
k11 <- max(Rocs_k11)
k11_best_I <- which(Rocs_k11 == k11)
k11_ensemble_names <- names(models_resampPredsAll)[k11_I[, k11_best_I]]

# k = 12

k12_I <- combn(1:length(models_resampPredsAll),12)

k12_roc <- function(preds_df, k12_index, obs) {
  roc_k12 <- list()
  for (i in 1:dim(k12_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k12_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k12[i] <- tmp
  }
  return(roc_k12)
}

roc12 <- k12_roc(models_resampPredsAll, k12_I, obs_train)
Rocs_k12 <- sapply(roc12, '[[', "auc") ### extract aucs
k12 <- max(Rocs_k12)
k12_best_I <- which(Rocs_k12 == k12)
k12_ensemble_names <- names(models_resampPredsAll)[k12_I[, k12_best_I]]

# k = 13

k13_I <- combn(1:length(models_resampPredsAll),13)

k13_roc <- function(preds_df, k13_index, obs) {
  roc_k13 <- list()
  for (i in 1:dim(k13_index)[2]){
    roc_tmp <- roc(obs, rowMeans(preds_df[k13_index[,i]]), ci = TRUE, levels = c("control", "disease"))
    tmp <- list(a = roc_tmp)
    roc_k13[i] <- tmp
  }
  return(roc_k13)
}

roc13 <- k13_roc(models_resampPredsAll, k13_I, obs_train)
Rocs_k13 <- sapply(roc13, '[[', "auc") # extract aucs
k13 <- max(Rocs_k13)
k13_best_I <- which(Rocs_k13 == k13)
k13_ensemble_names <- names(models_resampPredsAll)[k13_I[, k13_best_I]]

# collect best AUCs from each group: 

training_k_roc <- list(k2_1 = roc2[k2_best_I][[1]],
                       k2_2 = roc2[k2_best_I][[2]],
                       k3 = roc3[k3_best_I][[1]],
                       k4_1 = roc4[k4_best_I][[1]],
                       k4_2 = roc4[k4_best_I][[2]],
                       k5_1 = roc5[k5_best_I][[1]],
                       k5_2 = roc5[k5_best_I][[2]],
                       k5_3 = roc5[k5_best_I][[3]],
                       k5_4 = roc5[k5_best_I][[4]],
                       k6 = roc6[k6_best_I][[1]],
                       k7_1 = roc7[k7_best_I][[1]],
                       k7_2 = roc7[k7_best_I][[2]],
                       k7_3 = roc7[k7_best_I][[3]],
                       k8 = roc8[k8_best_I][[1]],
                       k9 = roc9[k9_best_I][[1]],
                       k10_1 = roc10[k10_best_I][[1]],
                       k10_2 = roc10[k10_best_I][[2]],
                       k11 = roc11[k11_best_I][[1]],
                       k12 = roc12[k12_best_I][[1]],
                       k13 = roc13[k13_best_I][[1]])

rocs_train_k <- sapply(training_k_roc, '[[', "auc")
ci_train_k <- sapply(training_k_roc, '[[', "ci")
# to get ci, sub the mean from one end
ci_train_k <- ci_train_k[3,] - ci_train_k[2,]

rocs_k_bar <- barplot(sort(rocs_train_k, decreasing = TRUE),
                      ylab = "AUC", las = 2, ylim = c(0,1))
segments(rocs_k_bar, rocs_train_k - ci_train_k, rocs_k_bar, rocs_train_k + ci_train_k)
arrows(rocs_k_bar, rocs_train_k - ci_train_k, rocs_k_bar,
       rocs_train_k + ci_train_k, lwd = 1.5, angle = 90,
       code = 3, length = 0.05)
        
# observe ROCs to see performance across the resampling profile 
# using lists, and extractProb(list, ...) error occurs due to dummy vars 

Full_test_preds <- data.frame(sapply(models_Full_Ind,
                                     function(x) predict(x, testing, type = "prob")[1]))
colnames(Full_test_preds) <- names(models_Full_Ind)
RFE_test_preds <- data.frame(sapply(models_RFE_Ind,
                                    function(x) predict(x, testing)[2]))

colnames(RFE_test_preds) <- names(models_RFE_Ind)
test_preds_Ind <- cbind(Full_test_preds, RFE_test_preds)

test_roc_Ind <- lapply(test_preds_Ind,
                       function(x) roc(testing$response, x,ci = TRUE,
                                       levels = c("control", "disease")))

test_auc_Ind <- sapply(test_roc_Ind, '[[', "auc")

bestInd_roc <- max(test_auc_Ind)
bestInd_roc

barplot(sort(test_auc_Ind, decreasing = TRUE), ylim = c(0,1),
        main = "External Test", ylab = "AUC", las = 2)

# best Ind model is rfRFE with auc = 0.9

# print roc curves for each

lapply(test_roc_Ind, function(x) plot.roc(x, print.auc = TRUE))

# Section 6: External predictions =========================================================

# ensemble

ensemblePrediction <- function(model_Ensemblenames, modelsPreds_DF,obsTest) {
  ensemble_pred <- list()
  ensemble_testPred <- modelsPreds_DF[model_Ensemblenames]
  ensemble_testPred <- rowMeans(ensemble_testPred)
  ensemble_roc <- roc(obsTest, ensemble_testPred, ci = TRUE, levels = c("control", "disease"))
  ensemble_pred <- list(pred = ensemble_testPred, roc = ensemble_roc)
  return(ensemble_pred)
}


models_ensemble <- names(models_resampPredsAll)

# k = 2


k2_ensemble_names_1 <- models_ensemble[k2_I[, k2_best_I[1]]]
k2_ensemble_names_2 <- models_ensemble[k2_I[, k2_best_I[2]]]


k2_ensemblePred_1 <- ensemblePrediction(k2_ensemble_names_1, test_preds_Ind, testing$response)
k2_ensemblePred_2 <- ensemblePrediction(k2_ensemble_names_2, test_preds_Ind, testing$response)


# k = 3

k3_ensemble_names <- models_ensemble[k3_I[, k3_best_I]]

k3_ensemblePred <- ensemblePrediction(k3_ensemble_names, test_preds_Ind, testing$response)

# k = 4

k4_ensemble_names_1 <- models_ensemble[k4_I[, k4_best_I[1]]]
k4_ensemble_names_2 <- models_ensemble[k4_I[, k4_best_I[2]]]


k4_ensemblePred_1 <- ensemblePrediction(k4_ensemble_names_1, test_preds_Ind, testing$response)
k4_ensemblePred_2 <- ensemblePrediction(k4_ensemble_names_2, test_preds_Ind, testing$response)


# k = 5

k5_ensemble_names_1 <- models_ensemble[k5_I[, k5_best_I[1]]]
k5_ensemble_names_2 <- models_ensemble[k5_I[, k5_best_I[2]]]
k5_ensemble_names_3 <- models_ensemble[k5_I[, k5_best_I[3]]]
k5_ensemble_names_4 <- models_ensemble[k5_I[, k5_best_I[4]]]


k5_ensemblePred_1 <- ensemblePrediction(k5_ensemble_names_1, test_preds_Ind, testing$response)
k5_ensemblePred_2 <- ensemblePrediction(k5_ensemble_names_2, test_preds_Ind, testing$response)
k5_ensemblePred_3 <- ensemblePrediction(k5_ensemble_names_3, test_preds_Ind, testing$response)
k5_ensemblePred_4 <- ensemblePrediction(k5_ensemble_names_4, test_preds_Ind, testing$response)


# k = 6

k6_ensemble_names <- models_ensemble[k6_I[, k6_best_I]]

k6_ensemblePred <- ensemblePrediction(k6_ensemble_names, test_preds_Ind, testing$response)

### k = 7 ###

k7_ensemble_names_1 <- models_ensemble[k7_I[, k7_best_I[1]]]
k7_ensemble_names_2 <- models_ensemble[k7_I[, k7_best_I[2]]]
k7_ensemble_names_3 <- models_ensemble[k7_I[, k7_best_I[3]]]

k7_ensemblePred_1 <- ensemblePrediction(k7_ensemble_names_1, test_preds_Ind, testing$response)
k7_ensemblePred_2 <- ensemblePrediction(k7_ensemble_names_2, test_preds_Ind, testing$response)
k7_ensemblePred_3 <- ensemblePrediction(k7_ensemble_names_3, test_preds_Ind, testing$response)


# k = 8

k8_ensemble_names <- models_ensemble[k8_I[, k8_best_I]]

k8_ensemblePred <- ensemblePrediction(k8_ensemble_names, test_preds_Ind, testing$response)

# k = 9

k9_ensemble_names <- models_ensemble[k9_I[, k9_best_I]]

k9_ensemblePred <- ensemblePrediction(k9_ensemble_names, test_preds_Ind, testing$response)

# k = 10

k10_ensemble_names_1 <- models_ensemble[k10_I[, k10_best_I[1]]]
k10_ensemble_names_2 <- models_ensemble[k10_I[, k10_best_I[2]]]


k10_ensemblePred_1 <- ensemblePrediction(k10_ensemble_names_1, test_preds_Ind, testing$response)
k10_ensemblePred_2 <- ensemblePrediction(k10_ensemble_names_2, test_preds_Ind, testing$response)

# k = 11

k11_ensemble_names <- models_ensemble[k11_I[, k11_best_I]]

k11_ensemblePred <- ensemblePrediction(k11_ensemble_names, test_preds_Ind, testing$response)

# k = 12

k12_ensemble_names <- models_ensemble[k12_I[, k12_best_I]]

k12_ensemblePred <- ensemblePrediction(k12_ensemble_names, test_preds_Ind, testing$response)

# k = 13

k13_ensemble_names <- models_ensemble[k13_I[, k13_best_I]]

k13_ensemblePred <- ensemblePrediction(k13_ensemble_names, test_preds_Ind, testing$response)

test_preds_k_ensemble <- list(k2_1 = k2_ensemblePred_1,
                              k2_2 = k2_ensemblePred_2,
                              k3 = k3_ensemblePred,
                              k4_1 = k4_ensemblePred_1,
                              k4_2 = k4_ensemblePred_2,
                              k5_1 = k5_ensemblePred_1,
                              k5_2 = k5_ensemblePred_2,
                              k5_3 = k5_ensemblePred_3,
                              k5_4 = k5_ensemblePred_4,
                              k6 = k6_ensemblePred,
                              k7_1 = k7_ensemblePred_1,
                              k7_2 = k7_ensemblePred_2,
                              k7_3 = k7_ensemblePred_3,
                              k8 = k8_ensemblePred,
                              k9 = k9_ensemblePred,
                              k10_1 = k10_ensemblePred_1,
                              k10_2 = k10_ensemblePred_2,
                              k11 = k11_ensemblePred,
                              k12 = k12_ensemblePred,
                              k13 = k13_ensemblePred)

test_preds_k_auc <- rbind(test_preds_k_ensemble[["k2_1"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k2_2"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k3"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k4_1"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k4_2"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k5_1"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k5_2"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k5_3"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k5_4"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k6"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k7_1"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k7_2"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k7_3"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k8"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k9"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k10_1"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k10_2"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k11"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k12"]][["roc"]][["auc"]],
                          test_preds_k_ensemble[["k13"]][["roc"]][["auc"]])

names(test_preds_k_auc) <- names(test_preds_k_ensemble)                           

bestEnsemble_k_roc <- max(test_preds_k_auc)
bestEnsemble_k_roc
barplot(sort(test_preds_k_auc, decreasing = TRUE), las = 2, ylim = c(0,1),
        main = "External Test", ylab = "AUC")

# from ind models above

# stacked ensemble tests 

testing_ensemble <- cbind(test_preds_Ind, testing$response)
names(testing_ensemble)[14] <- "response"
Full_test_ensemble <- data.frame(
  sapply(stack_ensembleFull,
         function(x) predict(x,
                             testing_ensemble[,predMods],
                             type = "prob")[1])) # doesnt like just testing_ensemble
colnames(Full_test_ensemble) <- names(stack_ensembleFull)
RFE_test_ensemble <- data.frame(
  sapply(stack_ensembleRFE,
         function(x) predict(x, testing_ensemble)[2]))
RFE_test_ensemble[5] <- predict(lrRFE_ENS, testing_ensemble)[1] # lr predictions are out
# of order
colnames(RFE_test_ensemble) <- names(stack_ensembleRFE)
test_preds_Ensemble <- cbind(Full_test_ensemble, RFE_test_ensemble)
testsPreds_Ensemble_factor <- sapply(test_preds_Ensemble,
                                     function(x) ifelse(x >= 0.5, "disease", "control"))


test_roc_Ensemble <- lapply(
  test_preds_Ensemble,
  function(x) roc(testing_ensemble$response,
                  x, ci = TRUE,
                  levels = c("control", "disease")))
test_auc_Ensemble <- sapply(test_roc_Ensemble, '[[', "auc")
bestEns_roc <- max(test_auc_Ensemble)
bestEns_roc

# combine

aucs_all <- list(rfFull = as.numeric(test_auc_Ind[1]),
                 svmFull = as.numeric(test_auc_Ind[2]),
                 knnFull = as.numeric(test_auc_Ind[3]),
                 nbFull = as.numeric(test_auc_Ind[4]),
                 lrFull = as.numeric(test_auc_Ind[5]),
                 lr_netFull = as.numeric(test_auc_Ind[6]),
                 ldaFull = as.numeric(test_auc_Ind[7]),
                 rfRFE = as.numeric(test_auc_Ind[8]),
                 svmRFE = as.numeric(test_auc_Ind[9]),
                 knnRFE = as.numeric(test_auc_Ind[10]),
                 nbRFE = as.numeric(test_auc_Ind[11]),
                 lrRFE = as.numeric(test_auc_Ind[12]),
                 ldaRFE = as.numeric(test_auc_Ind[13]),
                 rfFull_ENS = as.numeric(test_auc_Ensemble[1]),
                 svmFull_ENS = as.numeric(test_auc_Ensemble[2]),
                 knnFull_ENS = as.numeric(test_auc_Ensemble[3]),
                 nbFull_ENS = as.numeric(test_auc_Ensemble[4]),
                 lrFull_ENS = as.numeric(test_auc_Ensemble[5]),
                 lr_netFull_ENS = as.numeric(test_auc_Ensemble[6]),
                 ldaFull_ENS = as.numeric(test_auc_Ensemble[7]),
                 rfRFE_ENS = as.numeric(test_auc_Ensemble[8]),
                 svmRFE_ENS = as.numeric(test_auc_Ensemble[9]),
                 knnRFE_ENS = as.numeric(test_auc_Ensemble[10]),
                 nbRFE_ENS = as.numeric(test_auc_Ensemble[11]),
                 lrRFE_ENS = as.numeric(test_auc_Ensemble[12]),
                 ldaRFE_ENS = as.numeric(test_auc_Ensemble[13]),
                 k2_1 = as.numeric(test_preds_k_auc[1]),
                 k2_2 = as.numeric(test_preds_k_auc[2]),
                 k3 = as.numeric(test_preds_k_auc[3]),
                 k4 = as.numeric(test_preds_k_auc[4]),
                 k5_1 = as.numeric(test_preds_k_auc[5]),
                 k5_2 = as.numeric(test_preds_k_auc[6]),
                 k6 = as.numeric(test_preds_k_auc[7]),
                 k7_1 = as.numeric(test_preds_k_auc[8]),
                 k7_2 = as.numeric(test_preds_k_auc[9]),
                 k8 = as.numeric(test_preds_k_auc[10]),
                 k9 = as.numeric(test_preds_k_auc[11]),
                 k10 = as.numeric(test_preds_k_auc[12]),
                 k11 = as.numeric(test_preds_k_auc[13]),
                 k12 = as.numeric(test_preds_k_auc[14]),
                 k13 = as.numeric(test_preds_k_auc[15]))

barplot(sort(unlist(aucs_all), decreasing = TRUE), ylim = c(0,1),
        ylab = "AUC", cex.names = 0.7, cex.axis = 0.7, las = 2)

# rfFull and rfRFE give best auc values on external tests

# considering resampling and testing, rfRFE is the best model

# rfRFE 
confusionMatrix(predict(rfRFE, newdata = testing)$pred,
                reference = testing$response)

# ldaRFE 
confusionMatrix(predict(ldaRFE, newdata = testing)$pred,
                reference = testing_ensemble$response)

# rfFull 
confusionMatrix(predict(rfFull, testing),
                reference = testing$response)
# same as rfRFE

# knnRFE
confusionMatrix(predict(knnRFE, testing)$pred,
                reference = testing$response)

# lr_netFull
confusionMatrix(predict(lr_netFull, testing),
                reference = testing_ensemble$response)

# Section 7: Figures for manuscript =========================================================

# setwd("G:/My Drive/WFU/Nails/r/manuscript/reviewers_commments")

# Figures for manuscript; in order of appearance

# Figures 1) and 2):

f_score <- read_excel("G:/My Drive/WFU/Nails/manuscript/manuscript_univariate.xlsx",
                      sheet = "F-score")
chi_square <- read_excel("G:/My Drive/WFU/Nails/manuscript/manuscript_univariate.xlsx",
                         sheet = "chi-square")

f_score <- arrange(f_score, desc(f_score$`F-score`))
chi_square <- arrange(chi_square, desc(chi_square$`Cramer's V`))

f_score$Feature <- factor(f_score$Feature,
                          levels = f_score$Feature[order(f_score$`F-score`)])
chi_square$Feature <- factor(chi_square$Feature,
                             levels = chi_square$Feature[order(chi_square$`Cramer's V`)])

f_score_ggBarplot <- ggplot(data = f_score,
                            aes(x = f_score$Feature, y = f_score$`F-score`)) +
  geom_bar(stat = "identity", width = 0.5, size = 0.5, color = "#000000", fill = "#000000") +
  xlab("") +
  ylab("") +
  theme_bw() +
  coord_flip() +
  theme(legend.direction = 'vertical',
        legend.position = 'right',
        text=element_text(face="bold", size = 10),
        panel.border = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"),
        axis.text.y = element_text(vjust = 0.5))
f_score_ggBarplot
ggsave("f_score_GeorgeEdit.tiff", units="in", width=6, height=4, dpi=1000, compression = 'lzw')

chi_square_ggBarplot <- ggplot(data = chi_square,
                               aes(x = Feature, y = chi_square$`Cramer's V`)) +
  geom_bar(stat = "identity", width = 0.5, size = 0.5, color = "#000000", fill = "#000000") +
  xlab("") +
  ylab("") +
  theme_bw() +
  coord_flip() +
  theme(legend.direction = 'vertical',
        legend.position = 'right',
        text=element_text(face="bold", size = 10),
        panel.border = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"),
        axis.text.y = element_text(vjust = 0.5))
chi_square_ggBarplot
ggsave("chi_square_GeorgeEdit.tiff", units="in", width=6, height=4, dpi=1000, compression = 'lzw')

# Figure 3)

DM_order <- DM[,order(names(DM))]
predVars_order_num <- names(DM_order)[!(names(DM_order) %in% c("Smoker", "Gender", "response"))] ### collect feats that are continuous (arent reponse and factors) ###

feats <- list()
for (i in 1:length(DM_order[predVars_order_num])) {
  #j=show[i]
  feats[[i]] <- ggplot(as.data.frame(DM_order[predVars_order_num]),
                       aes_string(factor(DM_order$response, levels = c("control", "disease")),
                                  color = factor(DM_order$response, levels = c("control", "disease")),
                                  colnames(as.data.frame(DM_order[predVars_order_num])[i])))  + 
    geom_boxplot(size = 0.25, outlier.size = 0.25, width = 0.5) +
    scale_color_manual(values = c("#000000", "#000000"), # from cividis(2)
                      guide = FALSE) +
    theme_bw() + 
    theme(plot.title = element_text(size = 8),
          axis.text.y = element_text(size = 5),
          text=element_text(face="bold", size = 8),
          panel.grid.major = element_blank()) +
    xlab(NULL) + 
    ylab(NULL) +
    ggtitle(colnames(as.data.frame(DM_order[predVars_order_num]))[i]) +
    theme(plot.title = element_text(hjust = 0.5))
}
box_plot <- arrangeGrob(grobs = feats, nrow = 5, ncol = 5)
ggsave("box_plot_GeorgeEdit.tiff", plot = box_plot, units="in", width=6, height=4, dpi=1000, compression = 'lzw')

# Figure 4)

DM_scale <-DM

DM_scale$Gender <- as.factor(DM_scale$Gender) ### 1) female; 2) male
DM_scale$Smoker <- as.factor(DM_scale$Smoker) ### 1) current; 2) former; 3) "never"
DM_scale$Gender <- as.numeric(DM_scale$Gender)
DM_scale$Smoker <- as.numeric(DM_scale$Smoker)

DM_scale <- scale(DM_scale[!names(DM_scale) %in% "response"])

DM_PCA <- prcomp(DM_scale)
DM_PCA$x[,2] <- DM_PCA$x[,2] * -1 # eigen values unique up to sign change.
# We change the sign to match original manuscript submission
DM_PCA$rotation[,2] <- DM_PCA$rotation[,2] * -1 

source("G:/My Drive/WFU/Nails/r/ggbiplot_custom_DM.R")

# We load in a modified ggbiplot function to produce the PCA biplot figure.
# See the following for the original function:
# Vincent Q. Vu (2011). ggbiplot: A ggplot2 based biplot. R package version 0.55.
# http://github.com/vqv/ggbiplot

DM_biplot <- ggbiplot_custom(DM_PCA, choices = c(1,2), segment.alpha = 0.5, varname.adjust = 3,
                             groups = factor(DM$response, levels = c("disease", "control")),
                             varname.size = 2, shape.size = 2,
                             alpha = 0.5)

DM_biplot <- DM_biplot +theme_bw() +
  scale_color_manual(name = "Response",
                     labels = c("Disease", "Control"),
                     # color comes from cividis(2, alpha = 0.5)
                     values = c("#000000", "#000000")) +
  scale_fill_manual(name = "Response",
                    labels = c("Disease", "Control"),
                    # color comes from cividis(2, alpha = 0.5)
                    # and inferno(3, alpha = 0.5)
                    values = c("#000000", "#BB375480")) +
  scale_shape_manual(name = "Response",
                     labels = c("Disease", "Control"),
                     # color comes from cividis(2, alpha = 0.5)
                     # and inferno(3, alpha = 0.5)
                     values = c(21, 24)) +
  xlab("PC1") +
  ylab("PC2") +
  theme(legend.direction = 'vertical',
        legend.position = 'right',
        text=element_text(face="bold", size = 10),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"))
DM_biplot
ggsave("DM_biplot_georgeEdit2.tiff", units="in", width=6, height=4, dpi=1000, compression = 'lzw')

# Figure 5)

RFI <- read_excel("G:/My Drive/WFU/Nails/manuscript/manuscript_univariate.xlsx",
                         sheet = "RFI")

RFI <- arrange(RFI, desc(RFI$gini))
RFI$Feature <- factor(RFI$Feature,
                             levels = RFI$Feature[order(RFI$gini)])

RFI_ggBarplot <- ggplot(data = RFI, aes(x = Feature, y = gini)) +
  geom_bar(stat = "identity", width = 0.5, size = 0.5, color = "#000000", fill = "#000000") +
  xlab("") +
  ylab("") +
  theme_bw() +
  coord_flip() +
  theme(legend.direction = 'vertical',
        legend.position = 'right',
        text=element_text(face="bold", size = 10),
        panel.border = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"),
        axis.text.y = element_text(vjust = 0.5))
RFI_ggBarplot
ggsave("RFI_GeorgeEdit.tiff", units="in", width=6, height=4, dpi=1000, compression = 'lzw')

# Figure 6)

# this function allows us to make ggplot object from resampling results.
# See topepo (Max Kuhn) caret::dotplot.resamples
# https://stackoverflow.com/questions/39489817/how-create-a-dotplot-equivalent-chart-using-ggplot-for-resample-class
# the function returns a data.frame
resamps_ggplot <- function(resamps_list, model_metric) {
  
  library(reshape2) 
  plotData <- melt(resamps_list$values, id.vars = "Resample")
  tmp <- strsplit(as.character(plotData$variable), "~", fixed = TRUE)
  plotData$Model <- unlist(lapply(tmp, function(x) x[1]))
  plotData$Metric <- unlist(lapply(tmp, function(x) x[2]))
  plotData <- subset(plotData, Metric == model_metric)
  plotData$variable <- factor(as.character(plotData$variable))
  plotData <- split(plotData, plotData$variable)
  results <- lapply(plotData, function(x, cl) {
    ttest <- try(t.test(x$value, conf.level = cl), silent = TRUE)
    if (class(ttest)[1] == "htest") {
      out <- c(ttest$conf.int, ttest$estimate)
      names(out) <- c("LowerLimit", "UpperLimit", "Estimate")
    }
    else out <- rep(NA, 3)
    out
  }, cl = 0.95)
  results <- as.data.frame(do.call("rbind", results))
  tmp <- strsplit(rownames(results), "~", fixed = TRUE)
  results$Model <- unlist(lapply(tmp, function(x) x[1]))
  
  results$Model <- factor(results$Model,
                          levels = results$Model[order(results$Estimate)])
  return(results)
}

models_all_resamp_ggplot <- resamps_ggplot(models_all_resamp, rfFull$metric)

# now factor variables according to type of model
# color according to stacked ensemble or not
# pch according to RFE or not

models_all_resamp_ggplot$`Model type` <- NA

for (i in 1:length(models_all_resamp_ggplot$`Model type`)){
  models_all_resamp_ggplot$`Feature selection`[i] <- ifelse(
    grepl("RFE",models_all_resamp_ggplot$Model[i],fixed = TRUE), "RFE", "None")
}

models_all_resamp_ggplot$`Feature selection` <- as.factor(
  models_all_resamp_ggplot$`Feature selection`)

for (i in 1:length(models_all_resamp_ggplot$`Model type`)){
  models_all_resamp_ggplot$`Model type`[i] <- ifelse(
    grepl("_ENS", models_all_resamp_ggplot$Model[i], fixed = TRUE),
    "Stacked ensemble", "Individual model")
}

models_all_resamp_ggplot$`Model type` <- as.factor(
  models_all_resamp_ggplot$`Model type`)

# make one more factor variable (4 levels) to show both model type
# and feature selection

models_all_resamp_ggplot$`Model type & Feature selection` <- NA

models_all_resamp_ggplot$`Model type & Feature selection`[which(models_all_resamp_ggplot$`Model type` == "Individual model" & models_all_resamp_ggplot$`Feature selection` == "None")] <- "Individual model, None"
  
models_all_resamp_ggplot$`Model type & Feature selection`[which(models_all_resamp_ggplot$`Model type` == "Stacked ensemble" & models_all_resamp_ggplot$`Feature selection` == "None")] <- "Stacked ensemble, None"
    
models_all_resamp_ggplot$`Model type & Feature selection`[which(models_all_resamp_ggplot$`Model type` == "Individual model" & models_all_resamp_ggplot$`Feature selection` == "RFE")] <- "Individual model, RFE"
      
models_all_resamp_ggplot$`Model type & Feature selection`[which(models_all_resamp_ggplot$`Model type` == "Stacked ensemble" & models_all_resamp_ggplot$`Feature selection` == "RFE")] <- "Stacked ensemble, RFE"
        
models_all_resamp_ggplot$`Model type & Feature selection` <- as.factor(
          models_all_resamp_ggplot$`Model type & Feature selection`)

# Models performance comparison
# We use color scale from viridis package for color blind friendly colors

Resamps_dotplot <- ggplot(models_all_resamp_ggplot,
                          aes(x = Model, y = Estimate,
                              color = `Model type & Feature selection`,
                              shape = `Model type & Feature selection`)) + 
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = LowerLimit, ymax = UpperLimit), size = 0.5, width = .1) +
  scale_color_manual(name = "Model type & Feature selection",
                     labels = c("Individual model, None", "Individual model, RFE",
                                "Stacked ensemble, None", "Stacked ensemble, RFE"),
                     values = c("#000000", "#000000", # color from cividis(2)
                                "#BB3754FF", "#BB3754FF")) +
  scale_shape_manual(name = "Model type & Feature selection",
                     labels = c("Individual model, None", "Individual model, RFE",
                                "Stacked ensemble, None", "Stacked ensemble, RFE"),
                     values = c(16, 17,
                                16, 17)) +
  xlab("") +
  ylab("AUC") +
  coord_flip() +
  theme_bw() +
  theme(text=element_text(face="bold", size = 10),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank())
Resamps_dotplot
ggsave("resamps_training_dot_GeorgeEdit2.tiff", units="in", width=6, height=4, dpi=1000, compression = 'lzw')

# Figure 7)

avg_ens_train_bar <- data.frame(rocs_train_k)
avg_ens_train_bar$model <- names(rocs_train_k)
colnames(avg_ens_train_bar) <- c("AUC", "model")

avg_ens_train_bar <- arrange(avg_ens_train_bar, desc(AUC))
avg_ens_train_bar$model <- factor(
  avg_ens_train_bar$model,
  levels = avg_ens_train_bar$model[order(-avg_ens_train_bar$AUC)])

aucs_k_training_barplot <- ggplot(data = avg_ens_train_bar, aes(x = model, y = AUC)) +
  geom_bar(stat = "identity", width = 0.75, size = 1, color = "#000000", fill = "#000000") +
  ylim(0,1) +
  xlab("") +
  theme_bw() +
  theme(legend.direction = 'vertical',
        legend.position = 'right',
        text=element_text(face="bold", size = 10),
        panel.border = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
aucs_k_training_barplot
ggsave("aucs_k_training_GeorgeEdits.tiff", units="in", width=6, height=4, dpi=1000, compression = 'lzw')

# Figure 8)

aucs_df <- data.frame(unlist(aucs_all, use.names = TRUE))

aucs_df <- data.frame(aucs_df)
colnames(aucs_df) <- "AUC"
aucs_df$model <- NA
aucs_df$model_type <- NA
aucs_df$RFE <- NA

aucs_df$model <- rownames(aucs_df)

aucs_df$model_type[startsWith(rownames(aucs_df), "k") &
                     !startsWith(rownames(aucs_df), "knn")] <- "Averaged ensemble"
aucs_df$model_type[endsWith(rownames(aucs_df), "_ENS")] <- "Stacked ensemble"
aucs_df$model_type[is.na(aucs_df$model_type)] <- "Individual model"
aucs_df$model_type <- factor(aucs_df$model_type,
                             c("Individual model", "Averaged ensemble",
                               "Stacked ensemble"))

for (i in 1:length(aucs_df$model)){
  aucs_df$RFE[i] <- ifelse(grepl("RFE", aucs_df$model[i], fixed = TRUE), "Yes", "No")
}

aucs_df$RFE <- as.factor(aucs_df$RFE)

aucs_df <- arrange(aucs_df, desc(AUC))

aucs_df$model <- factor(aucs_df$model, levels = aucs_df$model[order(-aucs_df$AUC)])

aucs_barplot <- ggplot(data = aucs_df,
                       aes(x = model, y = AUC,
                           color = model_type,
                           fill = model_type)) +
  geom_bar(stat = "identity", width = 0.5, aes(size = model_type)) +
  scale_color_manual(name = "Model type",
                     labels = c("Individual model", "Averaged ensemble",
                                "Stacked ensemble"),
                     values = c("#000000", "#000000",  "#BB3754FF")) +
  scale_fill_manual(name = "Model type",
                    labels = c("Individual model", "Averaged ensemble",
                               "Stacked ensemble"),
                    values = c("#000000", "#FFFFFF",  "#BB3754FF")) +
  scale_size_manual(name = "Model type",
                    labels = c("Individual model", "Averaged ensemble",
                               "Stacked ensemble"),
                    values = c(0.5, 0.5, 0.5)) +
  ylim(0,1) +
  xlab("") +
  theme_bw() +
  theme(legend.direction = 'vertical',
        legend.position = 'right',
        text=element_text(face = "bold", size = 8),
        panel.border = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
aucs_barplot
ggsave("aucs_external_GeorgeEdits2.tiff", units="in", width=6, height=4, dpi=1000, compression = 'lzw')

# Summary statistics

DM_stats_control <- DM %>%
  dplyr::filter(response == "control") %>%
  dplyr::select(-Smoker, - response, - Gender) %>%
  tidyr::gather(Element, conc) %>%
  dplyr::group_by(Element) %>%
  dplyr::mutate_all(funs(min = min,
                            max = max,
                            mean = mean,
                            sd = sd)) %>%
  dplyr::select(-conc) %>%
  dplyr::distinct(min, max, mean, sd) %>%
  dplyr::arrange(Element)

DM_stats_disease <- DM %>%
  dplyr::filter(response == "disease") %>%
  dplyr::select(-Smoker, - response, - Gender) %>%
  tidyr::gather(Element, conc) %>%
  dplyr::group_by(Element) %>%
  dplyr::mutate_all(funs(min = min,
                         max = max,
                         mean = mean,
                         sd = sd)) %>%
  dplyr::select(-conc) %>%
  dplyr::distinct(min, max, mean, sd) %>%
  dplyr::arrange(Element)

# write.table(DM_stats_control, file="DM_control_stats.csv",sep=",",row.names=F)
# write.table(DM_stats_disease, file="DM_disease_stats.csv",sep=",",row.names=F)
