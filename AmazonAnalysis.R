library(tidyverse)
library(ggplot2)
library(vroom)
library(tidymodels)
library(embed)
library(ranger)
library(discrim)
library(naivebayes)
library(kknn)
library(themis)

#Read in Data
train <- vroom('./train.csv/train.csv') %>%
  mutate(ACTION = as.factor(ACTION))
test <- vroom('./test.csv/test.csv')

#Create Two plots:::
#Include Action Variable
#DataExplorer::plot_intro(train)
#DataExplorer::plot_correlation(train) 
#DataExplorer::plot_bar(train) 
#DataExplorer::plot_histogram(train) 



#Recipe that does dummy variable encoding, combines factors that occur less than 1% into an "Other Category"
my_recipe <- recipe(ACTION ~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION))


# apply the recipe to your data
prep <- prep(my_recipe)
bake(prep, new_data = train)
bake(prep, new_data= test)




#Logistic Regression

my_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod) %>%
fit(data = train) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data=test,
                              type="prob")

#amazon_predictions <- amazon_predictions %>%
#  ifelse(.pred_1>=0.7, 1, 0)



amazon_predictions <- amazon_predictions %>% 
  bind_cols(., test) %>%
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(action =.pred_1) 
  


vroom_write(x=amazon_predictions, file="./amazon_logistic.csv", delim=",")




#Penalized Logistic Regression



my_mod_penalized <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")


amazon_workflow_penalized <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod_penalized)


## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities


## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_pen <- amazon_workflow_penalized %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc))

## Find Best Tuning Parameters

bestTune <- CV_results_pen %>%
select_best("roc_auc")


## Finalize the Workflow & fit it

final_wf <-
amazon_workflow_penalized %>%
finalize_workflow(bestTune) %>%
fit(data=train)


penalized_predictions <- final_wf %>%
  predict(test, type = "prob")

penalized_predictions <- penalized_predictions %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x=penalized_predictions, file="penalized_predictions.csv", delim=",")

bestTune
  



#Random Forest 
my_recipe_forest_final_model <- recipe(ACTION ~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION))

prep <- prep(my_recipe_forest_final_model)
bake(prep, new_data = train)

my_mod_forest <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
set_engine("ranger") %>%
set_mode("classification")

## Create a workflow with model & recipe
amazon_workflow_forest <- workflow() %>%
  add_recipe(my_recipe_forest_final_model) %>%
  add_model(my_mod_forest)


## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1, ncol(train) - 1)),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities



## Set up K-fold CV
folds <- vfold_cv(train, v = 10, repeats=1)


CV_results_forest <- amazon_workflow_forest %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune <- CV_results_forest %>%
  select_best("roc_auc")


## Finalize workflow and predict

final_wf_forest <-
  amazon_workflow_forest %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)


predictions_forest <- final_wf_forest %>%
  predict(test, type = "prob")

predictions_forest <- predictions_forest %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_forest, file="predictions_forest_final.csv", delim=",")


#Naive Bayes

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") # install discrim library for the naivebayes eng6

nb_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(nb_model)

## Tune smoothness and Laplace here
tuning_grid_nb <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_nb <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_nb,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_nb <- CV_results_nb %>%
  select_best("roc_auc")

final_wf_nb <-
  nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data=train)

## Predict

predictions_nb <- final_wf_nb %>%
  predict(test, type = "prob")

predictions_nb <- predictions_nb %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_nb, file="predictions_nb.csv", delim=",")






#K Nearest Neighbors
my_recipe_knn <- recipe(ACTION ~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())


# apply the recipe to your data
prep_knn <- prep(my_recipe_knn)
bake(prep_knn, new_data = train)
bake(prep_knn, new_data= test)


knn_model <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>%
set_engine("kknn")

knn_wf <- workflow() %>%
add_recipe(my_recipe_knn) %>%
add_model(knn_model)

## Fit or Tune Model HERE
tuning_grid_knn <- grid_regular(neighbors(),
                               levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_knn <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_knn,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_knn <- CV_results_knn %>%
  select_best("roc_auc")

final_wf_knn <- knn_wf %>%
  finalize_workflow(bestTune_knn) %>%
  fit(data=train)

## Predict

predictions_knn <- final_wf_knn %>%
  predict(test, type = "prob")

predictions_knn <- predictions_knn %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_knn, file="predictions_knn.csv", delim=",")


#PCA

my_recipe_pca <- recipe(ACTION ~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.9)


# apply the recipe to your data
prep_pca <- prep(my_recipe_pca)
bake(prep_pca, new_data = train)
bake(prep_pca, new_data= test)

nb_model_pca <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng6

nb_wf_pca <- workflow() %>%
  add_recipe(my_recipe_pca) %>%
  add_model(nb_model_pca)

## Tune smoothness and Laplace here
tuning_grid_nb_pca <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_nb_pca <- nb_wf_pca %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_nb_pca,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_nb_pca <- CV_results_nb_pca %>%
  select_best("roc_auc")

final_wf_nb_pca <-
  nb_wf_pca %>%
  finalize_workflow(bestTune_nb_pca) %>%
  fit(data=train)

## Predict

predictions_nb_pca <- final_wf_nb_pca %>%
  predict(test, type = "prob")

predictions_nb_pca <- predictions_nb_pca %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_nb_pca, file="predictions_nb_pca.csv", delim=",")


#Knn PCA


# apply the recipe to your data
prep_knn_pca <- prep(my_recipe_pca)
bake(prep_knn_pca, new_data = train)
bake(prep_knn, new_data= test)


knn_model_pca <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf_pca <- workflow() %>%
  add_recipe(my_recipe_pca) %>%
  add_model(knn_model_pca)

## Fit or Tune Model HERE
tuning_grid_knn_pca <- grid_regular(neighbors(),
                                levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_knn_pca <- knn_wf_pca %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_knn_pca,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_knn_pca <- CV_results_knn_pca %>%
  select_best("roc_auc")

final_wf_knn_pca <- knn_wf_pca %>%
  finalize_workflow(bestTune_knn_pca) %>%
  fit(data=train)

## Predict

predictions_knn_pca <- final_wf_knn_pca %>%
  predict(test, type = "prob")

predictions_knn_pca <- predictions_knn_pca %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_knn_pca, file="predictions_knn_pca.csv", delim=",")


#SVM
my_recipe_svm <- recipe(ACTION ~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION)) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors())


## SVM models
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

wf_svm <- workflow() %>%
  add_recipe(my_recipe_svm) %>%
  add_model(svmRadial)



## Fit or Tune Model HERE
tuning_grid_svm <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_svm <- wf_svm %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_svm,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_svm <- CV_results_svm %>%
  select_best("roc_auc")

final_wf_svm <- wf_svm %>%
  finalize_workflow(bestTune_svm) %>%
  fit(data=train)

## Predict

predictions_svm <- final_wf_svm %>%
  predict(test, type = "prob")

predictions_svm <- predictions_svm %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_svm, file="predictions_svm_radial.csv", delim=",")



#SMOTE - Imbalanced Model
library(tidymodels)
library(themis) # for smote

my_recipe_smote <- recipe(ACTION ~., data=train) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>%
step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION)) %>%
step_normalize(all_predictors()) %>% #Everything numeric for SMOTE so encode it here
step_smote(all_outcomes(), neighbors=5)


# apply the recipe to your data
prepped_recipe_smote <- prep(my_recipe_smote)
baked <- bake(prepped_recipe_smote, new_data = train)


#Smote Logistic 
my_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe_smote) %>%
  add_model(my_mod) %>%
  fit(data = train) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data=test,
                              type="prob")

amazon_predictions <- amazon_predictions %>% 
  bind_cols(., test) %>%
  select(id, .pred_1) %>% #Just keep datetime and predictions
  rename(action =.pred_1) 


vroom_write(x=amazon_predictions, file="./amazon_logistic_smote.csv", delim=",")

#Smote Logistic 

my_mod_penalized <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")


amazon_workflow_penalized <- workflow() %>%
  add_recipe(my_recipe_smote) %>%
  add_model(my_mod_penalized)


## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities


## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_pen <- amazon_workflow_penalized %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters

bestTune <- CV_results_pen %>%
  select_best("roc_auc")


## Finalize the Workflow & fit it

final_wf <-
  amazon_workflow_penalized %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)


penalized_predictions <- final_wf %>%
  predict(test, type = "prob")

penalized_predictions <- penalized_predictions %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x=penalized_predictions, file="penalized_predictions_smote.csv", delim=",")


#Smote Forest
my_mod_forest <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## Create a workflow with model & recipe
amazon_workflow_forest <- workflow() %>%
  add_recipe(my_recipe_smote) %>%
  add_model(my_mod_forest)


## Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1, 10)),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities



## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_forest <- amazon_workflow_forest %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune <- CV_results_forest %>%
  select_best("roc_auc")


## Finalize workflow and predict

final_wf_forest <-
  amazon_workflow_forest %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)


predictions_forest <- final_wf_forest %>%
  predict(test, type = "prob")

predictions_forest <- predictions_forest %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_forest, file="predictions_forest_smote.csv", delim=",")




#Naive Bayes Smote

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng6

nb_wf <- workflow() %>%
  add_recipe(my_recipe_smote) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
tuning_grid_nb <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_nb <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_nb,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_nb <- CV_results_nb %>%
  select_best("roc_auc")

final_wf_nb <-
  nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data=train)

## Predict

predictions_nb <- final_wf_nb %>%
  predict(test, type = "prob")

predictions_nb <- predictions_nb %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_nb, file="predictions_nb_smote.csv", delim=",")

#Knn SMOTE


# apply the recipe to your data


knn_model_smote <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf_smote <- workflow() %>%
  add_recipe(my_recipe_smote) %>%
  add_model(knn_model_smote)

## Fit or Tune Model HERE
tuning_grid_knn_pca <- grid_regular(neighbors(),
                                    levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_knn_smote <- knn_wf_smote %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_knn_pca,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_knn_pca <- CV_results_knn_smote %>%
  select_best("roc_auc")

final_wf_knn_pca <- knn_wf_smote %>%
  finalize_workflow(bestTune_knn_pca) %>%
  fit(data=train)

## Predict

predictions_knn_pca <- final_wf_knn_pca %>%
  predict(test, type = "prob")

predictions_knn_pca <- predictions_knn_pca %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_knn_pca, file="predictions_knn_smote.csv", delim=",")



#Smote Linear SVM
## SVM models
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

wf_svm <- workflow() %>%
  add_recipe(my_recipe_smote) %>%
  add_model(svmLinear)



## Fit or Tune Model HERE
tuning_grid_svm <- grid_regular(cost(),
                                levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_svm <- wf_svm %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_svm,
            metrics=metric_set(roc_auc))

## Find best tuning parameters
bestTune_svm <- CV_results_svm %>%
  select_best("roc_auc")

final_wf_svm <- wf_svm %>%
  finalize_workflow(bestTune_svm) %>%
  fit(data=train)

## Predict

predictions_svm <- final_wf_svm %>%
  predict(test, type = "prob")

predictions_svm <- predictions_svm %>%
  bind_cols(., test) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x= predictions_svm, file="predictions_svm_linear_smote.csv", delim=",")
