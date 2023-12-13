library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
# library(stacks)
library(embed)
library(discrim)
library(kknn)
library(themis)

all_train <- vroom("./train.csv")
all_test <- vroom("./test.csv")

all_train$loss <- log(all_train$loss)
### Initial_Split ####
boost <- boost_tree(mode = 'regression', 
                    learn_rate = tune(),
                    loss_reduction = tune(),
                    tree_depth = tune(),
                    min_n = tune()
) %>%
  set_engine('xgboost', objective = 'reg:absoluteerror')

my_recipe <- recipe(loss ~ ., all_train) %>% 
  update_role(id, new_role = 'ID') %>%
  step_scale(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .6) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  prep()

# RF
prep <- prep(my_recipe)
baked_train <- bake(prep, new_data = all_train)
baked_test <- bake(prep, new_data = all_test)

rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")


rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,ncol(all_train)-1)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(all_train, v = 5, repeats = 1)

CV_results <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mae))

bestTune <- CV_results %>%
  select_best("mae")

final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = all_train)

rf_predictions <- final_wf %>%
  predict(all_test)

rf_predictions <- rf_predictions %>%
  bind_cols(., all_test) %>%
  select(id, .pred) %>%
  rename(loss = .pred)

vroom_write(x=rf_predictions, file="rf_predictions.csv", delim=",")

# Penalized USED
my_recipe <- recipe(loss ~ ., all_train) %>% 
  update_role(id, new_role = 'ID') %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  prep()

pen_mod <- linear_reg(mixture = tune() , penalty = tune()) %>%
  set_engine("glmnet")

pen_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pen_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(all_train, v = 5, repeats = 1)

CV_results <- pen_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mae))

bestTune <- CV_results %>%
  select_best("mae")

final_wf <-
  pen_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = all_train)

pen_predictions <- final_wf %>%
  predict(amazon_test)

pen_predictions <- pen_predictions %>%
  bind_cols(., all_test) %>%
  select(id, .pred) %>%
  rename(action = .pred) %>%
  mutate(loss = exp(loss))

vroom_write(x=pen_predictions, file="pen_predictions.csv", delim=",")

# bart
my_recipe <- recipe(loss ~ ., all_train) %>% 
  update_role(id, new_role = 'ID') %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  prep()
  
bart_mod <- parsnip::bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("regression")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_mod)

tuning_grid <- grid_regular(trees(),
                            levels = 5)

folds <- vfold_cv(all_train, v = 5, repeats = 1)

CV_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mae))

bestTune <- CV_results %>%
  select_best("mae")

final_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = all_train)

bart_predictions <- final_wf %>%
  predict(all_test, type = "class")

bart_predictions <- bart_predictions %>%
  bind_cols(., all_test) %>%
  select(id, .pred) %>%
  rename(loss = .pred)

vroom_write(x=bart_predictions, file="bart_predictions.csv", delim=",")

# knn
knn_mod <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("regression") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_mod)

tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

folds <- vfold_cv(all_train, v = 10, repeats = 1)

CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mae))

bestTune <- CV_results %>%
  select_best("mae")

final_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = all_train)

knn_predictions <- final_wf %>%
  predict(all_test, type = "prob")

knn_predictions <- knn_predictions %>%
  bind_cols(., all_test) %>%
  select(id, .pred) %>%
  rename(loss = .pred)

vroom_write(x=knn_predictions, file="knn_predictions.csv", delim=",")

# Linear USED
my_recipe <- recipe(loss ~ ., all_train) %>% 
  update_role(id, new_role = 'ID') %>%
  step_scale(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .6) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  prep()

my_mod <- linear_reg() %>% # Type of model
  set_engine("lm")


lin_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = all_train) # Fit the workflow

lin_predictions <- predict(lin_wf,
                              new_data= all_test)

lin_predictions <- lin_predictions %>%
  bind_cols(., all_test) %>%
  select(id, .pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = exp(loss))

vroom_write(x=lin_predictions, file="linear_predictions.csv", delim=",")

# Poisson
all_train <- vroom("./train.csv")
all_test <- vroom("./test.csv")

all_train <- all_train %>%
  mutate(loss = log(loss))

all_train$loss <- abs(all_train$loss)

my_recipe <- recipe(loss ~ ., all_train) %>% 
  update_role(id, new_role = 'ID') %>%
  step_scale(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .6) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()

pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

pois_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = all_train) # Fit the workflow

pois_predictions <- predict(pois_wf,
                            new_data= all_test) # Use fit to predict

pois_predictions <- pois_predictions %>%
  bind_cols(., all_test) %>%
  select(id, .pred) %>%
  rename(loss = .pred) %>%
  mutate(loss = exp(loss))


vroom_write(x=pois_predictions, file="pois_predictions.csv", delim=",")









