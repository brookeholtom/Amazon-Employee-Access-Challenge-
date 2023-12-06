library(tidyverse)
library(ggplot2)
library(vroom)
library(tidymodels)
library(embed)


my_recipe <- recipe(rFormula, data=myDataset) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(var2, threshold = .05) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(target_var)) #target encoding
# also step_lencode_glm() and step_lencode_bayes()

# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)


