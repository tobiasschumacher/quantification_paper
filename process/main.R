rm(list = ls())
gc()

# load all functions to build results
source("build.R")


################################################################################
# BUILD RESULTS FRAMES FROM CSV-FILES
################################################################################

## build tables that combine all results of a data set and saves them to file
for(ds_name in DATASET_LIST){
  build(ds_name)
}

## enrich result frames with performance values

# results from main experiments
for(ds_name in DATASET_LIST){
  print(ds_name)
  add_metrics(ds_name, c("AE","NKLD"))
}


## enrich result frames with training-test-difference values

# results from main experiments
for(ds_name in DATASET_LIST){
  print(ds_name)
  add_ttdist(ds_name)
}
