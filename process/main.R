rm(list = ls())
gc()

# load all functions to build results
source("build.R")


mc################################################################################
# BUILD RESULTS FRAMES FROM CSV-FILES
################################################################################

## build tables that combine all results of a data set and saves them to file
for(ds_name in dtas){
  build(ds_name)
}


## enrich result frames with performance values

# results from main experiments
for(ds_name in dtas){
  print(ds_name)
  add_metrics(ds_name, c("AE","NKLD"), clf=FALSE)
}

# results using tuned classifiers
for(ds_name in clf_dtas){
  print(ds_name)
  add_metrics(ds_name, c("AE","NKLD"), clf=TRUE)
}


## enrich result frames with training-test-difference values

# results from main experiments
for(ds_name in dtas){
  print(ds_name)
  add_ttdist(ds_name, clf = FALSE)
}

# results using tuned classifiers
for(ds_name in clf_dtas){
  print(ds_name)
  add_ttdist(ds_name, clf = TRUE)
}


################################################################################
# BUILD MATRICES OF AVERAGE PERFORMANCES PER ALGORITHM/DATASET
################################################################################


## MAIN EXPRERIMENTS

# binary case
write_perf_mat(bin_labels, bin_args, bin_dtas, bin_dta_names, alg_names, measures, clf=FALSE)

# multiclass case
write_perf_mat(mc_labels, mc_args, mc_dtas, mc_dta_names, mc_names, measures, clf=FALSE)


## EXPRERIMENTS ON TUNED CLASSIFIERS

# binary case
write_perf_mat(clf_bin_labels, clf_args, clf_bin_dtas, clf_bin_dta_names, clf_bin_names, measures, clf=TRUE)

# multiclass case
write_perf_mat(clf_mc_labels, clf_args, clf_mc_dtas, clf_mc_dta_names, clf_mc_names, measures, clf=TRUE)
