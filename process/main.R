rm(list = ls())
gc()

# load all functions to build results
source("build.R")


################################################################################
# BUILD RESULTS FRAMES FROM CSV-FILES
################################################################################

# build tables that combine all result csv per data set
for(ds_name in dtas){
  build(ds_name)
}

# enrich result frames by performance values
for(ds_name in dtas){
  print(ds_name)
  add_metrics(ds_name, c("AE","NKLD"))
}

# enrich result frames by training-test-difference values
for(ds_name in dtas){
  print(ds_name)
  add_ttdist(ds_name)
}


################################################################################
# BUILD MATRICES OF AVERAGE PERFORMANCES PER ALGORITHM/DATASET
################################################################################


for (i in 1:length(bin_labels)) {
  dta_args <- bin_args[i]
  if (dta_args == "NULL"){
    dta_args <- NULL
  }
  lbl <- bin_labels[i]

  for (err_func in measures) {
    perf_mat <- build_perf_mat(bin_dtas, alg_names, err_func, dta_args=dta_args)
    colnames(perf_mat) <- alg_names

    rownames(perf_mat) <- bin_dta_names
    perf_mat <- perf_mat[order(row.names(perf_mat)),]
    cmeans <- colMeans(perf_mat)
    perf_mat <- rbind(perf_mat,cmeans)
    rownames(perf_mat)[nrow(perf_mat)] <- "Mean"
    fname <- paste0(lbl,"_",err_func,".csv")
    fwrite(as.data.frame(round(perf_mat,digits = 3)),
           file=paste0(table_path,fname),
           sep=',', row.names = TRUE)
    
  }
}


for (i in 1:length(mc_labels)) {
  dta_args <- mc_args[i]
  if (dta_args == "NULL"){
    dta_args <- NULL
  }
  lbl <- mc_labels[i]
  
  for (err_func in measures) {
    perf_mat <- build_perf_mat(mc_dtas, mc_names, err_func, dta_args=dta_args)
    colnames(perf_mat) <- mc_names
    rownames(perf_mat) <- mc_dta_names
    perf_mat <- perf_mat[order(row.names(perf_mat)),]
                    
    cmeans <- colMeans(perf_mat)
    perf_mat <- rbind(perf_mat,cmeans)
    rownames(perf_mat)[nrow(perf_mat)] <- "Mean"
    fname <- paste0(lbl,"_",err_func,".csv")
    fwrite(as.data.frame(round(perf_mat,digits = 3)),
           file=paste0(table_path,fname),
           sep=',', row.names = TRUE)
    
  }
}

