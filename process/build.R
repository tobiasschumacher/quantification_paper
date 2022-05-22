library(data.table)
library(stringr)

source("metrics.R")


################################################################################
# GLOBAL VARIABLES
################################################################################

# PATHS
res_path <- "../results/preprocessed/"
exp_path <- "../results/raw/"
table_path <- "../results/tables/"
res_files <- list.files(exp_path)
wd <- getwd()

# GENERAL VARIABLES
dta_index <- fread("../data/data_index.csv", sep = ';')
alg_index <- fread("../alg_index.csv", sep = ';')
clf_index <- fread("../clf_index.csv", sep = ';')

dtas <- dta_index$dataset
algs <- alg_index$algorithm
alg_names <- alg_index$abbr
measures <- c("AE","NKLD")

### VARIABLES FOR BINARY ANALYSIS
bin_dtas <- dta_index[classes == 2L,dataset]
bin_dta_names <- dta_index[classes == 2L,abbr]
bin_args <- c("NULL",
              "Drift_MAE < 0.2",
              "Drift_MAE >= 0.2 & Drift_MAE < 0.4",
              "Drift_MAE >= 0.5",
              "Training_Ratio < 0.2",
              "Training_Ratio > 0.2 & Training_Ratio < 0.4",
              "Training_Ratio > 0.4 & Training_Ratio < 0.6",
              "Training_Ratio > 0.6",
              "D_test %in% c('95/5','99/1','100/0')",
              "(D_train == '90/10' || D_train == '10/90') & Drift_MAE < 0.55",
              "D_train == '50/50'"
)
bin_labels <- paste0("binary_",
                     c("general",
                       "lowshift",
                       "medshift",
                       "highshift",
                       "littletrain",
                       "sometrain",
                       "solidtrain",
                       "muchtrain",
                       "fewtestpos",
                       "trainimba",
                       "trainbal"))


### VARIABLES FOR MULTICLASS ANALYSIS
mc_dtas <- dta_index[classes > 2L,dataset]
mc_algs <- alg_index[multiclass==1, algorithm]
mc_names <- alg_index[multiclass==1, abbr]
mc_dta_names <- dta_index[classes > 2L,abbr]
mc_args <- c("NULL",
             "Drift_MAE < 0.25",
             "Drift_MAE >= 0.25",
             "Training_Ratio < 0.4",
             "Training_Ratio >= 0.4"
)
mc_labels <- paste0("mc_",
                    c("general",
                      "lowshift",
                      "highshift",
                      "fewtrain",
                      "muchtrain"))

### VARIABLES FOR CLF-IMPACT ANALYSIS
clf_dtas <- dta_index[size <= 10000L,dataset]
clf_bin_dtas <- dta_index[(size <= 10000L) & classes == 2L,dataset]
clf_mc_dtas <- dta_index[(size <= 10000L) & classes > 2L,dataset]
clf_bin_algs <- clf_index$algorithm
clf_bin_names <- clf_index$abbr
clf_mc_algs <- clf_index[multiclass==1, algorithm]
clf_mc_names <- clf_index[multiclass==1, abbr]
clf_dta_names <- dta_index[size <= 10000L,abbr]
clf_bin_dta_names <- dta_index[(size <= 10000L) & classes == 2L,abbr]
clf_mc_dta_names <- dta_index[(size <= 10000L) & classes > 2L,abbr]
clf_args <- c("NULL")
clf_bin_labels <- c("clf_binary")
clf_mc_labels <- c("clf_multiclass")


## train/test-splits
train_splits <- vector(mode = "list", length = 4)
test_splits <- vector(mode = "list", length = 4)

train_splits[[1]] <- c("10/90","30/70","50/50","70/30","90/10", "95/5")
test_splits[[1]] <- c("10/90","20/80","30/70","40/60","50/50","60/40","70/30","80/20","90/10","95/5","99/1","100/0")

train_splits[[2]] <- c("20/50/30","5/80/15","35/30/35")
test_splits[[2]] <- c("10/70/20","55/10/35","35/55/10","40/25/35","0/5/95")

train_splits[[3]] <- c("50/30/10/10","70/10/10/10","25/25/25/25")
test_splits[[3]] <- c("65/25/05/05","20/25/30/25","45/15/20/20","20/0/0/80","30/25/35/10")

train_splits[[4]] <- c("5/20/10/20/45","5/10/70/10/5","20/20/20/20/20")
test_splits[[4]] <- c("15/10/65/10/0","45/10/30/5/10","20/25/25/10/20","35/5/5/5/50","5/25/15/15/40")

###
seeds = c(4711, 42, 4055, 666, 90210, 512, 1337, 879, 711, 1812)

key_cols <- c("Seed", "TT_split","D_train", "D_test")

base_cols <- c("Total_Samples_Used", "Training_Size","Test_Size","Training_Ratio","Test_Ratio")



################################################################################
# Merge all results or a specific dataset into a single dataframe
################################################################################
build <- function(ds_name){
  
  read_prep_data <- function(fname,L){
    get_TT_split <- function(col){
      tt_ratio <- round(col, digits = 1)*100
      tt_code <- paste0(tt_ratio,"/",100-tt_ratio)
      return(tt_code)
    }
    get_str_ratio <- function(t1,t2){
      d_vec <- round(dta[,intersect(t1,t2), with = FALSE],digits = 2)*100
      res_str  <-  as.character(d_vec[[1]])
      for(i in 2:L){
        res_str <- paste0(res_str,'/',as.character(d_vec[[i]]))
      }
      return(res_str)
    }
     
    dta <- fread(paste0(exp_path,fname), sep = ';')

    ## patch handling
    if(nrow(dta) < length(test_splits[[L-1]])*length(train_splits[[L-1]])*4){
      dta[,TT_split := get_TT_split(Training_Ratio)]
      trel <- grep("Relative",colnames(dta))
      t1 <- grep("Training_Class",colnames(dta))
      t2 <- grep("Test_Class",colnames(dta))
      dta[,D_train := get_str_ratio(trel,t1)]
      dta[,D_test := get_str_ratio(trel,t2)]
    }else{
      
      dta[,TT_split := get_TT_split(Training_Ratio)]
      
      combs <- expand.grid(test_splits[[L-1]], train_splits[[L-1]])
      n <- nrow(dta)/nrow(combs)
      combs <- do.call("rbind", replicate(n, combs, simplify = FALSE))
      dta[,D_train := combs[,2]]
      dta[,D_test := combs[,1]]
      # clean up situations where no samples could be drawn 
      # under given constraints (does not happen in given comparison)
      dta <- dta[Total_Samples_Used>0]
      
    }
    
    # add seed 
    s <- strsplit(fname,"_seed_")[[1]][2]
    ind <- str_locate_all(pattern ='_', s)[[1]][1,1]
    s <- as.integer(substr(s,0,ind-1))
    dta[,Seed:=s]
  }
  
  print(ds_name)
  
  labels <- dta_index[dataset == ds_name, class_labels]
  labels <- paste0("c",gsub("\\[","(",labels))
  labels <- gsub("\\]",")",labels)
  labels <- eval(parse(text = labels))
  L = length(labels)
  
  ds_files <- res_files[grep(paste0(ds_name,"_seed"), res_files)]
  dta_res <- read_prep_data(ds_files[1],L)
  setkeyv(dta_res, key_cols)
  
  for(f in ds_files[-1]){
    dta_tmp <- read_prep_data(f,L)
    setkeyv(dta_tmp, key_cols)
    # join dataframes
    dta_res <- merge(dta_res, dta_tmp, all=TRUE)
    x_cols <- colnames(dta_res)[grep("\\.x",colnames(dta_res))]
    y_cols <- str_replace(x_cols, "\\.x", ".y")
    i <- 1
    # fill in existing columns with new results
    while(i <= length(x_cols)){
      x_col <- x_cols[i]
      y_col <- y_cols[i]
      na_y <- apply(dta_res[,y_col, with = FALSE], MARGIN = 1,function(row) is.na(row))
      dta_res[!na_y, (x_col):=get(y_col)]
      i<-i + 1
      
    }
    dta_res[,(y_cols):=NULL ]
    setnames(dta_res, old = x_cols, new = gsub("\\.x","",x_cols))
    
  }
  
  fwrite(dta_res, paste0(res_path,ds_name,"_stats.csv"))
}


################################################################################
# Compute performance measures from metrics.R to dataframes resulting from build
################################################################################
add_metrics <- function(ds_name, metrics, clf = FALSE){
  dta <- fread(paste0(res_path,ds_name, "_stats.csv"))
  
  
  labels <- dta_index[dataset == ds_name, class_labels]
  labels <- paste0("c",gsub("\\[","(",labels))
  labels <- gsub("\\]",")",labels)
  labels <- eval(parse(text = labels))
  L = length(labels)
  
  if(clf == FALSE){
    if(L>2){
      algs <-  alg_index[multiclass == 1, algorithm]
      alg_names <- alg_index[multiclass == 1, abbr]
    } else{
      algs <- alg_index$algorithm
      alg_names <- alg_index$abbr
    }
    fname = paste0(res_path,ds_name,"_stats_metrics.csv")
  }
  else{
    if(L>2){
      algs <-  clf_index[multiclass == 1, algorithm]
      alg_names <- clf_index[multiclass == 1, abbr]
    } else{
      algs <- clf_index$algorithm
      alg_names <- clf_index$abbr
    }
    fname = paste0(res_path,ds_name,"_clf_stats_metrics.csv")
  }

  
  # alg_cols <- colnames(dta)[str_detect(colnames(dta), "_Prediction")]
  # alg_names <- str_split(alg_cols, "_Prediction")
  # alg_names <- unique(unlist(lapply(alg_names, function(s) return(s[[1]]))))
  
  train_ds <- paste0("Training_Class_",labels,"_Relative")
  test_ds <- paste0("Test_Class_",labels,"_Relative")
  alg_cols <- lapply(algs, function(alg) paste0(alg,"_Prediction_Class_",labels))
  names(alg_cols) <- alg_names
  
  train_mat <- dta[,train_ds, with = FALSE]
  test_mat <- dta[,test_ds, with = FALSE]
  for(alg in alg_names){
    pred_mat <- dta[,alg_cols[[alg]], with = FALSE]
    for(M in metrics){
      args <- "test_mat[i,],pred_mat[i,]"
      if(M=="TrainBiasQ") args <- paste0(args,",train_mat[i,]")
      res_col <- unlist(lapply(1:nrow(dta),
                               function(i){
                                  p <- pred_mat[i,]
                                  if(any(!is.finite(unlist(p))) || sum(p) == 0){return(NA)} 
                                  return(eval(parse(text=paste0(M,"(",args,")"))))
                               }))
      cname <- paste0(alg,"_",M)
      dta[,(cname):=res_col]
    }
  print(alg) 
  }
  
  fwrite(dta, fname)
}


################################################################################
# Compute MAE distance between training and test set
################################################################################
add_ttdist <- function(ds_name, clf = FALSE){
  tt_distance <- function(train_cols, test_cols){
    lapply(1:n,
           function(i){
             p1 <- eval(parse(text=paste0("c(",gsub("\\/",",",train_cols[i]),")")))
             p2 <- eval(parse(text=paste0("c(",gsub("\\/",",",test_cols[i]),")")))
             return(MAE(p1/100,p2/100))
           })
  }
  if(clf){
    fname <- paste0(res_path,ds_name, "_clf_stats_metrics.csv")
  }else{
    fname <- paste0(res_path,ds_name, "_stats_metrics.csv")
  }
  
  dta <- fread(fname)
  n <- nrow(dta)
  dta[, Drift_MAE := tt_distance(D_train, D_test)]
  
  fwrite(dta, fname)
}


################################################################################
# BUILD MATRICES OF AVERAGE PERFORMANCES PER ALGORITHM/DATASET
################################################################################


write_perf_mat <- function(labels, args, dtas, dta_names, alg_names, measures, clf=FALSE){
  for (i in 1:length(labels)) {
    print(labels[i])
    dta_args <- args[i]
    if (dta_args == "NULL"){
      dta_args <- NULL
    }
    lbl <- labels[i]
    
    for (err_func in measures) {
      print(err_func)
      perf_mat <- build_perf_mat(dtas, alg_names, err_func, dta_args=dta_args, clf=clf)
      colnames(perf_mat) <- alg_names
      
      rownames(perf_mat) <- dta_names
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
}

# helper function for performance matrix construction
build_perf_mat <- function(data_sets, algs, p_metric, dta_args = NULL, clf=FALSE){
  perf_mat <- do.call(rbind,lapply(data_sets, function(ds_name){
    if(clf){
      fname <- paste0(res_path,ds_name,"_clf_stats_metrics.csv")
    }
    else{
      fname <- paste0(res_path,ds_name,"_stats_metrics.csv")
    }
    print(ds_name)
    dta <- fread(fname)
    if(!is.null(dta_args)){ eval(parse(text=paste0("dta <- dta[",dta_args,"]"))) }
    agg_mat <- do.call(cbind,lapply(algs,
                                    function(alg){
                                      col <- paste0(alg,"_",p_metric)
                                      mean(dta[,col,with = FALSE][[1]], na.rm = TRUE)
                                    }))

  }))
  return(perf_mat)
}
