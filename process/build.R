library(data.table)
library(stringr)

source("metrics.R")


################################################################################
# GLOBAL VARIABLES
################################################################################

# PATHS
RES_PATH <- "../results/preprocessed/"
EXP_PATH <- "../results/raw/"
TABLE_PATH <- "../results/tables/"
RES_FILES <- list.files(EXP_PATH)
WD <- getwd()

# GENERAL VARIABLES
DATA_INDEX <- fread("../data/data_index.csv", sep = ';')
QUANTIFIER_INDEX <- fread("../quantifier_index.csv", sep = ';')

DATASET_LIST <- DATA_INDEX$dataset
QUANTIFIER_LIST <- QUANTIFIER_INDEX$algorithm
MEASURES <- c("AE","NKLD")


## train/test-splits
TRAINING_SPLITS <- vector(mode = "list", length = 4)
TEST_SPLITS <- vector(mode = "list", length = 4)

TRAINING_SPLITS[[1]] <- c("10/90","30/70","50/50","70/30","90/10", "95/5")
TEST_SPLITS[[1]] <- c("10/90","20/80","30/70","40/60","50/50","60/40","70/30","80/20","90/10","95/5","99/1","100/0")

TRAINING_SPLITS[[2]] <- c("20/50/30","5/80/15","35/30/35")
TEST_SPLITS[[2]] <- c("10/70/20","55/10/35","35/55/10","40/25/35","0/5/95")

TRAINING_SPLITS[[3]] <- c("50/30/10/10","70/10/10/10","25/25/25/25")
TEST_SPLITS[[3]] <- c("65/25/05/05","20/25/30/25","45/15/20/20","20/0/0/80","30/25/35/10")

TRAINING_SPLITS[[4]] <- c("5/20/10/20/45","5/10/70/10/5","20/20/20/20/20")
TEST_SPLITS[[4]] <- c("15/10/65/10/0","45/10/30/5/10","20/25/25/10/20","35/5/5/5/50","5/25/15/15/40")

###
GLOBAL_SEEDS = c(4711, 42, 4055, 666, 90210, 512, 1337, 879, 711, 1812)

KEY_COLS <- c("Seed", "TT_split","D_train", "D_test")



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
     
    dta <- fread(paste0(EXP_PATH,fname), sep = ';')

    ## patch handling
    if(nrow(dta) < length(TEST_SPLITS[[L-1]])*length(TRAINING_SPLITS[[L-1]])*4){
      dta[,TT_split := get_TT_split(Training_Ratio)]
      trel <- grep("Relative",colnames(dta))
      t1 <- grep("Training_Class",colnames(dta))
      t2 <- grep("Test_Class",colnames(dta))
      dta[,D_train := get_str_ratio(trel,t1)]
      dta[,D_test := get_str_ratio(trel,t2)]
    }else{
      
      dta[,TT_split := get_TT_split(Training_Ratio)]
      
      combs <- expand.grid(TEST_SPLITS[[L-1]], TRAINING_SPLITS[[L-1]])
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
  
  labels <- DATA_INDEX[dataset == ds_name, class_labels]
  labels <- paste0("c",gsub("\\[","(",labels))
  labels <- gsub("\\]",")",labels)
  labels <- eval(parse(text = labels))
  L = length(labels)
  
  ds_files <- RES_FILES[grep(paste0(ds_name,"_seed"), RES_FILES)]
  dta_res <- read_prep_data(ds_files[1],L)
  setkeyv(dta_res, KEY_COLS)
  
  for(f in ds_files[-1]){
    dta_tmp <- read_prep_data(f,L)
    setkeyv(dta_tmp, KEY_COLS)
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
  
  fwrite(dta_res, paste0(RES_PATH,ds_name,"_stats.csv"))
}


################################################################################
# Compute performance MEASURES from metrics.R to dataframes resulting from build
################################################################################
add_metrics <- function(ds_name, metrics){
  dta <- fread(paste0(RES_PATH,ds_name, "_stats.csv"))
  
  
  labels <- DATA_INDEX[dataset == ds_name, class_labels]
  labels <- paste0("c",gsub("\\[","(",labels))
  labels <- gsub("\\]",")",labels)
  labels <- eval(parse(text = labels))
  L = length(labels)

  fname <- paste0(RES_PATH,ds_name,"_stats_metrics.csv")
  alg_cols <- colnames(dta)[str_detect(colnames(dta), "_Prediction")]
  alg_names <- str_split(alg_cols, "_Prediction")
  alg_names <- unique(unlist(lapply(alg_names, function(s) return(s[[1]]))))

  train_ds <- paste0("Training_Class_",labels,"_Relative")
  test_ds <- paste0("Test_Class_",labels,"_Relative")
  alg_cols <- lapply(alg_names, function(alg) paste0(alg,"_Prediction_Class_",labels))
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
add_ttdist <- function(ds_name){
  tt_distance <- function(train_cols, test_cols){
    lapply(1:n,
           function(i){
             p1 <- eval(parse(text=paste0("c(",gsub("\\/",",",train_cols[i]),")")))
             p2 <- eval(parse(text=paste0("c(",gsub("\\/",",",test_cols[i]),")")))
             return(MAE(p1/100,p2/100))
           })
  }
  fname <- paste0(RES_PATH,ds_name, "_stats_metrics.csv")
  
  dta <- fread(fname)
  n <- nrow(dta)
  dta[, Drift_MAE := tt_distance(D_train, D_test)]
  
  fwrite(dta, fname)
}