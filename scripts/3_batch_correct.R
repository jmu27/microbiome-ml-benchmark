#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(sva)        # ComBat
  library(MMUPHin)    # adjust_batch
  library(foreach)
})

# ===== CLI =====
option_list <- list(
  make_option("--method", type="character", default="none",
              help="Batch method: none|ComBat|MMUPHin"),
  make_option("--train", type="character", help="Path to TRAIN csv (samples x columns incl. metadata)"),
  make_option("--test",  type="character", help="Path to TEST csv (samples x columns incl. metadata)"),
  make_option("--batch_col", type="character", default="pair_rank", help="Batch column name"),
  make_option("--label_col", type="character", default="Group", help="Label column (optional for covariates)"),
  make_option("--out_train", type="character", help="Output csv for corrected TRAIN features"),
  make_option("--out_test",  type="character", help="Output csv for corrected TEST features")
)
opt <- parse_args(OptionParser(option_list = option_list))

# ===== Read =====
train_df <- readr::read_csv(opt$train, show_col_types = FALSE)
test_df  <- readr::read_csv(opt$test,  show_col_types = FALSE)

# ===== Feature columns =====
feat_cols <- names(train_df)[grepl("^ncbi", names(train_df))]
if (!length(feat_cols)) stop("No feature columns starting with 'ncbi' in train.")
if (!all(feat_cols %in% names(test_df))) stop("Feature columns differ between train/test.")

# ===== Sanity on metadata columns =====
for (col in c(opt$batch_col, opt$label_col)) {
  if (!col %in% names(train_df)) stop("Missing column in train: ", col)
  if (!col %in% names(test_df))  stop("Missing column in test: ",  col)
}

# ===== Combine features & metadata; keep split flag =====
X_all <- dplyr::bind_rows(
  dplyr::mutate(train_df[, feat_cols, drop = FALSE], .__set = "train"),
  dplyr::mutate(test_df[,  feat_cols, drop = FALSE], .__set = "test")
)
meta_all <- dplyr::bind_rows(
  dplyr::select(train_df, all_of(c(opt$label_col, opt$batch_col))),
  dplyr::select(test_df,  all_of(c(opt$label_col, opt$batch_col)))
)
stopifnot(nrow(X_all) == nrow(meta_all))

# ===== Create unified SampleID; tibble -> data.frame to set rownames =====
SampleID <- sprintf("S%06d", seq_len(nrow(X_all)))
X_all$SampleID <- SampleID
meta_all <- meta_all %>%
  dplyr::mutate(SampleID = SampleID) %>%
  as.data.frame()
rownames(meta_all) <- meta_all$SampleID

# ----- Force batch as factor; possibly drop covariate if constant -----
meta_all[[opt$batch_col]] <- as.factor(meta_all[[opt$batch_col]])
n_batch <- nlevels(meta_all[[opt$batch_col]])

cov_name <- opt$label_col
if (length(unique(meta_all[[opt$label_col]])) < 2) {
  message("[MMUPHin] covariate '", opt$label_col, "' has <2 levels. Using covariates = NULL.")
  cov_name <- NULL
}

# ----- Diag prints -----
cat("[Diag] batch_col '", opt$batch_col, "' -> levels: ", n_batch, "\n", sep = "")
print(head(table(meta_all[[opt$batch_col]])))
if (!is.null(cov_name)) {
  cat("[Diag] label_col '", opt$label_col, "' levels: ", length(unique(meta_all[[opt$label_col]])), "\n", sep = "")
  print(head(table(meta_all[[opt$label_col]])))
} else {
  cat("[Diag] No covariate used.\n")
}

transpose_to_pxN <- function(df_np) {
  m <- t(as.matrix(df_np)); colnames(m) <- SampleID; m
}

corrected <- NULL

if (opt$method == "none") {
  corrected <- X_all[, feat_cols, drop = FALSE]; rownames(corrected) <- SampleID

} else if (opt$method == "ComBat") {
  mat <- transpose_to_pxN(X_all[, feat_cols, drop = FALSE])
  if (n_batch < 2) {
    message("[ComBat] Only one batch detected. Returning original features.")
    corrected <- X_all[, feat_cols, drop = FALSE]; rownames(corrected) <- SampleID
  } else {
    adj <- sva::ComBat(dat = mat, batch = meta_all[[opt$batch_col]])
    corrected <- as.data.frame(t(adj)); colnames(corrected) <- feat_cols
  }

} else if (opt$method == "MMUPHin") {

  # Guard: must have >=2 batches
  if (n_batch < 2) {
    message("[MMUPHin] Only one batch detected. Returning original features (no correction).")
    corrected <- X_all[, feat_cols, drop = FALSE]; rownames(corrected) <- SampleID
  } else {
    feat_mat <- transpose_to_pxN(X_all[, feat_cols, drop = FALSE])  # features x samples
    if (!identical(colnames(feat_mat), rownames(meta_all))) {
      cat("MMUPHin name check failed. Heads:\n")
      print(head(colnames(feat_mat))); print(head(rownames(meta_all)))
      stop("For MMUPHin: colnames(feature_abd) must equal rownames(data).")
    }

    # Build call with/without covariate
    if (is.null(cov_name)) {
      fit <- MMUPHin::adjust_batch(
        feature_abd = feat_mat,
        batch       = opt$batch_col,
        data        = meta_all,
        control     = list(verbose = FALSE)
      )
    } else {
      fit <- MMUPHin::adjust_batch(
        feature_abd = feat_mat,
        batch       = opt$batch_col,
        covariates  = cov_name,
        data        = meta_all,
        control     = list(verbose = FALSE)
      )
    }

    corrected <- as.data.frame(t(fit$feature_abd_adj))
    colnames(corrected) <- feat_cols
  }

} else {
  stop("Unsupported method: ", opt$method)
}

# ===== Split & Write =====
corrected <- corrected %>%
  mutate(SampleID = rownames(corrected)) %>%
  left_join(X_all[, c("SampleID", ".__set")], by = "SampleID")

train_out <- corrected[corrected$.__set == "train", feat_cols, drop = FALSE]
test_out  <- corrected[corrected$.__set == "test",  feat_cols, drop = FALSE]

readr::write_csv(train_out, opt$out_train)
readr::write_csv(test_out,  opt$out_test)

cat("[OK] Method:", opt$method, "\n")
cat("[OK] Train/Test written:\n -", opt$out_train, "\n -", opt$out_test, "\n")
