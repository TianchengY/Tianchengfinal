#' @title Create Table
#' @description This function selects specific columns from the results data frame and creates a table.
#' @param results A data frame that contains the results of the simulations. It should have columns for "embedding_dim", "context_size", "train_accuracy", "mcse_train_accuracy", "test_accuracy", and "mcse_test_accuracy"
#' @return A table that contains the selected columns.
#' @import dplyr
#' @export
create_table <- function(results) {
  # Check if 'results' is a data frame
  if (!is.data.frame(results)) {
    stop("results must be a data frame")
  }

  # Add assertion to check if required columns exist in the data frame
  required_columns <- c("embedding_dim", "param_value", "train_accuracy", "mcse_train_accuracy", "test_accuracy", "mcse_test_accuracy")
  if (!all(required_columns %in% colnames(results))) {
    stop(paste("results must contain the following columns:", paste(required_columns, collapse = ", ")))
  }

  # Select columns and create table
  results$context_size <- results$param_value
  results_table <- dplyr::select(results, embedding_dim, context_size, train_accuracy, mcse_train_accuracy, test_accuracy, mcse_test_accuracy)

  return(results_table)
}
