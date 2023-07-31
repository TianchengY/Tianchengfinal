#' @title Plot Test Accuracy with MCSE
#' @description This function plots the test accuracy with Monte Carlo Standard Error (MCSE) for different parameter values and embedding dimensions.
#' @param results A data frame that contains the results of the simulations. It should have columns for 'param_value', 'test_accuracy', 'test_lower_bound', 'test_upper_bound', and 'embedding_dim'.
#' @param param_name A string that specifies the name of the parameter that was varied in the simulations.
#' @param param_values A numeric vector that contains the values of the parameter that were used in the simulations.
#' @param colors A character vector that specifies the colors to be used in the plot. Default is c("red", "blue", "yellow", "red", "blue").
#' @param linetypes A character vector that specifies the line types to be used in the plot. Default is c("solid", "dashed", "solid", "dashed", "solid").
#' @param plot_title A string that specifies the title of the plot. Default is "Test Accuracy with MCSE".
#' @return A ggplot object that represents the plot.
#' @export
plot_test_accuracy <- function(results, param_name, param_values, colors = c("red", "blue", "yellow", "red", "blue"), linetypes = c("solid", "dashed", "solid", "dashed", "solid")) {
  # Assertions
  if (!is.data.frame(results)) {
    stop("results must be a data frame")
  }
  # Add assertion to check if required columns exist in the data frame
  required_columns <- c("embedding_dim", "param_value", "test_accuracy", "test_lower_bound", "test_upper_bound")
  if (!all(required_columns %in% colnames(results))) {
    stop(paste("results must contain the following columns:", paste(required_columns, collapse = ", ")))
  }
  if (!is.character(param_name) || length(param_name) != 1) {
    stop("param_name must be a single string")
  }
  if (!is.numeric(param_values) || length(param_values) < 1) {
    stop("param_values must be a non-empty numeric vector")
  }
  if (!is.character(colors) || length(colors) < 1) {
    stop("color_palette must be a non-empty character vector")
  }
  if (!is.character(linetypes) || length(linetypes) < 1) {
    stop("linetype_palette must be a non-empty character vector")
  }
  if (!is.character(plot_title) || length(plot_title) != 1) {
    stop("plot_title must be a single string")
  }

  # plot a line chart for test accuracy
  plot <- ggplot(results, aes(x = param_value, y = test_accuracy, colour = as.factor(embedding_dim), linetype = as.factor(embedding_dim))) +
    geom_ribbon(aes(ymin = test_lower_bound, ymax = test_upper_bound), alpha = 0.2) +
    geom_line() +
    scale_x_continuous(breaks = param_values) +
    scale_color_manual(values = colors) +
    scale_linetype_manual(values = linetypes) +
    labs(x = param_name, y = "Accuracy", colour = "Embedding Dimension", linetype = "Embedding Dimension") +
    theme_bw() +
    ggtitle("Test Accuracy with MCSE")

  return(plot)
}

#' @title Plot Training Accuracy with MCSE
#' @description This function plots the train accuracy with Monte Carlo Standard Error (MCSE) for different parameter values and embedding dimensions.
#' @param results A data frame that contains the results of the simulations. It should have columns for 'param_value', 'train_accuracy', 'train_lower_bound', 'train_upper_bound', and 'embedding_dim'.
#' @param param_name A string that specifies the name of the parameter that was varied in the simulations.
#' @param param_values A numeric vector that contains the values of the parameter that were used in the simulations.
#' @param colors A character vector that specifies the colors to be used in the plot. Default is c("red", "blue", "yellow", "red", "blue").
#' @param linetypes A character vector that specifies the line types to be used in the plot. Default is c("solid", "dashed", "solid", "dashed", "solid").
#' @param plot_title A string that specifies the title of the plot. Default is "Training Accuracy with MCSE".
#' @return A ggplot object that represents the plot.
#' @export
plot_train_accuracy <- function(results, param_name, param_values, colors = c("red", "blue", "yellow", "red", "blue"), linetypes = c("solid", "dashed", "solid", "dashed", "solid")) {
  # Assertions
  if (!is.data.frame(results)) {
    stop("results must be a data frame")
  }
  # Add assertion to check if required columns exist in the data frame
  required_columns <- c("embedding_dim", "param_value", "train_accuracy", "train_lower_bound", "train_upper_bound")
  if (!all(required_columns %in% colnames(results))) {
    stop(paste("results must contain the following columns:", paste(required_columns, collapse = ", ")))
  }
  if (!is.character(param_name) || length(param_name) != 1) {
    stop("param_name must be a single string")
  }
  if (!is.numeric(param_values) || length(param_values) < 1) {
    stop("param_values must be a non-empty numeric vector")
  }
  if (!is.character(colors) || length(colors) < 1) {
    stop("color_palette must be a non-empty character vector")
  }
  if (!is.character(linetypes) || length(linetypes) < 1) {
    stop("linetype_palette must be a non-empty character vector")
  }
  if (!is.character(plot_title) || length(plot_title) != 1) {
    stop("plot_title must be a single string")
  }

  # plot a line chart for train accuracy
  plot <- ggplot(results, aes(x = param_value, y = train_accuracy, colour = as.factor(embedding_dim), linetype = as.factor(embedding_dim))) +
    geom_ribbon(aes(ymin = train_lower_bound, ymax = train_upper_bound), alpha = 0.2) +
    geom_line() +
    scale_x_continuous(breaks = param_values) +
    scale_color_manual(values = colors) +
    scale_linetype_manual(values = linetypes) +
    labs(x = param_name, y = "Accuracy", colour = "Embedding Dimension", linetype = "Embedding Dimension") +
    theme_bw() +
    ggtitle("Training Accuracy with MCSE")

  return(plot)
}
