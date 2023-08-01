#' @title Run Simulations
#' @description This function runs simulations for combinations of embedding size and target hyperparameters.
#' @param embedding_dim_values A numeric vector that contains the values of the embedding dimension to be tested.
#' @param param_values A numeric vector that contains the values of the target hyperparameter to be tested.
#' @param param_name A character string that specifies the name of the target hyperparameter. Could be one of "context_size", "h", or "batch_size". Default is "context_size".
#' @param other_params A list that contains the values of the other hyperparameters. The list should include the following elements:
#'   * data: A data frame that contains the data to be processed.
#'   * n_simulations: An integer that specifies the number of simulations to run. Default is 50.
#'   * random_seed: An integer that sets the seed for reproducibility. Default is 900.
#'   * train_portion: A numeric value that specifies the portion of the data to be used for training. Default is 0.8.
#'   * val_portion: A numeric value that specifies the portion of the data to be used for validation. Default is 0.1.
#'   * test_portion: A numeric value that specifies the portion of the data to be used for testing. Default is 0.1.
#'   * lowest_frequency: An integer that specifies the lowest frequency of words to be included in the vocabulary. Default is 3.
#'   * batch_size: An integer that specifies the batch size. Default is 256.
#'   * epochs: An integer that specifies the number of epochs. Default is 20.
#'   * h: An integer that specifies the number of units in the hidden layer. Default is 50.
#'   * learning_rate: A numeric value that specifies the learning rate. Default is 5e-3.
#'   * early_stop_min_delta: A numeric value that specifies the minimum change in the monitored quantity to qualify as an improvement. Default is 0.01.
#'   * early_stop_patience: An integer that specifies the number of epochs with no improvement after which training will be stopped. Default is 2.
#'   * verbose: An integer that specifies the verbosity mode. Default is 1.
#' @return A data frame that includes the results of the simulations.
#' @examples
#' # define hyperparameters you want to explore
#' embedding_dim_values <- c(10000,3319,300,30,3)
#' context_size_values <- c(2, 3, 4)
#' data("rocstories", package = "Tianchengfinal")
#' # must do: define the parameters passed to the run_simulation function
#' other_params <- list(data = rocstories, n_simulations = 50, random_seed = 900, train_portion = 0.8, val_portion = 0.1, test_portion = 0.1, lowest_frequency = 3, batch_size=256, epochs=20, h=50, learning_rate=5e-3, early_stop_min_delta=0.01,early_stop_patience=2,verbose=1)
#' results <- run_simulations(embedding_dim_values=embedding_dim_values, param_values=context_size_values, param_name="context_size", other_params)
#' @import stringr
#' @import tm
#' @import dplyr
#' @import tensorflow
#' @import keras
#' @importFrom mcmcse mcse
#'
#' @export
run_simulations <- function(embedding_dim_values, param_values, param_name, other_params) {
  # Assertions
  if (!is.numeric(embedding_dim_values) || length(embedding_dim_values) < 1) {
    stop("embedding_dim_values must be a non-empty numeric vector")
  }
  if (!is.numeric(param_values) || length(param_values) < 1) {
    stop("param_values must be a non-empty numeric vector")
  }
  if (!is.character(param_name) || length(param_name) != 1) {
    stop("param_name must be a single character string")
  }
  if (!is.list(other_params)) {
    stop("other_params must be a list")
  }

  # create an empty data frame to store the results
  results <- data.frame()

  # loop over the parameter values
  for (embedding_dim in embedding_dim_values) {
    for (param_value in param_values) {
      # create a list of parameters for the run_simulation function
      params <- c(list(embedding_dim = embedding_dim), other_params)
      if (param_name == "context_size") {
        params$context_size <- param_value
      } else if (param_name == "h") {
        params$h <- param_value
      } else if (param_name == "batch_size") {
        params$batch_size <- param_value
      }

      # run the simulation with the appropriate parameters
      sim_results <- do.call(run_simulation, params)

      # add the results to the data frame
      results <- rbind(results, data.frame(embedding_dim = embedding_dim,
                                           param_value = param_value,
                                           test_accuracy = sim_results$mean_test_accuracy,
                                           mcse_test_accuracy =  sim_results$mcse_test_accuracy,
                                           test_upper_bound = sim_results$mean_test_accuracy + sim_results$mcse_test_accuracy,
                                           test_lower_bound = sim_results$mean_test_accuracy - sim_results$mcse_test_accuracy,
                                           train_accuracy = sim_results$mean_train_accuracy,
                                           mcse_train_accuracy =  sim_results$mcse_train_accuracy,
                                           train_upper_bound = sim_results$mean_train_accuracy + sim_results$mcse_train_accuracy,
                                           train_lower_bound = sim_results$mean_train_accuracy - sim_results$mcse_train_accuracy))
    }
  }

  # return the results
  return(results)
}


#' @title Run Simulation
#' @description This function runs a series of simulations for the neural network model.
#' @param data A data frame that contains the data to be processed.
#' @param n_simulations An integer that specifies the number of simulations to run. Default is 10.
#' @param random_seed An integer that sets the seed for reproducibility. Default is 900.
#' @param context_size An integer that specifies the context size. Default is 3.
#' @param train_portion A numeric value that specifies the portion of the data to be used for training. Default is 0.8.
#' @param val_portion A numeric value that specifies the portion of the data to be used for validation. Default is 0.1.
#' @param test_portion A numeric value that specifies the portion of the data to be used for testing. Default is 0.1.
#' @param lowest_frequency An integer that specifies the lowest frequency. Default is 3.
#' @param embedding_dim An integer that specifies the dimension of the embedding layer. Default is 30.
#' @param batch_size An integer that specifies the batch size. Default is 256.
#' @param epochs An integer that specifies the number of epochs. Default is 10.
#' @param h An integer that specifies the number of units in the hidden layer. Default is 50.
#' @param learning_rate A numeric value that specifies the learning rate. Default is 5e-3.
#' @param early_stop_min_delta A numeric value that specifies the minimum change in the monitored quantity to qualify as an improvement. Default is 0.05.
#' @param early_stop_patience An integer that specifies the number of epochs with no improvement after which training will be stopped. Default is 3.
#' @param verbose An integer that specifies the verbosity mode. Default is 1.
#' @return A list that includes the mean test accuracy, the Monte Carlo Standard Error of the test accuracy, the mean train accuracy, and the Monte Carlo Standard Error of the train accuracy.
#' @import stringr
#' @import tm
#' @import dplyr
#' @import tensorflow
#' @import keras
#' @importFrom mcmcse mcse
#' @examples
#' # load data
#' data("rocstories", package = "Tianchengfinal")
#' # run simulation with default hyperparameter settings
#' run_simulation(data = rocstories, n_simulations=10, random_seed=900)
#'
#' @export
run_simulation <- function(data,n_simulations=10, random_seed=900, context_size=3, train_portion=0.8, val_portion=0.1, test_portion=0.1, lowest_frequency=3, embedding_dim=30, batch_size=256, epochs=20, h=50, learning_rate=5e-3, early_stop_min_delta=0.05,early_stop_patience=3,verbose=1) {
  # Assertions
  if (!is.numeric(n_simulations) || length(n_simulations) != 1 || round(n_simulations) != n_simulations || n_simulations < 0) {
    stop("n_simulations must be a non-negative integer")
  }
  if (!is.data.frame(data)) {
    stop("data must be a data frame")
  }
  if (!is.numeric(random_seed) || length(random_seed) != 1 || round(random_seed) != random_seed || random_seed < 0) {
    stop("random_seed must be a non-negative integer")
  }
  if (!is.numeric(context_size) || length(context_size) != 1 || round(context_size) != context_size || context_size < 0) {
    stop("context_size must be a non-negative integer")
  }
  if (!is.numeric(train_portion) || length(train_portion) != 1 || train_portion < 0 || train_portion > 1) {
    stop("train_portion must be a numeric value between 0 and 1")
  }
  if (!is.numeric(val_portion) || length(val_portion) != 1 || val_portion < 0 || val_portion > 1) {
    stop("val_portion must be a numeric value between 0 and 1")
  }
  if (!is.numeric(test_portion) || length(test_portion) != 1 || test_portion < 0 || test_portion > 1) {
    stop("test_portion must be a numeric value between 0 and 1")
  }
  if (!is.numeric(lowest_frequency) || length(lowest_frequency) != 1 || round(lowest_frequency) != lowest_frequency || lowest_frequency < 0) {
    stop("lowest_frequency must be a non-negative integer")
  }
  if (!is.numeric(embedding_dim) || length(embedding_dim) != 1 || round(embedding_dim) != embedding_dim || embedding_dim < 0) {
    stop("embedding_dim must be a non-negative integer")
  }
  if (!is.numeric(batch_size) || length(batch_size) != 1 || round(batch_size) != batch_size || batch_size < 0) {
    stop("batch_size must be a non-negative integer")
  }
  if (!is.numeric(epochs) || length(epochs) != 1 || round(epochs) != epochs || epochs < 0) {
    stop("epochs must be a non-negative integer")
  }
  if (!is.numeric(h) || length(h) != 1 || round(h) != h || h < 0) {
    stop("h must be a non-negative integer")
  }
  if (!is.numeric(learning_rate) || length(learning_rate) != 1 || learning_rate < 0) {
    stop("learning_rate must be a non-negative numeric value")
  }
  if (!is.numeric(early_stop_min_delta) || length(early_stop_min_delta) != 1 || early_stop_min_delta < 0) {
    stop("early_stop_min_delta must be a non-negative numeric value")
  }
  if (!is.numeric(early_stop_patience) || length(early_stop_patience) != 1 || round(early_stop_patience) != early_stop_patience || early_stop_patience < 0) {
    stop("early_stop_patience must be a non-negative integer")
  }
  if (!is.numeric(verbose) || length(verbose) != 1 || round(verbose) != verbose || verbose < 0) {
    stop("verbose must be a non-negative integer")
  }

  set.seed(random_seed)
  tensorflow::set_random_seed(random_seed)
  test_accuracies <- numeric(n_simulations)
  train_accuracies <- numeric(n_simulations)

  processed_data <- process_data(data=data, context_size=context_size, lowest_frequency=lowest_frequency)

  for (i in 1:n_simulations) {
    print(paste("Running simulation", i,"context_size: ",context_size, "embedding size: ",embedding_dim))

    # Set a different random seed for each simulation
    seed <- sample.int(10000, 1)

    # Preprocess the data
    data <- split_data (x_data=processed_data$x_data, y_data=processed_data$y_data,vocab=processed_data$vocab, random_seed=seed, train_portion=train_portion, val_portion=val_portion, test_portion=test_portion)

    # Train the model
    model_results <- model_training(data$x_train, data$y_train_onehot, data$x_val, data$y_val_onehot, data$vocab, embedding_dim=embedding_dim, context_size=context_size, random_seed=seed, batch_size=batch_size, epochs=epochs, h=h, learning_rate=learning_rate, early_stop_min_delta=early_stop_min_delta,early_stop_patience=early_stop_patience,verbose=verbose)

    # Extract the model and average training accuracy
    model <- model_results$model
    final_train_acc <- model_results$final_train_acc

    # Predict on the test set
    y_pred <- model_prediction(model, data$x_test)

    # Calculate accuracy
    accuracy <- calculate_accuracy(y_pred, data$y_test)

    # Store the accuracy
    train_accuracies[i] <- final_train_acc
    test_accuracies[i] <- accuracy

    gc()
  }

  # Calculate the mean and standard deviation of the accuracies
  mean_train_accuracy <- mean(train_accuracies)
  mcse_train_accuracy <- mcse(train_accuracies)$se

  mean_test_accuracy <- mean(test_accuracies)
  mcse_test_accuracy <- mcse(test_accuracies)$se

  print(paste("Mean train accuracy: ", mean_train_accuracy))
  print(paste("Monte Carlo Standard Error: ", mcse_train_accuracy))

  print(paste("Mean test accuracy: ", mean_test_accuracy))
  print(paste("Monte Carlo Standard Error: ", mcse_test_accuracy))


  return(list(mean_test_accuracy = mean_test_accuracy, mcse_test_accuracy = mcse_test_accuracy,
              mean_train_accuracy = mean_train_accuracy, mcse_train_accuracy = mcse_train_accuracy))
}

