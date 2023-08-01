#' @title Model Training
#' @description This function creates and trains a neural network model.
#' @param x_train A data frame that contains the training data.
#' @param y_train_onehot A data frame that contains the one-hot encoded training labels.
#' @param x_val A data frame that contains the validation data.
#' @param y_val_onehot A data frame that contains the one-hot encoded validation labels.
#' @param vocab A character vector that contains the vocabulary.
#' @param embedding_dim An integer that specifies the dimension of the embedding layer. Default is 30.
#' @param context_size An integer that specifies the context size. Default is 2.
#' @param random_seed An integer that sets the seed for reproducibility. Default is 900.
#' @param batch_size An integer that specifies the batch size. Default is 256.
#' @param epochs An integer that specifies the number of epochs. Default is 10.
#' @param h An integer that specifies the number of units in the hidden layer. Default is 50.
#' @param learning_rate A numeric value that specifies the learning rate. Default is 5e-3.
#' @param early_stop_min_delta A numeric value that specifies the minimum change in the monitored quantity to qualify as an improvement. Default is 0.05.
#' @param early_stop_patience An integer that specifies the number of epochs with no improvement after which training will be stopped. Default is 3.
#' @param verbose An integer that specifies the verbosity mode. Default is 1.
#' @return A list that includes the trained model and the final training accuracy.
#' @export
model_training <- function(x_train, y_train_onehot, x_val, y_val_onehot, vocab, embedding_dim=30, context_size=2, random_seed=900, batch_size=256,epochs=10,h=50,learning_rate=5e-3,early_stop_min_delta=0.05,early_stop_patience=3,verbose=1) {
  # Assertions
  if (!is.character(vocab) || length(vocab) < 1) {
    stop("vocab must be a non-empty character vector")
  }
  if (!is.numeric(embedding_dim) || length(embedding_dim) != 1 || round(embedding_dim) != embedding_dim || embedding_dim < 0) {
    stop("embedding_dim must be a non-negative integer")
  }
  if (!is.numeric(context_size) || length(context_size) != 1 || round(context_size) != context_size || context_size < 0) {
    stop("context_size must be a non-negative integer")
  }
  if (!is.numeric(random_seed) || length(random_seed) != 1 || round(random_seed) != random_seed || random_seed < 0) {
    stop("random_seed must be a non-negative integer")
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
  if (!is.numeric(learning_rate) || length(learning_rate) != 1 || learning_rate <= 0) {
    stop("learning_rate must be a positive number")
  }
  if (!is.numeric(early_stop_min_delta) || length(early_stop_min_delta) != 1 || early_stop_min_delta < 0) {
    stop("early_stop_min_delta must be a non-negative number")
  }
  if (!is.numeric(early_stop_patience) || length(early_stop_patience) != 1 || round(early_stop_patience) != early_stop_patience || early_stop_patience < 0) {
    stop("early_stop_patience must be a non-negative integer")
  }
  if (!is.numeric(verbose) || length(verbose) != 1 || round(verbose) != verbose || verbose < 0) {
    stop("verbose must be a non-negative integer")
  }

  set.seed(random_seed)

  # Create model
  model <- keras_model_sequential() %>%
    layer_embedding(input_dim = length(vocab), output_dim = embedding_dim, input_length = context_size) %>%
    layer_flatten() %>%
    layer_dense(units = h, activation = 'tanh') %>%
    layer_dense(units = length(vocab), activation = 'softmax')

  # Compile model
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = c('accuracy')
  )

  # Define early stopping callback
  early_stopping <- callback_early_stopping(monitor = "val_loss", min_delta = early_stop_min_delta, patience = early_stop_patience, mode = "min")

  # Fit model
  history <- model %>% fit(
    x_train, y_train_onehot,
    epochs = epochs,  # Set a large number of epochs, training will stop early if no improvement
    batch_size = batch_size,
    validation_data = list(x_val, y_val_onehot),
    callbacks = list(early_stopping),  # Add early stopping callback
    verbose=verbose
  )

  # Save model
  # save_model_hdf5(model, "best_model.h5")

  # Get final training accuracy
  final_train_acc <- history$metrics$accuracy[length(history$metrics$accuracy)]

  return(list(model = model, final_train_acc = final_train_acc))
}

#' @title Model Prediction
#' @description This function used trained model to make predictions on the test set.
#' @param model A keras model object.
#' @param x_test A data frame that contains the test data.
#' @return A vector that contains the predicted labels.
#' @export
model_prediction <- function(model, x_test) {
  # Predict on test set
  y_pred <- model %>% predict(x_test) %>% k_argmax()

  return(y_pred)
}


#' @title Calculate Accuracy
#' @description This function calculates the accuracy of the model's predictions.
#' @param y_pred A vector that contains the predicted labels.
#' @param y_test A vector that contains the true labels.
#' @return A numeric value that represents the accuracy.
#' @export
calculate_accuracy <- function(y_pred, y_test) {
  # Calculate accuracy
  accuracy <- sum(as.vector(y_pred == y_test)) / length(y_test)

  #print(paste("Test accuracy: ", accuracy))

  return(accuracy)
}

#' @title Calculate Similarities
#' @description This function calculates the cosine similarity between word vectors.
#' @param model A keras model object.
#' @param words A list of character vectors that represent word pairs. Default: list(c("her", "his"), c("her", "the"))
#' @param vocab A character vector that contains the vocabulary.
#' @return A list that contains the cosine similarities for each word pair.
#' @export
calculate_similarities <- function(model, words=list(c("her", "his"), c("her", "the")), vocab) {
  # Assertions
  if (!is.list(words) || length(words) < 1) {
    stop("words must be a non-empty list of character vectors")
  }
  if (!is.character(vocab) || length(vocab) < 1) {
    stop("vocab must be a non-empty character vector")
  }

  # Function to calculate cosine similarity
  cosine_similarity <- function(a, b) {
    return(sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b))))
  }

  # Function to get id for a given word
  # Return <UNK> id if not found
  get_id_of_word <- function(word) {
    ifelse(word %in% names(word_to_id_mappings), word_to_id_mappings[word], word_to_id_mappings[UNK_symbol])
  }

  # Calculate LM similarities using cosine similarity
  lm_similarities <- list()
  for (word_pairs in words) {
    w1 <- word_pairs[1]
    w2 <- word_pairs[2]
    words_vector <- c(get_id_of_word(w1, vocab), get_id_of_word(w2, vocab))

    # Get word embeddings from the best model
    words_embeds <- get_weights(model$layers[[1]])[[1]][words_vector,]

    # Calculate cosine similarity between word vectors
    sim <- cosine_similarity(words_embeds[1,], words_embeds[2,])
    lm_similarities[[paste(word_pairs, collapse = ",")]] <- sim
  }

  #print(lm_similarities)

  return(lm_similarities)
}


