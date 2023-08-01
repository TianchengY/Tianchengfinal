#' @title Process Data
#' @description This function loads the data, converts all sentences to lowercase, creates term frequency of the words, and converts words to unique ids.
#' @param data A data frame that contains the data to be processed.
#' @param context_size An integer that specifies the context size which is the length of the input word sequence to the model. Default is 3.
#' @param lowest_frequency An integer that specifies the lowest frequency for a word to be included in the vocabulary.
#'                        The word with frequency less than 3 will be replaced to a special token <UNK> . Default is 3.
#' @return A list that includes the processed data (input word sequence and target subsequent word) and
#'        the vocabulary (a table for mapping a word with its unique word id).
#' @import stringr
#' @import tm
#' @import dplyr
#' @export
process_data <- function(data, context_size=3, lowest_frequency=3) {
  # Assertions
  if (!is.data.frame(data)) {
    stop("data must be a data frame")
  }
  if (!is.numeric(context_size) || length(context_size) != 1 || round(context_size) != context_size || context_size < 0) {
    stop("context_size must be a non-negative integer")
  }
  if (!is.numeric(lowest_frequency) || length(lowest_frequency) != 1 || round(lowest_frequency) != lowest_frequency || lowest_frequency < 0) {
    stop("lowest_frequency must be a non-negative integer")
  }

  UNK_symbol <- "<UNK>"
  vocab <- c(UNK_symbol)

  # Convert all sentences to lowercase
  data_corpus <- tolower(data[,1])

  # Create term frequency of the words
  words_term_frequency <- table(unlist(str_split(data_corpus, " ")))

  # Create vocabulary
  vocab <- c(vocab, names(words_term_frequency[words_term_frequency >= lowest_frequency]))

  #print(length(vocab))

  # Create required lists
  x_data <- list()
  y_data <- list()

  # Create word to id mappings
  word_to_id_mappings <- setNames(1:length(vocab), vocab)

  # Function to get id for a given word
  # Return <UNK> id if not found
  get_id_of_word <- function(word) {
    ifelse(word %in% names(word_to_id_mappings), word_to_id_mappings[word], word_to_id_mappings[UNK_symbol])
  }

  # Creating the dataset
  for (idx in 1:nrow(data)) {
    sentence <- unlist(str_split(data[idx,1], " "))
    for (i in 1:(length(sentence) - context_size)) {
      if (i + context_size > length(sentence)) {
        # Sentence boundary reached
        # Ignoring sentence less than context_size words
        break
      }
      # Convert word to id
      x_extract <- sapply(sentence[i:(i + context_size - 1)], get_id_of_word)
      y_extract <- c(get_id_of_word(sentence[i + context_size]))

      x_data <- c(x_data, list(x_extract))
      y_data <- c(y_data, list(y_extract))
    }
  }

  # Making data frames
  x_data <- do.call(rbind, x_data)
  y_data <- do.call(rbind, y_data)

  # fill out NA if exists
  x_data <- apply(x_data, 2, function(col) {
    if (any(is.na(col))) {
      col[is.na(col)] <- 1
    }
    return(col)
  })

  y_data[is.na(y_data)] <- 1



  return(list(x_data = x_data, y_data = y_data, vocab = vocab))
}


#' @title Split Data
#' @description This function shuffles and splits the data into training, validation, and test sets.
#' @param x_data A data frame that contains the processed data.
#' @param y_data A data frame that contains the labels.
#' @param vocab A table for mapping a word with its unique word id.
#' @param random_seed An integer that sets the seed for reproducibility. Default is 900.
#' @param train_portion A numeric value between 0 and 1 that specifies the portion of the data to be used for training. Default is 0.8.
#' @param val_portion A numeric value between 0 and 1 that specifies the portion of the data to be used for validation. Default is 0.1.
#' @param test_portion A numeric value between 0 and 1 that specifies the portion of the data to be used for testing. Default is 0.1.
#' @return A list that includes the training, validation, and test sets, and the vocabulary (a table for mapping a word with its unique word id).
#' @import stringr
#' @import tm
#' @import dplyr
#' @export
split_data <- function(x_data, y_data, vocab, random_seed=900, train_portion=0.8, val_portion=0.1, test_portion=0.1) {
  # Assertions
  if (!is.character(vocab) || length(vocab) < 1) {
    stop("vocab must be a non-empty character vector")
  }
  if (!is.numeric(random_seed) || length(random_seed) != 1 || round(random_seed) != random_seed || random_seed < 0) {
    stop("random_seed must be a non-negative integer")
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

  # Shuffle data for Monte Carlo simulation
  set.seed(random_seed)  # for reproducibility
  indices <- sample(1:nrow(x_data))

  # Split data into training, validation, and test sets
  train_indices <- indices[1:floor(train_portion * length(indices))]
  val_indices <- indices[(floor(train_portion * length(indices)) + 1):(floor((train_portion+val_portion) * length(indices)))]
  test_indices <- indices[(floor((train_portion+val_portion) * length(indices)) + 1):length(indices)]

  x_train <- x_data[train_indices, ]
  y_train_onehot <- to_categorical(y_data[train_indices, ], num_classes = length(vocab))
  x_val <- x_data[val_indices, ]
  y_val_onehot <- to_categorical(y_data[val_indices, ], num_classes = length(vocab))
  x_test <- x_data[test_indices, ]
  y_test <- y_data[test_indices, ]


  return(list(x_train = x_train, y_train_onehot = y_train_onehot, x_val = x_val, y_val_onehot = y_val_onehot, x_test = x_test, y_test = y_test,
              vocab=vocab))
}
