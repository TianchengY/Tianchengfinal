library(testthat)
library(Tianchengfinal)

# Load the data
data("rocstories")

# Test process_data function
test_that("process_data function works correctly", {
  result <- process_data(rocstories, context_size = 3, lowest_frequency = 3)

  # Check if the result is a list
  expect_is(result, "list")

  # Check if the list has 3 elements
  expect_equal(length(result), 3)

})

# Test split_data function
test_that("split_data function works correctly", {
  result <- process_data(rocstories, context_size = 3, lowest_frequency = 3)
  split_result <- split_data(result$x_data, result$y_data, result$vocab, random_seed = 900, train_portion = 0.8, val_portion = 0.1, test_portion = 0.1)

  # Check if the result is a list
  expect_is(split_result, "list")

  # Check if the list has 7 elements
  expect_equal(length(split_result), 7)

})


test_that("process_data function works correctly", {
  processed_data <- process_data(data, context_size=3, lowest_frequency=3)

  # Check that the function returns a list
  expect_is(processed_data, "list")

  # Check that the list has the correct names
  expect_equal(names(processed_data), c("x_data", "y_data", "vocab"))

})

test_that("split_data function works correctly", {
  processed_data <- process_data(rocstories, context_size=3, lowest_frequency=3)
  split_data_result <- split_data(processed_data$x_data, processed_data$y_data, processed_data$vocab, random_seed=900, train_portion=0.8, val_portion=0.1, test_portion=0.1)

  # Check that the function returns a list
  expect_is(split_data_result, "list")

  # Check that the list has the correct names
  expect_equal(names(split_data_result), c("x_train", "y_train_onehot", "x_val", "y_val_onehot", "x_test", "y_test", "vocab"))

})


# Process the data
processed_data <- process_data(data = rocstories, context_size = 3, lowest_frequency = 3)
x_data <- processed_data$x_data
y_data <- processed_data$y_data
vocab <- processed_data$vocab

# Test the function
test_that("split_data function throws an error when arguments are incorrect", {
  # Test that an error is thrown when vocab is not a character vector
  expect_error(split_data(x_data = x_data, y_data = y_data, vocab = 1, random_seed = 900, train_portion = 0.8, val_portion = 0.1, test_portion = 0.1))

  # Test that an error is thrown when random_seed is not a non-negative integer
  expect_error(split_data(x_data = x_data, y_data = y_data, vocab = vocab, random_seed = "900", train_portion = 0.8, val_portion = 0.1, test_portion = 0.1))

  # Test that an error is thrown when train_portion is not a numeric value between 0 and 1
  expect_error(split_data(x_data = x_data, y_data = y_data, vocab = vocab, random_seed = 900, train_portion = 1.5, val_portion = 0.1, test_portion = 0.1))

  # Test that an error is thrown when val_portion is not a numeric value between 0 and 1
  expect_error(split_data(x_data = x_data, y_data = y_data, vocab = vocab, random_seed = 900, train_portion = 0.8, val_portion = 1.5, test_portion = 0.1))

  # Test that an error is thrown when test_portion is not a numeric value between 0 and 1
  expect_error(split_data(x_data = x_data, y_data = y_data, vocab = vocab, random_seed = 900, train_portion = 0.8, val_portion = 0.1, test_portion = 1.5))
})

