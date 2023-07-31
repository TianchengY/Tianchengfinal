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

  # Check if the list elements are of correct types
  expect_is(result$x_data, "data.frame")
  expect_is(result$y_data, "data.frame")
  expect_is(result$vocab, "character")
})

# Test split_data function
test_that("split_data function works correctly", {
  result <- process_data(rocstories, context_size = 3, lowest_frequency = 3)
  split_result <- split_data(result$x_data, result$y_data, result$vocab, random_seed = 900, train_portion = 0.8, val_portion = 0.1, test_portion = 0.1)

  # Check if the result is a list
  expect_is(split_result, "list")

  # Check if the list has 7 elements
  expect_equal(length(split_result), 7)

  # Check if the list elements are of correct types
  expect_is(split_result$x_train, "data.frame")
  expect_is(split_result$y_train_onehot, "matrix")
  expect_is(split_result$x_val, "data.frame")
  expect_is(split_result$y_val_onehot, "matrix")
  expect_is(split_result$x_test, "data.frame")
  expect_is(split_result$y_test, "data.frame")
  expect_is(split_result$vocab, "character")
})


test_that("process_data function works correctly", {
  processed_data <- process_data(data, context_size=3, lowest_frequency=3)

  # Check that the function returns a list
  expect_is(processed_data, "list")

  # Check that the list has the correct names
  expect_equal(names(processed_data), c("x_data", "y_data", "vocab"))

})

test_that("split_data function works correctly", {
  processed_data <- process_data(data, context_size=3, lowest_frequency=3)
  split_data_result <- split_data(processed_data$x_data, processed_data$y_data, processed_data$vocab, random_seed=900, train_portion=0.8, val_portion=0.1, test_portion=0.1)

  # Check that the function returns a list
  expect_is(split_data_result, "list")

  # Check that the list has the correct names
  expect_equal(names(split_data_result), c("x_train", "y_train_onehot", "x_val", "y_val_onehot", "x_test", "y_test", "vocab"))

})
