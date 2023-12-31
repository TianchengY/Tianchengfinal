% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rocstories.R
\docType{data}
\name{rocstories}
\alias{rocstories}
\title{ROCStories Text Corpus}
\format{
A data frame with 9426 rows and 1 column:
\describe{
  \item{X1}{The text of the sentence.}
}
}
\source{
Mostafazadeh, N., Chambers, N., He, X., Parikh, A., Batra, D., Vanderwende, L., Kohli, P., & Allen, J. (2016, June). A corpus and cloze evaluation for deeper understanding of commonsense stories. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 839-849). San Diego, California: Association for Computational Linguistics.
}
\usage{
rocstories
}
\description{
A text corpus derived from the ROCStories dataset. It comprises 9426 lines of English sentences. Each six adjacent sentences in this dataset form a small story.
}
\details{
All sentences in the ROCStories dataset were converted to lowercase first and a word frequency table was created. Words with a frequency of 3 or above were selected to form the vocabulary table (words with frequency less than 3 was replaced with a special character 'UNK'), and each word was assigned a unique ID. Then, for each simulation, the dataset was shuffled and partitioned into training, validation, and testing sets with a proportion of 80%, 10% and 10%, and each word was replaced with its corresponding ID. A series of training instances and corresponding labels were created as well, where each training instance consisted of 'context size' continuous words, and the corresponding label was the word following these words.
}
\keyword{datasets}
