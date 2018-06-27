Machine learning competition
================
David Benkeser
June 27, 2018

<!-- README.md is generated from README.Rmd. Please edit that file -->

## Assignment

The file `mashable/mashable.csv` in the GitHub repository contains 3,000
observations on articles published by Mashable (www.mashable.com). The
data set consists of 57 features that describe key aspects of the
article (details below). Additionally there are two outcome variables:
`shares` and `viral` – the latter is an indicator that an article was
shared more than 10,000 times. Your task is to develop two prediction
algorithms: one to predict the absolute number of times an article is
shared and one to predict whether an article will go ‘viral’. For the
former, you should consider building a prediction function that does
well in terms of mean squared-error. For the latter, you should consider
building a prediction function that does well in terms of negative
log-likelihood loss.

You need to turn in four things: 1. an `R` object for your continuous
outcome prediction model that can be used to predict on new data (e.g.,
`predict(you_object, newdata = ...))`); 2. an `R` object for your binary
outcome prediction model that can be used to predict on new data; 3. an
estimate of the true mean squared-error of your continuous outcome
prediction model; 4. an estimate of the true mean negative
log-likelihood of your binary outcome prediction model.

Additionally, each group will make a short (2-3 minute presentation)
detailing their approach and their results.

## Judging

Your group will be ranked relative to the other groups on each of four
criteria: 1. the true mean squared-error, as calculated on a held-out
data set, of your continuous outcome prediction model; 2. the true
negative log-likelihood, as calculated on a held-out data set, of your
binary outcome prediction model; 3. how close your estimate of
mean-squared error is to the truth (ranked as a percentage of the true
value); 4. how close your estimate of negative log-likelihood is to the
truth (ranked as a percentage of the true value).

The group with the lowest average ranking in the four categories wins\!

## Reading in the data

The data may be read into `R` using the following commands.

``` r
library(RCurl)
#> Loading required package: methods
#> Loading required package: bitops
web_address <- getURL("https://raw.githubusercontent.com/benkeser/sibsml/master/mashable/mashable.csv")
full_data <- read.csv(text = web_address, header = TRUE)
```

## Variable descriptions

A description of the features and outcomes.

  - `n_tokens_title`: Number of words in the title
  - `n_tokens_content`: Number of words in the content
  - `n_unique_tokens`: Rate of unique words in the content
  - `n_non_stop_words`: Rate of non-stop words in the content
  - `n_non_stop_unique_tokens`: Rate of unique non-stop words in the
    content
  - `num_hrefs`: Number of links
  - `num_self_hrefs`: Number of links to other articles published by
    Mashable
  - `num_imgs`: Number of images
  - `num_videos`: Number of videos
  - `average_token_length`: Average length of the words in the content
  - `num_keywords`: Number of keywords in the metadata
  - `data_channel_is_lifestyle`: Is data channel ‘Lifestyle’?
  - `data_channel_is_entertainment`: Is data channel ‘Entertainment’?
  - `data_channel_is_bus`: Is data channel ‘Business’?
  - `data_channel_is_socmed`: Is data channel ‘Social Media’?
  - `data_channel_is_tech`: Is data channel ‘Tech’?
  - `data_channel_is_world`: Is data channel ‘World’?
  - `kw_min_min`: Worst keyword (min shares)
  - `kw_max_min`: Worst keyword (max shares)
  - `kw_avg_min`: Worst keyword (avg shares)
  - `kw_min_max`: Best keyword (min shares)
  - `kw_max_max`: Best keyword (max shares)
  - `kw_avg_max`: Best keyword (avg shares)
  - `kw_min_avg`: Avg keyword (min shares)
  - `kw_max_avg`: Avg keyword (max shares)
  - `kw_avg_avg`: Avg keyword (avg shares)
  - `self_reference_min_shares`: Min shares of referenced articles in
    Mashable
  - `self_reference_max_shares`: Max shares of referenced articles in
    Mashable
  - `self_reference_avg_sharess`: Avg shares of referenced articles in
    Mashable
  - `weekday_is_monday`: Was the article published on a Monday?
  - `weekday_is_tuesday`: Was the article published on a Tuesday?
  - `weekday_is_wednesday`: Was the article published on a Wednesday?
  - `weekday_is_thursday`: Was the article published on a Thursday?
  - `weekday_is_friday`: Was the article published on a Friday?
  - `weekday_is_saturday`: Was the article published on a Saturday?
  - `weekday_is_sunday`: Was the article published on a Sunday?
  - `is_weekend`: Was the article published on the weekend?
  - `LDA_00`: Closeness to LDA topic 0
  - `LDA_01`: Closeness to LDA topic 1
  - `LDA_02`: Closeness to LDA topic 2
  - `LDA_03`: Closeness to LDA topic 3
  - `LDA_04`: Closeness to LDA topic 4
  - `global_subjectivity`: Text subjectivity
  - `global_sentiment_polarity`: Text sentiment polarity
  - `global_rate_positive_words`: Rate of positive words in the content
  - `global_rate_negative_words`: Rate of negative words in the content
  - `rate_positive_words`: Rate of positive words among non-neutral
    tokens
  - `rate_negative_words`: Rate of negative words among non-neutral
    tokens
  - `avg_positive_polarity`: Avg polarity of positive words
  - `min_positive_polarity`: Min polarity of positive words
  - `max_positive_polarity`: Max polarity of positive words
  - `avg_negative_polarity`: Avg polarity of negative words
  - `min_negative_polarity`: Min polarity of negative words
  - `max_negative_polarity`: Max polarity of negative words
  - `title_subjectivity`: Title subjectivity
  - `title_sentiment_polarity`: Title polarity
  - `abs_title_subjectivity`: Absolute subjectivity level
  - `abs_title_sentiment_polarity`: Absolute polarity level
  - `shares`: Number of shares (continuous outcome)
  - `viral`: Did the article ‘go viral’ (more than 10k shares) (binary
    outcome)

## Hints

Hint 1: Recall that if we have an outcome vector `Y` and a vector of
predictions `p`, mean squared-error and negative log-likelihood may be
computed as follows:

``` r
# make up 100 data points
Y <- rnorm(100)
p <- runif(100)

# mean squared-error
mse <- mean((Y - p)^2)

# make up 100 binary outcomes and predictions
Y <- rbinom(100, 1, 0.5)
p <- runif(100, 0, 1)

# negative log-likelihood
nloglik <- - mean(Y*log(p) + (1-Y)*log(1-p))
```

Hint 2: If you are using the `SuperLearner` package, to use mean
squared-error as your risk function, specify `method = "method.NNLS"` or
`method = "method.CC_LS"`. To use negative log-likelihood as your risk
function, specify `method = "method.NNloglik"` or `method =
"method.CC_nloglik"`.

Hint 3: If you are using the `SuperLearner` package, depending on what
algorithms you include, you may need to decrease the number of
cross-validation folds via the `CV.control` option of `SuperLearner`.