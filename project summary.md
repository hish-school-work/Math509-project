# Project Report: Predicting Tesla Stock Prices Using Tweet Sentiment

## Introduction

In the modern financial ecosystem, investor behavior and market reactions are increasingly influenced by real-time information streams, particularly from social media. Tesla, known for its volatile stock behavior and high media visibility—fueled in part by the public persona of its CEO Elon Musk—presents a compelling case for studying the relationship between public sentiment and stock price movements. This project was designed to rigorously evaluate whether and how the general sentiment expressed in social media, specifically Twitter, correlates with or predicts changes in Tesla's daily stock returns.

Our central research question was: **"How does news sentiment affect stock prices?"** To answer this, we built a robust, multi-stage data science workflow that included collecting a large dataset of Tesla-related tweets, cleaning and processing the textual data, performing sentiment analysis, engineering features related to sentiment and market dynamics, and finally applying both inferential and predictive modeling strategies. We used linear regression to establish a baseline understanding of the sentiment-return relationship, Bayesian regression to capture parameter uncertainty and quantify effect credibility, and Random Forest modeling to enhance prediction by introducing nonlinearity and interactions between features.

## Dataset Description and Preprocessing

### 1. Tesla-Related Tweets (2015–2019)

We used a pre-collected dataset comprising over 1.4 million tweets spanning five years, from January 1, 2015 to December 31, 2019. These tweets were filtered for relevance using keyword matching. Keywords included "Tesla", "TSLA", "Elon Musk", "EV", and "Electric Vehicle", ensuring a broad capture of both direct and indirect discussions related to the company. Each tweet included a body text and a `post_date` timestamp in UNIX format.

#### Preprocessing Steps:
- **Timestamp Conversion**: UNIX timestamps were converted into standard `datetime` format using pandas' `to_datetime()` function. This enabled daily-level aggregation of sentiment scores and matching with stock market data.
- **Text Cleaning**: Raw tweets were cleaned using regular expressions and text preprocessing techniques:
  - Removal of URLs, user mentions (e.g., @elonmusk), hashtags, emojis, numbers, and punctuation.
  - Lowercasing the text to ensure consistency.
  - Elimination of common stopwords (e.g., "and", "the", "is") using NLTK's stopword corpus.

This resulted in a clean corpus of text, which was ready for sentiment analysis. An example:
- **Raw Tweet**: "@elonmusk Tesla's new EV rollout is amazing! #innovation https://t.co/abc123"
- **Cleaned Text**: "tesla new ev rollout amazing innovation"

### 2. Tesla Stock Price Data (2015–2019)

This dataset consisted of daily trading metrics, including `Open`, `High`, `Low`, `Close`, `Adjusted Close`, and `Volume`. For the purposes of modeling stock behavior, we computed the **daily return**, defined as:
\[ \text{return}_t = \frac{\text{AdjClose}_t - \text{AdjClose}_{t-1}}{\text{AdjClose}_{t-1}} \]
This measure captures percentage change in value from one trading day to the next and is a common target variable in financial forecasting models.

## Sentiment Analysis and Feature Engineering

### VADER Sentiment Scoring
We used the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer, optimized for social media and short text, to assign each cleaned tweet a **compound sentiment score** between -1 (very negative) and +1 (very positive). This allowed us to classify each tweet into one of three categories:
- **Positive**: Compound score ≥ 0.05
- **Neutral**: -0.05 < Compound score < 0.05
- **Negative**: Compound score ≤ -0.05

We then aggregated these scores **daily**, producing:
- `avg_sentiment`: The mean compound score across all Tesla-related tweets on a given day.
- `tweet_count`: The number of Tesla-related tweets per day, representing public interest or attention volume.

Additional features were created to capture time trends:
- `lagged_sentiment`: Previous day’s average sentiment.
- `rolling_sentiment_3`: 3-day rolling average of daily sentiment.

## Exploratory Data Analysis

We first examined the sentiment label distribution:
- **Positive Tweets**: 41%
- **Neutral Tweets**: 36%
- **Negative Tweets**: 22%
This indicated a generally optimistic tone in the Tesla tweet universe, which aligns with the company’s innovative image during this period.

We then computed the **Pearson correlation** between daily average sentiment and stock return, which yielded a modest value of **0.23**. This suggested a weak but meaningful linear relationship. A scatter plot showed that while most data points were concentrated near zero sentiment and zero return, higher sentiment scores often corresponded to positive stock returns.

Interestingly, a bar plot of average return by sentiment label revealed that days with **negative sentiment** had higher average returns than neutral days. This may be explained by investor overreaction to negative sentiment followed by market corrections, or perhaps strategic buying opportunities during pessimistic news cycles.

## Modeling and Results

### 1. Linear Regression
We first applied simple linear regression using `avg_sentiment` to predict `return`:
- **Train/Test Split**: 80/20
- **Results**:
  - R²: 0.08
  - RMSE: 0.0276

While the R² was low, this was expected due to the high volatility and noise in stock return data. Nonetheless, the positive coefficient suggested that sentiment had a modest upward effect on stock return.

### 2. Bayesian Linear Regression (PyMC)
To improve our understanding of the sentiment-return relationship and account for uncertainty, we implemented a Bayesian linear regression model.

#### Priors:
- \( \alpha \sim \mathcal{N}(0, 0.02) \)
- \( \beta \sim \mathcal{N}(0, 0.2) \)
- \( \sigma \sim \text{HalfNormal}(0.02) \)

We conducted **prior predictive checks** to ensure the priors produced realistic return values. Then we sampled from the **posterior distribution** using MCMC:
- **Posterior Mean (\( \beta \))**: 0.21
- **94% HDI**: [0.16, 0.25]

This strongly supported the hypothesis that sentiment positively affects stock returns. Furthermore, **posterior predictive checks** showed that simulated returns closely mirrored the real return distribution.

### 3. Random Forest Regression
To focus on improving predictive accuracy, we trained a Random Forest Regressor with the following features:
- `avg_sentiment`
- `lagged_sentiment`
- `rolling_sentiment_3`
- `tweet_count`

Results:
- **R²**: 0.17
- **RMSE**: 0.0267

#### Feature Importance:
- `tweet_count`: 32%
- `avg_sentiment`: 29%
- `rolling_sentiment_3`: 20%
- `lagged_sentiment`: 18%

This indicated that **tweet volume** was more predictive than sentiment itself, suggesting that market attention may be as crucial as sentiment tone.

## Interpretation and Discussion

The modeling results highlight key insights:
- Sentiment alone, while correlated with return, is insufficient for precise prediction.
- The combination of **sentiment, volume, and time-trend features** yields stronger performance.
- Bayesian modeling is particularly valuable in financial contexts, where understanding uncertainty is critical.
- Machine learning models like Random Forests are better suited for handling noisy, nonlinear relationships present in stock data.

## Limitations
1. **Univariate Sentiment Source**: Only Twitter data was analyzed. Including Reddit, financial news, and stock forums could yield more comprehensive sentiment.
2. **Contextual Misinterpretation**: VADER, though powerful, may misclassify sarcasm or slang common in financial discussions.
3. **Event Insensitivity**: Models do not factor in macroeconomic events, earnings calls, or press releases that strongly move markets.
4. **Short-Term Focus**: We focused on **next-day returns** only; lag effects over several days may also be meaningful.
5. **Generalizability**: This model is specific to Tesla and may not generalize across sectors or other stocks without adaptation.

## Future Work
- Incorporate **multi-platform sentiment**, combining Twitter, Reddit, and news.
- Use **classification models** (e.g., logistic regression) to predict up/down movement.
- Introduce **lag structures and time series models** (LSTM, ARIMA) to account for delayed sentiment effects.
- Add **technical indicators** like volatility, RSI, or moving averages.
- Develop a **real-time pipeline** for live tweet ingestion and prediction.

## Conclusion
This project explored the intersection of finance and NLP, revealing that public sentiment—particularly when measured through tweet volume and rolling trends—can offer predictive insight into stock market behavior. Linear regression served as a simple baseline, while Bayesian analysis quantified our confidence in the sentiment-return effect. The Random Forest model demonstrated that combining sentiment with engineered features produces substantially better predictive results.

The findings suggest that while stock returns are noisy and influenced by many external factors, public sentiment is a **valuable, exploitable signal** when combined with thoughtful feature engineering and robust modeling strategies. The project lays a solid foundation for future improvements through broader sentiment sourcing, deeper time-based modeling, and real-time analytics applications.

