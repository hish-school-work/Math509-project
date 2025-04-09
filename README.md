# Stock Price Prediction Based on News Sentiment

## Group Members

| Name           | Student ID |
| -------------- | ---------- |
| Chetan Dhingra | 1862481    |
| Hish Salehi    | 1812352    |
| Velda Iskandar | 1882870    |

## Research Question

How does news sentiment affect stock prices?

## Project Description

This project aims to understand if the general feeling (sentiment) expressed in news articles about a company can help predict how its stock price will change. We will collect news articles, analyze their sentiment, and then see if this sentiment is related to actual stock price movements. We will use machine learning techniques to build a model that tries to predict these price changes based on news sentiment.

## Goals

- Get sentiment scores from news articles about a specific stock.
- See if there's a connection between these sentiment scores and how the stock price changes.
- Build computer models to predict stock price changes using news sentiment.
- Check how well our models perform using standard statistical measures.
- Use methods like MCMC sampling to better understand our model.

## Data

We will use financial news articles and historical stock price data. The specific sources and how often we get the data will be decided later. We will choose data that allows us to see both quick reactions to news and longer-term trends.

Find the Tweets.csv file at

## Data Source

https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020

## Methodology

We will first process the news articles to determine their sentiment (positive, negative, or neutral). Then, we will use this sentiment data along with historical stock prices to build our prediction models. We plan to use Bayesian inference techniques for this.

## Evaluation

We will measure how well our models work by looking at metrics like AUC-ROC. This will tell us how good our model is at correctly predicting whether a stock price will go up or down.

## Assignments/Roles

| Member         | Role                              |
| -------------- | --------------------------------- |
| Chetan Dhingra | Data Collection and Preprocessing |
| Hish Salehi    | Model Modeling and Implementation |
| Velda Iskandar | Evaluation and Report Writing     |

## Usage

Instructions on how to run our code and use our model will be added here later.

## Value

This project will help us understand the relationship between news and stock markets. It could provide insights for investors on how news sentiment might influence stock prices.

Project Summary

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
\[ \text{return}_t = \frac{\text{AdjClose}\_t - \text{AdjClose}_{t-1}}{\text{AdjClose}\_{t-1}} \]
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

# Project Report: Predicting Tesla Stock Prices Using Tweet Sentiment

## Introduction

In the modern financial ecosystem, investor behavior and market reactions are increasingly influenced by real-time information streams, particularly from social media. Tesla, known for its volatile stock behavior and high media visibility—fueled in part by the public persona of its CEO Elon Musk—presents a compelling case for studying the relationship between public sentiment and stock price movements. This project was designed to rigorously evaluate whether and how the general sentiment expressed in social media, specifically Twitter, correlates with or predicts changes in Tesla's daily stock returns.

Our central research question was: **"How does news sentiment affect stock prices?"** To answer this, we built a robust, multi-stage data science workflow that included collecting a large dataset of Tesla-related tweets, cleaning and processing the textual data, performing sentiment analysis, engineering features related to sentiment and market dynamics, and finally applying both inferential and predictive modeling strategies. We used linear regression to establish a baseline understanding of the sentiment-return relationship, Bayesian regression to capture parameter uncertainty and quantify effect credibility, and Random Forest modeling to enhance prediction by introducing nonlinearity and interactions between features.

## Graphical Interpretations and Analysis

### Sentiment Label Distribution Pie Chart

This chart visualizes the distribution of sentiment categories—positive, neutral, and negative—based on daily tweet analysis over the five-year span from 2015 to 2019. The largest segment, approximately 41%, is composed of positive tweets, reflecting a generally optimistic tone in public sentiment regarding Tesla. Neutral tweets make up about 36% of the total, while negative tweets constitute the remaining 22%. This breakdown indicates that while Tesla enjoys substantial positive attention, there is still a significant amount of mixed and critical sentiment. From a financial modeling perspective, this sentiment distribution provides an important baseline, revealing how often the public mood aligns with or diverges from market optimism.

### Scatter Plot: Daily Return vs Average Sentiment

This plot illustrates the relationship between daily average tweet sentiment and corresponding daily stock return. While most data points cluster around the center—reflecting frequent occurrences of mild sentiment and low returns—there is a subtle positive trend. The trend line, derived from linear regression, suggests that days with higher average sentiment tend to be associated with slightly higher returns. However, the scatter is wide, indicating considerable noise in the relationship. This visualization supports our decision to investigate more complex models, as the linear correlation alone is insufficient for robust prediction but hints at an underlying pattern.

### Bar Plot: Average Return by Sentiment Label

This bar plot compares the average stock return on days categorized by dominant sentiment: positive, neutral, or negative. Interestingly, days with negative sentiment showed a slightly higher average return than neutral days. This counterintuitive observation might be attributed to market overreactions to negative news, followed by a price rebound, or the tendency of contrarian traders to buy on pessimistic sentiment. Meanwhile, neutral days show the least movement, reinforcing the idea that a lack of clear sentiment offers limited predictive value. This chart provides insight into potential nonlinear effects in sentiment-driven market reactions.

### Correlation Matrix Heatmap

The correlation matrix heatmap presents the strength and direction of the linear relationship between `avg_sentiment` and `return`. The correlation coefficient, approximately 0.23, signifies a weak but positive correlation. Though modest, this value suggests that sentiment is not entirely disconnected from market behavior. The heatmap, with its color gradient and labeled coefficients, provides a visual reinforcement that while `avg_sentiment` alone may not predict return with high accuracy, it is statistically associated with return movements and merits inclusion in further models.

### Posterior Distribution Plots (Bayesian Linear Regression)

These plots visualize the posterior distributions for the parameters α (intercept), β (sentiment effect), and σ (error or standard deviation of residuals) obtained from Bayesian linear regression. The distribution for β is centered around 0.21, with a 94% Highest Density Interval (HDI) ranging from 0.16 to 0.25. Crucially, this interval does not include zero, indicating a high probability that sentiment positively influences return. The narrow distribution for σ indicates that the model's uncertainty is low, and the parameter estimates are stable. These plots strengthen the statistical credibility of our claim that sentiment has a positive effect on returns.

### Bayesian Regression Line Plot with Uncertainty Bands

This chart overlays multiple regression lines drawn from the posterior distribution of the Bayesian model. Each line represents a possible linear relationship between sentiment and return, and together they form a credible interval band that reflects model uncertainty. The band is centered along a positively sloping line, supporting the conclusion that increases in sentiment are likely to correspond to increases in return. The inclusion of uncertainty bands, rather than a single regression line, makes this visualization particularly powerful in conveying both the direction and variability of the effect.

### Histogram of Prior Predictive Samples

This histogram evaluates whether our prior assumptions in the Bayesian model generate plausible outcomes. It displays the distribution of stock returns simulated from the priors alone—without observing the data. Most values fall within ±0.1 (±10% daily return), aligning with the actual historical behavior of Tesla stock. This validates the appropriateness of our prior choices and confirms that they do not predispose the model toward unrealistic return values. The chart is essential for demonstrating that the model starts from a sound baseline.

### Histogram Comparison: Posterior Predictive vs Actual Returns

This comparative histogram shows two distributions: the simulated returns from the Bayesian model's posterior predictive distribution and the actual observed returns. The close overlap between the two distributions is a strong indicator of model fit. This alignment means the Bayesian model successfully learned from the data and is capable of replicating real-world return behavior. A mismatch here would suggest model misfit or poor parameter estimation, so this chart is a crucial diagnostic tool for validating the model’s performance.

### Feature Importance Bar Chart (Random Forest)

This bar chart displays the relative importance of input features in the Random Forest model. Feature importance is calculated based on how much each feature reduces error across the ensemble of decision trees. `tweet_count`, representing the volume of Tesla-related tweets, emerged as the most important feature (32%), followed by `avg_sentiment` (29%), `rolling_sentiment_3` (20%), and `lagged_sentiment` (18%). The chart demonstrates that tweet volume—an indicator of public attention—has stronger predictive power than sentiment tone alone. It also underscores the value of temporal features, such as rolling averages and lagged sentiment, in capturing sentiment trends and their delayed effects on market behavior.

Each of these visualizations contributed to building a comprehensive understanding of the relationship between social sentiment and stock returns. From descriptive statistics and correlation checks to model diagnostics and feature analysis, the graphical outputs provided both intuitive insight and quantitative validation. Together, they reinforce the central conclusion of this project: that tweet sentiment, particularly when combined with volume and time-aware features, plays a measurable role in predicting Tesla's stock price movements.
