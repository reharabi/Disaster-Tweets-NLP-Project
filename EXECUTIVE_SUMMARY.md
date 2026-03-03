# Executive Summary — Disaster Tweets Classification

## The Problem

When disasters happen, people tweet about them in real-time — earthquakes, wildfires, floods. But not every tweet mentioning "fire" or "crash" is about a real emergency. A tweet saying "my mixtape is fire" is very different from "forest fire near La Ronge." Emergency response teams need a way to automatically tell the difference so they can act on real reports faster.

## What We Built

We built a machine learning model that reads a tweet and predicts whether it describes a real disaster or not. The model takes raw tweet text as input and outputs a prediction (disaster or not) along with a confidence score.

## The Data

We used 7,613 real tweets from Kaggle's disaster tweets dataset, each labeled by humans as either a real disaster (43%) or not (57%). The data was split 80/20 into training (6,090 tweets) and testing (1,523 tweets).

## What We Tried

We tested three different model types, each representing a different approach to classification:

- **Logistic Regression** — a simple, fast model that draws a straight line between disaster and non-disaster tweets
- **Support Vector Machine (SVM)** — finds the best possible separation boundary between the two classes
- **Random Forest** — uses hundreds of decision trees that vote on each prediction

For SVM and Random Forest, we also created tuned versions to address overfitting (when a model memorizes training data instead of learning general patterns).

## The Winner

**SVM Tuned (C=0.1)** came out on top with a Test F1 score of 0.77. This means the model correctly identifies about 77% of disaster tweets while keeping false alarms low. Key reasons it won:

- It generalizes well — the gap between training and testing performance is only 6%, meaning it learned real patterns rather than memorizing
- It is the most confident — when it predicts disaster, it is more decisive than the other models
- Cross-validation confirmed this result is reliable across different data splits

Logistic Regression performed almost identically (Test F1: 0.77), making it a strong backup choice due to its simplicity and interpretability.

## Deployment Flexibility

The model includes an adjustable confidence threshold. By default, a tweet is flagged as a disaster if the model is 50% or more confident. Lowering this to 40% catches 80% of real disasters (up from 74%) at the cost of more false alarms. This trade-off can be configured based on the use case:

- **Emergency alert systems** (where missing a disaster is costly) — use the 40% threshold
- **General social media monitoring** (where false alarms are disruptive) — use the default 50%

## Where It Falls Short

The model misclassifies 18.3% of tweets. Most errors come from tweets that use disaster-related words metaphorically ("sinking in my thoughts") or describe real disasters without common disaster vocabulary ("pray for the people affected"). This is a fundamental limitation of treating each word independently — the model sees "fire" but cannot tell if it means a wildfire or a hot take.

## What Would Improve It

The main bottleneck is not the model — it is how we represent the text. Our current approach (TF-IDF) treats each word in isolation. Moving to word embeddings or transformer models like BERT would let the model understand context, which is exactly where our errors come from. However, this would require a larger dataset (10,000+ tweets) to be effective.

## Bottom Line

We built a reliable, deployment-ready tweet classifier that achieves 77% F1 on disaster detection using fast, interpretable traditional machine learning. It runs in seconds with no GPU required, can classify any new tweet in real-time, and includes a tunable threshold for different use cases.
