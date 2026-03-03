# Technical Analysis — Disaster Tweets Classification

## 1. Data Overview

The dataset contains 7,613 tweets from Kaggle's "Natural Language Processing with Disaster Tweets" competition. Each tweet has two fields: `text` (the raw tweet content) and `target` (binary label — 1 for disaster, 0 for not). There are zero missing values. Class distribution is 4,342 non-disaster (57.0%) and 3,271 disaster (43.0%) — a mild imbalance that does not require oversampling but does justify using F1 score over accuracy.

Average tweet length is similar across both classes (disaster tweets slightly longer), confirming that content, not length, drives the classification.

## 2. Preprocessing Pipeline

The text preprocessing follows a six-step cleaning function applied to all tweets:

1. **URL removal** — regex pattern `http\S+|www\S+|https\S+` strips all links
2. **@mention removal** — regex `@\w+` removes Twitter handles
3. **Hashtag cleaning** — removes the `#` symbol but keeps the word (e.g., `#earthquake` → `earthquake`)
4. **Lowercasing** — standardizes all text
5. **Number removal** — regex `\d+` strips digits (numbers do not help TF-IDF)
6. **Special character removal** — regex `[^\w\s]` keeps only word characters and spaces

After cleaning, text goes through tokenization, stop word removal, and POS-aware lemmatization.

### POS-Aware Lemmatization

Standard lemmatization defaults to treating every word as a noun, which produces incorrect results for verbs and adjectives. Our implementation uses NLTK's `pos_tag` to identify each word's part of speech first, then maps it to WordNet's POS categories (noun, verb, adjective, adverb) before lemmatizing. This correctly produces: "running" → "run" (verb), "destroyed" → "destroy" (verb), "residents" → "resident" (noun).

### Custom Stop Words

After standard stop word removal (NLTK English), we performed a frequency analysis across both classes to find words that appear frequently in both disaster and non-disaster tweets. Two custom stop words were identified: "amp" (HTML artifact from `&amp;` in tweets) and "get" (generic verb appearing equally in both classes). Removing these sharpened the discriminative power of the remaining vocabulary.

## 3. Feature Extraction

We used `TfidfVectorizer` with the following configuration:

- `max_features=5000` — vocabulary limited to 5,000 most informative terms
- `ngram_range=(1,2)` — captures both unigrams ("earthquake") and bigrams ("forest fire")

Each tweet is transformed into a 5,000-dimensional sparse vector where each dimension represents the TF-IDF weight of a specific term. TF-IDF upweights words that are distinctive to a document (high term frequency, low document frequency) and downweights common words.

The vectorizer is wrapped inside an sklearn `Pipeline` with the classifier, ensuring the vocabulary is learned only from training data. This prevents data leakage — the test set vocabulary is never seen during fitting.

## 4. Train-Test Split

The data was split 80/20 using `train_test_split` with `stratify=train_df['target']` and `random_state=42`:

- Training: 6,090 tweets (57.0% non-disaster / 43.0% disaster)
- Testing: 1,523 tweets (57.1% non-disaster / 42.9% disaster)

Stratification preserves the original class distribution in both sets.

### Note on EDA Before Split

EDA was performed on the full dataset before splitting. This is acceptable because the EDA was purely exploratory — class distribution, word frequencies, and tweet lengths. No features were engineered from test data, and the stratified split ensures all EDA findings hold equally for the training set. In a production setting, splitting first would be the safer approach.

## 5. Models and Results

### 5.1 Logistic Regression

Configuration: `LogisticRegression(max_iter=1000)` with default `C=1.0`.

| Metric | Train | Test |
|--------|-------|------|
| F1 (Disaster) | 0.8328 | 0.7718 |
| Precision | 0.9161 | 0.8333 |
| Recall | 0.7635 | 0.7187 |

Gap: 0.06 — healthy generalization, no significant overfitting. LR serves as a strong baseline.

**Why we did not tune LR further**: The train-test gap is already small (0.06), meaning overfitting is not an issue. The real limitation is the TF-IDF feature representation (treats words independently), not the model's capacity. Tuning C or adding class weights would yield marginal improvement at best.

### 5.2 SVM Default (LinearSVC)

Configuration: `CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=5)`. CalibratedClassifierCV wraps LinearSVC to provide `predict_proba` support via Platt scaling.

| Metric | Train | Test |
|--------|-------|------|
| F1 (Disaster) | 0.9029 | 0.7629 |
| Precision | 0.9548 | 0.7924 |
| Recall | 0.8563 | 0.7355 |

Gap: 0.14 — moderate overfitting. The default C=1.0 creates a tight margin that fits training data too closely.

### 5.3 SVM Tuned (C=0.1)

Configuration: `CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=2000), cv=5)`.

| Metric | Train | Test |
|--------|-------|------|
| F1 (Disaster) | 0.8296 | 0.7728 |
| Precision | 0.8820 | 0.8104 |
| Recall | 0.7830 | 0.7385 |

Gap: 0.06 — overfitting reduced from 0.14 to 0.06 by widening the margin (lower C). Test F1 also improved (0.76 → 0.77), confirming that reducing overfitting helped generalization.

### 5.4 Random Forest Default

Configuration: `RandomForestClassifier(n_estimators=200, random_state=42)`.

| Metric | Train | Test |
|--------|-------|------|
| F1 (Disaster) | 0.9809 | 0.7563 |
| Precision | 0.9910 | 0.8038 |
| Recall | 0.9710 | 0.7141 |

Gap: 0.22 — severe overfitting. The unlimited tree depth allowed the 200 trees to memorize training examples (98% Train F1) with poor generalization.

### 5.5 Random Forest Tuned

Configuration: `RandomForestClassifier(n_estimators=1000, max_depth=30, min_samples_split=5, class_weight='balanced', random_state=42)`.

| Metric | Train | Test |
|--------|-------|------|
| F1 (Disaster) | 0.7589 | 0.7009 |
| Precision | 0.9326 | 0.8406 |
| Recall | 0.5602 | 0.5566 |

Gap: 0.06 — overfitting eliminated, but test F1 dropped significantly (0.76 → 0.70). The depth and split constraints restricted the trees too much, causing low recall (0.56). An earlier attempt with `max_depth=8` performed even worse (Test F1: 0.65). Random Forest struggles on sparse, high-dimensional TF-IDF data — it is better suited for dense, lower-dimensional features.

## 6. Why SVM Tuned Wins

Three converging lines of evidence:

**Test Performance** — Highest Test F1 (0.7728), narrowly beating LR (0.7718). Both outperform the other models on unseen data.

**Probability Calibration** — SVM Tuned has the highest average confidence (0.29, meaning predictions deviate furthest from 0.5) and the highest correct class probability (0.71). By contrast, RF Tuned's confidence is just 0.056 — barely above random guessing.

**Cross-Validation** — 5-fold CV on training data confirms: SVM Tuned F1 = 0.7402 ± 0.0122, LR F1 = 0.7400 ± 0.0145. Both are stable across folds. RF Tuned trails at 0.6341 ± 0.0209.

## 7. Threshold Tuning

Applied to SVM Tuned (the best model). Default threshold: 0.5. Custom threshold: 0.4.

| Metric | Threshold 0.5 | Threshold 0.4 | Change |
|--------|--------------|--------------|--------|
| F1 | 0.7728 | 0.7630 | -0.0098 |
| Precision | 0.8104 | 0.7334 | -0.0770 |
| Recall | 0.7385 | 0.7951 | +0.0566 |

Lowering the threshold trades precision for recall: 37 additional disaster tweets are correctly identified, at the cost of 67 additional false alarms. This is a deployment decision — in emergency systems, the recall gain justifies the precision cost.

## 8. Error Analysis

The model misclassifies 278 of 1,523 test tweets (18.3%).

**False Positives** (predicted disaster, was not): Tweets using disaster-adjacent words metaphorically — "firefighter act like cop" (metaphorical scenario), "mass murderer portrait" (art reference), "oil spill" (used casually). TF-IDF sees the individual words but cannot determine context.

**False Negatives** (missed real disaster): Tweets describing disasters without typical vocabulary — "trapped miner" (uncommon term), "pray for the people affected" (indirect language), "mudslide" (less frequent in training data). The model lacks the word frequency signal to flag these.

**Root cause**: TF-IDF treats words independently. The word "fire" in "forest fire" and "you're fired" produces identical feature values. This is the fundamental limitation of bag-of-words approaches.

## 9. Decisions Not Taken (and Why)

**GridSearchCV**: With ~7,600 tweets, exhaustive search across 36+ parameter combinations (5-fold CV each = 180+ model trainings) is computationally wasteful for marginal gains. Manual tuning of the most impactful parameters (C for SVM, max_depth for RF) was sufficient.

**Non-linear SVM (RBF kernel)**: With 5,000 TF-IDF features, the data is high-dimensional enough that a linear boundary is effective. RBF kernel scales O(n²) to O(n³) with sample count, adds a gamma hyperparameter, and would not fix the core issue (words treated independently).

**Neural Networks / BERT**: BERT (110M parameters) on 6,090 training samples risks severe overfitting — the same pattern we saw with Random Forest. It also requires GPU, longer training, and sacrifices interpretability. Research shows traditional models perform comparably on small text datasets (<10,000 samples).

**XGBoost**: Similar overfitting profile to Random Forest on sparse TF-IDF features. More hyperparameters to tune. Marginal improvement expected given the data size and feature representation.

**90/10 Split**: Considered but rejected. It would shrink the test set from 1,523 to ~762 tweets, making metrics less stable. The extra 761 training tweets would not meaningfully improve SVM or LR performance.

## 10. Reproducibility

All random processes use `random_state=42`. The sklearn Pipeline architecture ensures the TF-IDF vocabulary is fitted only on training data. The notebook can be re-run end-to-end in Google Colab with identical results.

## 11. Future Work

The most impactful improvements in order of expected benefit:

1. **Word embeddings** (GloVe/Word2Vec) — replace TF-IDF to capture semantic similarity between words, directly addressing the "fire" ambiguity problem
2. **BERT fine-tuning** — requires a larger dataset (10,000+ tweets) to avoid overfitting, but would understand full sentence context
3. **Feature engineering** — the Kaggle dataset includes `keyword` and `location` columns that we did not use. Keywords like "earthquake" or "flood" could serve as strong additional signals
4. **Ensemble methods** — combining SVM Tuned and LR predictions (both strong, potentially complementary errors) through stacking or voting
5. **API deployment** — the sklearn Pipeline can be serialized with `joblib` and served as a REST API for real-time tweet classification
