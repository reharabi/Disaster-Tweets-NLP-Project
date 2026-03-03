# Disaster Tweets Classification — NLP Project

An NLP binary classification project that predicts whether a tweet is about a **real disaster** or not, using traditional machine learning on the Kaggle "Natural Language Processing with Disaster Tweets" dataset.

**Best Model: SVM Tuned (C=0.1) — Test F1: 0.7728**

## Why This Matters

During emergencies, people tweet about disasters in real-time. Automatically identifying which tweets report real disasters helps emergency services respond faster. The challenge: distinguishing "Forest fire near La Ronge" (real disaster) from "my career is on fire" (metaphor).

## Dataset

| Stat | Value |
|------|-------|
| Total tweets | 7,613 |
| Class split | 57% non-disaster / 43% disaster |
| Columns | `text`, `target` |
| Missing values | 0 |
| Train set | 6,090 (80%) |
| Test set | 1,523 (20%) |

Source: [Kaggle — NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

## Approach

**Preprocessing** — Cleaned tweets (set everything to lowercase, removed URLs, @mentions, special characters, numbers), tokenized, removed stop words, applied POS-aware lemmatization, and removed 2 custom stop words ("amp", "get") that appeared frequently in both classes.

**Feature Extraction** — TF-IDF Vectorization (`max_features=5000`, `ngram_range=(1,2)`) wrapped in sklearn Pipelines to prevent data leakage.

**Models Trained** — Logistic Regression, SVM (LinearSVC + CalibratedClassifierCV), and Random Forest, plus tuned versions of SVM and RF to address overfitting.

**Evaluation** — F1 score as primary metric (Recall in some cases). Train AND test metrics reported for every model. 5-fold cross-validation for reliability.

## Results
<img width="855" height="150" alt="Screenshot 2026-03-03 at 19 53 00" src="https://github.com/user-attachments/assets/435c3825-379c-48a7-a991-41c42c62a783" />



SVM Tuned wins: highest test F1

## Key Findings

- **Tuning SVM worked** — Lowering C from 1.0 to 0.1 reduced overfitting (gap: 0.14 → 0.06) AND improved test performance (F1: 0.76 → 0.77)
- **Tuning RF had mixed results** — Overfitting was eliminated (gap: 0.22 → 0.06), but test performance dropped (F1: 0.76 → 0.70)
- **Threshold tuning** — Lowering from 0.5 to 0.4 on SVM Tuned increased recall from 0.74 to 0.80 at the cost of precision (0.81 → 0.73) — useful for emergency systems where missing a disaster is costly
- **Error analysis** — 18.3% error rate, mainly from metaphorical language and indirect disaster descriptions
- **Cross-validation confirmed** — SVM Tuned and LR tied at top (0.74 ± 0.01), consistent across all 5 folds

## Project Structure

```
├── Disaster_Tweets_NLP.ipynb            # Full notebook with code, outputs, and analysis
├── Disaster_Tweets_Presentation.pptx    # 14-slide presentation deck
├── Disaster_Tweets_Talking_Script.docx  # Slide-by-slide speaking notes
├── README.md                            # This file
├── EXECUTIVE_SUMMARY.md                 # Non-technical project summary
└── TECHNICAL_ANALYSIS.md                # Detailed technical deep-dive
```
## 📁 Project Files

| File | Description |
|------|-------------|
| [Disaster_Tweets_NLP_Project.ipynb](Disaster_Tweets_NLP_Project.ipynb) | Main analysis notebook |
| [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) | Project summary |
| [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md) | Detailed technical analysis |
| [Presentation](Disaster_Tweets_Presentation%202.key) | Project presentation |

## How to Run

1. Upload `Disaster_Tweets_NLP.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload `train.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started) to your Google Drive
3. Update the file path in cell 4 to match your Drive location
4. Run all cells sequentially

**Requirements**: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk — all pre-installed in Google Colab.

## Why Not Deep Learning?

With only 7,600 tweets, traditional ML models (SVM, LR) perform comparably to deep learning. BERT's 110M parameters on 6,090 training samples risks severe overfitting — the same issue we saw with the default Random Forest (Train F1: 0.98 vs Test F1: 0.76). The real bottleneck is feature representation (TF-IDF treats words independently), not model complexity. On small text datasets, well-tuned traditional models match or come close to transformer performance.

## Future Improvements

- Word embeddings (GloVe/Word2Vec) to capture semantic relationships between words
- BERT fine-tuning with a larger dataset (10,000+ tweets)
- Feature engineering from keyword and location columns
- Deploy the sklearn Pipeline as an API endpoint for real-time tweet classification

## Author

**Reha** — Masterschool, Data Science & Machine Learning
