# AIDI1002_Group_Project
Predicting stock market using natural language processing
**Abstract**
We applied machine learning techniques to evaluate historical financial news headlines and their relationship to stock market movements. Our goal was to reproduce the methodology of an existing research paper and extend it with a significant contribution. Using Natural Language Processing (NLP), we built models that predict whether the Dow Jones Industrial Average (DJIA) will rise or fall based on daily news headlines. Baseline models included Random Forest, Logistic Regression, and Naive Bayes. As our contribution, we introduced a Multi‑Layer Perceptron (MLP) classifier with hyperparameter tuning. This addition allowed us to explore nonlinear feature interactions and evaluate whether neural architectures improve predictive accuracy compared to traditional baselines.
Introduction
Natural Language Processing (NLP) enables computers to interpret and analyze human language. In this project, we use NLP to transform daily news headlines into structured features that can be used for stock market prediction. Sentiment analysis and text vectorization techniques allow us to capture the “emotional” tone of headlines and link them to market movements. Our work reproduces the baseline models from the original paper and extends them with a neural network approach.

**Data Collection and Wrangling**
We used the Combined_News_DJIA.csv dataset (Kaggle, courtesy Aaron7sun), which spans 2008–2016. Each row contains the top 25 news headlines for a given day, along with a binary label indicating whether the DJIA increased (1) or decreased (0) the following day.
News data: Reddit World News headlines (Top 25 per day).
Stock data: DJIA index values from Yahoo Finance.
Labels: 1 if DJIA rose the next day, 0 otherwise.

**Preprocessing steps:**
Concatenated Top1–Top25 into a single text field.
Cleaned text: lowercasing, removing punctuation/HTML, stopword removal, lemmatization.
Vectorization: Bag‑of‑Words with n‑grams (1–2), max 5000 features.
Train/test split: 80% training, 20% testing, ordered by date.

**Workflow:**

**1) Text Cleaning:** Remove noise, stopwords, lemmatize tokens.

**2) Vectorization:** Bag‑of‑Words and n‑gram features.

**3) Baseline Models:** Random Forest, Logistic Regression, Multinomial Naive Bayes

**4) Contribution Model:**
Multi‑Layer Perceptron (MLP) with GridSearchCV for hyperparameter tuning.
Parameters tuned: hidden layer sizes, activation functions, regularization (alpha).

**5) Evaluation:**
Metrics: Accuracy, Precision, Recall, F1 Score.
Confusion matrices for error analysis.
Comparative bar plots of F1 scores across models.

**Environment**
We used Jupyter Notebook for implementation. Libraries include:
Python 3.10+
NumPy, Pandas
Scikit‑learn
NLTK
Matplotlib, Seaborn
Joblib

**Results**
Baseline models achieved F1 scores in the range of ~0.55–0.65.
The MLP classifier, after hyperparameter tuning, showed competitive performance and highlighted the potential of neural architectures for text‑based financial prediction.
Random Forest remained strong on n‑gram features, but MLP provided a valuable nonlinear benchmark.
Example visualization:
Bar chart comparing F1 scores of all models.
Confusion matrix for MLP predictions.

**Conclusion**
This project demonstrates how NLP can be applied to financial news headlines to predict stock market movements. By reproducing baseline models and adding an MLP classifier, we explored the impact of neural architectures on prediction accuracy. While results are not yet sufficient for real‑world trading, the work provides a solid foundation for understanding text preprocessing, feature engineering, and model evaluation in financial NLP tasks.

**Future Work**
Experiment with TF‑IDF and pretrained embeddings (e.g., GloVe, FinBERT).
Extend to multi‑class classification (e.g., strong rise, weak rise, fall).
Apply time‑series models (LSTM, Transformers) to capture sequential dependencies.
Incorporate sentiment scores from external APIs.

**References**
Original paper: Predicting stock market using natural language processing (Emerald Publishing, 2023).
Kaggle dataset: Combined_News_DJIA.
Scikit‑learn documentation.
NLTK documentation.

## Results

After training and evaluation, we compared the baseline models (Random Forest, Logistic Regression, Naive Bayes) with our contribution (MLP Classifier). The table below summarizes the performance metrics:

| Model              | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Random Forest      | 0.62     | 0.61      | 0.63   | 0.62     |
| Logistic Regression| 0.60     | 0.59      | 0.61   | 0.60     |
| Multinomial NB     | 0.58     | 0.57      | 0.58   | 0.58     |
| MLP Classifier     | 0.64     | 0.63      | 0.65   | 0.64     |

### Visualizations
- **Bar chart of F1 scores** comparing all models.  
- **Confusion matrix** for MLP predictions to show error distribution.  

These plots are saved in the `results/` folder and referenced in the README for clarity.
