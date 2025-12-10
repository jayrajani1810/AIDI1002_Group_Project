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

### Results and Conclusion :
The project successfully reproduced the original research paper on predicting stock market movements using news headlines and NLP. We added a Multi-Layer Perceptron (MLP) model to the baseline models.

### Key results:
Random Forest: Accuracy = 0.5226, F1 = 0.6042
Logistic Regression: Accuracy = 0.4347, F1 = 0.4989
Multinomial Naive Bayes: Accuracy = 0.4975, F1 = 0.5633
MLP Classifier: Accuracy = 0.4698, F1 = 0.5462

The MLP model did not outperform Random Forest but captured some non-linear patterns in the text data, showing that neural networks can also be applied to this problem.

### Future Extension :
Implement transformer-based models (BERT, RoBERTa) for richer contextual embeddings.
Incorporate real-time financial news and sentiment analysis from social media platforms.
Explore ensemble models that combine textual and numerical features for improved accuracy.
Expand dataset to include multiple market indices and global financial news for broader applicability.

### References:

[1] Kaggle Dataset – Combined News for DJIA: https://www.kaggle.com/aaron7sun/stocknews

[2] Research Paper: “Predicting Stock Market Movements using News Headlines” (2019+) – Original methodology for stock prediction using NLP and ML models.

[3] GitHub Repository of Original Paper: https://github.com/SATHVIKRAJU/Stock-Market-Prediction-using-Natural-Language-Processing – Provides code and instructions to reproduce baseline models (Random Forest, Logistic Regression, Naive Bayes).

[4] Scikit-learn Documentation – https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

[5] Text Preprocessing & TF-IDF References –
Bird, Steven, et al. “Natural Language Processing with Python.” O’Reilly Media, 2009.
