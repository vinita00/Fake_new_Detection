## Fake News Detection Using Machine Learning

This Python script aims to detect fake news using machine learning techniques. It utilizes various classification algorithms such as Logistic Regression, Decision Tree Classifier, Gradient Boosting Classifier, and Random Forest Classifier.

### Requirements:

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

### Setup:

1. Install Python (if not already installed) from [python.org](https://www.python.org/).
2. Install required libraries using pip:

    ```
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```

3. Ensure the datasets `Fake.csv` and `True.csv` are available. (Datasets not provided in the code snippet)

### Usage:

1. Run the script (`fake_news_detection.py`).
2. The script will read the provided datasets `Fake.csv` and `True.csv` containing fake and true news articles, respectively.
3. It preprocesses the data, including cleaning text and labeling fake news as class 0 and true news as class 1.
4. Splits the data into training and testing sets.
5. Uses TF-IDF vectorization to convert text data into numerical form.
6. Trains various classification models on the training data.
7. Evaluates the models' performance using classification reports, including precision, recall, and F1-score.
8. Provides a manual testing function where you can input news text to predict whether it's fake or not using the trained models.

### Key Functions:

- `word_drop(text)`: Preprocessing function to clean the text data.
- `manual_testing(news)`: Function to manually input news text and predict its authenticity using the trained models.

### Important Notes:

- Ensure the datasets `Fake.csv` and `True.csv` are correctly formatted and contain relevant news articles.
- Adjustments to the code may be necessary based on the structure and quality of the datasets.
- Fine-tuning of models and hyperparameters can be performed for better performance.
- Use caution when interpreting the results and predictions, as machine learning models are not perfect and may produce false positives or false negatives.

---
