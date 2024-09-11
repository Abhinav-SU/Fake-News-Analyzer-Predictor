
# Fake News Analyzer and Predictor

This project focuses on developing a machine learning model to accurately detect and classify news articles from various platforms as either legitimate or fake. The goal is to combat the spread of misinformation by providing a tool that assesses the authenticity of news content. The model was trained using a diverse dataset of labeled news articles, and we applied several machine learning techniques to evaluate performance.

## Project Overview

In the digital age, the proliferation of fake news is a significant challenge. Our team developed a machine learning solution to address this by classifying news articles using multiple algorithms. This solution provides insights into the classification process and aims to contribute to reducing the spread of misinformation.

### Algorithms Implemented:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Naive Bayes Classifier**

## Dataset

The dataset contains labeled news articles, categorized as either true (authentic) or false (fake). The data is divided into two categories:
- **True**: Legitimate news articles.
- **False**: Fake or fabricated news articles.

### Data Preprocessing:
- Cleaned and normalized text data.
- Used techniques like tokenization and vectorization (TF-IDF) to prepare the dataset for model training.

## Dependencies

Make sure the following Python libraries are installed before running the project:

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Regular Expression (re)

Install these dependencies using pip:

```bash
pip install pandas numpy matplotlib sklearn seaborn re
```

## Usage

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/Fake-News-Analyzer-Predictor.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd Fake-News-Analyzer-Predictor
   ```

3. **Run the Jupyter Notebook**:
   Open the `fakeNewsDetection.ipynb` notebook and run the cells to train and test the models.

4. **Evaluate the results**:
   The code will output performance metrics (accuracy, precision, recall, F1-score) and will allow you to input custom news data for classification as real or fake.

## License

This project is licensed under the MIT License.
