
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
### Data Source and License
The CSV files in the `datasets/` directory originate from the
[**Fake and Real News Dataset**](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
available on Kaggle. The dataset is distributed under the
[**CC0 1.0 license**](https://creativecommons.org/publicdomain/zero/1.0/), so
these files may be freely copied or modified. The repository includes the
original `Fake.csv` and `True.csv` files from Kaggle, renamed here as
`FakeNewsData.csv` and `TrueNewsData.csv`.

The dataset contains **23,481** fake news entries and **21,417** real news
entries. No rows or columns were removed when copying the data into this
repository.

### Data Preprocessing
- Dropped the `title`, `subject`, and `date` columns from the original files.
- Removed punctuation, lowercased the text and stripped HTML artifacts.
- Filtered out common stop words, tokenized articles and applied TF-IDF vectorization for model training.

## Dependencies

Make sure the following Python libraries are installed before running the project:

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib

Install these dependencies using pip:

```bash
pip install -r requirements.txt
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

3. **Run the training script**:
   Execute `main.py` to train and evaluate the models. You can optionally specify the dataset directory if your CSV files are stored elsewhere.
   ```bash
   python main.py --dataset_dir path/to/datasets
   ```

4. **Evaluate the results**:
   The script prints the classification report and accuracy for each algorithm after training.

## License

This project is licensed under the MIT License.
