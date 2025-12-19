# Sentiment Analysis with LSTM

This repository contains a complete end-to-end machine learning project for sentiment analysis on Amazon Kindle book reviews. The project utilizes a Bidirectional Long Short-Term Memory (LSTM) network to classify review text into three categories: **Positive**, **Neutral**, and **Negative**.

Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Notebooks](#notebooks)
- [Dataset Description](#data)
- [Requirements](#requirements)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This project demonstrates how to train an LSTM (Long Short-Term Memory) model to perform sentiment classification (for example: positive/negative) on short to medium-length text data. The approach includes standard text preprocessing, tokenization, constructing an LSTM model with Keras/TensorFlow, training with class-balanced batches, and evaluating metrics such as accuracy, precision, recall, and F1-score.

## Repository Structure
```
├── .github/workflows/deploy.yml     # GitHub Actions CI/CD workflow for EC2 deployment
├── Exploratory_Data_Analysis/
│   ├── EDA.pdf                      # Exploratory Data Analysis
│   ├── kindle_review_visualisation.pbix 
├── api/
│   ├── lstm_sentiment_model.keras   # The trained LSTM model
│   ├── tokenizer.pkl                # The Keras tokenizer for text preprocessing
│   ├── main.py                      # FastAPI application logic
│   ├── Dockerfile                   # Dockerfile for the API service
│   └── requirements.txt             # Python dependencies for the API
├── lstm_preprocessing_training/
│   ├── Part1.ipynb                  # Notebook for EDA and text preprocessing
│   └── Part2.ipynb                  # Notebook for model building, training, and evaluation
├── streamlit_app/
│   ├── app.py                       # Streamlit application logic
│   ├── Dockerfile                   # Dockerfile for the Streamlit service
│   └── requirements.txt             # Python dependencies for the UI
└── docker-compose.yml               # Docker Compose file to orchestrate the services
```

## Notebooks
Typical notebooks include:
- Data cleaning and exploratory data analysis (EDA)
- Text preprocessing and tokenization
- Model building (embedding layer, LSTM, dense layers)
- Model training and callbacks (early stopping, model checkpoint)
- Evaluation and visualization (confusion matrix, ROC, metrics)

Open the notebooks in Jupyter Notebook, JupyterLab, or Google Colab to run and modify experiments interactively.

## Dataset Description

The dataset contains product reviews from Amazon's Kindle Store (you can find it here https://www.kaggle.com/datasets/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis ), where each reviewer has written at least five reviews, and each product has at least five reviews. The columns in the dataset include:

1. **asin** - Product ID (e.g., `B000FA64PK`).
2. **helpful** - Helpfulness rating of the review (e.g., `2/3` indicates two users found it helpful out of three).
3. **overall** - Rating of the product, typically ranging from 1 to 5 stars.
4. **reviewText** - Text content of the review (the main body of the review).
5. **reviewTime** - Raw time when the review was submitted.
6. **reviewerID** - ID of the reviewer (e.g., `A3SPTOKDG7WBLN`).
7. **reviewerName** - Name of the reviewer.
8. **summary** - Summary or brief description of the review.
9. **unixReviewTime** - Unix timestamp of when the review was posted.

## Requirements
The notebooks were built using the Python scientific stack and TensorFlow/Keras. 
Package list:
- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tensorflow (>=2.0)
- keras (if using standalone Keras)
- nltk or spaCy (for tokenization / stopwords)
- jupyterlab or notebook

If you see a `requirements.txt`, prefer installing from it.

## Installation
1. Clone the repository
   git clone https://github.com/JihaneFettoukh/sentiment-analysis-lstm.git
2. Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate     # Windows
3. Install dependencies
   pip install -r requirements.txt
   or
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow jupyter nltk

## Contributing
Contributions are welcome. Suggested ways to contribute:
- Add a new notebook experimenting with different architectures (e.g., GRU, Transformer)
- Add a requirements.txt or environment.yml
- Provide example datasets or preprocessing scripts 
- Improve README/documentation and add usage examples

When contributing, please open an issue or a pull request describing the change.

## Contact
Authors / Maintainers: JihaneFettoukh and NohaLakhdimi (check the repository owner and collaborator profiles)
For questions or suggestions, open an issue in this repository.

Happy experimenting!
