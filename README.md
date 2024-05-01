**Research Project: Industrial Economics - Stock Prediction Based on Tweet Sentiment Analysis**

**Objective:**
In this research project, our aim is to identify the most effective model for stock prediction using sentiment analysis of tweets. We utilized the StockNet dataset for our analysis, selecting it over AStock due to its superior performance across various metrics.

**Dataset:**
We employed the StockNet dataset, sourced from GitHub, which consisted of raw data in JSON format. Prior to analysis, we preprocessed the data and assigned sentiment scores using the Vader Sentiment Intensity Analyzer.

**Training Data Size:**
The training dataset comprised 38,000 sentences extracted from the StockNet dataset.
You can download the dataset from this [link](https://github.com/yumoxu/stocknet-dataset)

**Models Evaluated:**
Our analysis involved evaluating the performance of seven distinct models:

    ULMfit
    BART
    BERT
    RoBERTa
    XLNet
    ELECTRA
    Support Vector Machines (SVMs)

**Performance Evaluation:**
We meticulously calculated and documented the accuracy metrics for each model, providing insights into their effectiveness for stock prediction based on tweet sentiment analysis.

**Requirements:**
To replicate our analysis, the following software versions were utilized:

    NumPy Version: 1.25.2
    Pandas Version: 2.0.3
    Matplotlib Version: 3.7.1
    Scikit-Learn Version: 1.2.2
    PyTorch Version: 2.2.1 + cu121
    Hugging Face Transformers Version: 4.29.2

**Instructions for Replication:**

    Download the StockNet dataset from the provided GitHub repository.
    Preprocess the dataset and assign sentiment scores using the Vader Sentiment Intensity Analyzer.
    Implement and train the selected models using the processed dataset.
    Evaluate the performance of each model based on accuracy metrics.
    Ensure the specified software versions are installed to replicate the analysis accurately.

Note: For further details on the project methodology, data preprocessing steps, and model training procedures, please refer to the project documentation and code repository.

This README provides a formal overview of our research project, outlining the objectives, methodology, requirements, and instructions for replication.
