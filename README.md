# Publication

Classification and Detection of Phishing URLs using Machine Learning Approach
Published in the International Journal of Innovative Research in Computer and Communication Engineering (IJIRCCE), Volume 10, Issue 6, June 2022

🔗 Publication Link:
https://doi.org/10.15680/IJIRCCE.2022.1006142

# Project Description

This project is based on our published research paper “Classification and Detection of Phishing URLs using Machine Learning Approach.”
It presents a complete end-to-end system that detects phishing websites using an ensemble of machine learning models combined through a Voting Classifier.

The model analyzes 24 carefully selected URL, domain, and HTML-based features, extracted using WHOIS, Alexa, ipaddress, BeautifulSoup, and custom parsing techniques. After evaluating nine machine learning algorithms—including Random Forest, Gradient Boosting, XGBoost, SVM, and Logistic Regression—we show that an ensemble of Random Forest + Gradient Boosting + AdaBoost achieves the highest accuracy (in a 70:30 train-test split).

The system includes:

Automated feature extraction from any real-time URL

Machine-learning based phishing prediction using an optimized ensemble

Interactive web interface (Flask) to test URLs instantly

Class probability output with clear safe/unsafe indicators

This research demonstrates that combining bagging and boosting techniques significantly improves phishing URL detection accuracy, outperforming individual classifiers.
