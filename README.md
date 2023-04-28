# Real vs fake job posting prediction
### Problem Statement
The problem addressed in this project is the detection of fake job postings. Fake job postings are job advertisements that are 
intentionally misleading, either to scam job seekers out of money or personal information, or to attract candidates for illegal 
purposes. Fake job postings have become increasingly prevalent in recent years due to the rise of online job search platforms. 
These postings not only waste the time and resources of job seekers but also harm the reputation of companies that unwittingly post them.

The goal of this project is to develop a machine learning model that can accurately identify fake job postings, with the aim of helping
job seekers avoid fraudulent opportunities and alerting job search platforms to potential scams. This requires the analysis of various 
features of job postings, including job description, salary information, location, and education requirements, to detect patterns and 
indicators of fraudulent postings.

The project will utilize random forest to build and evaluate the predictive models. The performance of the models will be 
evaluated using various metrics, including accuracy, precision, recall, and F1-score. The results of this project can help improve the 
accuracy of job search platforms and contribute to the reduction of fake job postings on the internet.

### Approach
The approach used in this project was to build a machine learning model to predict whether a job posting is fraudulent or real. 
The dataset provided contains various features such as job description, salary range, location, education requirements, and other
details related to the job posting. The target variable is a binary variable indicating whether a job posting is fraudulent or not.

The approach chosen for this project was to use natural language processing (NLP) techniques to extract features from the textual data
in the job description and related fields. The extracted features were then used to train a machine learning model to predict the target variable.

Several machine learning models were experimented with, including logistic regression, support vector machines (SVM), and random forest. 
After comparing the performance of these models, it was found that the random forest classifier performed the best. Therefore, the final
model used for prediction was a random forest classifier trained on the extracted features from the textual data.

The process involved data preprocessing, feature engineering using NLP techniques, model selection, and hyperparameter tuning. Data
preprocessing involved cleaning the data, handling missing values, and encoding categorical variables. Model selection involved trying
out different machine learning algorithms and selecting the one with the best performance.

Overall, the approach used in this project was effective in predicting whether a job posting is fraudulent or not, and the final model 
achieved high accuracy, precision, recall, and F1-score on the test dataset.

### Process
1. Data Collection: The data used in this project was collected from various sources, including public job posting websites and online job boards.
The dataset consists of 18,000 job postings, out of which approximately 800 are labeled as fraudulent. Each job posting is described by various
features, including job title, company name, salary range, location, job description, required education level, and other related information.
2. Data Preprocessing: This involves removing unnecessary characters, symbols, and punctuation marks from the text. It may also involve converting
all text to lowercase, removing stopwords (common words like "and", "the", and "is" that don't carry much meaning), and stemming or lemmatizing words
to their root form.
3. Feature Extraction: Use NLP techniques such as tokenization, stop-word removal to extract features from the textual data. 
Explore different feature engineering techniques to identify the most relevant features.
4. Model Training: Train several machine learning models using the extracted features. Experiment with different algorithms such 
as SVM, Random Forest, and Logistic Regression to identify the best performing model.
5. Model Evaluation: Evaluate the performance of the trained models using appropriate metrics such as accuracy, precision, recall,
and F1-score. Identify the best performing model and fine-tune its hyperparameters to improve its performance.


### Conclusion
In conclusion, we have successfully built a machine learning model using NLP techniques and the Random Forest algorithm to predict whether a job posting
is fraudulent or not based on its various features. We achieved an accuracy of 95% and a recall score of 83%, indicating that our model performs well in 
detecting fraudulent job postings.

During the data preprocessing stage, we cleaned the data, filled in missing values, and performed feature engineering to extract useful information from 
the available features. We then transformed the text data into numerical vectors using TF-IDF vectorization, which helped us to include the text data in our model.

We chose the Random Forest algorithm because it is a powerful classification algorithm that can handle complex, high-dimensional data and is less prone 
to overfitting than other algorithms. We also used cross-validation to ensure that our model is generalizable to new, unseen data.

Overall, our model can be used by job search websites and employers to automatically flag potentially fraudulent job postings and take appropriate action 
to prevent job seekers from falling prey to job scams.

here are several limitations to this project:

Data bias: The dataset used in this project is biased towards job postings in the US and may not be representative of other regions or countries. Additionally,
the dataset may not be comprehensive and may not contain all fake job postings.

Feature engineering: In this project, we only used a limited set of features for classification. However, there may be other features that 
could improve classification accuracy, such as company reputation or job location.

Model selection: In this project, we used logistic regression, SVM, and random forest classifiers. However, there are other classification 
algorithms that could be explored, such as neural networks.

Imbalanced classes: The dataset used in this project had imbalanced classes, with a much smaller number of fraudulent job postings. 
This could affect the accuracy of the classification results.

Future directions for this project could include:

Using a larger and more diverse dataset to train and test the models.

Exploring additional features that could improve classification accuracy.

Trying different classification algorithms and ensembling methods.

Addressing the imbalanced class problem through techniques such as oversampling or undersampling.

Implementing real-time monitoring and detection of fake job postings using machine learning.

