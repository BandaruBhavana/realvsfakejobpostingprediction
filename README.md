# Real vs fake job posting prediction
## Problem Statement
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

## Approach
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
## 1. Data Collection: The data used in this project was collected from various sources, including public job posting websites and online job boards.
The dataset consists of 18,000 job postings, out of which approximately 800 are labeled as fraudulent. Each job posting is described by various
features, including job title, company name, salary range, location, job description, required education level, and other related information.
### Code:
import pandas as pd

df=pd.read_csv('fake_job_postings.csv')

df

<img width="734" alt="image" src="https://user-images.githubusercontent.com/111446938/235056029-d7cf2e4c-68d0-47d8-babb-51a62612be5b.png">


## 2. Data Preprocessing: This involves removing unnecessary characters, symbols, and punctuation marks from the text. It may also involve converting
all text to lowercase, removing stopwords (common words like "and", "the", and "is" that don't carry much meaning), and stemming or lemmatizing words
to their root form.

### Code:
def get_country_name(code):

    try:
    
        country = pycountry.countries.get(alpha_2=code)
        
        return country.name
        
    except:
    
        return code


df = df.dropna(subset=['location'])


df['country_code'] = df['location'].apply(lambda x: x.split(',')[0].strip())


df['country'] = df['country_code'].apply(get_country_name)


country_counts = df.groupby('country')['fraudulent'].value_counts(normalize=True).loc[:, 1] * 100

## 3.Data Visualization

country_counts = df.groupby('country')['fraudulent'].value_counts(normalize=True).loc[:, 1] * 100

plt.figure(figsize=(12,8))

ax = country_counts.plot(kind='bar')

ax.set_xlabel('Country')

ax.set_ylabel('% of fraudulent job postings')

ax.set_title('Percentage of Fraudulent Job Postings by Country')

ax.set_yticks(range(0, 101, 10))

for p in ax.patches:

    ax.annotate(str(round(p.get_height(), 2)) + '%', (p.get_x() + 0.1, p.get_height() + 1))
    

labels = [get_country_name(code.get_text()) for code in ax.get_xticklabels()]

ax.set_xticklabels(labels)

plt.show()

![image](https://user-images.githubusercontent.com/111446938/235064835-21e2a70e-591d-4f4a-adf9-7c430fe7843a.png)

import matplotlib.pyplot as plt

top_industries = df['industry'].value_counts(normalize=True).nlargest(10) * 100

plt.figure(figsize=(12,6))

plt.bar(top_industries.index, top_industries.values)

plt.title('Top 10 Industries by Job Postings')

plt.xlabel('Industry')

plt.ylabel('Percentage of Job Postings')

plt.xticks(rotation=90)

for i, v in enumerate(top_industries):

    plt.text(i, v+1, f"{v:.1f}%", ha='center')
    
plt.show()

![image](https://user-images.githubusercontent.com/111446938/235065061-8e021391-11b3-4191-9484-22cd955d4837.png)

fraudulent_df = df[df['fraudulent'] == 1]

top_fraudulent_industries = fraudulent_df['industry'].value_counts(normalize=True).nlargest(10) * 100

plt.figure(figsize=(12,6))

plt.bar(top_fraudulent_industries.index, top_fraudulent_industries.values)

plt.title('Top 10 Industries with Fake Job Postings')

plt.xlabel('Industry')

plt.ylabel('Percentage of Fake Job Postings')

plt.xticks(rotation=90)

for i, v in enumerate(top_fraudulent_industries):

    plt.text(i, v+1, f"{v:.1f}%", ha='center')
    
plt.show()

![image](https://user-images.githubusercontent.com/111446938/235065208-1489bc56-7a41-4a45-8556-9129122db944.png)

emp_type_counts = df.groupby(['employment_type', 'fraudulent']).size().unstack(fill_value=0)

emp_type_percents = emp_type_counts.apply(lambda x: 100 * x / x.sum(), axis=0)

ax = emp_type_percents.plot(kind='bar', stacked=True, figsize=(12,6))

ax.set_title('Employment Type by Fraudulent vs Non-Fraudulent Job Postings')

ax.set_xlabel('Employment Type')

ax.set_ylabel('Percentage of Job Postings')

for container in ax.containers:

 ax.bar_label(container, label_type='edge', labels=[f"{val:.1f}%" if val != 0 else "" for val in container.datavalues], padding=5)
 
plt.show()

![image](https://user-images.githubusercontent.com/111446938/235065445-1b0af9c4-c66a-466e-add4-f27deeacca11.png)

du_counts = df.groupby(['required_education', 'fraudulent'])['fraudulent'].count().reset_index(name='counts')

sns.barplot(x='required_education', y='counts', hue='fraudulent', data=edu_counts)

plt.xticks(rotation=90)

plt.title('Number of Real and Fraudulent Jobs for each Education Level')

plt.show()

![image](https://user-images.githubusercontent.com/111446938/235065557-1c38514e-052e-41f3-9797-97a60bf2d1f9.png)

plt.figure(figsize=(16, 10))

df_fraud = df[df['fraudulent'] == 1]

df_fraud_by_emp_type = df_fraud.groupby('employment_type').size().reset_index(name='count')

df_fraud_by_emp_type['percentage'] = df_fraud_by_emp_type['count'] / df_fraud_by_emp_type['count'].sum() * 100

sns.barplot(x='employment_type', y='percentage', data=df_fraud_by_emp_type)

plt.title('Fraudulent Job Postings by Employment Type')

plt.xlabel('Employment Type')

plt.ylabel('Percentage of Fraudulent Job Postings')

for index, row in df_fraud_by_emp_type.iterrows():

 plt.text(row.name, row.percentage + 1, f"{row.percentage:.1f}%", ha='center')
 
plt.show()

![image](https://user-images.githubusercontent.com/111446938/235065675-112cc9fb-f8c8-4e5c-ad2a-eb8106b7edab.png)

## 4. Feature Extraction: Use NLP techniques such as tokenization, stop-word removal to extract features from the textual data. 
Explore different feature engineering techniques to identify the most relevant features.
import seaborn as sns

import pandas as pd

cols = ['telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']

corr = df[cols].corr()

sns.heatmap(corr, cmap='magma', annot=True, fmt='.2f')

![image](https://user-images.githubusercontent.com/111446938/235065888-3aae948f-9605-4beb-a3c5-3075eeb4279e.png)

## 5. Model Training: Train several machine learning models using the extracted features. Experiment with different algorithms such 
as SVM, Random Forest, and Logistic Regression to identify the best performing model.



df = df[['salary_min', 'salary_max', 'country', 'description', 'department', 'required_education', 'requirements', 'fraudulent']]

df.dropna(inplace=True)

text = df['description'] + ' ' + df['department'].fillna('') + ' ' + df['required_education'].fillna('') + ' ' + df['requirements'].fillna('')

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(text)

y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train


## 6. Model Evaluation: Evaluate the performance of the trained models using appropriate metrics such as accuracy, precision, recall, and F1-score. Identify the best performing model and fine-tune its hyperparameters to improve its performance.

y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))

<img width="276" alt="image" src="https://user-images.githubusercontent.com/111446938/235067281-daf46e29-2ef5-455d-84b1-4e5310dbe14e.png">




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

### Limitations of this project:

1. Data bias: The dataset used in this project is biased towards job postings in the US and may not be representative of other regions or countries. Additionally,
the dataset may not be comprehensive and may not contain all fake job postings.
2. Feature engineering: In this project, we only used a limited set of features for classification. However, there may be other features that 
could improve classification accuracy, such as company reputation or job location.
3. Model selection: In this project, we used logistic regression, SVM, and random forest classifiers. However, there are other classification 
algorithms that could be explored, such as neural networks.
4. Imbalanced classes: The dataset used in this project had imbalanced classes, with a much smaller number of fraudulent job postings. 
This could affect the accuracy of the classification results.

### Future directions for this project could include:

1. Using a larger and more diverse dataset to train and test the models.
2. Exploring additional features that could improve classification accuracy.
3. Trying different classification algorithms and ensembling methods.
4. Addressing the imbalanced class problem through techniques such as oversampling or undersampling.
5. Implementing real-time monitoring and detection of fake job postings using machine learning.

