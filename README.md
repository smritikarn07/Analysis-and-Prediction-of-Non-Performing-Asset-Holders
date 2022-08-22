# Analysis-and-Prediction-of-Non-Performing-Asset-Holders

## Definition of 'Non Performing Assets'

**Definition:** A non performing asset (NPA) is a loan or advance for which the principal or interest payment remained overdue for a period of 90 days.

**Description:** Banks are required to classify NPAs further into atleast Substandard and Doubtful or Loss assets.

**Motivation:** Non-performing assets are a reflection of the bank’s overall efficiency while performing its business of converting deposits into loans and recovering these loans. Non-recovery or partial recovery of loans has an impact on the bank’s balance sheet and income statement items in the form of reduction in interest earned on loan assets, increase in provision on NPAs, increase in capital requirement and lower profits. Hence, rising NPAs are a concern for a bank and determinants of NPAs should be identified prior to loans turning into NPAs.

**Objective:** To predict the borrowers capability to repay his loan or become involved in the group of Non Performing Asset Holders.

## Dataset:
**Data Source:** The Bank Indessa's NPA has reached an all time high which after a careful analysis concluded that a majority of NPA was contributed by loan defaulters. There is an immediate need to restrict the can-be-loan defaulters to improvise the current scenario. The Bank have decided to apply data analysis to analyze the pattern of loan defaulters and machine learning to predict if the new customers can-be the new loan defualters.
The data was collected by the Bank while performing due diligence on requested loan applications. The data is, however too messy to directly apply on machine learning models. Thus feature engineering is used to align the data.

The dataset available had around 45 features and ~5,00,000 datapoints. However for training such a huge dataset I did not have enough resources, so I decided to use a random 50K data points with 50 features.

## Steps Involved

### Exploratory Data Analysis and data visualization (EDA)
Exploratory data analysis (EDA) is used by data scientists to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions.

![image (1)](https://user-images.githubusercontent.com/109594714/185826932-c8cda36e-3231-4d0b-9452-1260ee7813f4.png)

EDA is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a provides a better understanding of data set variables and the relationships between them. It can also help determine if the statistical techniques you are considering for data analysis are appropriate.

![0](https://user-images.githubusercontent.com/109594714/185826421-30e48005-a6ef-4487-b99c-42074588b08c.png)


### Feature Engineering

This is the most important step for any data science based application. We have to select the relevant features.

Feature engineering refers to a process of selecting and transforming variables when creating a predictive model using machine learning or statistical modeling (such as deep learning, decision trees, or regression). The process involves a combination of data analysis, applying rules of thumb, and judgement. It is sometimes referred to as pre-processing, although that term can have a more general meaning.

The data used to create a predictive model consists of an outcome variable, which contains data that needs to be predicted, and a series of predictor variables that contain data believed to be predictive of the outcome variable. For example, in a model predicting property prices, the data showing the actual prices is the outcome variable. The data showing things, such as the size of the house, number of bedrooms, and location, are the predictor variables. These are believed to determine the value of the property.
A "feature" in the context of predictive modeling is just another name for a predictor variable. Feature engineering is the general term for creating and manipulating predictors so that a good predictive model can be created.

The column is test_member_id is removed as it is a unique ID for the loan and not hold any relevant information
loan_status is selected as our target variable or outcome variable.
member_id, emp_length, loan_amnt, funded_amnt, funded_amnt_inv, int_rate, annual_inc, dti, mths_since_last_delinq, mths_since_last_record, open_acc, revol_bal, revol_util, total_acc, total_rec_int, total_rec_late_fee, mths_since_last_major_derog, last_week_pay, tot_cur_bal, total_rev_hi_lim, tot_coll_amt, recoveries, collection_recovery_fee, term, acc_now_delinq, collections_12_mths_ex_med columns are selected based on EDA.

The steps followed in this particular dataset are:

#### Feature Scaling
XGBoost is not sensitive to monotonic transformations of its features for the same reason that decision trees and random forests are not: the model only needs to pick "cut points" on features to split a node. Splits are not sensitive to monotonic transformations: defining a split on one scale has a corresponding split on the transformed scale.

#### Data Cleanup
* Stripping datatype to numeric values: In this step the extra textual parts are striped off Convert the datatype to numeric type. Features where this technique is applied: term, emp_length, last_week_pay.

* Categorical to Numerical: Further, categorical values are replaced with numerical for enmp_length, purpose.

* Common 'Other' string for categories that need not be considered: All the cells with values 'OTHER', 'NONE', 'ANY', 'other', 'major_purchase', 'small_business', 'medical','car', 'moving', 'vacation', 'house', 'wedding', 'renewable_energy', 'educational' are replaced with a common 'Other' string.

* Dropping unncessary features: Dropping 'member_id', 'batch_enrolled', 'sub_grade', 'emp_title', 'pymnt_plan', 'desc', 'title', 'zip_code', 'addr_state', 'application_type', 'verification_status_joint' features from the dataframe.

#### Dealing with Missing Values
* Features where ***median imputaion*** is applied: emp_length, annual_inc, open_acc, pub_rec, revol_util, total_acc, mths_since_last_delinq, mths_since_last_record, tot_coll_amt, tot_cur_bal, total_rev_hi_lim

* Features where ***zero imputaion*** is applied: delinq_2yrs, inq_last_6mths, collections_12_mths_ex_med, mths_since_last_major_derog, acc_now_delinq

* Features where ***mean imputaion*** is applied: total_rev_hi_lim, int_rate, annual_inc, dti, delinq_2yrs, inq_last_6mths, open_acc, revol_bal, revol_util, total_acc, total_rec_int, tot_cur_bal, total_rec_late_fee, recoveries, collection_recovery_fee, collections_12_mths_ex_med, acc_now_delinq, funded_amnt_inv

#### One Hot Encoding:
One hot encoding is one method of converting data to prepare it for an algorithm and get a better prediction. With one-hot, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns. Each integer value is represented as a binary vector. 
Following features were one hot encoded: 'term', 'grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status'

### Train Test Split
The transformed data is splitted into 30% test and 70% train.

## Models
A total of 8 models were used named: Logistic Regression, Decision Tree, Random Forest Classifier, KNN (with neighbor 1, 5, 7 and 10) and Gradient Boosting. For Hyperparameter Optimization GridSearchCV and RandomizedSearchCV is used to optimize the learning rate, min samples split, min samples leaf, max depth, max features, subsample, and n_estimators parameters. 

* RandomizedSearchCV was run with 1 iteration for 5 fold cross validation resulting in a total o f 5 fits.
The best parameters that are selected by RandomizedSearchCV: {learning_rate=0.01, max_features='log2', min_samples_leaf=0.1, min_samples_split=0.1, n_estimators=10, subsample=0.5}

* GridSearchCV was run with 216 iterations for 5 fold cross validation resulting in a total of 1080 fits.
The best parameters that are selected by RandomizedSearchCV: {'n_estimators': 700, 'min_child_weight': 7, 'max_depth': 20, 'learning_rate': 0.15, 'gamma': 10, 'colsample_bytree': 0.5}

### Comparing the models
All the models were compared amongst each other using the following as performance metrics: Accuracy, Precision, Recall and F1 Score.
Since F1 Score is the Harmonic mean of precision and recall, it takes into account both sensitivity and specificity. Thus the major performance metrics here would be F1 Score. Clearly Decision Tree has the best F1 Score followed by Gradient Boosting.

The best Accuracy Score is: 83.41% (Random Forest Classifier)
