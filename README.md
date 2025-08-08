# Bank Deposito Marketing Campaign Model Prediction
---
## *Final Project Delta Team - JCDS2804*
---
Purwadhika Digital Technology School - Job Connector Data Science
* Ikhsan Herdi Fariyanto
* Sulaeman Nurhakim
* Muhammad Naufal Maahir
---

## *1. Background*
**Bank marketing campaigns dataset analysis Opening a Term Deposit**
It is a dataset that describing Portugal bank marketing campaigns results.
Conducted campaigns were based mostly on direct phone calls, offering bank client to place a term deposit.
If after all marking efforts client had agreed to place deposit - target variable was marked 'yes', otherwise 'no'

Sourse of the data
https://archive.ics.uci.edu/ml/datasets/bank+marketing

## *2. Business Understanding*
**Context**
A bank in Portugal conducted a marketing campaign via telephone to offer term deposit products. However, the campaign achieved relatively low success rates (small conversion ratio). Management aims to enhance campaign effectiveness by identifying client characteristics associated with accepting deposit offers, allowing marketing strategies to become more targeted and cost-effective.

**Target**
* `yes` → client accepts the deposit offer
* `no` → client rejects the offer
This represents a binary classification target.

## *3. Problem Statements*
This project will try to answer the following questions:

How can we predict the likelihood of a customer accepting a term deposit offer based on historical marketing campaign data and their profiles, enabling the bank to target campaigns more effectively?

## *4. Goals* 
The main goals of this project are: 

Based on these issues, the company aims to develop the capability to predict a customer's likelihood of accepting a term deposit offer using historical customer data by:
* Building a classification model to predict whether a customer will accept a deposit offer.
* Identifying key factors influencing customer decisions.
* Providing strategic recommendations to improve campaign success rates.

## *5. Stakeholders*
The stakeholders involved in this project include:

* **Bank Management** → makes business decisions based on predictive outcomes.
* **Marketing/Telemarketing Team** → optimizes strategies and customer segmentation.
* **End Customers** → indirectly benefit from more relevant campaigns.

## *6. Analytical Approach*
This project will follow these steps:

* **Data Understanding & Cleaning:** Examine outliers, handle missing values, encode categorical variables, etc.
* **Exploratory Data Analysis (EDA):** Analyze data distributions and relationships between features and the target variable.
* **Feature Engineering:** Create derived features (e.g., age groups, campaign timing segmentation).
* **Modeling:**

  * **Baseline Model:** Logistic Regression
  * **Advanced Models:** Random Forest, XGBoost, etc.
* **Model Evaluation:** Assess performance using classification metrics.


## *7. Metric Evaluation*
![0_wPxmui5uaThY8_ab](https://github.com/user-attachments/assets/7bb4e387-aae5-4a6f-8e60-95896a33cdb0)
* **True Negative**: The model predicts the customer will not accept the deposit offer (`no`), and in reality, they don't (`no`).
* **False Negative**: The model predicts the customer will not accept the deposit offer (`no`), but in reality, they do (`yes`).
* **False Positive**: The model predicts the customer will accept the deposit offer (`yes`), but in reality, they don't (`no`).
* **True Positive**: The model predicts the customer will accept the deposit offer (`yes`), and in reality, they do (`yes`).

### **Type I Error – False Positive**

**Case**: The model predicts the customer will accept the deposit offer (`yes`), but in reality, they won’t (`no`).
**Business Consequences**:

* Wasted time and cost on telemarketing
* Marketing resources allocated to uninterested clients
* Reduced campaign efficiency

### **Type II Error – False Negative**

**Case**: The model predicts the customer will not accept the deposit offer (`no`), but in reality, they would (`yes`).
**Business Consequences**:

* Missed golden opportunities to convert potential clients
* Lower overall campaign success rate
* Overlooking high-potential customers

Given these consequences, our goal is to build a model that minimizes business inefficiencies. Therefore, we aim to maximize the number of correct positive predictions (true positives), while minimizing false positives as much as possible.


## *8. Data Understanding*
**Bank Client Data:**
| **No** | **Attribute**                   | **Data Type** | **Description**                                    |
| -------| --------------------------- | --------- | ---------------------------------------------- |
|1. | `age`                         | Int       | Umur pelanggan                                 |
|2. | `job`                         | Object    | type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")                     |
|3. | `marital`                     | Object    | marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)                          |
|4. | `education`                   | Object    | categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"                    |
|5. | `default`                     | Object    | has credit in default (categorical: "no","yes","unknown")          |
|6. | `housing`                     | Object    | has housing loan? (categorical: "no","yes","unknown")        |
|7. | `loan`                        | Object    | has personal loan? (categorical: "no","yes","unknown")                        |

**related with the last contact of the current campaign:**
| **No** | **Attribute**                   | **Data Type** | **Description**                                    |
| -------| --------------------------- | --------- | ---------------------------------------------- |
|8. | `contact`                     | Object    | contact communication type (categorical: "cellular","telephone")                        |
|9. | `month`                       | Object    | last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec")   |
|10. | `day_of_week`                 | Object    | last contact day of the week (categorical: "mon","tue","wed","thu","fri")                |
|11. | `duration`                    | Int       | last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.             |

**other attributes:**
| **No** | **Attribute**                   | **Data Type** | **Description**                                    |
| -------| --------------------------- | --------- | ---------------------------------------------- |
|12. | `campaign`                    | Int       | number of contacts performed during this campaign and for this client (numeric, includes last contact)                        |
|13. | `pdays`                       | Int       | number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)                    |
|14. | `previous`                    | Int       | number of contacts performed before this campaign and for this client (numeric)                 |
|15. | `poutcome`                    | Object    | outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")       |

**other attributes:**
| **No** | **Attribute**                   | **Data Type** | **Description**                                    |
| -------| --------------------------- | --------- | ---------------------------------------------- |
|16. | `emp.var.rate`                | Float     | employment variation rate - quarterly indicator (numeric) |
|17. | `cons.price.idx`              | Float     | consumer price index - monthly indicator (numeric)                    |
|18. | `cons.conf.idx`               | Float     | consumer confidence index - monthly indicator (numeric)                           |
|19. | `euribor3m`                   | Float     | euribor 3 month rate - daily indicator (numeric)                    |
|20. | `nr.employed`                 | Float     | number of employees - quarterly indicator (numeric)         |
|21. | `y`                           | Object    | has the client subscribed a term deposit? (binary: "yes","no")         |

## *9. Summary of EDA*
Based on the previous analysis, we can summarize several key points as follows:

1. Prospective customers aged **30–45 years** are the bank's main target, as this age group is considered financially mature.
2. Analysis results show that the age groups **56–74 years** and **15–24 years** have the highest conversion rates, at **0.43** and **0.24**, which are higher than the productive age group.
3. Although **admin** and **blue-collar** workers are offered term deposits the most, occupations such as **students** and **retirees** have higher conversion rates, at **0.31** and **0.25**, respectively.
4. Although **married** customers are contacted most frequently, **single** customers have a higher conversion rate of **0.14**.
5. Those with **university degrees** are offered term deposits the most and have a high conversion rate of **0.14**.
6. Customers with a history of **default** have a **0% conversion rate**, and only **3 customers (0.0072%)** fall into this category.
7. There is no significant difference in conversion rates between customers who **have housing loans** and those who do not.
8. In the loan column, although customers **without loans** are contacted more often, the conversion rate is nearly the same as those **with loans**.
9. The bank’s strategy of contacting customers via **mobile phone (cellular)** proves effective, with a **0.15** conversion rate compared to **landlines** at **0.05**.
10. **March, September, October, and December** have high conversion rates despite low contact volume. On the other hand, **May** has high contact volume but a low conversion rate.
11. **Thursday** is the best day for the last contact, with a **0.12** conversion rate.
12. **Very long call durations** yield the highest conversion rates, followed by **medium-long**, **medium-short**, and **very short calls**.
13. Most customers have never been contacted before (category **999**).
14. The median duration of the last contact does not show a significant difference between customers who subscribed and those who didn’t.
15. Most customers were not contacted in the previous campaign (category **0**).

These insights can help improve the bank’s conversion rate and profitability, and are expected to further enhance the bank’s overall performance.

## *10. EDA Recommendation*
After analyzing the data, we have identified several recommended actions to improve the conversion rate, which in turn will increase the company's profitability:

1. **More Precise Demographic Targeting:** Focus marketing efforts on two specific age groups, individuals aged 17 to 25 and those aged 65 to 74. Additionally, prioritize customers who are students or retirees.
2. **Target Single Customers:** Market more actively to customers with a single marital status, as this group has a high conversion rate but relatively low marketing exposure.
3. **Provide Educational Materials:** Offer more information and educational materials on the importance of term deposits to customers with low education levels, such as those categorized as "Illiterate" or with only basic education, to potentially boost conversion rates.
4. **Avoid Contacting Customers with Poor Payment History:** Be more selective in contacting customers with a history of credit default, as they may be less interested in subscribing to term deposits.
5. **Prefer Mobile Phone Communication:** Use mobile phones for contacting customers more often than landlines.
6. **Monitor Euribor Interest Rate:** Increase contact with customers when the Euribor interest rate is declining, to assess any relationship between interest rate trends and conversion rates.
7. **Prioritize Contact on Thursdays:** Give priority to contacting customers on Thursdays, as this day shows a higher conversion rate, indicating better chances for positive responses.
8. **Creative and Interactive Approach:** Strive to make conversations more engaging and interactive rather than too brief. Although this may take more time, longer calls tend to attract more customers to subscribe to term deposits.
9. **Limit Campaign Contacts:** Restrict the number of campaigns per customer to a maximum of 20 times, as excessive campaigns can become ineffective.
10. **Focus on Previous Term Deposit Subscribers:** Pay special attention to customers who have subscribed to term deposits in previous campaigns, as they are more likely to subscribe again.
11. **Monitor Economic Indicators:** Contact customers when the "Employment Variation Rate" shows a negative trend (increasing layoffs), "CPI" (inflation rate) is low, and "CCI" (Consumer Confidence Index) is high.
12. **Prioritize Communication During Low Euribor Rates:** Focus on contacting customers when the 3-month Euribor interest rate is low, as this may have a stronger influence on their decision.
13. **Consider Number of Employees:** Contact customers when the "Number of Employees" indicator shows a low number of employees, as this situation might affect their decision related to term deposits.

By implementing these recommendations and adapting them into a more detailed marketing strategy, we are optimistic that they will help increase the conversion rate and support the company’s business growth.
