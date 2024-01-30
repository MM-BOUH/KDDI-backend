#ignore warnings
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Without encoding='shift_jis', there will be an error:  'utf-8' codec can't decode byte 0x81 in position
df=pd.read_csv("small_phc_data_after_processing.csv",encoding='shift_jis')

### Data Preprocessing

#select the variables(the columns) to be used
df = df.loc[:, ~df.columns.isin(['Bar code Id', 'Company Name', '5. What is your monthly family expenditure?','Mobile Number', '1. What is your occupation?', 'Blood Hemoglobin', 'Blood Hemoglobin Color',  'Blood cholesterol', 'Blood cholesterol Color'],)]
df = df.iloc[:,1:61]
#remove missing values
df=df.dropna()
# DataTable.max_columns = 80
## Replace missing values by 0
# df = df.fillna(0)
## Replace the categorical data by numeric data 
le = LabelEncoder()
### The data here is ordinal => the order is important 
edu = ['1. No education (no school entered)', '2. Primary school completed', '3. Secondary school completed', '4. High school completed',
       'Diploma', '5. Vocation school completed', '6. College/University completed', '7. Higher (Master or Doctor) completed']
ordi = OrdinalEncoder(categories=[edu])
df['2. What education did you complete?'] = ordi.fit_transform(df[['2. What education did you complete?']])

## Replace the categorical data by numeric data 
df['10. Do you drink sugar contained drinks (Coke, Fanta, Soda, Fruit Juice, other Sweet/Sugar contained drinks) three or more times a week?'] = le.fit_transform(df['10. Do you drink sugar contained drinks (Coke, Fanta, Soda, Fruit Juice, other Sweet/Sugar contained drinks) three or more times a week?'])
df['11. Do you eat fast foods such as Pizza, Hamburger, Deep Fried Foods (e.g. Singara, Samosa, Moglai Parata, etc.) three or more time a week?'] = le.fit_transform(df['11. Do you eat fast foods such as Pizza, Hamburger, Deep Fried Foods (e.g. Singara, Samosa, Moglai Parata, etc.) three or more time a week?'])

# df.to_csv(r'small_phc_data_after_processing.csv', index=False)


# # Selecting the independent variables(features)
X = df[['Age ','Gender', '2. What education did you complete?','10. Do you drink sugar contained drinks (Coke, Fanta, Soda, Fruit Juice, other Sweet/Sugar contained drinks) three or more times a week?','11. Do you eat fast foods such as Pizza, Hamburger, Deep Fried Foods (e.g. Singara, Samosa, Moglai Parata, etc.) three or more time a week?','BMI', 'Waist', 'Waist/Hip Ratio', 'Body Temperature','SpO2','Blood Pressure(sys)','Blood Pressure(dia)','Blood Glucose','Urinary Glucose','Urinary Protein', 'Pulse Rate', 'Color']]

# # # Select the target of the prediction(blood uric acid)
y_blood_uric_acid = df['Blood uric acid']



# # # Prepare the train dataset
X_train_blood_uric_acid, X_test_blood_uric_acid, y_train_blood_uric_acid, y_test_blood_uric_acid = train_test_split(X, y_blood_uric_acid,test_size=0.3, random_state=1234)

# # # Regression
reg_blood_pressure = RandomForestRegressor()

# #Build a forest of trees from the training set(X_train_blood_uric_acid,y_train_blood_uric_acid)
reg_blood_pressure.fit(X_train_blood_uric_acid, y_train_blood_uric_acid)

# # # Enter the data for the features
data_features = pd.DataFrame(
    data={
        'Age':[59],
        'Gender':[1],
        'BMI':[20.85],
        'Waist':[85],
        'Waist/Hip Ratio':[1.06],
        'Body Temperature':[95],
        'SpO2':[96],
        'Blood Pressure(sys)':[138],
        'Blood Pressure(dia)':[78],
        'Blood Glucose':[234],
        'Urinary Glucose':[1],
        'Urinary Protein':[1],
        'Pulse Rate':[94],
        'Color':[3],
        '2. What education did you complete?': [7],
        '10. Do you drink sugar contained drinks (Coke, Fanta, Soda, Fruit Juice, other Sweet/Sugar contained drinks) three or more times a week?':[0],
        '11. Do you eat fast foods such as Pizza, Hamburger, Deep Fried Foods (e.g. Singara, Samosa, Moglai Parata, etc.) three or more time a week?':[0]
    }
)

# # # Prediction result
y_pred_result_blood_uric_acid = reg_blood_pressure.predict(data_features)
y_pred_result_blood_uric_acid
