#ignore warnings
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Without encoding='shift_jis', there will be an error:  'utf-8' codec can't decode byte 0x81 in position
df=pd.read_csv("small_phc_data.csv",encoding='shift_jis')

#select the variables(the columns) to be used
df=df.iloc[:,[2,3,38,40,43,45,47,49,51,53,57,59,61,67,69]]

#remove missing values
df=df.dropna()

# # Select the features(independent variables) to be used
X = df[['Age ','Gender', 'BMI', 'Waist', 'Waist/Hip Ratio', 'Body Temperature','SpO2','Blood Pressure(sys)','Blood Pressure(dia)','Blood Glucose','Urinary Glucose','Urinary Protein', 'Pulse Rate', 'Color']]

# Select the target of the prediction(blood uric acid)
y_blood_uric_acid = df['Blood uric acid']

# Prepare the train dataset
X_train_blood_uric_acid, X_test_blood_uric_acid, y_train_blood_uric_acid, y_test_blood_uric_acid = train_test_split(X, y_blood_uric_acid,test_size=0.3, random_state=1234)

# Regression
reg_blood_pressure = RandomForestRegressor()
print("TEST1")
#Build a forest of trees from the training set(X_train_blood_uric_acid,y_train_blood_uric_acid)
reg_blood_pressure.fit(X_train_blood_uric_acid, y_train_blood_uric_acid)
print("X_train_blood_uric_acid")
# X_train_blood_uric_acid
# Enter the data for the features
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
    }
)

# Prediction result
result_blood_uric_acid = reg_blood_pressure.predict(data_features)
result_blood_uric_acid
