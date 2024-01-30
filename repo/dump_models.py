#ignore warnings
import os
import warnings

from schemas import PredictionRequest

warnings.filterwarnings('ignore')
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def save_model(model, model_name):

    # Save the model to the 'dumped_models' folder
    script_path = os.path.abspath(__file__)
    model_path = os.path.join(os.path.dirname(script_path), 'dumped_models', model_name)

    # Use try-except to handle file existence
    try:
        joblib.dump(model, model_path)
    except FileNotFoundError:
        # If the 'dumped_models' folder doesn't exist, create it first
        os.makedirs(os.path.dirname(model_path))
        joblib.dump(model, model_path)


def getPredictions(request:PredictionRequest):
  print("Test2")
  # df = read_csv_file()
  df=pd.read_csv("Gapnext_v'2.csv")
  #select the variable to use
  df=df.iloc[:,[0,1,5,9,10,11,12,18]]
  #°F→℃
  df['temperature']=(df['temperature']-32)*5/9 
  #remove missing values
  df=df.dropna()
  if request.current_step == 1:
    #select the fetures
    X_1st=df[['gender','age_oncheckup']]
    #select the target of prediction
    y_1st_bmi=df['bmi']
    y_1st_oxy=df['oxygen_of_blood']
    y_1st_pulse=df['pulse_rate']
    y_1st_temp=df['temperature']
    y_1st_bps=df['bp_sys']
    y_1st_bpd=df['bp_dia']
    #prepare the train dataset
    X_train_1st_bmi, X_test_1st_bmi, y_train_1st_bmi, y_test_1st_bmi = train_test_split(X_1st, y_1st_bmi,test_size=0.3, random_state=1234)
    X_train_1st_oxy, X_test_1st_oxy, y_train_1st_oxy, y_test_1st_oxy = train_test_split(X_1st, y_1st_oxy,test_size=0.3, random_state=1234)
    X_train_1st_pulse, X_test_1st_pulse, y_train_1st_pulse, y_test_1st_pulse = train_test_split(X_1st, y_1st_pulse,test_size=0.3, random_state=1234)
    X_train_1st_temp, X_test_1st_temp, y_train_1st_temp, y_test_1st_temp = train_test_split(X_1st, y_1st_temp,test_size=0.3, random_state=1234)
    X_train_1st_bps, X_test_1st_bps, y_train_1st_bps, y_test_1st_bps = train_test_split(X_1st, y_1st_bps,test_size=0.3, random_state=1234)
    X_train_1st_bpd, X_test_1st_bpd, y_train_1st_bpd, y_test_1st_bpd = train_test_split(X_1st, y_1st_bpd,test_size=0.3, random_state=1234)
    #Regression
    reg1_bmi=RandomForestRegressor()
    #Build a forest of trees from the training set (X_train_1st_bmi, y_train_1st_bmi).
    reg1_bmi.fit(X_train_1st_bmi,y_train_1st_bmi)
    reg1_oxy=RandomForestRegressor()
    reg1_oxy.fit(X_train_1st_oxy,y_train_1st_oxy)
    reg1_pulse=RandomForestRegressor()
    reg1_pulse.fit(X_train_1st_pulse,y_train_1st_pulse)
    reg1_temp=RandomForestRegressor()
    reg1_temp.fit(X_train_1st_temp,y_train_1st_temp)
    reg1_bps=RandomForestRegressor()
    reg1_bps.fit(X_train_1st_bps,y_train_1st_bps)
    reg1_bpd=RandomForestRegressor()
    reg1_bpd.fit(X_train_1st_bpd,y_train_1st_bpd)

    #Dump the models
    save_model(reg1_bmi, 'reg1_bmi_model.joblib')
    save_model(reg1_oxy, 'reg1_oxy_model.joblib')
    save_model(reg1_pulse, 'reg1_pulse_model.joblib')
    save_model(reg1_temp, 'reg1_temp_model.joblib')
    save_model(reg1_bps, 'reg1_bps_model.joblib')
    save_model(reg1_bpd, 'reg1_bpd_model.joblib')

    #patient data
    p1 = pd.DataFrame(
        data={
        'gender': [request.gender], 
        'age_oncheckup': [request.age]}
    )
    #result of prediction
    result1_bmi=reg1_bmi.predict(p1)
    result1_oxy=reg1_oxy.predict(p1)
    result1_pulse=reg1_pulse.predict(p1)
    result1_temp=reg1_temp.predict(p1)
    result1_bps=reg1_bps.predict(p1)
    result1_bpd=reg1_bpd.predict(p1)
    actual_data = {
      "age": request.age,
      "gender": request.gender
    }
    predicted_data = {
      "bmi": float(result1_bmi[0]),
      "spo2":  float(result1_oxy[0]),
      "pulse_rate": float(result1_pulse[0]), 
      "temperature": float(result1_temp[0]),
      "bp_sys": float(result1_bps[0]),
      "bp_dia": float(result1_bpd[0]),
    }
    # verify the health status
    health_status = {
      "currentStep" : 1
      # "health_status"
    }    
  elif request.current_step == 2:
    print("****** Current step = 2 ******")
    X_2nd=df[['gender','age_oncheckup','bmi']]
    y_2nd_oxy=df['oxygen_of_blood']
    y_2nd_pulse=df['pulse_rate']
    y_2nd_temp=df['temperature']
    y_2nd_bps=df['bp_sys']
    y_2nd_bpd=df['bp_dia']

    X_train_2nd_oxy, X_test_2nd_oxy, y_train_2nd_oxy, y_test_2nd_oxy = train_test_split(X_2nd, y_2nd_oxy,test_size=0.3, random_state=1234)
    X_train_2nd_pulse, X_test_2nd_pulse, y_train_2nd_pulse, y_test_2nd_pulse = train_test_split(X_2nd, y_2nd_pulse,test_size=0.3, random_state=1234)
    X_train_2nd_temp, X_test_2nd_temp, y_train_2nd_temp, y_test_2nd_temp = train_test_split(X_2nd, y_2nd_temp,test_size=0.3, random_state=1234)
    X_train_2nd_bps, X_test_2nd_bps, y_train_2nd_bps, y_test_2nd_bps = train_test_split(X_2nd, y_2nd_bps,test_size=0.3, random_state=1234)
    X_train_2nd_bpd, X_test_2nd_bpd, y_train_2nd_bpd, y_test_2nd_bpd = train_test_split(X_2nd, y_2nd_bpd,test_size=0.3, random_state=1234)
    reg2_oxy=RandomForestRegressor()
    reg2_oxy.fit(X_train_2nd_oxy,y_train_2nd_oxy)
    reg2_pulse=RandomForestRegressor()
    reg2_pulse.fit(X_train_2nd_pulse,y_train_2nd_pulse)
    reg2_temp=RandomForestRegressor()
    reg2_temp.fit(X_train_2nd_temp,y_train_2nd_temp)
    reg2_bps=RandomForestRegressor()
    reg2_bps.fit(X_train_2nd_bps,y_train_2nd_bps)
    reg2_bpd=RandomForestRegressor()
    reg2_bpd.fit(X_train_2nd_bpd,y_train_2nd_bpd)

    #Dump the models

    #Dump the models
    save_model(reg2_oxy, 'reg2_oxy_model.joblib')
    save_model(reg2_pulse, 'reg2_pulse_model.joblib')
    save_model(reg2_temp, 'reg2_temp_model.joblib')
    save_model(reg2_bps, 'reg2_bps_model.joblib')
    save_model(reg2_bpd, 'reg2_bpd_model.joblib')

    p2 = pd.DataFrame(
     data={
     'gender': [request.gender], 
     'age_oncheckup': [request.age],
     'bmi':[request.bmi]}
    )
    result2_oxy=reg2_oxy.predict(p2)
    result2_pulse=reg2_pulse.predict(p2)
    result2_temp=reg2_temp.predict(p2)
    result2_bps=reg2_bps.predict(p2)
    result2_bpd=reg2_bpd.predict(p2)
    actual_data = {
      "age": request.age,
      "gender": request.gender,
      "bmi": request.bmi
    }
    predicted_data = {
      "bmi": None,
      "spo2": float(result2_oxy[0]),
      "pulse_rate": float(result2_pulse[0]), 
      "temperature": float(result2_temp[0]),
      "bp_sys": float(result2_bps[0]),
      "bp_dia": float(result2_bpd[0]),
    }
    health_status = {
      "currentStep" : 2
    }
  elif request.current_step == 3:
    print("***** Current step = 3")
    X_3rd=df[['gender','age_oncheckup','bmi','oxygen_of_blood']]
    y_3rd_pulse=df['pulse_rate']
    y_3rd_temp=df['temperature']
    y_3rd_bps=df['bp_sys']
    y_3rd_bpd=df['bp_dia']
    X_train_3rd_pulse, X_test_3rd_pulse, y_train_3rd_pulse, y_test_3rd_pulse = train_test_split(X_3rd, y_3rd_pulse,test_size=0.3, random_state=1234)
    X_train_3rd_temp, X_test_3rd_temp, y_train_3rd_temp, y_test_3rd_temp = train_test_split(X_3rd, y_3rd_temp,test_size=0.3, random_state=1234)
    X_train_3rd_bps, X_test_3rd_bps, y_train_3rd_bps, y_test_3rd_bps = train_test_split(X_3rd, y_3rd_bps,test_size=0.3, random_state=1234)
    X_train_3rd_bpd, X_test_3rd_bpd, y_train_3rd_bpd, y_test_3rd_bpd = train_test_split(X_3rd, y_3rd_bpd,test_size=0.3, random_state=1234)
    reg3_pulse=RandomForestRegressor()
    reg3_pulse.fit(X_train_3rd_pulse,y_train_3rd_pulse)
    reg3_temp=RandomForestRegressor()
    reg3_temp.fit(X_train_3rd_temp,y_train_3rd_temp)
    reg3_bps=RandomForestRegressor()
    reg3_bps.fit(X_train_3rd_bps,y_train_3rd_bps)
    reg3_bpd=RandomForestRegressor()
    reg3_bpd.fit(X_train_3rd_bpd,y_train_3rd_bpd)
    
    #Dump the models
    save_model(reg3_pulse, 'reg3_pulse_model.joblib')
    save_model(reg3_temp, 'reg3_temp_model.joblib')
    save_model(reg3_bps, 'reg3_bps_model.joblib')
    save_model(reg3_bpd, 'reg3_bpd_model.joblib')

    p3 = pd.DataFrame(
        data={
        'gender': [request.gender], 
        'age_oncheckup': [request.age],
        'bmi':[request.bmi],
        'oxygen_of_blood':[request.spo2]}
    )
    result3_pulse=reg3_pulse.predict(p3)
    result3_temp=reg3_temp.predict(p3)
    result3_bps=reg3_bps.predict(p3)
    result3_bpd=reg3_bpd.predict(p3)
    actual_data = {
      "age": request.age,
      "gender": request.gender,
      "bmi": request.bmi, 
      "spo2": request.spo2,
    }
    predicted_data = {
      "bmi": None, 
      "spo2": None,
      "pulse_rate": float(result3_pulse[0]),
      "temperature": float(result3_temp[0]),
      "bp_sys": float(result3_bps[0]),
      "bp_dia": float(result3_bpd[0]),
    }
    health_status = {
      "currentStep" : 3
    }
  elif request.current_step == 4:
    print("***** Current step = 4")
    X_4th=df[['gender','age_oncheckup','bmi','oxygen_of_blood','pulse_rate']]
    y_4th_temp=df['temperature']
    y_4th_bps=df['bp_sys']
    y_4th_bpd=df['bp_dia']
    X_train_4th_temp, X_test_4th_temp, y_train_4th_temp, y_test_4th_temp = train_test_split(X_4th, y_4th_temp,test_size=0.3, random_state=1234)
    X_train_4th_bps, X_test_4th_bps, y_train_4th_bps, y_test_4th_bps = train_test_split(X_4th, y_4th_bps,test_size=0.3, random_state=1234)
    X_train_4th_bpd, X_test_4th_bpd, y_train_4th_bpd, y_test_4th_bpd = train_test_split(X_4th, y_4th_bpd,test_size=0.3, random_state=1234)
    reg4_temp=RandomForestRegressor()
    reg4_temp.fit(X_train_4th_temp,y_train_4th_temp)
    reg4_bps=RandomForestRegressor()
    reg4_bps.fit(X_train_4th_bps,y_train_4th_bps)
    reg4_bpd=RandomForestRegressor()
    reg4_bpd.fit(X_train_4th_bpd,y_train_4th_bpd)

    #Dump the models
    save_model(reg4_temp, 'reg4_temp_model.joblib')
    save_model(reg4_bps, 'reg4_bps_model.joblib')
    save_model(reg4_bpd, 'reg4_bpd_model.joblib')

    p4= pd.DataFrame(
        data={
        'gender': [request.gender], 
        'age_oncheckup': [request.age],
        'bmi':[request.bmi],
        'oxygen_of_blood':[request.spo2],
        'pulse_rate':[request.pulse_rate]}
    )
    
    result4_temp=reg4_temp.predict(p4)
    result4_bps=reg4_bps.predict(p4)
    result4_bpd=reg4_bpd.predict(p4)
    actual_data = {
      "age": request.age,
      "gender": request.gender,
      "bmi": request.bmi,
      "spo2": request.spo2,
      "pulse_rate": request.pulse_rate
    }

    predicted_data = {
      "bmi": None,
      "spo2": None,
      "pulse_rate": None,
      "temperature": float(result4_temp[0]),
      "bp_sys": float(result4_bps[0]),
      "bp_dia": float(result4_bpd[0]),
    }
    health_status = {
      "currentStep" : 4
    }
  elif request.current_step == 5:
    print("***** Current step = 5")
    X_5th=df[['gender','age_oncheckup','bmi','oxygen_of_blood','pulse_rate','temperature']]
    y_5th_bps=df['bp_sys']
    y_5th_bpd=df['bp_dia']
    X_train_5th_bps, X_test_5th_bps, y_train_5th_bps, y_test_5th_bps = train_test_split(X_5th, y_5th_bps,test_size=0.3, random_state=1234)
    X_train_5th_bpd, X_test_5th_bpd, y_train_5th_bpd, y_test_5th_bpd = train_test_split(X_5th, y_5th_bpd,test_size=0.3, random_state=1234)
    reg5_bps=RandomForestRegressor()
    reg5_bps.fit(X_train_5th_bps,y_train_5th_bps)
    reg5_bpd=RandomForestRegressor()
    reg5_bpd.fit(X_train_5th_bpd,y_train_5th_bpd)

    #Dump the models
    save_model(reg5_bps, 'reg5_bps_model.joblib')
    save_model(reg5_bpd, 'reg5_bpd_model.joblib')

    p5= pd.DataFrame(
        data={
        'gender': [request.gender], 
        'age_oncheckup': [request.age],
        'bmi':[request.bmi],
        'oxygen_of_blood':[request.spo2],
        'pulse_rate':[request.pulse_rate],
        'temperature':[request.temperature]}
    )
    result5_bps=reg5_bps.predict(p5)
    result5_bpd=reg5_bpd.predict(p5)
    actual_data = {
      "age": request.age,
      "gender": request.gender,
      "bmi": request.bmi,
      "spo2": request.spo2,
      "pulse_rate": request.pulse_rate,
      "temperature": request.temperature
    }
    predicted_data = {
      "bmi": None,
      "spo2": None,
      "pulse_rate": None,
      "temperature": None,
      "bp_sys": float(result5_bps[0]),
      "bp_dia": float(result5_bpd[0]),
    }
    health_status = {
      "currentStep" : 5
    }
  elif request.current_step == 6:
    print("***** Current step = 6")
    X_6th=df[['gender','age_oncheckup','bmi','oxygen_of_blood','pulse_rate','temperature','bp_sys']]
    y_6th_bpd=df['bp_dia']
    X_train_6th_bpd, X_test_6th_bpd, y_train_6th_bpd, y_test_6th_bpd = train_test_split(X_6th, y_6th_bpd,test_size=0.3, random_state=1234)
    reg6_bpd=RandomForestRegressor()
    reg6_bpd.fit(X_train_6th_bpd,y_train_6th_bpd)
    #Dump the models
    save_model(reg6_bpd, 'reg6_bpd_model.joblib')


    p6= pd.DataFrame(
     data={
     'gender': [request.gender], 
     'age_oncheckup': [request.age],
     'bmi':[request.bmi],
     'oxygen_of_blood':[request.spo2],
     'pulse_rate':[request.pulse_rate],
     'temperature':[request.temperature],
     'bp_sys':[request.bp_sys]}
    )
    result6_bpd=reg6_bpd.predict(p6)
    actual_data = {
      "age": request.age,
      "gender": request.gender,
      "bmi": request.bmi,
      "spo2": request.spo2,
      "pulse_rate": request.pulse_rate,
      "temperature": request.temperature,
      "bp_sys": request.bp_sys,
    }
    predicted_data = {
      "bmi": None,
      "spo2": None,
      "pulse_rate": None,
      "temperature": None,
      "bp_sys": None,
      "bp_dia": float(result6_bpd[0])
    }
    health_status = {
      "currentStep" : 6
    }  
  elif request.current_step == 7:
    print("***** Current step = 7")
    actual_data = {
      "age": request.age,
      "gender": request.gender,
      "bmi": request.bmi,
      "spo2": request.spo2,
      "pulse_rate": request.pulse_rate,
      "temperature": request.temperature,
      "bp_sys": request.bp_sys,
      "bp_dia": request.bp_dia,
    }
    predicted_data = {
      "bmi": None,
      "spo2": None,
      "pulse_rate": None,
      "temperature": None,
      "bp_sys": None,
      "bp_dia": None,
    }
    health_status = {
      "currentStep" : 7
    }  
  print("Test12")
  return {
    "actual_data":actual_data,
    "predicted_data":predicted_data,
    "health_status":health_status
  }
