#ignore warnings
import warnings

from schemas import PredictionRequest

warnings.filterwarnings('ignore')
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def load_model(model_name):
    script_path = os.path.abspath(__file__)
    print("Test4")

    model_path = os.path.join(os.path.dirname(script_path), 'dumped_models', model_name)
    print("Test5")

    return joblib.load(model_path)

def getPredictions(request:PredictionRequest):
  print("Test3")

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
    #patient data
    p1 = pd.DataFrame(
        data={
        'gender': [request.gender], 
        'age_oncheckup': [request.age]}
    )
    #Load the dumped models
    reg1_bmi = load_model('reg1_bmi_model.joblib')
    reg1_oxy = load_model('reg1_oxy_model.joblib')
    reg1_pulse = load_model('reg1_pulse_model.joblib')
    reg1_temp = load_model('reg1_temp_model.joblib')
    reg1_bps = load_model('reg1_bps_model.joblib')
    reg1_bpd = load_model('reg1_bpd_model.joblib')
    
    print("Test6")

    #Start prediction
    result1_bmi=reg1_bmi.predict(p1)
    print("Test7")

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
    print("Test8")

    # verify the health status
    health_status = {
      "currentStep" : 1
      # "health_status"
    }    
    print("Test9")

  elif request.current_step == 2:
    print("****** Current step = 2 ******")
    
    p2 = pd.DataFrame(
     data={
     'gender': [request.gender], 
     'age_oncheckup': [request.age],
     'bmi':[request.bmi]}
    )
      #Load the dumped models
    reg2_oxy = load_model('reg2_oxy_model.joblib')
    reg2_pulse = load_model('reg2_pulse_model.joblib')
    reg2_temp = load_model('reg2_temp_model.joblib')
    reg2_bps = load_model('reg2_bps_model.joblib')
    reg2_bpd = load_model('reg2_bpd_model.joblib')
    
    # Start Prediction
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
    

    p3 = pd.DataFrame(
        data={
        'gender': [request.gender], 
        'age_oncheckup': [request.age],
        'bmi':[request.bmi],
        'oxygen_of_blood':[request.spo2]}
    )
    #Load the dumped models
    reg3_pulse = load_model('reg3_pulse_model.joblib')
    reg3_temp = load_model('reg3_temp_model.joblib')
    reg3_bps = load_model('reg3_bps_model.joblib')
    reg3_bpd = load_model('reg3_bpd_model.joblib')
    
    #Start prediction
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
   
    p4= pd.DataFrame(
        data={
        'gender': [request.gender], 
        'age_oncheckup': [request.age],
        'bmi':[request.bmi],
        'oxygen_of_blood':[request.spo2],
        'pulse_rate':[request.pulse_rate]}
    )
    #Load the dumped models
    reg4_temp = load_model('reg4_temp_model.joblib')
    reg4_bps = load_model('reg4_bps_model.joblib')
    reg4_bpd = load_model('reg4_bpd_model.joblib')

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
    
    p5= pd.DataFrame(
        data={
        'gender': [request.gender], 
        'age_oncheckup': [request.age],
        'bmi':[request.bmi],
        'oxygen_of_blood':[request.spo2],
        'pulse_rate':[request.pulse_rate],
        'temperature':[request.temperature]}
    )
     #Load the dumped models
    reg5_bps = load_model('reg5_bps_model.joblib')
    reg5_bpd = load_model('reg5_bpd_model.joblib')

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
    
    reg6_bpd = load_model('reg6_bpd_model.joblib')

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
