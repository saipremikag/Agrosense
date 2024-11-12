import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from model import cd
import sqlite3
import os
import joblib
 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

yield_model = pickle.load(open('model/model.pkl', 'rb'))
leaf_model =load_model("model/model-leaf.h5")
#model1 =load_model("model/model-apple.h5")

def pred_cot_dieas(cott_plant):
  test_image = load_img(cott_plant, target_size = (224, 224)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = leaf_model.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return "Bacteria", '1.html' 
  elif pred == 1:
      return 'Fungi', '2.html' 
  elif pred == 2:
      return 'Nematodes', '3.html'
  elif pred == 3:
      return 'Normal', '4.html'
  elif pred == 4:
      return 'Virus', '5.html'  
  else:
    return "Invaild Image", 'index11.html' # if index 3


@app.route("/")
def intro():
    return render_template('intro.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("home.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signup.html")

@app.route("/home")
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/index3')
def index3():
    return render_template('index3.html')

# render index.html page
@app.route("/index2", methods=['GET', 'POST'])
def index2():
        return render_template('index11.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    com_fea=['Area',                                                                 
                'percent_of_production',                     
                'State_Name_Andaman and Nicobar Islands',   
                'State_Name_Andhra Pradesh',                 
                'State_Name_Arunachal Pradesh',             
                'Season_Autumn',                        
                'Season_Kharif',                            
                'Season_Rabi',                               
                'Season_Whole Year',                         
                'Crop_Arecanut',                             
                'Crop_Arhar/Tur',                            
                'Crop_Bajra',                                
                'Crop_Banana',                              
                'Crop_Beans & Mutter(Vegetable)',            
                'Crop_Bhindi',                              
                'Crop_Black pepper',                         
                'Crop_Bottle Gourd',                         
                'Crop_Brinjal',                             
                'Crop_Cabbage',                              
                'Crop_Cashewnut',                            
                'Crop_Castor seed',                          
                'Crop_Citrus Fruit',                         
                'Crop_Coconut',                              
                'Crop_Coriander',                            
                'Crop_Cotton(lint)',                         
                'Crop_Cowpea(Lobia)',                        
                'Crop_Cucumber',                             
                'Crop_Dry chillies',                         
                'Crop_Dry ginger',                           
                'Crop_Garlic',                               
                'Crop_Ginger',                              
                'Crop_Gram',                                 
                'Crop_Grapes',                               
                'Crop_Groundnut',                            
                'Crop_Horse-gram',                           
                'Crop_Jowar',                                
                'Crop_Korra',                                
                'Crop_Lemon',                                
                'Crop_Linseed',                              
                'Crop_Maize',                                
                'Crop_Mango',                                
                'Crop_Masoor',                               
                'Crop_Mesta',                                
                'Crop_Moong(Green Gram)',                    
                'Crop_Niger seed',                           
                'Crop_Oilseeds total',                       
                'Crop_Onion',                                
                'Crop_Orange',                               
                'Crop_Other  Rabi pulses',                  
                'Crop_Other Fresh Fruits',                   
                'Crop_Other Kharif pulses',                  
                'Crop_Other Vegetables',
                'Crop_Papaya',                               
                'Crop_Peas  (vegetable)',
                'Crop_Pome Fruit',                           
                'Crop_Pome Granet',                         
                'Crop_Potato',                              
                'Crop_Pulses total',                         
                'Crop_Ragi',                                 
                'Crop_Rapeseed &Mustard',                    
                'Crop_Rice',                                 
                'Crop_Safflower',                            
                'Crop_Samai',                                
                'Crop_Sapota',                               
                'Crop_Sesamum',                              
                'Crop_Small millets',                        
                'Crop_Soyabean',                             
                'Crop_Sugarcane',                            
                'Crop_Sunflower',                            
                'Crop_Sweet potato',                        
                'Crop_Tapioca',                              
                'Crop_Tobacco',                              
                'Crop_Tomato',                               
                'Crop_Turmeric',                             
                'Crop_Urad',                                 
                'Crop_Varagu',                               
                'Crop_Wheat',                                
                'Crop_other fibres',                         
                'Crop_other misc. pulses',                   
                'Crop_other oilseeds'                       ]


    dataf=pd.DataFrame(columns=com_fea)
    dataf.loc[len(dataf)]=0
    
    
    features = [x for x in request.form.values()] #state | season | crop | area
    dataf['Area']=float(features[3])
    
    for j in com_fea:
        test_list=j.strip().split('_')
        
        if (test_list[0]=='State'):
            if(test_list[-1]==features[0]):
                dataf[j]=1
                
        elif (test_list[0]=='Season'):
            if(test_list[-1]==features[1]):
                dataf[j]=1
                
        elif (test_list[0]=='Crop'):            
            if(test_list[-1]==features[2]):
                dataf[j]=1
                    
    
    prediction = yield_model.predict(dataf)
    print(prediction)
    output = round(prediction[0], 2)


    return render_template('result.html', prediction_text='Predicted rate = {}'.format(output))

@app.route('/predict1',methods=['POST'])
def predict1():
    N = request.form['N']
    print(N)
    P = request.form['P']
    K = request.form['K']
    T = request.form['T']
    H = request.form['H']
    P = request.form['P']
    R = request.form['R']

    Soil_composition_list = np.array([N,P,K,T,H,P,R]).reshape(1,7)
    print(Soil_composition_list)
    loaded_model = pickle.load(open('model/Crop_recommendation_model.sav', 'rb'))
    crop = loaded_model.predict(Soil_composition_list)
    print(crop)

    if crop == 'rice':
        return render_template('rice.html')
    
    elif crop == 'maize':
        return render_template('maize.html')
    
    else:
        predicted_crop = crop[0]
        crop_discription = cd[crop[0]]['crop_dis']
        crop_link = cd[crop[0]]['crop_link']
        crop_img = cd[crop[0]]['crop_img']


        return render_template('prediction.html', predicted_crop = predicted_crop , crop_discription = crop_discription , crop_link = crop_link , crop_img = crop_img )

@app.route("/predict2", methods = ['GET','POST'])
def predict2():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)
               
        return render_template(output_page, pred_output = pred, user_image = file_path)

@app.route('/predict3',methods=['POST'])
def predict3():
    N = request.form['N']
    P = request.form['P']
    K = request.form['K']
    T = request.form['T']
    H = request.form['H']
    P = request.form['P']
    R = request.form['R']
    P1 = request.form['P1']
    R1 = request.form['R1']

    Soil_composition_list = np.array([N,P,K,T,H,P,R,P1,R1]).reshape(1,9)
    
    loaded_model1 = joblib.load("model/model.sav")
    crop = loaded_model1.predict(Soil_composition_list)
    

    if crop[0] == 0:
        predicted_crop = 'Moisture of Soil is High! so the STatus of Tap is OFF'
        return render_template('prediction1.html', predicted_crop = predicted_crop )
    
    elif crop[0] == 1:
        predicted_crop = 'Moisture of Soil is Low! so the STatus of Tap is ON'
        return render_template('prediction1.html', predicted_crop = predicted_crop )
    
    else:
        predicted_crop = 'Moisture of Soil is Medium! so the STatus of Tap is ON'
        return render_template('prediction1.html', predicted_crop = predicted_crop)


if __name__ == "__main__":
 app.run(host='127.0.0.1', port=5000,debug=True)