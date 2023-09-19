from django.shortcuts import render
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
def Home(request):
     if(request.method=="POST"):
        data=request.POST
        age=data.get('age')
        sex=data.get('sex')
        weight_kg=data.get('kg')
        height_cm=data.get('cm')
        medication=data.get('med')
        bmd=data.get('bmd')
        path="C:\\Users\\abhis\\Desktop\\Iris\\Basics\\bmd.csv"
        data=pd.read_csv(path)
        data['sex']=data['sex'].map({'M':1,'F':0})
        data['medication']=data['medication'].map({'No medication':1,'Glucocorticoids':0,'Anticonvulsant':0})
    

        inputs=data.drop(['fracture','id','waiting_time'],'columns')
        output=data.drop(['id','age','sex','weight_kg','height_cm','medication','waiting_time','bmd'],'columns')
        x_train,x_test,y_train,y_test= train_test_split(inputs,output,train_size=0.9)

        model=GaussianNB()
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        result=model.predict([[age,sex,weight_kg,height_cm,medication,bmd]])
        if len(result)!=0:
            return render(request,"Home.html",context={'result':result})
        
        
     return render(request,'Home.html')