import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

st.title("LSTM ile Günlük Suç Sayısı Tahmini :chart_with_upwards_trend:")
model=keras.models.load_model('suc_lstm_model.keras')
scaler=pickle.load(open('lstm_scaler.pkl','rb'))
son=pickle.load(open('lstm_son.pkl','rb'))

gun_sayisi=st.number_input('Kaç gün sonrasına kadar tahmin edilsin?',1,60,7)
if st.button('Tahmin et'):
    pencere=scaler.transform(np.array(son['son_degerler']).reshape(-1,1)).flatten().tolist()
    tarih=son['son_tarih']
    sonuc=[]
    for i in range(int(gun_sayisi)):
        x=np.array(pencere[-30:]).reshape(1,30,1)
        t=float(model.predict(x,verbose=0)[0][0])
        pencere.append(t)
        tarih=tarih+pd.Timedelta(days=1)
        gercek=scaler.inverse_transform([[t]])[0][0]
        sonuc.append({'tarih':tarih.strftime('%Y-%m-%d'),'tahmini_suc_sayisi':round(float(gercek))})
    sonuc=pd.DataFrame(sonuc)
    st.line_chart(sonuc.set_index('tarih'))
    st.write(sonuc)
