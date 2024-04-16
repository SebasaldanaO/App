import streamlit as st 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
np.random.seed(12345)
st.title('El Precio Es Correcto')
datos = np.random.normal(0,1,size=(100,4))
data = pd.DataFrame(datos,
                    columns = list('ABCD'))
st.dataframe(data)
e = np.random.normal(0,1,size=100)
y = data['A']*2 + data['B']*3 + data['C']*4 + data['D']*0.3 + 10 + e
model = DecisionTreeRegressor(max_depth = 4)
model.fit(data,y)
st.subheader('A')
val_a = st.slider('Seleccione El Valor De A',
          data['A'].min(),
          data['A'].max())
st.subheader('B')
val_b = st.slider('Seleccione El Valor De B',
          data['B'].min(),
          data['B'].max())
st.subheader('C')
val_c = st.slider('Seleccione El Valor De C',
          data['C'].min(),
          data['C'].max())
st.subheader('D')
val_d = st.slider('Seleccione El Valor De D',
          data['D'].min(),
          data['D'].max())

Valores = np.array([[val_a,val_b,val_c,val_d]])
pre = model.predict(Valores)
st.write(pre)












