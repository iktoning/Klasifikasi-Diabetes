# Import library
import numpy as np
import streamlit as st
import pickle

# Membuat function untuk prediksi diabetes
def diabetes_prediction(input_data):
    # Merubah input_data ke numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Membentuk kembali array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    #  Hasil prediksi
    if prediction[0] == 0:
        return 'Anda tidak terkena diabetes'
    else:
        return 'Anda terkena diabetes'    

if __name__ == '__main__':
    #Load model
    loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))
    
    # Tambahkan judul aplikasi
    st.title("Aplikasi Klasifikasi Penyakit Diabetes")

    # Menerima input dari pengguna
    age = st.text_input("Umur")
    glucose = st.text_input("Glukosa")
    insulin = st.text_input("Insulin")
    
    #Prediksi
    diagnosis = ''

    # Menampilkan hasil prediksi
    if st.button('Hasil Tes Diabetes'):
        try:
            diagnosis = diabetes_prediction(
                [age, glucose, insulin]
            )
        except:
            diagnosis = "Pastikan input valid!"
            
    st.success(diagnosis)