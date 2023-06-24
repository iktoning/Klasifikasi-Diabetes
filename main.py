# Import library
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Mendapatkan path direktori proyek saat ini
base_dir = os.path.dirname(os.path.abspath(__file__))

# Mendapatkan path file dataset
dataset_path = os.path.join(base_dir, 'Dataset', 'Diabetes.csv')

# Membaca dataset
data = pd.read_csv(dataset_path)

# Memilih atribut yang akan digunakan
selected_features = ['Glucose', 'Insulin', 'Age']
X = data[selected_features]
y = data['Outcome']

# Mengambil subset data dengan atribut yang dipilih
selected_data = data[selected_features]

# Menampilkan data selected_features
print(selected_data)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Membuat objek klasifikasi KNN dengan nilai K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model dengan data latih
knn.fit(X_train, y_train)

# Memprediksi label dari data uji
y_pred = knn.predict(X_test)

# Menghitung dan menampilkan akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi: {:.2f}%".format(accuracy * 100))

# Menghitung dan menampilkan precision
precision = precision_score(y_test, y_pred)
print("Precision: {:.2f}%".format(precision * 100))

# Menghitung dan menampilkan recall
recall = recall_score(y_test, y_pred)
print("Recall: {:.2f}%".format(recall * 100))

# Menghitung dan menampilkan F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score: {:.2f}%".format(f1 * 100))

# Import streamlit untuk UI
import streamlit as st

# Tambahkan judul dan deskripsi aplikasi
st.title("Aplikasi Klasifikasi Penyakit Diabetes")

# Menerima input dari pengguna
glucose = st.number_input("Glucose", value=0.0)
insulin = st.number_input("Insulin", value=0.0)
age = st.number_input("Age", value=0.0)

# Membuat data baru berdasarkan input pengguna
new_data = pd.DataFrame([[glucose, insulin, age]], columns=selected_features)

# Melakukan prediksi dengan model KNN
prediction = knn.predict(new_data)

# Menampilkan hasil prediksi
result = "Tidak terkena diabetes." if prediction[0] == 0 else "Terkena diabetes."
st.write("Hasil Prediksi:", result)
