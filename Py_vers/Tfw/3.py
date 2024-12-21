import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# on isleme alinmis veriyi yukle
file_path = 'processed_data.xlsx'
data = pd.read_excel(file_path)

# ozellikleri ve hedefi belirle
X = data[['Km', 'hp', 'year']]
y = data['price']

# yine veriyi 80 train 20 test olmak uzere bol
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# veriyi normalize et
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# noral neti olustur
def model_insasi():
    model = Sequential()

    # 3 noronlu input katmanı (km, beygir gucu, yil)
    model.add(Dense(64, input_dim=3, activation='relu'))  # 64 neurons in the first hidden layer

    # belirlenmis sayida noron iceren gizli katmanlar eklenmesi, modeli ekstra guclendirmek icin, her katmanda asamli olarak daha gelismis soyutlamalar yapabilecek ancak bu bir hiperparametre ve deneylerle degistirilmeli
    model.add(Dense(32, activation='relu'))  # ikinci gizli layer
    model.add(Dense(16, activation='relu'))  # ucuncu gizli layer
    # hedef olan fiyat tahminlemesi bir norona sahip
    model.add(Dense(1))

    # modeli compile et
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

# modelin insasini calistir
model = model_insasi()

# egitimini calistir
model.fit(X_train, y_train, epochs=103, batch_size=32, validation_data=(X_test, y_test))

# degerlendir
kayip = model.evaluate(X_test, y_test)
print(f"Test kaybı: {kayip}")

# tahminler
tahminler = model.predict(X_test)

# Show a few predictions
for i in range(5):
    print(f"Tahmini fiyat: {tahminler[i][0]}, Asil fiyat: {y_test.iloc[i]}")

# kullanici inputu ile deney
print("\nArabanin ozelliklerini girin:")
yil = float(input("Arabanin yili: "))
kilometre = float(input("Arabanin km'sini yazin: "))
beygirgucu = float(input("Arabanin beygir gucunu yazin: "))

# bu deneyin tek bir data olarak 2b numpy dizisi olarak hazirlanmasi
input_data = np.array([[kilometre, beygirgucu, yil]])

# deney datasinin normalizasyonu
input_data_scaled = scaler.transform(input_data)

# modelimizin deney datasina tahmini
predicted_price = model.predict(input_data_scaled)

# sonucu goster
print(f"Tahmini fiyat: ${predicted_price[0][0]:.2f}")

