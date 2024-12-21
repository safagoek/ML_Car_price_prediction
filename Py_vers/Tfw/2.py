import pandas as pd
from sklearn.model_selection import train_test_split

# verileri on isleme tabi tutan fonksiyon
def veri_onislem(dosya):
    # islenmis veriyi yukle
    data = pd.read_excel("processed_data.xlsx")
    
    # ihtiyacimiz olan ozellikleri ve hedefi belirle
    X = data[['Km', 'hp', 'year']]  # input
    y = data['price']  # hedef

    # datayi train ve test icin bolumle (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Veri on isleme tamamlandi")
    print(f"train verisi icin shape: {X_train.shape}")
    print(f"testing verisi icin shape: {X_test.shape}")

    # bolumlenmis veriyi model trainingi icin returnle
    return X_train, X_test, y_train, y_test

# dosya direkt execute edilirse on islemeyi yap
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = veri_onislem('processed_data.xlsx')

