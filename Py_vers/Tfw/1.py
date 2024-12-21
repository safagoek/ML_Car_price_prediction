import pandas as pd
from sklearn.preprocessing import StandardScaler

def veri_yukle(dosya):
    data = pd.read_excel("cleaned_data.xlsx")
    print("Veriler yüklendi")
    print(data.head())
    return data

def veri_incele(data):
    print("\nKayip Degerler:\n", data.isnull().sum())
    print("\nVeri Infosu:")
    print(data.info())
    print("\nVeri Istatistigi:\n", data.describe())

def kayip_degerlerle_ilgilen(data):
    data['Km'].fillna(data['Km'].median(), inplace=True)
    data['hp'].fillna(data['hp'].median(), inplace=True)
    data['year'].fillna(data['year'].median(), inplace=True)
    data.dropna(subset=['price'], inplace=True)
    return data

def aykiri_degerleri_kaldir(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64'])
    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = data[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"\nAykiriliklar kaldirildi, kalan veri bicimi: {filtered_data.shape}")
    return filtered_data

def ozellikleri_normalize_et(data, features):
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    print("\nOzellikler normalize edildi")
    print(data.head())
    return data

if __name__ == "__main__":
    dosya = 'cleaned_data.xlsx'
    data = veri_yukle(dosya)
    veri_incele(data)
    data = kayip_degerlerle_ilgilen(data)
    data = aykiri_degerleri_kaldir(data)
    data = ozellikleri_normalize_et(data, ['Km', 'hp', 'year'])
    
    output_file = 'processed_data.xlsx'
    data.to_excel(output_file, index=False)
    print(f"\nVerinin hazirlanmasi tamamlandi. Hazır veri bu sekilde kaydedildi '{output_file}'.")

