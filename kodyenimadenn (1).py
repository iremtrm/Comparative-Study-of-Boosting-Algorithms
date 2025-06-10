#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:22:25 2024

@author: simalvize
"""
pip install pandas openpyxl
pip install xlrd


import pandas as pd
veriSeti = pd.read_excel(r"C:\Users\irem\Downloads\data.xls", engine="xlrd")

print(veriSeti.head())
print(veriSeti.columns)
# Veri setindeki her bir değişkenin veri türünü (class) öğrenmek
print(veriSeti.dtypes)

# Eksik veri kontrolü (NaN değerler)
print(veriSeti.isnull().sum())

# Kategorik degiskenlerin veri tipinin category yapilmasi
veriSeti.dtypes

veriSeti["cınsıyet"] = veriSeti["cınsıyet"].astype("category")
veriSeti["egıtımduzeyı"] = veriSeti["egıtımduzeyı"].astype("category")
veriSeti["meslegı"] = veriSeti["meslegı"].astype("category")
veriSeti["sıgarakullanımı"] = veriSeti["sıgarakullanımı"].astype("category")
veriSeti["tanı"] = veriSeti["tanı"].astype("category")
veriSeti["ailedekoahveyaastımTanılıHastavarmı"] = veriSeti["ailedekoahveyaastımTanılıHastavarmı"].astype("category")
veriSeti["varsakımde ANNE"] = veriSeti["varsakımde ANNE"].astype("category")
veriSeti["varsakımde BABA"] = veriSeti["varsakımde BABA"].astype("category")
veriSeti["varsakımde KARDES"] = veriSeti["varsakımde KARDES"].astype("category")
veriSeti["varsakımde DİĞER"] = veriSeti["varsakımde DİĞER"].astype("category")    
veriSeti["hastaneyeyattımı"] = veriSeti["hastaneyeyattımı"].astype("category")    

# Sayısal sütunları seçme
sayisal_sutunlar = veriSeti.select_dtypes(include=['float64', 'int64']).columns

# Sayısal sütunlardaki NaN değerlerini medyan ile doldurma
for sutun in sayisal_sutunlar:
    veriSeti[sutun] = veriSeti[sutun].fillna(veriSeti[sutun].median())
veriSeti.isnull().sum()  # NaN'ların kontrolü

yaş_k2 = pd.cut(veriSeti["YAŞ"], bins = 3)
yaş_k2.cat.categories
yaş_k2 = yaş_k2.cat.rename_categories(["GENÇ", "YETİŞKİN", "YAŞLI"])
yaş_k2.value_counts()
pd.concat([veriSeti["YAŞ"].head(10), yaş_k2[0:10]], axis=1)
    veriSeti = veriSeti.drop(columns=["YAŞ"])  # YAŞ değişkenini çıkar
    veriSeti["YAŞ_GRUBU"] = yaş_k2  # Yeni grubu ekle
    print(veriSeti)
    print(veriSeti["YAŞ_GRUBU"].dtype)

# Nabız aralıklarını belirleme (Düşük, Normal, Yüksek)
veriSeti['NaNbız'] = pd.cut(
    veriSeti['NaNbız'],
    bins=[0, 60, 100, float('inf')],  # Aralık sınırları: Düşük, Normal, Yüksek
    labels=['DÜŞÜK', 'NORMAL', 'YÜKSEK']  # Kategoriler
)

# Sonuçları görüntüleme
print(veriSeti)
print("\n'NaNbız' Sütununun Veri Tipi:")
print(veriSeti["NaNbız"].dtype)
print(veriSeti.columns)


# Değişiklikleri kontrol etme
veriSeti.isnull().sum()  # NaN'ların kontrolü

veriSeti.describe()

# Saat cinsine dönüştürme işlemi
veriSeti['acılservısetoplamyatıssuresısaat'] = veriSeti['acilservistoplamyatışsüresigün'] * 24
veriSeti['yogumbakımatoplamyatıssuresısaat'] = veriSeti['yogumbakımatoplamyatıssüresıgun'] * 24
veriSeti['servıseoplamyatıssuresısaat'] = veriSeti['servisetoplamyatıssüresıgun'] * 24

# Eski sütunları kaldırmak isterseniz
veriSeti = veriSeti.drop(columns=['acilservistoplamyatışsüresigün', 'yogumbakımatoplamyatıssüresıgun', 'servisetoplamyatıssüresıgun'])

# Sonuçları kontrol etme
print(veriSeti.head())

# 'tanısuresıyıl' sütununu sayısala dönüştürün
veriSeti['tanısuresıyıl'] = pd.to_numeric(veriSeti['tanısuresıyıl'], errors='coerce')


# Yıl'ı Ay'a çevirmek için her bir yıl değerini 12 ile çarpın
veriSeti['tanısuresıay'] = veriSeti['tanısuresıyıl'] * 12
# Eski sütunları kaldırmak isterseniz
veriSeti = veriSeti.drop(columns=['tanısuresıyıl'])


# Boyu metreye çevirme (Eğer boy santimetre olarak verilmişse)
veriSeti['boy'] = veriSeti['boy'] / 100  # Boyu santimetreden metreye çevirme (eğer boy santimetre cinsindense)

# Kitle Endeksi (BMI) hesaplama
veriSeti['kitleendeksi'] = veriSeti['vucutagırlıgı'] / (veriSeti['boy'] ** 2)

# Sonuçları kontrol etme
veriSeti[['boy', 'vucutagırlıgı', 'kitleendeksi']].head()

# Boy ve vücut ağırlığı sütunlarını silme
veriSeti = veriSeti.drop(columns=['boy', 'vucutagırlıgı'])

# Silinen sütunları kontrol etme
veriSeti.head()

print(veriSeti.isnull().sum())

veriSeti = veriSeti.drop(columns=['basvurutarıhı'])

from sklearn.decomposition import PCA

# PCA uygulamak için sistolik ve diastolik kan basıncını seçin
pca = PCA(n_components=1)
veriSeti['PCA_KanBasıncı'] = pca.fit_transform(veriSeti[['kanbasıncısıstolık', 'kanbasıncıdıastolık']])

# PCA sonucu, sistolik ve diastolik kan basıncının tek bir bileşeni olacak
print(veriSeti[['PCA_KanBasıncı']].describe())
veriSeti = veriSeti.drop(columns=['kanbasıncısıstolık', 'kanbasıncıdıastolık'])

# FEV1 ve PEF sütunlarını veri setinden çıkarma
veriSeti = veriSeti.drop(columns=['FEV1', 'PEF'])
veriSeti.head()

import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

# Sayısal tanıları etiketlere dönüştürme
veriSeti['tanı'] = veriSeti['tanı'].replace({1: 'Astım', 2: 'KOAH'})

# Veri setinde bulunan tanıları kontrol etme
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# İlgili iki grup seçimi
group1 = veriSeti[veriSeti['tanı'] == 'Astım']['solunumsayısı']
group2 = veriSeti[veriSeti['tanı'] == 'KOAH']['solunumsayısı']
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# Mann-Whitney U testi
stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# Görselleştirme
plt.figure(figsize=(8, 5))
sns.boxplot(x='tanı', y='solunumsayısı', data=veriSeti)
plt.title('Tanı ve Solunum Sayısı İlişkisi')
plt.show()

# İlgili iki grup seçimi
group1_1 = veriSeti[veriSeti['tanı'] == 'Astım']['sıgarayıbırakannekadarGÜNıcmıs']
group2_1 = veriSeti[veriSeti['tanı'] == 'KOAH']['sıgarayıbırakannekadarGÜNıcmıs']
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# Mann-Whitney U testi
stat, p_value = mannwhitneyu(group1_1, group2_1, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# Sonuç yorumlama
if p_value < 0.05:
    print("İki grup arasında sigarayı bırakan ne kadar gün içtiği açısından anlamlı bir fark vardır.")
else:
    print("İki grup arasında sigarayı bırakan ne kadar gün içtiği açısından anlamlı bir fark vardır.")

# Görselleştirme
plt.figure(figsize=(8, 5))
sns.boxplot(x='tanı', y='solunumsayısı', data=veriSeti)
plt.title('Tanı ve Kaç Gün İçtiği İlişkisi')
plt.show()

# İlgili iki grup seçimi
group1_2 = veriSeti[veriSeti['tanı'] == 'Astım']['sıgarabırakangundekacadetıcmıs']
group2_2 = veriSeti[veriSeti['tanı'] == 'KOAH']['sıgarabırakangundekacadetıcmıs']
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# Mann-Whitney U testi
stat, p_value = mannwhitneyu(group1_2, group2_2, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# Sonuç yorumlama
if p_value < 0.05:
    print("İki grup arasında sigarayı bırakan günde kaç adet içtiği açısından anlamlı bir fark vardır.")
else:
    print("İki grup arasında sigarayı bırakan günde kaç adet içtiği açısından anlamlı bir fark vardır.")

# Görselleştirme
plt.figure(figsize=(8, 5))
sns.boxplot(x='tanı', y='solunumsayısı', data=veriSeti)
plt.title('Tanı ve Günde Kaç Adet İçtiği İlişkisi')
plt.show()


# İlgili iki grup seçimi
group1_3 = veriSeti[veriSeti['tanı'] == 'Astım']['nezamanbırakmısGÜN']
group2_3 = veriSeti[veriSeti['tanı'] == 'KOAH']['nezamanbırakmısGÜN']
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# Mann-Whitney U testi
stat, p_value = mannwhitneyu(group1_3, group2_3, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# Görselleştirme
plt.figure(figsize=(8, 5))
sns.boxplot(x='tanı', y='solunumsayısı', data=veriSeti)
plt.title('Tanı ve Kaç Gün Olmuş Bırakalı İlişkisi')
plt.show()

# İlgili iki grup seçimi
group1_4 = veriSeti[veriSeti['tanı'] == 'Astım']['sıgarayadevamedengundekacadetıcıyo']
group2_4= veriSeti[veriSeti['tanı'] == 'KOAH']['sıgarayadevamedengundekacadetıcıyo']
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# Mann-Whitney U testi
stat, p_value = mannwhitneyu(group1_4, group2_4, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# Görselleştirme
plt.figure(figsize=(8, 5))
sns.boxplot(x='tanı', y='solunumsayısı', data=veriSeti)
plt.title('Tanı ve Kaç Gün Olmuş Bırakalı İlişkisi')
plt.show()

# İlgili iki grup seçimi
group1_5 = veriSeti[veriSeti['tanı'] == 'Astım']['tanısuresıay']
group2_5 = veriSeti[veriSeti['tanı'] == 'KOAH']['tanısuresıay']
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# Mann-Whitney U testi
stat, p_value = mannwhitneyu(group1_5, group2_5, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# Görselleştirme
plt.figure(figsize=(8, 5))
sns.boxplot(x='tanı', y='solunumsayısı', data=veriSeti)
plt.title('Tanı ve Kaç Gün Olmuş Bırakalı İlişkisi')
plt.show()

# İlgili iki grup seçimi
group1_6 = veriSeti[veriSeti['tanı'] == 'Astım']['acılservıseyatıssayısı']
group2_6 = veriSeti[veriSeti['tanı'] == 'KOAH']['acılservıseyatıssayısı']
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# Mann-Whitney U testi
stat, p_value = mannwhitneyu(group1_6, group2_6, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")


# İlgili iki grup seçimi
group1_7 = veriSeti[veriSeti['tanı'] == 'Astım']['acılservısetoplamyatıssuresısaat']
group2_7 = veriSeti[veriSeti['tanı'] == 'KOAH']['acılservısetoplamyatıssuresısaat']
print("Veri setinde bulunan tanılar:", veriSeti['tanı'].unique())

# Mann-Whitney U testi
stat, p_value = mannwhitneyu(group1_7, group2_7, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_8 = veriSeti[veriSeti['tanı'] == 'Astım']['yogumbakımayatıssayısı']
group2_8 = veriSeti[veriSeti['tanı'] == 'KOAH']['yogumbakımayatıssayısı']

stat, p_value = mannwhitneyu(group1_8, group2_8, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_9 = veriSeti[veriSeti['tanı'] == 'Astım']['yogumbakımatoplamyatıssuresısaat']
group2_9 = veriSeti[veriSeti['tanı'] == 'KOAH']['yogumbakımatoplamyatıssuresısaat']

stat, p_value = mannwhitneyu(group1_9, group2_9, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_10 = veriSeti[veriSeti['tanı'] == 'Astım']['servıseyatıssayısı']
group2_10 = veriSeti[veriSeti['tanı'] == 'KOAH']['servıseyatıssayısı']

stat, p_value = mannwhitneyu(group1_10, group2_10, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_11 = veriSeti[veriSeti['tanı'] == 'Astım']['servıseoplamyatıssuresısaat']
group2_11 = veriSeti[veriSeti['tanı'] == 'KOAH']['servıseoplamyatıssuresısaat']

stat, p_value = mannwhitneyu(group1_11, group2_11, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_12 = veriSeti[veriSeti['tanı'] == 'Astım']['FEV1 %']
group2_12 = veriSeti[veriSeti['tanı'] == 'KOAH']['FEV1 %']

stat, p_value = mannwhitneyu(group1_12, group2_12, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_13 = veriSeti[veriSeti['tanı'] == 'Astım']['PEF %']
group2_13 = veriSeti[veriSeti['tanı'] == 'KOAH']['PEF %']

stat, p_value = mannwhitneyu(group1_13, group2_13, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_14 = veriSeti[veriSeti['tanı'] == 'Astım']['FEV1/FVC Değeri']
group2_14 = veriSeti[veriSeti['tanı'] == 'KOAH']['FEV1/FVC Değeri']

stat, p_value = mannwhitneyu(group1_14, group2_14, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_15 = veriSeti[veriSeti['tanı'] == 'Astım']['kitleendeksi']
group2_15 = veriSeti[veriSeti['tanı'] == 'KOAH']['kitleendeksi']

stat, p_value = mannwhitneyu(group1_15, group2_15, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

# İlgili iki grup seçimi
group1_16 = veriSeti[veriSeti['tanı'] == 'Astım']['PCA_KanBasıncı']
group2_16 = veriSeti[veriSeti['tanı'] == 'KOAH']['PCA_KanBasıncı']

stat, p_value = mannwhitneyu(group1_16, group2_16, alternative='two-sided')
print(f"Mann-Whitney U Testi sonucu: U-istatistiği = {stat}, p-değeri = {p_value}")

#MANN-WHITNEY 'E GÖRE VERİ ÇIKARMA
veriSeti = veriSeti.drop(columns=['nezamanbırakmısGÜN', 'tanısuresıay', 'acılservısetoplamyatıssuresısaat', 'yogumbakımayatıssayısı','yogumbakımatoplamyatıssuresısaat'])
veriSeti.head() 

#ki-kare testi
   #kategorik sütunların kendi içinde karşılaştırılması
        
        import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Kategorik sütunları seçme
kategorik_sutunlar = veriSeti.select_dtypes(include=['category', 'object']).columns

# Kategorik değişkenlerin birbirleriyle ilişkisini analiz etme
for i, sutun1 in enumerate(kategorik_sutunlar):
    for sutun2 in kategorik_sutunlar[i + 1:]:  # Aynı sütunun kendisiyle karşılaştırılmasını engeller
        # Çapraz tablo oluşturma
        tablo = pd.crosstab(veriSeti[sutun1], veriSeti[sutun2])

        # Ki-Kare testi uygulama
        try:
            chi2, p, dof, expected = chi2_contingency(tablo)
        except ValueError as e:
            print(f"Hata: {sutun1} ve {sutun2} arasında test uygulanamadı - {e}")
            continue

        # Ki-Kare Değeri ve P-Değeri yazdırma
        print(f"İlişki: {sutun1} ve {sutun2}")
        print(f"Ki-Kare Değeri: {chi2:.2f}, P-Değeri: {p:.4f}\n")

#tanı ve kategorik sütunlar arası korelasyon
hedef_degisken = 'tanı'  # Hedef değişken (örneğin tanı)
print(veriSeti.columns)
for sutun in kategorik_sutunlar:
    if sutun != hedef_degisken:
        tablo = pd.crosstab(veriSeti[sutun], veriSeti[hedef_degisken])  # Çapraz tablo
        chi2, p, dof, expected = chi2_contingency(tablo)  # Ki-Kare testi
        print(f"İlişki: {sutun} ve {hedef_degisken}")
        print(f"Ki-Kare Değeri: {chi2}, P-Değeri: {p}\n")
        


#ki-kareye göre silme
veriSeti = veriSeti.drop(columns=['egıtımduzeyı','varsakımde ANNE','varsakımde BABA','varsakımde KARDES','varsakımde DİĞER'])
veriSeti.head() 

# Sayısal sütunları seçme, ancak 'hastano' sütununu hariç tutma
sayisal_sutunlar = veriSeti.select_dtypes(include=['number']).columns
# 'hastano' sütununu sayısal sütunlar listesinden çıkarma
sayisal_sutunlar = sayisal_sutunlar[sayisal_sutunlar != 'hastaNo']
# Sayısal sütunları veri setinden çekme
sayisal_veri = veriSeti[sayisal_sutunlar]
# Sonuçları kontrol etme
print(sayisal_veri)

#robust standartlaştırma
from sklearn.preprocessing import RobustScaler

# RobustScaler oluştur
scaler = RobustScaler()
# Ölçeklendirme işlemi
veriSeti[sayisal_sutunlar] = scaler.fit_transform(veriSeti[sayisal_sutunlar])

# Sonuçları kontrol etme
print(veriSeti.head())


# Kategorik sütunları seçme
kategorik_sutunlar = veriSeti.select_dtypes(include=['category', 'object']).columns

# Kategorik sütunların değerlerini inceleme
kategorik_veriler = veriSeti[kategorik_sutunlar]
print("\nKategorik Veriler:")
print(kategorik_veriler)

# One-Hot Encoding işlemi
df_encoded = pd.get_dummies(kategorik_veriler, columns=kategorik_sutunlar)

print("\nOne-Hot Encoded VeriFrame:")
print(df_encoded)
# True/False değerlerini 1/0 olarak değiştirme
df_encoded = df_encoded.astype(int)

# Sonuçları kontrol etme
print(df_encoded)
# Sayısal veriler ve One-Hot Encoded veriyi birleştirme
veriSeti = pd.concat([veriSeti[sayisal_sutunlar], df_encoded], axis=1)

# Sonuçları kontrol etme
print("\nBirleştirilmiş VeriFrame:")
print(veriSeti.head())


# Çapraz tablo oluşturma
crosstab = pd.crosstab(veriSeti['tanı_Astım'], veriSeti['tanı_KOAH'])
print("\nÇapraz Tablo:")
print(crosstab)

#LOJİSİTK

# Bağımsız (özellik) değişkenler
X = veriSeti.drop(columns=['tanı_Astım', 'tanı_KOAH'])  # Her iki hedef değişkeni çıkarıyoruz.
import pandas as pd

# Eksik değer kontrolü
print(X_test_astim.isnull().sum())

# Bağımlı değişkenler (iki tane hedef)
y_astim = veriSeti['tanı_Astım']  # Astım için hedef değişken
y_koah = veriSeti['tanı_KOAH']   # KOAH için hedef değişken

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Eğitim ve test setlerine ayırma (Astım için)
X_train_astim, X_test_astim, y_train_astim, y_test_astim = train_test_split(X, y_astim, test_size=0.3, random_state=42)

# Eğitim ve test setlerine ayırma (KOAH için)
X_train_koah, X_test_koah, y_train_koah, y_test_koah = train_test_split(X, y_koah, test_size=0.3, random_state=42)

# Astım için Logistik Regresyon modeli
logreg_model_astim = LogisticRegression()
logreg_model_astim.fit(X_train_astim, y_train_astim)
y_pred_astim = logreg_model_astim.predict(X_test_astim)

# KOAH için Logistik Regresyon modeli
logreg_model_koah = LogisticRegression()
logreg_model_koah.fit(X_train_koah, y_train_koah)
y_pred_koah = logreg_model_koah.predict(X_test_koah)

# Sonuçları değerlendirme (Astım)
print("Astım Modeli - Doğruluk:", accuracy_score(y_test_astim, y_pred_astim))
print("Astım Modeli - Karışıklık Matrisi:")
print(confusion_matrix(y_test_astim, y_pred_astim))
print("Astım Modeli - Sınıflandırma Raporu:")
print(classification_report(y_test_astim, y_pred_astim))

# Sonuçları değerlendirme (KOAH)
print("KOAH Modeli - Doğruluk:", accuracy_score(y_test_koah, y_pred_koah))
print("KOAH Modeli - Karışıklık Matrisi:")
print(confusion_matrix(y_test_koah, y_pred_koah))
print("KOAH Modeli - Sınıflandırma Raporu:")
print(classification_report(y_test_koah, y_pred_koah))

#Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Astım için karışıklık matrisi
conf_matrix_astim = confusion_matrix(y_test_astim, y_pred_astim)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_astim, annot=True, fmt='d', cmap='Blues', xticklabels=['Astım Yok', 'Astım Var'], yticklabels=['Astım Yok', 'Astım Var'])
plt.title('Astım Modeli - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# KOAH için karışıklık matrisi
conf_matrix_koah = confusion_matrix(y_test_koah, y_pred_koah)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_koah, annot=True, fmt='d', cmap='Greens', xticklabels=['KOAH Yok', 'KOAH Var'], yticklabels=['KOAH Yok', 'KOAH Var'])
plt.title('KOAH Modeli - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

#RANDOMFOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Bağımsız (özellik) değişkenler
X = veriSeti.drop(columns=['tanı_Astım', 'tanı_KOAH'])  # Her iki hedef değişkeni çıkarıyoruz.
# Bağımlı değişkenler (iki tane hedef)
y_astim = veriSeti['tanı_Astım']  # Astım için hedef değişken
y_koah = veriSeti['tanı_KOAH']   # KOAH için hedef değişken
# Eğitim ve test setlerine ayırma (Astım için)
X_train_astim, X_test_astim, y_train_astim, y_test_astim = train_test_split(X, y_astim, test_size=0.3, random_state=42)
# Eğitim ve test setlerine ayırma (KOAH için)
X_train_koah, X_test_koah, y_train_koah, y_test_koah = train_test_split(X, y_koah, test_size=0.3, random_state=42)
# Astım Modeli için Random Forest
rf_astim = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
# Modeli eğitme
rf_astim.fit(X_train_astim, y_train_astim)
# Test setinde tahmin yapma
y_pred_astim = rf_astim.predict(X_test_astim)

# Performans metrikleri
print("Astım Modeli - Random Forest")
print("Doğruluk:", accuracy_score(y_test_astim, y_pred_astim))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_astim, y_pred_astim))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_astim, y_pred_astim))

# KOAH Modeli için Random Forest
rf_koah = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
# Modeli eğitme
rf_koah.fit(X_train_koah, y_train_koah)
# Test setinde tahmin yapma
y_pred_koah = rf_koah.predict(X_test_koah)

# Performans metrikleri
print("\nKOAH Modeli - Random Forest")
print("Doğruluk:", accuracy_score(y_test_koah, y_pred_koah))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_koah, y_pred_koah))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_koah, y_pred_koah))

#Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Astım için karışıklık matrisi
from sklearn.metrics import roc_curve, roc_auc_score

# Astım için karışıklık matrisi
conf_matrix_astim = confusion_matrix(y_test_astim, y_pred_astim)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_astim, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Astım Yok', 'Astım Var'], 
            yticklabels=['Astım Yok', 'Astım Var'])
plt.title('Astım Modeli - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# KOAH için karışıklık matrisi
conf_matrix_koah = confusion_matrix(y_test_koah, y_pred_koah)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_koah, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['KOAH Yok', 'KOAH Var'], 
            yticklabels=['KOAH Yok', 'KOAH Var'])
plt.title('KOAH Modeli - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

import pandas as pd
from sklearn.metrics import classification_report

# Modelin tahminlerini ve gerçek değerleri kullanarak classification_report'u elde etme
report = classification_report(y_test_astim, y_pred_astim, output_dict=True)

# DataFrame oluşturma
classification_df = pd.DataFrame(report).transpose()

# Tablonun yazdırılması
print(classification_df)


# Modelin tahminlerini ve gerçek değerleri kullanarak classification_report'u elde etme
report2 = classification_report(y_test_koah, y_pred_koah, output_dict=True)

# DataFrame oluşturma
classification_df2 = pd.DataFrame(report2).transpose()

# Tablonun yazdırılması
print(classification_df2)


# Astım için ROC eğrisi ve AUC
y_pred_proba_astim = rf_astim.predict_proba(X_test_astim)[:, 1]
fpr_astim, tpr_astim, _ = roc_curve(y_test_astim, y_pred_proba_astim)
auc_astim = roc_auc_score(y_test_astim, y_pred_proba_astim)

plt.figure(figsize=(8, 6))
plt.plot(fpr_astim, tpr_astim, color='blue', label=f'ROC AUC = {auc_astim:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Astım Modeli - ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend(loc='lower right')
plt.show()

# KOAH için ROC eğrisi ve AUC
y_pred_proba_koah = rf_koah.predict_proba(X_test_koah)[:, 1]
fpr_koah, tpr_koah, _ = roc_curve(y_test_koah, y_pred_proba_koah)
auc_koah = roc_auc_score(y_test_koah, y_pred_proba_koah)

plt.figure(figsize=(8, 6))
plt.plot(fpr_koah, tpr_koah, color='blue', label=f'ROC AUC = {auc_koah:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('KOAH Modeli - ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend(loc='lower right')
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

# ROC eğrisini çizme
fpr, tpr, thresholds = roc_curve(y_test_astim, y_pred_astim)
auc = roc_auc_score(y_test_astim, y_pred_astim)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Rastgele model
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Astım Modeli - ROC Eğrisi')
plt.legend(loc='lower right')
plt.show()

# Özelliklerin önem düzeyini elde etme
importances = rf_astim.feature_importances_

# Özellikleri görselleştirme
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances)
plt.xlabel('Özellik Önem Düzeyi')
plt.title('Random Forest - Özelliklerin Önem Düzeyleri')
plt.show()

import numpy as np

# Hataları hesaplayalım
errors = y_test_astim - y_pred_astim

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_astim, errors, color='blue')
plt.hlines(0, xmin=min(y_pred_astim), xmax=max(y_pred_astim), colors='red', linestyles='dashed')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar (Residuals)')
plt.title('Model Hatalarının Görselleştirilmesi(Astım)')
plt.show()

# Hataları hesaplayalım
errors = y_test_koah - y_pred_koah

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_astim, errors, color='blue')
plt.hlines(0, xmin=min(y_pred_koah), xmax=max(y_pred_koah), colors='red', linestyles='dashed')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar (Residuals)')
plt.title('Model Hatalarının Görselleştirilmesi(KOAH)')
plt.show()

#overfitting kontrolü(KOAH)
train_accuracy = rf_koah.score(X_train_koah, y_train_koah)
test_accuracy = rf_koah.score(X_test_koah, y_test_koah)
print(f"Eğitim Doğruluğu: {train_accuracy}")
print(f"Test Doğruluğu: {test_accuracy}")
# Çapraz doğrulama ile modelin doğruluğunu hesapla
scores = cross_val_score(rf_koah, X, y_koah, cv=5)
print(f"Çapraz Doğrulama Sonuçları: {scores}")
print(f"Ortalama Çapraz Doğrulama Doğruluğu: {scores.mean()}")

#overfitting kontrolü(Astım)
train_accuracy = rf_astim.score(X_train_astim, y_train_astim)
test_accuracy = rf_astim.score(X_test_astim, y_test_astim)
print(f"Eğitim Doğruluğu: {train_accuracy}")
print(f"Test Doğruluğu: {test_accuracy}")
# Çapraz doğrulama ile modelin doğruluğunu hesapla
scores = cross_val_score(rf_astim, X, y_koah, cv=5)
print(f"Çapraz Doğrulama Sonuçları: {scores}")
print(f"Ortalama Çapraz Doğrulama Doğruluğu: {scores.mean()}")

#GradientBoosting Classifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# Bağımsız (özellik) değişkenler
X = veriSeti.drop(columns=['tanı_Astım', 'tanı_KOAH'])  # Hedef değişkenleri çıkarıyoruz.

# Bağımlı değişkenler (iki tane hedef)
y_astim = veriSeti['tanı_Astım']  # Astım için hedef değişken
y_koah = veriSeti['tanı_KOAH']   # KOAH için hedef değişken

# Eğitim ve test setlerine ayırma (Astım için)
X_train_astim, X_test_astim, y_train_astim, y_test_astim = train_test_split(X, y_astim, test_size=0.3, random_state=42)

# Eğitim ve test setlerine ayırma (KOAH için)
X_train_koah, X_test_koah, y_train_koah, y_test_koah = train_test_split(X, y_koah, test_size=0.3, random_state=42)

# Astım Modeli için Gradient Boosting
gb_astim = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)

# Modeli eğitme
gb_astim.fit(X_train_astim, y_train_astim)

# Test setinde tahmin yapma
y_pred_astim = gb_astim.predict(X_test_astim)

# Performans metrikleri
print("Astım Modeli - Gradient Boosting")
print("Doğruluk:", accuracy_score(y_test_astim, y_pred_astim))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_astim, y_pred_astim))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_astim, y_pred_astim))


# KOAH Modeli için Gradient Boosting
gb_koah = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)

# Modeli eğitme
gb_koah.fit(X_train_koah, y_train_koah)

# Test setinde tahmin yapma
y_pred_koah = gb_koah.predict(X_test_koah)

# Performans metrikleri
print("\nKOAH Modeli - Gradient Boosting")
print("Doğruluk:", accuracy_score(y_test_koah, y_pred_koah))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_koah, y_pred_koah))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_koah, y_pred_koah))

# ROC Eğrisi (Astım)
fpr, tpr, thresholds = roc_curve(y_test_astim, gb_astim.predict_proba(X_test_astim)[:, 1])
auc = roc_auc_score(y_test_astim, gb_astim.predict_proba(X_test_astim)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Astım Modeli - ROC Eğrisi')
plt.legend(loc='lower right')
plt.show()

# Özelliklerin önem düzeyi (Astım)
importances = gb_astim.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances)
plt.xlabel('Özellik Önem Düzeyi')
plt.title('Gradient Boosting - Özelliklerin Önem Düzeyleri (Astım)')
plt.show()

# Hataların görselleştirilmesi (Astım)
errors = y_test_astim - y_pred_astim

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_astim, errors, color='blue')
plt.hlines(0, xmin=min(y_pred_astim), xmax=max(y_pred_astim), colors='red', linestyles='dashed')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar (Residuals)')
plt.title('Model Hatalarının Görselleştirilmesi (Astım)')
plt.show()

# Overfitting kontrolü (KOAH)
train_accuracy = gb_koah.score(X_train_koah, y_train_koah)
test_accuracy = gb_koah.score(X_test_koah, y_test_koah)

print(f"Eğitim Doğruluğu: {train_accuracy}")
print(f"Test Doğruluğu: {test_accuracy}")

# Çapraz doğrulama (KOAH)
scores = cross_val_score(gb_koah, X, y_koah, cv=5)
print(f"Çapraz Doğrulama Sonuçları: {scores}")
print(f"Ortalama Çapraz Doğrulama Doğruluğu: {scores.mean()}")

#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

#en iyi k belirleme
from sklearn.model_selection import GridSearchCV
# Bağımsız (özellik) değişkenler
X = veriSeti.drop(columns=['tanı_Astım', 'tanı_KOAH'])  # Hedef değişkenleri çıkarıyoruz.
# Bağımlı değişkenler (iki tane hedef)
y_astim = veriSeti['tanı_Astım']  # Astım için hedef değişken
y_koah = veriSeti['tanı_KOAH']   # KOAH için hedef değişken
# Eğitim ve test setlerine ayırma (Astım için)
X_train_astim, X_test_astim, y_train_astim, y_test_astim = train_test_split(X, y_astim, test_size=0.3, random_state=42)
# Eğitim ve test setlerine ayırma (KOAH için)
X_train_koah, X_test_koah, y_train_koah, y_test_koah = train_test_split(X, y_koah, test_size=0.3, random_state=42
                                                                        
param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_astim, y_train_astim)
print("En iyi parametreler:", grid_search.best_params_)

# Astım Modeli için KNN
knn_astim = KNeighborsClassifier(n_neighbors=6)  
# Modeli eğitme
knn_astim.fit(X_train_astim, y_train_astim)
# Test setinde tahmin yapma
y_pred_astim = knn_astim.predict(X_test_astim)

# Performans metrikleri
print("Astım Modeli - KNN")
print("Doğruluk:", accuracy_score(y_test_astim, y_pred_astim))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_astim, y_pred_astim))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_astim, y_pred_astim))

param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_koah, y_train_koah)
print("En iyi parametreler:", grid_search.best_params_)


# KOAH Modeli için KNN
knn_koah = KNeighborsClassifier(n_neighbors=3)
# Modeli eğitme
knn_koah.fit(X_train_koah, y_train_koah)
# Test setinde tahmin yapma
y_pred_koah = knn_koah.predict(X_test_koah)

# Performans metrikleri
print("\nKOAH Modeli - KNN")
print("Doğruluk:", accuracy_score(y_test_koah, y_pred_koah))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_koah, y_pred_koah))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_koah, y_pred_koah))

# Çapraz doğrulama (Astım ve KOAH için ayrı ayrı)
scores_astim = cross_val_score(knn_astim, X, y_astim, cv=5)
scores_koah = cross_val_score(knn_koah, X, y_koah, cv=5)

print(f"Astım için Çapraz Doğrulama Sonuçları: {scores_astim}")
print(f"Astım için Ortalama Çapraz Doğrulama Doğruluğu: {scores_astim.mean()}")

print(f"KOAH için Çapraz Doğrulama Sonuçları: {scores_koah}")
print(f"KOAH için Ortalama Çapraz Doğrulama Doğruluğu: {scores_koah.mean()}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Astım için karışıklık matrisi
conf_matrix_astim = confusion_matrix(y_test_astim, y_pred_astim)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_astim, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Astım Yok', 'Astım Var'], 
            yticklabels=['Astım Yok', 'Astım Var'])
plt.title('Astım Modeli - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# KOAH için karışıklık matrisi
conf_matrix_koah = confusion_matrix(y_test_koah, y_pred_koah)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_koah, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['KOAH Yok', 'KOAH Var'], 
            yticklabels=['KOAH Yok', 'KOAH Var'])
plt.title('KOAH Modeli - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

import pandas as pd
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import classification_report

# Astım için classification_report
report2 = classification_report(y_test_astim, y_pred_astim, output_dict=True)

# Astım için DataFrame oluşturma
classification_df3 = pd.DataFrame(report2).transpose()

# Astım raporunun yazdırılması
print("Astım Modeli - KNN Sınıflandırma Raporu:")
print(classification_df3)

# KOAH için classification_report
report2 = classification_report(y_test_koah, y_pred_koah, output_dict=True)

# KOAH için DataFrame oluşturma
classification_df2 = pd.DataFrame(report2).transpose()

# KOAH raporunun yazdırılması
print("\nKOAH Modeli - KNN Sınıflandırma Raporu:")
print(classification_df2)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score

# Astım için ROC eğrisi ve AUC
y_pred_proba_astim = knn_astim.predict_proba(X_test_astim)[:, 1]  # KNN modelinin probabilistik tahminleri
fpr_astim, tpr_astim, _ = roc_curve(y_test_astim, y_pred_proba_astim)
auc_astim = roc_auc_score(y_test_astim, y_pred_proba_astim)

plt.figure(figsize=(8, 6))
plt.plot(fpr_astim, tpr_astim, color='blue', label=f'ROC AUC = {auc_astim:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Astım Modeli - ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend(loc='lower right')
plt.show()

# KOAH için ROC eğrisi ve AUC
y_pred_proba_koah = knn_koah.predict_proba(X_test_koah)[:, 1]  # KNN modelinin probabilistik tahminleri
fpr_koah, tpr_koah, _ = roc_curve(y_test_koah, y_pred_proba_koah)
auc_koah = roc_auc_score(y_test_koah, y_pred_proba_koah)

plt.figure(figsize=(8, 6))
plt.plot(fpr_koah, tpr_koah, color='blue', label=f'ROC AUC = {auc_koah:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('KOAH Modeli - ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend(loc='lower right')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Astım için hataları hesapla
errors_astim = y_test_astim - y_pred_astim

# Astım modelinin hatalarının görselleştirilmesi
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_astim, errors_astim, color='blue')
plt.hlines(0, xmin=min(y_pred_astim), xmax=max(y_pred_astim), colors='red', linestyles='dashed')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar (Residuals)')
plt.title('Astım Modeli Hatalarının Görselleştirilmesi')
plt.show()

# KOAH için hataları hesapla
errors_koah = y_test_koah - y_pred_koah

# KOAH modelinin hatalarının görselleştirilmesi
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_koah, errors_koah, color='blue')
plt.hlines(0, xmin=min(y_pred_koah), xmax=max(y_pred_koah), colors='red', linestyles='dashed')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar (Residuals)')
plt.title('KOAH Modeli Hatalarının Görselleştirilmesi')
plt.show()





# Overfitting kontrolü (Astım)
train_accuracy_astim = knn_astim.score(X_train_astim, y_train_astim)
test_accuracy_astim = knn_astim.score(X_test_astim, y_test_astim)
print(f"Astım Modeli - Eğitim Doğruluğu: {train_accuracy_astim}")
print(f"Astım Modeli - Test Doğruluğu: {test_accuracy_astim}")

# Çapraz doğrulama ile modelin doğruluğunu hesapla (Astım)
scores_astim = cross_val_score(knn_astim, X, y_astim, cv=5)
print(f"Astım Modeli - Çapraz Doğrulama Sonuçları: {scores_astim}")
print(f"Astım Modeli - Ortalama Çapraz Doğrulama Doğruluğu: {scores_astim.mean()}")

# Overfitting kontrolü (KOAH)
train_accuracy_koah = knn_koah.score(X_train_koah, y_train_koah)
test_accuracy_koah = knn_koah.score(X_test_koah, y_test_koah)
print(f"KOAH Modeli - Eğitim Doğruluğu: {train_accuracy_koah}")
print(f"KOAH Modeli - Test Doğruluğu: {test_accuracy_koah}")

# Çapraz doğrulama ile modelin doğruluğunu hesapla (KOAH)
scores_koah = cross_val_score(knn_koah, X, y_koah, cv=5)
print(f"KOAH Modeli - Çapraz Doğrulama Sonuçları: {scores_koah}")
print(f"KOAH Modeli - Ortalama Çapraz Doğrulama Doğruluğu: {scores_koah.mean()}")



#XGBOOST
# Gerekli kütüphaneler
pip install xgboost

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Bağımsız (özellik) değişkenler
X = veriSeti.drop(columns=['tanı_Astım', 'tanı_KOAH'])  # Hedef değişkenleri çıkarıyoruz.

# Bağımlı değişkenler (iki tane hedef)
y_astim = veriSeti['tanı_Astım']  # Astım için hedef değişken
y_koah = veriSeti['tanı_KOAH']   # KOAH için hedef değişken

# Eğitim ve test setlerine ayırma (Astım için)
X_train_astim, X_test_astim, y_train_astim, y_test_astim = train_test_split(X, y_astim, test_size=0.3, random_state=42)

# Eğitim ve test setlerine ayırma (KOAH için)
X_train_koah, X_test_koah, y_train_koah, y_test_koah = train_test_split(X, y_koah, test_size=0.3, random_state=42)

# Astım Modeli için XGBoost
xgb_astim = XGBClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)

# Modeli eğitme
xgb_astim.fit(X_train_astim, y_train_astim)

# Test setinde tahmin yapma
y_pred_astim = xgb_astim.predict(X_test_astim)

# Performans metrikleri
print("Astım Modeli - XGBoost")
print("Doğruluk:", accuracy_score(y_test_astim, y_pred_astim))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_astim, y_pred_astim))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_astim, y_pred_astim))

# KOAH Modeli için XGBoost
xgb_koah = XGBClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)

# Modeli eğitme
xgb_koah.fit(X_train_koah, y_train_koah)

# Test setinde tahmin yapma
y_pred_koah = xgb_koah.predict(X_test_koah)

# Performans metrikleri
print("\nKOAH Modeli - XGBoost")
print("Doğruluk:", accuracy_score(y_test_koah, y_pred_koah))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_koah, y_pred_koah))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_koah, y_pred_koah))

#GÖRSELLEŞTİRME
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

# XGBoost Modeli
xgb_astim = XGBClassifier(random_state=42)
xgb_astim.fit(X_train_astim, y_train_astim)

# Özellik Önemini Görselleştirme
plt.figure(figsize=(10, 8))
plot_importance(xgb_astim, importance_type='weight', max_num_features=10)  # İlk 10 özelliği göster
plt.title("Özellik Önem Grafiği (Astım)")
plt.show()

# XGBoost Modeli (KOAH)
xgb_koah = XGBClassifier(random_state=42)
xgb_koah.fit(X_train_koah, y_train_koah)

# Özellik Önemini Görselleştirme
plt.figure(figsize=(10, 8))
plot_importance(xgb_koah, importance_type='weight', max_num_features=10)  # İlk 10 özelliği göster
plt.title("Özellik Önem Grafiği (KOAH)")
plt.show()


from sklearn.metrics import roc_curve, auc

# Test tahmin olasılıkları
y_pred_proba_astim = xgb_astim.predict_proba(X_test_astim)[:, 1]

# ROC eğrisi için metrikler
fpr, tpr, _ = roc_curve(y_test_astim, y_pred_proba_astim)
roc_auc = auc(fpr, tpr)

# ROC Eğrisini Çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.title('ROC Eğrisi (Astım)')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Test tahmin olasılıkları
y_pred_proba_koah = xgb_koah.predict_proba(X_test_koah)[:, 1]

# ROC eğrisi için metrikler
fpr, tpr, _ = roc_curve(y_test_koah, y_pred_proba_koah)
roc_auc = auc(fpr, tpr)

# ROC Eğrisini Çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.title('ROC Eğrisi (KOAH)')
plt.legend(loc="lower right")
plt.grid()
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Tahminler
y_pred_koah = xgb_koah.predict(X_test_koah)

# Karışıklık Matrisi
cm = confusion_matrix(y_test_koah, y_pred_koah)

# Görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi (KOAH)')
plt.show()

# Tahminler
y_pred_astim = xgb_astim.predict(X_test_astim)

# Karışıklık Matrisi
cm = confusion_matrix(y_test_astim, y_pred_astim)

# Görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi (Astım)')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Astım modeli için Karar Ağacı
dt_astim = DecisionTreeClassifier(random_state=42, max_depth=5)  # max_depth ile derinlik kontrolü
dt_astim.fit(X_train_astim, y_train_astim)

# Astım için tahminler
y_pred_astim = dt_astim.predict(X_test_astim)

# Performans metrikleri (Astım)
print("Astım Modeli - Karar Ağacı")
print("Doğruluk:", accuracy_score(y_test_astim, y_pred_astim))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_astim, y_pred_astim))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_astim, y_pred_astim))

# KOAH modeli için Karar Ağacı
dt_koah = DecisionTreeClassifier(random_state=42, max_depth=5)  # max_depth ile derinlik kontrolü
dt_koah.fit(X_train_koah, y_train_koah)

# KOAH için tahminler
y_pred_koah = dt_koah.predict(X_test_koah)

# Performans metrikleri (KOAH)
print("\nKOAH Modeli - Karar Ağacı")
print("Doğruluk:", accuracy_score(y_test_koah, y_pred_koah))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_koah, y_pred_koah))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_koah, y_pred_koah))

# Karar ağacını görselleştirme (Astım)
plt.figure(figsize=(50, 40))
tree.plot_tree(dt_astim, feature_names=X_train_astim.columns, class_names=['0', '1'], filled=True, fontsize=12)  # Font boyutunu artırdık
plt.title("Karar Ağacı (Astım)", fontsize=14)
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Modeli oluştururken max_depth parametresini belirleyelim
dt_astim = DecisionTreeClassifier(max_depth=3, random_state=42)

# Modeli eğitelim
dt_astim.fit(X_train_astim, y_train_astim)

# Karar ağacını görselleştirme
plt.figure(figsize=(20, 10))
tree.plot_tree(dt_astim, feature_names=X_train_astim.columns, class_names=['0', '1'], filled=True, fontsize=12)
plt.title("Karar Ağacı (Astım)")
plt.show()


# Karar ağacını görselleştirme (KOAH)
plt.figure(figsize=(50, 40))
tree.plot_tree(dt_koah, feature_names=X_train_koah.columns, class_names=['0', '1'], filled=True)
plt.title("Karar Ağacı (KOAH)")
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# KOAH Modeli için karar ağacını oluşturma ve max_depth parametresini belirleme
dt_koah = DecisionTreeClassifier(max_depth=3, random_state=42)

# Modeli eğitme
dt_koah.fit(X_train_koah, y_train_koah)

# Karar ağacını görselleştirme
plt.figure(figsize=(20, 10))  # Grafik boyutunu ayarlama
tree.plot_tree(dt_koah, feature_names=X_train_koah.columns, class_names=['0', '1'], filled=True, fontsize=12)
plt.title("Karar Ağacı (KOAH)", fontsize=14)
plt.show()

#SVM

# Gerekli kütüphaneleri import et
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Bağımsız değişkenler (özellikler) ve bağımlı değişkeni ayırma
X = veriSeti.drop(columns=['tanı_Astım', 'tanı_KOAH'])  # Hedef değişkenleri çıkarıyoruz.
y_astim = veriSeti['tanı_Astım']  # Astım için hedef değişken
y_koah = veriSeti['tanı_KOAH']   # KOAH için hedef değişken
# Eğitim ve test setlerine ayırma (Astım için)
X_train_astim, X_test_astim, y_train_astim, y_test_astim = train_test_split(X, y_astim, test_size=0.3, random_state=42)
# Eğitim ve test setlerine ayırma (KOAH için)
X_train_koah, X_test_koah, y_train_koah, y_test_koah = train_test_split(X, y_koah, test_size=0.3, random_state=42)

# Destek Vektör Makineleri (SVM) modelini oluştur (RBF kernel)
svm_astim = SVC(kernel='rbf', random_state=42, C=1, gamma='scale')  # C: Ceza parametresi, gamma: kernel parametresi
# Modeli eğit
svm_astim.fit(X_train_astim, y_train_astim)
# Test setinde tahmin yapma
y_pred_astim = svm_astim.predict(X_test_astim)



import pandas as pd
from sklearn.metrics import classification_report

# Modelin tahminlerini ve gerçek değerleri kullanarak classification_report'u elde etme
report = classification_report(y_test_astim, y_pred_astim, output_dict=True)

# DataFrame oluşturma
classification_df = pd.DataFrame(report).transpose()

# Tablonun yazdırılması
print(classification_df)


# Modelin tahminlerini ve gerçek değerleri kullanarak classification_report'u elde etme
report2 = classification_report(y_test_koah, y_pred_koah, output_dict=True)

# DataFrame oluşturma
classification_df2 = pd.DataFrame(report2).transpose()

# Tablonun yazdırılması
print(classification_df2)


# Performans metrikleri
print("Astım Modeli - SVM (RBF Kernel)")
print("Doğruluk:", accuracy_score(y_test_astim, y_pred_astim))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_astim, y_pred_astim))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_astim, y_pred_astim))

# KOAH Modeli için
svm_koah = SVC(kernel='rbf', random_state=42, C=1, gamma='scale')
# Modeli eğit
svm_koah.fit(X_train_koah, y_train_koah)
# Test setinde tahmin yapma
y_pred_koah = svm_koah.predict(X_test_koah)

# Performans metrikleri
print("KOAH Modeli - SVM (RBF Kernel)")
print("Doğruluk:", accuracy_score(y_test_koah, y_pred_koah))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_koah, y_pred_koah))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_koah, y_pred_koah))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Astım için karışıklık matrisi
cm_astim = confusion_matrix(y_test_astim, y_pred_astim)

# Karışıklık matrisini görselleştirme (Astım)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_astim, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.title("Astım Modeli - Karışıklık Matrisi (SVM)")
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.show()

# KOAH için karışıklık matrisi
cm_koah = confusion_matrix(y_test_koah, y_pred_koah)

# Karışıklık matrisini görselleştirme (KOAH)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_koah, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.title("KOAH Modeli - Karışıklık Matrisi (SVM)")
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.show()

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Astım modeli için SVM
svm_astim = SVC(probability=True, random_state=42)
svm_astim.fit(X_train_astim, y_train_astim)

# Astım için ROC eğrisi ve AUC
y_pred_proba_astim = svm_astim.predict_proba(X_test_astim)[:, 1]  # Pozitif sınıf olasılıkları
fpr_astim, tpr_astim, _ = roc_curve(y_test_astim, y_pred_proba_astim)
auc_astim = roc_auc_score(y_test_astim, y_pred_proba_astim)

plt.figure(figsize=(8, 6))
plt.plot(fpr_astim, tpr_astim, color='blue', label=f'ROC AUC = {auc_astim:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Astım Modeli - ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend(loc='lower right')
plt.show()

# KOAH modeli için SVM
svm_koah = SVC(probability=True, random_state=42)
svm_koah.fit(X_train_koah, y_train_koah)

# KOAH için ROC eğrisi ve AUC
y_pred_proba_koah = svm_koah.predict_proba(X_test_koah)[:, 1]  # Pozitif sınıf olasılıkları
fpr_koah, tpr_koah, _ = roc_curve(y_test_koah, y_pred_proba_koah)
auc_koah = roc_auc_score(y_test_koah, y_pred_proba_koah)

plt.figure(figsize=(8, 6))
plt.plot(fpr_koah, tpr_koah, color='blue', label=f'ROC AUC = {auc_koah:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('KOAH Modeli - ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend(loc='lower right')
plt.show()

# Overfitting kontrolü (Astım)
train_accuracy_astim = svm_astim.score(X_train_astim, y_train_astim)
test_accuracy_astim = svm_astim.score(X_test_astim, y_test_astim)
print(f"Astım Modeli - Eğitim Doğruluğu: {train_accuracy_astim}")
print(f"Astım Modeli - Test Doğruluğu: {test_accuracy_astim}")

# Çapraz doğrulama ile modelin doğruluğunu hesapla (Astım)
from sklearn.model_selection import cross_val_score
scores_astim = cross_val_score(svm_astim, X, y_astim, cv=5)
print(f"Astım - Çapraz Doğrulama Sonuçları: {scores_astim}")
print(f"Astım - Ortalama Çapraz Doğrulama Doğruluğu: {scores_astim.mean()}")

# Overfitting kontrolü (KOAH)
train_accuracy_koah = svm_koah.score(X_train_koah, y_train_koah)
test_accuracy_koah = svm_koah.score(X_test_koah, y_test_koah)
print(f"KOAH Modeli - Eğitim Doğruluğu: {train_accuracy_koah}")
print(f"KOAH Modeli - Test Doğruluğu: {test_accuracy_koah}")

# Çapraz doğrulama ile modelin doğruluğunu hesapla (KOAH)
scores_koah = cross_val_score(svm_koah, X, y_koah, cv=5)
print(f"KOAH - Çapraz Doğrulama Sonuçları: {scores_koah}")
print(f"KOAH - Ortalama Çapraz Doğrulama Doğruluğu: {scores_koah.mean()}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# SVM modelleri - Astım ve KOAH için tahminler yapılır
svm_astim = SVC(probability=True)  # SVM modelini oluştur
svm_astim.fit(X_train_astim, y_train_astim)  # Modeli eğit

y_pred_astim = svm_astim.predict(X_test_astim)  # Test verisi ile tahmin yap

# Hataları hesaplayalım (Astım)
errors_astim = y_test_astim - y_pred_astim

# Astım modelinin residuals görselleştirilmesi
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_astim, errors_astim, color='blue')
plt.hlines(0, xmin=min(y_pred_astim), xmax=max(y_pred_astim), colors='red', linestyles='dashed')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar (Residuals)')
plt.title('SVM Astım Modeli - Model Hatalarının Görselleştirilmesi')
plt.show()

# SVM modeli - KOAH için tahminler yapılır
svm_koah = SVC(probability=True)  # SVM modelini oluştur
svm_koah.fit(X_train_koah, y_train_koah)  # Modeli eğit

y_pred_koah = svm_koah.predict(X_test_koah)  # Test verisi ile tahmin yap

# Hataları hesaplayalım (KOAH)
errors_koah = y_test_koah - y_pred_koah

# KOAH modelinin residuals görselleştirilmesi
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_koah, errors_koah, color='blue')
plt.hlines(0, xmin=min(y_pred_koah), xmax=max(y_pred_koah), colors='red', linestyles='dashed')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar (Residuals)')
plt.title('SVM KOAH Modeli - Model Hatalarının Görselleştirilmesi')
plt.show()


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Astım modeli için SVM
svm_astim = SVC(probability=True, random_state=42)
svm_astim.fit(X_train_astim, y_train_astim)

# Overfitting kontrolü (Astım)
train_accuracy_astim = svm_astim.score(X_train_astim, y_train_astim)
test_accuracy_astim = svm_astim.score(X_test_astim, y_test_astim)
print(f"Astım Modeli - Eğitim Doğruluğu: {train_accuracy_astim}")
print(f"Astım Modeli - Test Doğruluğu: {test_accuracy_astim}")

# Çapraz doğrulama ile modelin doğruluğunu hesapla (Astım)
scores_astim = cross_val_score(svm_astim, X_train_astim, y_train_astim, cv=5)
print(f"Astım Modeli - Çapraz Doğrulama Sonuçları: {scores_astim}")
print(f"Astım Modeli - Ortalama Çapraz Doğrulama Doğruluğu: {scores_astim.mean()}")

# KOAH modeli için SVM
svm_koah = SVC(probability=True, random_state=42)
svm_koah.fit(X_train_koah, y_train_koah)

# Overfitting kontrolü (KOAH)
train_accuracy_koah = svm_koah.score(X_train_koah, y_train_koah)
test_accuracy_koah = svm_koah.score(X_test_koah, y_test_koah)
print(f"KOAH Modeli - Eğitim Doğruluğu: {train_accuracy_koah}")
print(f"KOAH Modeli - Test Doğruluğu: {test_accuracy_koah}")

# Çapraz doğrulama ile modelin doğruluğunu hesapla (KOAH)
scores_koah = cross_val_score(svm_koah, X_train_koah, y_train_koah, cv=5)
print(f"KOAH Modeli - Çapraz Doğrulama Sonuçları: {scores_koah}")
print(f"KOAH Modeli - Ortalama Çapraz Doğrulama Doğruluğu: {scores_koah.mean()}")

#yapay sinir ağları
pip install tensorflow

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Bağımsız değişkenler (özellikler) ve bağımlı değişkeni ayırma
X = veriSeti.drop(columns=['tanı_Astım', 'tanı_KOAH'])  # Hedef değişkenleri çıkarıyoruz.
y_astim = veriSeti['tanı_Astım']  # Astım için hedef değişken
y_koah = veriSeti['tanı_KOAH']   # KOAH için hedef değişken

# Eğitim ve test setlerine ayırma (Astım için)
X_train_astim, X_test_astim, y_train_astim, y_test_astim = train_test_split(X, y_astim, test_size=0.3, random_state=42)

# Eğitim ve test setlerine ayırma (KOAH için)
X_train_koah, X_test_koah, y_train_koah, y_test_koah = train_test_split(X, y_koah, test_size=0.3, random_state=42)

# Yapay Sinir Ağı Modelini Kurma (Astım)
model_astim = Sequential()

# İlk katman (gizli katman)
model_astim.add(Dense(units=64, activation='relu', input_dim=X_train_astim.shape[1]))

# İkinci katman (gizli katman)
model_astim.add(Dense(units=32, activation='relu'))

# Çıktı katmanı
model_astim.add(Dense(units=1, activation='sigmoid'))  # Sigmoid aktivasyon fonksiyonu çünkü ikili sınıflandırma

# Modeli derleme
model_astim.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history_astim = model_astim.fit(X_train_astim, y_train_astim, epochs=50, batch_size=32, validation_data=(X_test_astim, y_test_astim))

# Test setinde tahmin yapma
y_pred_astim = (model_astim.predict(X_test_astim) > 0.5).astype("int32")  # Sigmoid çıktıyı 0 veya 1'e dönüştür

# Performans metrikleri (Astım)
print("Astım Modeli - Yapay Sinir Ağı")
print("Doğruluk:", accuracy_score(y_test_astim, y_pred_astim))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_astim, y_pred_astim))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_astim, y_pred_astim))


pip install pydot
pip install graphviz
from tensorflow.keras.utils import plot_model

# Modelin yapısını görselleştir
plot_model(model_astim, to_file='model_astim.png', show_shapes=True, show_layer_names=True)

import matplotlib.pyplot as plt

# Eğitim süreci doğruluğu
plt.plot(history_astim.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_astim.history['val_accuracy'], label='Test Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Eğitim süreci kaybı
plt.plot(history_astim.history['loss'], label='Eğitim Kaybı')
plt.plot(history_astim.history['val_loss'], label='Test Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()


# KOAH Modeli için aynı işlemi tekrarlayabilirsiniz
model_koah = Sequential()
model_koah.add(Dense(units=64, activation='relu', input_dim=X_train_koah.shape[1]))
model_koah.add(Dense(units=32, activation='relu'))
model_koah.add(Dense(units=1, activation='sigmoid'))

model_koah.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

history_koah = model_koah.fit(X_train_koah, y_train_koah, epochs=50, batch_size=32, validation_data=(X_test_koah, y_test_koah))

y_pred_koah = (model_koah.predict(X_test_koah) > 0.5).astype("int32")

print("KOAH Modeli - Yapay Sinir Ağı")
print("Doğruluk:", accuracy_score(y_test_koah, y_pred_koah))
print("\nKarışıklık Matrisi:")
print(confusion_matrix(y_test_koah, y_pred_koah))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test_koah, y_pred_koah))

# Karışıklık matrisi görselleştirme (Astım)
cm_astim = confusion_matrix(y_test_astim, y_pred_astim)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_astim, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.title("Astım Modeli - Karışıklık Matrisi (Yapay Sinir Ağı)")
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.show()

# Karışıklık matrisi görselleştirme (KOAH)
cm_koah = confusion_matrix(y_test_koah, y_pred_koah)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_koah, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.title("KOAH Modeli - Karışıklık Matrisi (Yapay Sinir Ağı)")
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.show()

# Eğitim sürecini görselleştirme (Astım)
plt.plot(history_astim.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_astim.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title("Astım Modeli - Eğitim ve Doğrulama Doğruluğu")
plt.xlabel("Epok")
plt.ylabel("Doğruluk")
plt.legend()
plt.show()

# Eğitim sürecini görselleştirme (KOAH)
plt.plot(history_koah.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_koah.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title("KOAH Modeli - Eğitim ve Doğrulama Doğruluğu")
plt.xlabel("Epok")
plt.ylabel("Doğruluk")
plt.legend()
plt.show()

#EN İYİ MODEL SEÇİMİ
from sklearn.metrics import accuracy_score

# Her modelin doğruluk sonuçları
models = {
    'Lojistik Regresyon': accuracy_score(y_test_astim, y_pred_astim),
    'Random Forest': accuracy_score(y_test_astim, y_pred_astim),
    'Gradient Boosting': accuracy_score(y_test_astim, y_pred_astim),
    'KNN': accuracy_score(y_test_astim, y_pred_astim),
    'XGBoost': accuracy_score(y_test_astim, y_pred_astim),
    'Karar Ağacı': accuracy_score(y_test_astim, y_pred_astim),
    'SVM (RBF Kernel)': accuracy_score(y_test_astim, y_pred_astim),
    'Yapay Sinir Ağı': accuracy_score(y_test_astim, y_pred_astim)
}

# Astım modelleri için en iyi modeli seçme
best_model_astim = max(models, key=models.get)
best_accuracy_astim = models[best_model_astim]

# KOAH modelleri için en iyi modeli seçme
models_koah = {
    'Lojistik Regresyon': accuracy_score(y_test_koah, y_pred_koah),
    'Random Forest': accuracy_score(y_test_koah, y_pred_koah),
    'Gradient Boosting': accuracy_score(y_test_koah, y_pred_koah),
    'KNN': accuracy_score(y_test_koah, y_pred_koah),
    'XGBoost': accuracy_score(y_test_koah, y_pred_koah),
    'Karar Ağacı': accuracy_score(y_test_koah, y_pred_koah),
    'SVM (RBF Kernel)': accuracy_score(y_test_koah, y_pred_koah),
    'Yapay Sinir Ağı': accuracy_score(y_test_koah, y_pred_koah)
}

best_model_koah = max(models_koah, key=models_koah.get)
best_accuracy_koah = models_koah[best_model_koah]

# Sonuçları yazdırma
print("Astım için en iyi model: ", best_model_astim)
print("Doğruluk: ", best_accuracy_astim)

print("KOAH için en iyi model: ", best_model_koah)
print("Doğruluk: ", best_accuracy_koah)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Astım ve KOAH için en iyi modeli seçmek için kullanılacak metrikler
models = {
    'Lojistik Regresyon': (accuracy_score(y_test_astim, y_pred_astim), classification_report(y_test_astim, y_pred_astim, output_dict=True), confusion_matrix(y_test_astim, y_pred_astim), roc_auc_score(y_test_astim, y_pred_astim)),
    'Random Forest': (accuracy_score(y_test_astim, y_pred_astim), classification_report(y_test_astim, y_pred_astim, output_dict=True), confusion_matrix(y_test_astim, y_pred_astim), roc_auc_score(y_test_astim, y_pred_astim)),
    'Gradient Boosting': (accuracy_score(y_test_astim, y_pred_astim), classification_report(y_test_astim, y_pred_astim, output_dict=True), confusion_matrix(y_test_astim, y_pred_astim), roc_auc_score(y_test_astim, y_pred_astim)),
    'KNN': (accuracy_score(y_test_astim, y_pred_astim), classification_report(y_test_astim, y_pred_astim, output_dict=True), confusion_matrix(y_test_astim, y_pred_astim), roc_auc_score(y_test_astim, y_pred_astim)),
    'XGBoost': (accuracy_score(y_test_astim, y_pred_astim), classification_report(y_test_astim, y_pred_astim, output_dict=True), confusion_matrix(y_test_astim, y_pred_astim), roc_auc_score(y_test_astim, y_pred_astim)),
    'Karar Ağacı': (accuracy_score(y_test_astim, y_pred_astim), classification_report(y_test_astim, y_pred_astim, output_dict=True), confusion_matrix(y_test_astim, y_pred_astim), roc_auc_score(y_test_astim, y_pred_astim)),
    'SVM (RBF Kernel)': (accuracy_score(y_test_astim, y_pred_astim), classification_report(y_test_astim, y_pred_astim, output_dict=True), confusion_matrix(y_test_astim, y_pred_astim), roc_auc_score(y_test_astim, y_pred_astim)),
    'Yapay Sinir Ağı': (accuracy_score(y_test_astim, y_pred_astim), classification_report(y_test_astim, y_pred_astim, output_dict=True), confusion_matrix(y_test_astim, y_pred_astim), roc_auc_score(y_test_astim, y_pred_astim))
}

# En iyi modeli seçme
best_model = max(models, key=lambda model: models[model][0])  # Doğruluk skoruna göre en iyiyi seç
best_model_report = models[best_model]

# Sonuçları yazdırma
print(f"En iyi model: {best_model}")
print(f"Doğruluk: {best_model_report[0]}")
print(f"Karışıklık Matrisi:\n{best_model_report[2]}")
print(f"Sınıflandırma Raporu:\n{best_model_report[1]}")
print(f"ROC-AUC: {best_model_report[3]}")

# Performans metrikleri gradient için
accuracy_astim = accuracy_score(y_test_astim, y_pred_astim)
test_error_astim = 1 - accuracy_astim  # Test hatasını hesaplama
print("Test Hatası:", test_error_astim)

# Performans metrikleri
accuracy_koah = accuracy_score(y_test_koah, y_pred_koah)
test_error_koah = 1 - accuracy_koah  # Test hatasını hesaplama
print("Test Hatası:", test_error_koah)

# Performans metrikleri lojistik için
accuracy_astim = accuracy_score(y_test_astim, y_pred_astim)
test_error_astim = 1 - accuracy_astim  # Test hatasını hesaplama
print("Test Hatası:", test_error_astim)

accuracy_koah = accuracy_score(y_test_koah, y_pred_koah)
test_error_koah = 1 - accuracy_koah  # Test hatasını hesaplama
print("Test Hatası:", test_error_koah)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Logistik Regresyon için Cross-Validation
logreg_model_astim = LogisticRegression()
logreg_model_koah = LogisticRegression()

# Gradient Boosting için Cross-Validation
gb_model_astim = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)
gb_model_koah = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1)

# Cross-validation uygulanacak modelin performansını değerlendiriyoruz (5 katlamalı cross-validation)
cv_results_logreg_astim = cross_val_score(logreg_model_astim, X, y_astim, cv=5, scoring='accuracy')
cv_results_logreg_koah = cross_val_score(logreg_model_koah, X, y_koah, cv=5, scoring='accuracy')

cv_results_gb_astim = cross_val_score(gb_model_astim, X, y_astim, cv=5, scoring='accuracy')
cv_results_gb_koah = cross_val_score(gb_model_koah, X, y_koah, cv=5, scoring='accuracy')

# Sonuçları yazdırma
print("Logistik Regresyon - Astım: Cross-Validation Doğruluk Skorları: ", cv_results_logreg_astim)
print("Logistik Regresyon - KOAH: Cross-Validation Doğruluk Skorları: ", cv_results_logreg_koah)
print("Gradient Boosting - Astım: Cross-Validation Doğruluk Skorları: ", cv_results_gb_astim)
print("Gradient Boosting - KOAH: Cross-Validation Doğruluk Skorları: ", cv_results_gb_koah)

# Ortalama doğrulukları yazdırma
print("\nLogistik Regresyon - Astım: Ortalama Doğruluk: ", np.mean(cv_results_logreg_astim))
print("Logistik Regresyon - KOAH: Ortalama Doğruluk: ", np.mean(cv_results_logreg_koah))
print("Gradient Boosting - Astım: Ortalama Doğruluk: ", np.mean(cv_results_gb_astim))
print("Gradient Boosting - KOAH: Ortalama Doğruluk: ", np.mean(cv_results_gb_koah))

# Test hatasını hesaplamak için doğrulukları 1'den çıkarıyoruz
test_hatasi_logreg_astim = 1 - np.mean(cv_results_logreg_astim)
test_hatasi_logreg_koah = 1 - np.mean(cv_results_logreg_koah)
test_hatasi_gb_astim = 1 - np.mean(cv_results_gb_astim)
test_hatasi_gb_koah = 1 - np.mean(cv_results_gb_koah)

# Test hatalarını yazdırma
print("\nLogistik Regresyon - Astım: Test Hatası: ", test_hatasi_logreg_astim)
print("Logistik Regresyon - KOAH: Test Hatası: ", test_hatasi_logreg_koah)
print("Gradient Boosting - Astım: Test Hatası: ", test_hatasi_gb_astim)
print("Gradient Boosting - KOAH: Test Hatası: ", test_hatasi_gb_koah)

#karar ağacı için yorum
from sklearn.model_selection import cross_val_score
import numpy as np

# Karar Ağacı modelleri
dt_model_astim = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model_koah = DecisionTreeClassifier(random_state=42, max_depth=5)
# Astım için Cross-Validation
cv_results_dt_astim = cross_val_score(dt_model_astim, X, y_astim, cv=5, scoring='accuracy')
# KOAH için Cross-Validation
cv_results_dt_koah = cross_val_score(dt_model_koah, X, y_koah, cv=5, scoring='accuracy')

# Cross-Validation sonuçlarını yazdırma
print("Karar Ağacı - Astım: Cross-Validation Doğruluk Skorları: ", cv_results_dt_astim)
print("Karar Ağacı - KOAH: Cross-Validation Doğruluk Skorları: ", cv_results_dt_koah)

# Ortalama doğrulukları yazdırma
print("\nKarar Ağacı - Astım: Ortalama Doğruluk: ", np.mean(cv_results_dt_astim))
print("Karar Ağacı - KOAH: Ortalama Doğruluk: ", np.mean(cv_results_dt_koah))

# Test hatalarını hesaplama
test_hatasi_dt_astim = 1 - np.mean(cv_results_dt_astim)
test_hatasi_dt_koah = 1 - np.mean(cv_results_dt_koah)

# Test hatalarını yazdırma
print("\nKarar Ağacı - Astım: Test Hatası: ", test_hatasi_dt_astim)
print("Karar Ağacı - KOAH: Test Hatası: ", test_hatasi_dt_koah)





#EN SON GRADIENT BOOSTING E KARAR VERILMIŞTIR.
#test verisini düzenleme
import pandas as pd
veritest = pd.read_excel("~/Desktop/data2.xls", engine="xlrd")

print(veritest.head())
print(veritest.columns)
# Veri setindeki her bir değişkenin veri türünü (class) öğrenmek
print(veritest.dtypes)

# Eksik veri kontrolü (NaN değerler)
print(veritest.isnull().sum())

# Kategorik degiskenlerin veri tipinin category yapilmasi
veritest.dtypes

veritest["cınsıyet"] = veritest["cınsıyet"].astype("category")
veritest["meslegı"] = veritest["meslegı"].astype("category")
veritest["sıgarakullanımı"] = veritest["sıgarakullanımı"].astype("category")
veritest["ailedekoahveyaastımTanılıHastavarmı"] = veritest["ailedekoahveyaastımTanılıHastavarmı"].astype("category")
veritest["hastaneyeyattımı"] = veritest["hastaneyeyattımı"].astype("category")    

veritest['FEV1 %'] = pd.to_numeric(veritest['FEV1 %'], errors='coerce')
veritest['PEF %'] = pd.to_numeric(veritest['PEF %'], errors='coerce')
veritest['FEV1/FVC Değeri'] = pd.to_numeric(veritest['FEV1/FVC Değeri'], errors='coerce')

#veriyi doldurma
import pandas as pd
from sklearn.impute import KNNImputer
# Sayısal kolonları seç
numeric_data = veritest.select_dtypes(include=['float64', 'int64'])

# KNN Imputer'ı oluştur
imputer = KNNImputer(n_neighbors=5)

# Sadece sayısal verileri doldur
numeric_imputed = imputer.fit_transform(numeric_data)

# Doldurulmuş sayısal veriyi DataFrame'e geri çevir
numeric_imputed_df = pd.DataFrame(numeric_imputed, columns=numeric_data.columns)

# Kategorik veriyi orijinal veri çerçevesinden almak için indexi kullanın
categorical_data = veritest.select_dtypes(include=['category'])

# Son olarak sayısal verileri ve kategorik veriyi birleştirin
data_imputed = pd.concat([numeric_imputed_df, categorical_data], axis=1)

# Sonuç
print(data_imputed)
print(data_imputed.isnull().sum())

yaş_k2 = pd.cut(data_imputed["YAŞ"], bins = 3)
yaş_k2.cat.categories
yaş_k2 = yaş_k2.cat.rename_categories(["GENÇ", "YETİŞKİN", "YAŞLI"])
yaş_k2.value_counts()
pd.concat([data_imputed["YAŞ"].head(10), yaş_k2[0:10]], axis=1)
    data_imputed = data_imputed.drop(columns=["YAŞ"])  # YAŞ değişkenini çıkar
    data_imputed["YAŞ_GRUBU"] = yaş_k2  # Yeni grubu ekle
    print(data_imputed)
    print(data_imputed["YAŞ_GRUBU"].dtype)

# Nabız aralıklarını belirleme (Düşük, Normal, Yüksek)
data_imputed['NaNbız'] = pd.to_numeric(data_imputed['NaNbız'], errors='coerce')

data_imputed['NaNbız'] = pd.cut(
    data_imputed['NaNbız'],
    bins=[0, 60, 100, float('inf')],  # Aralık sınırları: Düşük, Normal, Yüksek
    labels=['DÜŞÜK', 'NORMAL', 'YÜKSEK']  # Kategoriler
)

# Sonuçları görüntüleme
print(data_imputed)
print("\n'Nabız' Sütununun Veri Tipi:")
print(data_imputed["NaNbız"].dtype)
print(data_imputed.columns)

# Saat cinsine dönüştürme işlemi
data_imputed['acılservısetoplamyatıssuresısaat'] = data_imputed['acilservistoplamyatışsüresigün'] * 24
data_imputed['yogumbakımatoplamyatıssuresısaat'] = data_imputed['yogumbakımatoplamyatıssüresıgun'] * 24
data_imputed['servıseoplamyatıssuresısaat'] = data_imputed['servisetoplamyatıssüresıgun'] * 24

# Eski sütunları kaldırmak isterseniz
data_imputed = data_imputed.drop(columns=['acilservistoplamyatışsüresigün', 'yogumbakımatoplamyatıssüresıgun', 'servisetoplamyatıssüresıgun'])

# Sonuçları kontrol etme
print(data_imputed.head())

# 'tanısuresıyıl' sütununu sayısala dönüştürün
data_imputed['tanısuresıyıl'] = pd.to_numeric(data_imputed['tanısuresıyıl'], errors='coerce')


# Yıl'ı Ay'a çevirmek için her bir yıl değerini 12 ile çarpın
data_imputed['tanısuresıay'] = data_imputed['tanısuresıyıl'] * 12
# Eski sütunları kaldırmak isterseniz
data_imputed = data_imputed.drop(columns=['tanısuresıyıl'])


# Boyu metreye çevirme (Eğer boy santimetre olarak verilmişse)
data_imputed['boy'] = data_imputed['boy'] / 100  # Boyu santimetreden metreye çevirme (eğer boy santimetre cinsindense)

# Kitle Endeksi (BMI) hesaplama
data_imputed['kitleendeksi'] = data_imputed['vucutagırlıgı'] / (data_imputed['boy'] ** 2)

# Sonuçları kontrol etme
data_imputed[['boy', 'vucutagırlıgı', 'kitleendeksi']].head()

# Boy ve vücut ağırlığı sütunlarını silme
data_imputed = data_imputed.drop(columns=['boy', 'vucutagırlıgı'])

# Silinen sütunları kontrol etme
data_imputed.head()

from sklearn.decomposition import PCA

# PCA uygulamak için sistolik ve diastolik kan basıncını seçin
pca = PCA(n_components=1)
data_imputed['PCA_KanBasıncı'] = pca.fit_transform(data_imputed[['kanbasıncısıstolık', 'kanbasıncıdıastolık']])

# PCA sonucu, sistolik ve diastolik kan basıncının tek bir bileşeni olacak
print(data_imputed[['PCA_KanBasıncı']].describe())
data_imputed = data_imputed.drop(columns=['kanbasıncısıstolık', 'kanbasıncıdıastolık'])

data_imputed = data_imputed.drop(columns=['tanısuresıay', 'acılservısetoplamyatıssuresısaat','yogumbakımayatıssayısı','yogumbakımatoplamyatıssuresısaat'])

import numpy as np
import scipy.stats as stats

# Sayısal sütunları seçme, ancak 'hastano' sütununu hariç tutma
sayisal_sutunlar1 = data_imputed.select_dtypes(include=['number']).columns
# 'hastano' sütununu sayısal sütunlar listesinden çıkarma
sayisal_sutunlar1 = sayisal_sutunlar1[sayisal_sutunlar1 != 'hastaNo']
# Sayısal sütunları veri setinden çekme
sayisal_veri1 = data_imputed[sayisal_sutunlar1]
# Sonuçları kontrol etme
print(sayisal_veri1)

# Normal dağılıma uygunluk testi (tek örneklem)
statistic, p_value = stats.kstest(sayisal_veri1, 'norm')

# Sonuçları yazdır
print(f"Test İstatistiği: {statistic}")
print(f"P-değeri: {p_value}")

#normal değil.

#robust standartlaştırma
from sklearn.preprocessing import RobustScaler

# RobustScaler oluştur
scaler = RobustScaler()
# Ölçeklendirme işlemi
data_imputed[sayisal_sutunlar1] = scaler.fit_transform(data_imputed[sayisal_sutunlar1])

# Sonuçları kontrol etme
print(data_imputed.head())

# Kategorik sütunları seçme
kategorik_sutunlar1 = data_imputed.select_dtypes(include=['category', 'object']).columns

# Kategorik sütunların değerlerini inceleme
kategorik_veriler1 = data_imputed[kategorik_sutunlar1]
print("\nKategorik Veriler:")
print(kategorik_veriler1)

# One-Hot Encoding işlemi
df_encoded1 = pd.get_dummies(kategorik_veriler1, columns=kategorik_sutunlar1)

print("\nOne-Hot Encoded VeriFrame:")
print(df_encoded1)
# True/False değerlerini 1/0 olarak değiştirme
df_encoded1 = df_encoded1.astype(int)

# Sonuçları kontrol etme
print(df_encoded1)
df_encoded1 = df_encoded1.rename(columns={'ailedekoahveyaastımTanılıHastavarmı_1.0': 'ailedekoahveyaastımTanılıHastavarmı_1', 'ailedekoahveyaastımTanılıHastavarmı_2.0': 'ailedekoahveyaastımTanılıHastavarmı_2'})
print(df_encoded1)
# Sayısal veriler ve One-Hot Encoded veriyi birleştirme
data_test = pd.concat([data_imputed[sayisal_sutunlar1], df_encoded1], axis=1)

# Sonuçları kontrol etme
print("\nBirleştirilmiş VeriFrame:")
print(data_test.head())

# Gradient Boosting 
# Eğitim verisindeki sütun sırasını kaydedin
train_columns = X.columns
# Test verisinin sütunlarını eğitim verisindeki sıralamaya göre düzenleyin
data_test = data_test[train_columns]

# Koah modelini test verisi üzerinde tahmin yapın
koah_tahmin = gb_koah.predict(data_test)

# Astım modelini test verisi üzerinde tahmin yapın
astim_tahmin = gb_astim .predict(data_test)

# Tahmin sonuçlarını yazdır
print("Test Verisi - Astım Tahminleri:", astim_tahmin)
print("Test Verisi - KOAH Tahminleri:", koah_tahmin)

import pandas as pd

# Tahmin sonuçlarını test verisine ekleyin
data_test['Tahmin_Astım'] = astim_tahmin
data_test['Tahmin_KOAH'] = koah_tahmin

# DataFrame'i Excel dosyasına kaydet
data_test.to_excel('data_test_sonuclari.xlsx', index=False)

print("Excel dosyası oluşturuldu: data_test_sonuclari.xlsx")

import os

# Mevcut çalışma dizinini kontrol edin
print("Mevcut Çalışma Dizini:", os.getcwd())


