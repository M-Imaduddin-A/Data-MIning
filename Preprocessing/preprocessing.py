# impor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# impor dataset
dataset = pd.read_csv(
        'Data.csv',
        delimiter=';', 
        header='infer', 
        index_col=False
        )
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# menangani nilai kosong
from sklearn.impute import SimpleImputer

# ganti NaN dengan mean kolom itu
imputer = SimpleImputer(
        missing_values=np.nan, 
        strategy='mean'
        )
imputer = imputer.fit(X[:, 1:2])
X[:, 1:2] = imputer.transform(X[:, 1:2])

# kodekan data kategori
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# kode hanya sebatas penanda
encoder_X = ColumnTransformer(
        [('tk_encoder', OneHotEncoder(), [0])], 
        remainder='passthrough'
        )
X = encoder_X.fit_transform(X).astype(float) # mengembalikan ke dalam tipe 'float64'

# y adalah dependent, cukup kodekan ke angka
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# variabel dummy kode provinsi juga diskalakan
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
