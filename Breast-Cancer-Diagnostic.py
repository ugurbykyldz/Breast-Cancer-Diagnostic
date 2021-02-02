#kütüphaneleri yüklüyelim
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#csv okuma
data = pd.read_csv("data.csv")
# rasgele 5 gozleme bakalım
data.sample(5)
#gereksiz sutunları çıkartma
data.drop(["Unnamed: 32","id"],inplace = True, axis = 1)
#diagnosis ozlelligini target yapma
data = data.rename(columns = {"diagnosis":"target"})

#kaç tane sıfımız var
sns.countplot(data["target"])
print(data.target.value_counts()) # B :357 , M:212

#B : 0 , M : 1 yapıyoruz
data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]

# data uzunlugu
print(len(data)) # 569
#data ilk 5 eleaman
print(data.head())
#data shape
print("Data shape :",data.shape) #(569,31)
#data ilgili bilgileri bakalım
print("Data info :",data.info())
#data describe(count, mean ,std, min,max)
print("Data describe" , data.describe()) 

"""
normalization  yapmamız gerekiyor
"""
#eksik deger var mı?
print(data.isnull().sum()) # missing value = none

#aynı degerlere sahip sample var mı?
print(data.duplicated().sum()) # yok

#%%  Keşifci veri analizi "EDA"

# Correlation features aralarındaki ilişkiye bakalım
#%80 oranında ilişki bulunanlar data çıkarılabilir
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation between Features")
plt.show()

#box plot çizdirelim
data_melted = pd.melt(data,id_vars = "target"
                      ,var_name = "features"
                      ,value_name = "value")


plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()

#anlamlı degerler elde edemedik normalization yaptıktan sonra bakalım



# Outlier tespiti
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(data)

data_scores = clf.negative_outlier_factor_
np.sort(data_scores)[0:10]

#eşik deger
threshold_value = np.sort(data_scores)[5]
outlier_tf = data_scores > threshold_value
#aykırı degerleri siliyoruz
data = data[data_scores >threshold_value]

#datamızı böluyoruz
x = data.drop(["target"],axis = 1)
y = data.target.values
columns = x.columns.tolist()

# Normalization
x = (x - np.min(x)) / (np.max(x) - np.min(x))


#train test split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
X_train_df["target"] = Y_train

#box plot 
data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()


#piar plot
sns.pairplot(X_train_df,diag_kind = "kde",markers = "+", hue = "target")
plt.show()



#%% Modellerin karşılaştırılması
knn_model = KNeighborsClassifier(n_neighbors = 2)
loj_model = LogisticRegression(solver = "liblinear" )
svm_model = SVC()
Ann_model = MLPClassifier()
rf_model = RandomForestClassifier()

models = [knn_model, loj_model, svm_model, Ann_model, rf_model]
result =list()
results = pd.DataFrame(columns = ["Models", "Accuracy"])

for model in models:
    names = model.__class__.__name__
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    result = pd.DataFrame([[names, acc*100]], columns = ["Models", "Accuracy"])
    results = results.append(result)
plt.figure(figsize = (10,10))   
sns.barplot(x = "Accuracy", y = "Models", data = results, color = "blue")
plt.xlabel("Accuracy %")
plt.title("Models rate of accuracy")    

print(results)
"""
                   Models   Accuracy
    KNeighborsClassifier    % 95
      LogisticRegression    % 95
                     SVC    % 95
      MLPClassifier(Ann)    % 96
  RandomForestClassifier    % 95
  
  
  -Knn algoritmasını kullanarak devam edecegiz.

"""
    

#%% KNN

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_train,Y_train)
print("Confusion Matrix", cm)
print("Knn Acc :", acc)



# en iyi k degerini bulma

k_range = list(range(1,31))
weight_options = ["uniform", "distance"]
param_grid = dict(n_neighbors = k_range, weights = weight_options)

knn_g = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy")
grid.fit(X_train, Y_train)

best_traing_score = grid.best_score_ # %97
parameters = grid.best_params_

knn = KNeighborsClassifier(**grid.best_params_)
knn.fit(X_train, Y_train)

#• overfitting var mı ?
y_pred_test = knn.predict(X_test)
y_pred_train = knn.predict(X_train)

cm_test = confusion_matrix(Y_test, y_pred_test)
cm_train = confusion_matrix(Y_train, y_pred_train)

acc_test = accuracy_score(Y_test, y_pred_test) # %97
acc_train = accuracy_score(Y_train, y_pred_train) # %98
print("Test score : {} \n Train score : {}".format(acc_test, acc_train))

# farklar kuçuk oldugu için owerfitting var diyemeyiz

#Confusion matrix gorselleştirelim

plt.figure()
sns.heatmap(cm_test, annot = True)
plt.xlabel("Y Pred")
plt.ylabel("Y True")
plt.show()


"""
Modelimiz knn algoritmasını kullanarak %97 oranında başarı saglamıştır.
"""
























