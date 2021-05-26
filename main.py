import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model

rangeMax = 88
predictData = 14
weigh = 0.1

# 全体的にscoreが低いので過去7日間の気象情報から予測するようにする
columnsDate = ['年', '月', '日', '年月日']
columns = ['気温', '降水量', '日照時間', '風速', '雲量', '蒸気圧', '湿度', '海面気圧', '現地気圧', '最高気温', '最低気温', '天気']
NagoyaLearn = pd.read_csv('NagoyaLearn.csv', header=None)
NagoyaTest = pd.read_csv('NagoyaTest.csv', header=None)
NagoyaLearn.columns = columnsDate + columns * 7
NagoyaTest.columns = columnsDate + columns * 7

X_nagoya_train = NagoyaLearn.drop(index=19717).iloc[:, 4:rangeMax]
Y_train = NagoyaLearn.drop(index=0).iloc[:, predictData].values
X_nagoya_test = NagoyaTest.drop(index=24).iloc[:, 4:rangeMax]
Y_test = NagoyaTest.drop(index=0).iloc[:, predictData].values

pngName = NagoyaLearn.columns[predictData]

columnsA = ['蒸気圧', '湿度', '海面気圧', '現地気圧', '気温', '降水量', '日照時間', '風速', '最低気温', '最高気温']
columnsT = columnsDate + columnsA * 7
GifuLearn = pd.read_csv('GifuLearn.csv', header=None)
GifuTest = pd.read_csv('GifuTest.csv', header=None)
GifuLearn.columns = columnsT
GifuTest.columns = columnsT

HamamatuLearn = pd.read_csv('HamamatuLearn.csv', header=None)
HamamatuTest = pd.read_csv('HamamatuTest.csv', header=None)
HamamatuLearn.columns = columnsT
HamamatuTest.columns = columnsT

IragoLearn = pd.read_csv('IragoLearn.csv', header=None)
IragoTest = pd.read_csv('IragoTest.csv', header=None)
IragoLearn.columns = columnsT
IragoTest.columns = columnsT

OmaezakiLearn = pd.read_csv('OmaezakiLearn.csv', header=None)
OmaezakiTest = pd.read_csv('OmaezakiTest.csv', header=None)
OmaezakiLearn.columns = columnsT
OmaezakiTest.columns = columnsT

OwashiLearn = pd.read_csv('OwashiLearn.csv', header=None)
OwashiTest = pd.read_csv('OwashiTest.csv', header=None)
OwashiLearn.columns = columnsT
OwashiTest.columns = columnsT

TuLearn = pd.read_csv('TuLearn.csv', header=None)
TuTest = pd.read_csv('TuTest.csv', header=None)
TuLearn.columns = columnsT
TuTest.columns = columnsT

UenoLearn = pd.read_csv('UenoLearn.csv', header=None)
UenoTest = pd.read_csv('UenoTest.csv', header=None)
UenoLearn.columns = columnsT
UenoTest.columns = columnsT

YokkaichiLearn = pd.read_csv('YokkaichiLearn.csv', header=None)
YokkaichiTest = pd.read_csv('YokkaichiTest.csv', header=None)
YokkaichiLearn.columns = columnsT
YokkaichiTest.columns = columnsT

X_gifu_train = GifuLearn.drop(index=19717).iloc[:, 4:rangeMax]
X_gifu_test = GifuTest.drop(index=24).iloc[:, 4:rangeMax]

X_hamamatu_train = HamamatuLearn.drop(index=19717).iloc[:, 4:rangeMax]
X_hamamatu_test = HamamatuTest.drop(index=24).iloc[:, 4:rangeMax]

X_irago_train = IragoLearn.drop(index=19717).iloc[:, 4:rangeMax]
X_irago_test = IragoTest.drop(index=24).iloc[:, 4:rangeMax]

X_omaezaki_train = OmaezakiLearn.drop(index=19717).iloc[:, 4:rangeMax]
X_omaezaki_test = OmaezakiTest.drop(index=24).iloc[:, 4:rangeMax]

X_owashi_train = OwashiLearn.drop(index=19717).iloc[:, 4:rangeMax]
X_owashi_test = OwashiTest.drop(index=24).iloc[:, 4:rangeMax]

X_tu_train = TuLearn.drop(index=19717).iloc[:, 4:rangeMax]
X_tu_test = TuTest.drop(index=24).iloc[:, 4:rangeMax]

X_ueno_train = UenoLearn.drop(index=19717).iloc[:, 4:rangeMax]
X_ueno_test = UenoTest.drop(index=24).iloc[:, 4:rangeMax]

X_yokkaichi_train = YokkaichiLearn.drop(index=19717).iloc[:, 4:rangeMax]
X_yokkaichi_test = YokkaichiTest.drop(index=24).iloc[:, 4:rangeMax]

print(X_owashi_train.isnull().sum())
print(X_owashi_test.isnull().sum())

X_train = pd.concat([X_nagoya_train, X_gifu_train, X_hamamatu_train, X_irago_train, X_omaezaki_train,
                     X_tu_train, X_ueno_train, X_yokkaichi_train], axis=1)
X_test = pd.concat([X_nagoya_test, X_gifu_test, X_hamamatu_test, X_irago_test, X_omaezaki_test,
                    X_tu_test, X_ueno_test, X_yokkaichi_test], axis=1)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
Y_train = Y_train.astype(float)
Y_test = Y_test.astype(float)

print(X_train)
print(X_test)

sc = preprocessing.StandardScaler()
sc.fit(X_train)
X = sc.transform(X_train)


def result(algorithm):
    return algorithm.score(X_test, Y_test)


def difference(predict):
    diff_y = abs(predict - Y_test)
    return sum(diff_y) / len(diff_y), max(diff_y)


def png_show(predict, name):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(Y_test)
    plt.plot(predict, linestyle="dashed")
    plt.savefig(name)
    plt.show()


print("----main.py----")
clf_ridge = linear_model.Ridge(alpha=weigh)
clf_ridge.fit(X_train, Y_train)
print("\nRidgeでの係数")
print(clf_ridge.intercept_)
print(clf_ridge.coef_)
Y_ridge_pred = clf_ridge.predict(X_test)
print("\n「Ridgeの平均2乗誤差」")
RMS_ridge = np.mean((Y_ridge_pred - Y_test) ** 2)
print(RMS_ridge)
print("\n「Ridgeのscore」")
print(result(clf_ridge))
print("\n「Ridgeの平均誤差」「Ridgeの最大誤差」")
print(difference(Y_ridge_pred))
print("\n----main.py----")

png_show(Y_ridge_pred, pngName)
