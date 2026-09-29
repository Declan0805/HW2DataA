# python -m pip install pandas numpy scikit-learn mlxtend
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_excel('iris.xlsx', sheet_name='Sheet1')
x = np.array(df[df.columns[0:3]])
y = np.array(df[df.columns[3]])
x_Train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

linear = LinearRegression()
linear.fit(x_Train, y_train)
y_predict = linear.predict(x_test)
print("\ty_test\t\ty_predict")
for i in range(len(y_predict)):
    print(i, "\t", y_test[i], "\t\t", y_predict[i])
