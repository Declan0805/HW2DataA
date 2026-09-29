import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
sheet_nums = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4',]
test_size_num = 0.25
column_sets = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
results = []
for sheet in sheet_nums:
    df = pd.read_excel('iris.xlsx', sheet_name=sheet)
    for columns_needed in column_sets:

        x = np.array(df.iloc[:, columns_needed])
        y = np.array(df[df.columns[3]])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_num)

        linear = LinearRegression()
        linear.fit(x_train, y_train)
        y_predict = linear.predict(x_test)
        r2 = linear.score(x_test, y_test)

        results.append([sheet, columns_needed, test_size_num, r2])

        print("Run#\t y_test\t\t y_predict")
        for i in range(len(y_predict)):
            print(i, "\t\t", y_test[i], "\t\t", y_predict[i])
        print("Sheet used: " , sheet)
        print("Columns used: ", columns_needed)
        print("R-squared value: %.4f" % r2)