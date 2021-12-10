import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump

data = pd.read_csv("pong_data.csv")
training_data = [list(a) for a in zip(data['ball_y'].values,data['paddle_y'].values)]
target_values = data['paddle_direction'].values
print(target_values)

model = LinearRegression()
model.fit(training_data,target_values)
dump(model, 'mymodel.joblib')
# newData = data.iloc[:, [5,1]]
# print(newData)
# target_values = data.iloc[:, 4]
# print(target_values)

# model = LogisticRegression().fit(newData, target_values)

# from joblib import dump, load 
# dump(model, 'mymodel.joblib')