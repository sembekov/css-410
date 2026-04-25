import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = {
    "AI_Investment": [10, 20, 30, 40, 50, 60, 70, 80],
    "Employees": [50, 60, 70, 80, 90, 100, 110, 120],
    "Revenue": [100, 150, 200, 260, 320, 400, 480, 600]
}

df = pd.DataFrame(data)

X = df[["AI_Investment", "Employees"]]
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)

print("Predictions:", predictions)
print("Actual:", list(y_test))
print("Mean Absolute Error:", error)
