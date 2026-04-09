import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

file_path = 'Salary_Data.csv'
data = pd.read_csv(file_path)

X = data[['YearsExperience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "salary_model.pkl")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (средняя абсолютная ошибка): {mae:.3f}")
print(f"MSE (среднеквадратичная ошибка): {mse:.2f}")
print(f"R² (точность предсказания): {r2:.2f}")

example_exp = [[6.5]]
predicted_salary = model.predict(example_exp)
print(f"Предсказанная зарплата для 6.5 лет опыта: {predicted_salary[0]:.2f}")

plt.scatter(X, y, color='blue', label='Истинные данные')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Предсказания модели')
plt.xlabel('Years of Experience')
plt.ylabel('зарплата')
plt.title('Линейная регрессия: опыт vs зарплата')
plt.legend()
plt.show()


