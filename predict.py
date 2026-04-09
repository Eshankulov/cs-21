import joblib

model = joblib.load("salary_model.pkl")

while True:
    user_input = input("Тажрыйбаны киргизиңиз (0-10) : ")

    if user_input.lower() == "exit":
        print("Программа токтотулду.")
        break

    try:
        exp = float(user_input)

        if exp < 1 or exp > 10:
            print("1ден 10го чейин гана сан киргизиңиз!")
            continue

        salary = model.predict([[exp]])
        print(f"Болжолдуу айлык: {salary[0]:.2f}")

    except:
        print("Сан киргизиңиз!")
