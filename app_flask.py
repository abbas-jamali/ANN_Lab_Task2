# Lab 2: Logistic Regression with Flask UI
from flask import Flask, request, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ======================= DATASET 1: FLIGHT DELAY =======================
# Step 1: Load the data
df1 = pd.read_csv("flight_delay.csv")
print("========== FLIGHT DELAY DATASET ==========")
print(df1.head())
print("Shape:", df1.shape)
print(df1.describe())

# Step 2: Check missing values and duplicate rows
print("Missing values:\n", df1.isnull().sum())
print("Duplicate rows:", df1.duplicated().sum())

# Step 3: Clean the data
df1["Weather_Severity"] = df1["Weather_Severity"].fillna(df1["Weather_Severity"].median())
df1["Previous_Delay_Min"] = df1["Previous_Delay_Min"].fillna(df1["Previous_Delay_Min"].median())
df1 = df1.drop_duplicates()
print("Missing values after cleaning:", df1.isnull().sum().sum())
print("Rows after cleaning:", len(df1))

# Step 4: Separate inputs (X) and output (y), then split into train and test
X1 = df1[["Distance_km", "Departure_Hour", "Weather_Severity", "Previous_Delay_Min"]].values
y1 = df1["Delayed"].values
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Step 5: Scaling (brings all columns to the same scale)
scaler1 = StandardScaler()
X1_train = scaler1.fit_transform(X1_train)
X1_test = scaler1.transform(X1_test)

# Step 6: Train the Logistic Regression model
model1 = LogisticRegression()
model1.fit(X1_train, y1_train)

# Step 7: Test the model
y1_pred = model1.predict(X1_test)
print("Flight Delay Accuracy:", accuracy_score(y1_test, y1_pred))
print("Confusion Matrix:\n", confusion_matrix(y1_test, y1_pred))
print("Classification Report:\n", classification_report(y1_test, y1_pred))


# ======================= DATASET 2: GYM MEMBERSHIP RENEWAL =======================
# Step 1: Load the data
df2 = pd.read_csv("gym_renewal.csv")
print("========== GYM RENEWAL DATASET ==========")
print(df2.head())
print("Shape:", df2.shape)
print(df2.describe())

# Step 2: Check missing values and duplicate rows
print("Missing values:\n", df2.isnull().sum())
print("Duplicate rows:", df2.duplicated().sum())

# Step 3: Clean the data
df2["Visits_Per_Month"] = df2["Visits_Per_Month"].fillna(df2["Visits_Per_Month"].median())
df2["Months_Enrolled"] = df2["Months_Enrolled"].fillna(df2["Months_Enrolled"].median())
df2["Plan_Type"] = df2["Plan_Type"].fillna(df2["Plan_Type"].mode()[0])
df2 = df2.drop_duplicates()
print("Missing values after cleaning:", df2.isnull().sum().sum())
print("Rows after cleaning:", len(df2))

# Step 4: Encoding (convert text to numbers): Premium = 1, Basic = 0
df2["Plan_Type"] = df2["Plan_Type"].map({"Premium": 1, "Basic": 0})

# Step 5: Separate inputs (X) and output (y), then split into train and test
X2 = df2[["Age", "Visits_Per_Month", "Months_Enrolled", "Plan_Type"]].values
y2 = df2["Renewed"].values
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Step 6: Scaling
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)

# Step 7: Train the Logistic Regression model
model2 = LogisticRegression()
model2.fit(X2_train, y2_train)

# Step 8: Test the model
y2_pred = model2.predict(X2_test)
print("Gym Renewal Accuracy:", accuracy_score(y2_test, y2_pred))
print("Confusion Matrix:\n", confusion_matrix(y2_test, y2_pred))
print("Classification Report:\n", classification_report(y2_test, y2_pred))


# ======================= PLOTS =======================
# (Close the plot windows to start the app)

# Flight delay plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
df1["Delayed"].value_counts().plot(kind="bar")
plt.title("Flights: Delayed (1) vs On Time (0)")
plt.subplot(1, 3, 2)
plt.hist(df1["Distance_km"])
plt.title("Flights: Distance (km)")
plt.subplot(1, 3, 3)
plt.scatter(df1["Weather_Severity"], df1["Previous_Delay_Min"], c=df1["Delayed"])
plt.xlabel("Weather Severity")
plt.ylabel("Previous Delay (min)")
plt.title("Flights: colour = Delayed/On Time")
plt.tight_layout()

# Gym renewal plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
df2["Renewed"].value_counts().plot(kind="bar")
plt.title("Gym: Renewed (1) vs Not Renewed (0)")
plt.subplot(1, 3, 2)
plt.hist(df2["Visits_Per_Month"])
plt.title("Gym: Visits Per Month")
plt.subplot(1, 3, 3)
plt.scatter(df2["Visits_Per_Month"], df2["Months_Enrolled"], c=df2["Renewed"])
plt.xlabel("Visits Per Month")
plt.ylabel("Months Enrolled")
plt.title("Gym: colour = Renewed/Not Renewed")
plt.tight_layout()

plt.show()


# ======================= PREDICTION FUNCTIONS =======================
def predict_flight(distance, hour, weather, previous_delay):
    data = scaler1.transform([[distance, hour, weather, previous_delay]])
    result = model1.predict(data)[0]
    confidence = model1.predict_proba(data)[0][result] * 100
    if result == 1:
        return {"text": f"⏰ Flight will be DELAYED (confidence {confidence:.1f}%)", "color": "red"}
    else:
        return {"text": f"✅ Flight will be ON TIME (confidence {confidence:.1f}%)", "color": "green"}


def predict_gym(age, visits, months, plan):
    if plan == "Premium":
        plan = 1
    else:
        plan = 0
    data = scaler2.transform([[age, visits, months, plan]])
    result = model2.predict(data)[0]
    confidence = model2.predict_proba(data)[0][result] * 100
    if result == 1:
        return {"text": f"✅ Member will RENEW (confidence {confidence:.1f}%)", "color": "green"}
    else:
        return {"text": f"❌ Member will NOT renew (confidence {confidence:.1f}%)", "color": "red"}


# ======================= FLASK UI =======================
app = Flask(__name__)

# The web page: HTML (structure) + CSS (colours and design)
page = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Logistic Regression Predictor</title>
<style>
    body {
        font-family: 'Segoe UI', Arial, sans-serif;
        margin: 0;
        min-height: 100vh;
        background: linear-gradient(135deg, #4f46e5, #9333ea);
    }
    .header { text-align: center; color: white; padding: 30px 10px 10px 10px; }
    .header h1 { margin: 0; font-size: 36px; }
    .header p { margin-top: 8px; font-size: 17px; opacity: 0.9; }
    .container { display: flex; flex-wrap: wrap; gap: 25px; justify-content: center; padding: 20px 20px 40px 20px; }
    .card { background: white; border-radius: 16px; padding: 25px; width: 380px; box-shadow: 0 10px 25px rgba(0,0,0,0.25); }
    .card h2 { margin-top: 0; color: #4f46e5; }
    label { display: block; margin-top: 14px; font-weight: 600; color: #374151; }
    input, select { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #d1d5db; border-radius: 8px; box-sizing: border-box; font-size: 15px; }
    input:focus, select:focus { outline: none; border-color: #4f46e5; }
    button { width: 100%; margin-top: 22px; padding: 12px; background: #4f46e5; color: white; border: none; border-radius: 8px; font-size: 17px; cursor: pointer; }
    button:hover { background: #3730a3; }
    .result { margin-top: 18px; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; font-weight: bold; }
    .green { background: #dcfce7; color: #166534; }
    .red { background: #fee2e2; color: #991b1b; }
</style>
</head>
<body>

<div class="header">
    <h1>🧠 Logistic Regression Predictor</h1>
    <p>ANN Lab Task 2 | Flight Delay &amp; Gym Membership Renewal</p>
</div>

<div class="container">

    <div class="card">
        <h2>✈️ Flight Delay Prediction</h2>
        <form method="post">
            <input type="hidden" name="form_type" value="flight">
            <label>📏 Distance (km)</label>
            <input type="number" step="any" name="distance" placeholder="e.g. 1500" required>
            <label>🕒 Departure Hour (0-23)</label>
            <input type="number" name="hour" min="0" max="23" placeholder="e.g. 14" required>
            <label>🌦️ Weather Severity (1-10)</label>
            <input type="number" name="weather" min="1" max="10" placeholder="e.g. 5" required>
            <label>⏱️ Previous Flight Delay (minutes)</label>
            <input type="number" step="any" name="previous" placeholder="e.g. 30" required>
            <button type="submit">🔍 Predict</button>
        </form>
        {% if flight_result %}
        <div class="result {{ flight_result.color }}">{{ flight_result.text }}</div>
        {% endif %}
    </div>

    <div class="card">
        <h2>🏋️ Gym Renewal Prediction</h2>
        <form method="post">
            <input type="hidden" name="form_type" value="gym">
            <label>🎂 Age</label>
            <input type="number" name="age" placeholder="e.g. 30" required>
            <label>🏃 Gym Visits Per Month</label>
            <input type="number" name="visits" placeholder="e.g. 10" required>
            <label>📅 Months Enrolled</label>
            <input type="number" name="months" placeholder="e.g. 12" required>
            <label>💳 Plan Type</label>
            <select name="plan">
                <option>Basic</option>
                <option>Premium</option>
            </select>
            <button type="submit">🔍 Predict</button>
        </form>
        {% if gym_result %}
        <div class="result {{ gym_result.color }}">{{ gym_result.text }}</div>
        {% endif %}
    </div>

</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    flight_result = None
    gym_result = None
    if request.method == "POST":
        if request.form["form_type"] == "flight":
            distance = float(request.form["distance"])
            hour = float(request.form["hour"])
            weather = float(request.form["weather"])
            previous = float(request.form["previous"])
            flight_result = predict_flight(distance, hour, weather, previous)
        else:
            age = float(request.form["age"])
            visits = float(request.form["visits"])
            months = float(request.form["months"])
            plan = request.form["plan"]
            gym_result = predict_gym(age, visits, months, plan)
    return render_template_string(page, flight_result=flight_result, gym_result=gym_result)


if __name__ == "__main__":
    app.run(debug=True)
