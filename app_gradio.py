# Lab 2: Logistic Regression with Gradio UI
import gradio as gr
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
    probability = model1.predict_proba(data)[0]
    return {"On Time ✅": float(probability[0]), "Delayed ⏰": float(probability[1])}


def predict_gym(age, visits, months, plan):
    if plan == "Premium":
        plan = 1
    else:
        plan = 0
    data = scaler2.transform([[age, visits, months, plan]])
    probability = model2.predict_proba(data)[0]
    return {"Not Renewed ❌": float(probability[0]), "Renewed ✅": float(probability[1])}


# ======================= GRADIO UI =======================
with gr.Blocks() as app:

    # Colourful header banner
    gr.HTML("""
    <div style="text-align:center; padding:25px; border-radius:15px;
                background:linear-gradient(90deg,#4f46e5,#9333ea); color:white;">
        <h1 style="margin:0;">🧠 Logistic Regression Predictor</h1>
        <p style="margin:5px 0 0 0;">ANN Lab Task 2 &nbsp;|&nbsp; Flight Delay &amp; Gym Membership Renewal</p>
    </div>
    """)

    # ---------- Tab 1: Flight Delay ----------
    with gr.Tab("✈️ Flight Delay"):
        gr.Markdown("### Will the flight be delayed?\nMove the sliders and click **Predict**.")
        with gr.Row():
            with gr.Column():
                distance = gr.Slider(200, 5000, value=1500, step=50, label="📏 Distance (km)")
                hour = gr.Slider(0, 23, value=12, step=1, label="🕒 Departure Hour (0-23)")
            with gr.Column():
                weather = gr.Slider(1, 10, value=5, step=1, label="🌦️ Weather Severity (1 = clear, 10 = storm)")
                previous = gr.Slider(0, 120, value=30, step=5, label="⏱️ Previous Flight Delay (minutes)")
        flight_button = gr.Button("🔍 Predict", variant="primary")
        flight_output = gr.Label(label="Prediction")
        flight_button.click(predict_flight, [distance, hour, weather, previous], flight_output)

    # ---------- Tab 2: Gym Renewal ----------
    with gr.Tab("🏋️ Gym Membership"):
        gr.Markdown("### Will the member renew the membership?\nEnter the member details and click **Predict**.")
        with gr.Row():
            with gr.Column():
                age = gr.Slider(18, 60, value=30, step=1, label="🎂 Age")
                visits = gr.Slider(0, 30, value=10, step=1, label="🏃 Gym Visits Per Month")
            with gr.Column():
                months = gr.Slider(1, 36, value=12, step=1, label="📅 Months Enrolled")
                plan = gr.Dropdown(["Basic", "Premium"], value="Basic", label="💳 Plan Type")
        gym_button = gr.Button("🔍 Predict", variant="primary")
        gym_output = gr.Label(label="Prediction")
        gym_button.click(predict_gym, [age, visits, months, plan], gym_output)

app.launch()
