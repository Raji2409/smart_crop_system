import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib, numpy as np, cv2, pandas as pd, sqlite3
from flask import Flask, render_template, request, redirect, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gradcam import make_gradcam_heatmap, overlay_heatmap
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = "crop_secret"

# ---------------- MODELS ----------------
disease_model = load_model("models/disease_model.h5")
stress_model = joblib.load("models/stress_model.pkl")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("crop.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        plant_name TEXT,
        disease TEXT,
        stress TEXT,
        severity TEXT,
        severity_percent REAL,
        soil REAL,
        humidity REAL,
        temperature REAL,
        ph REAL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()
# ---------------- DISEASE & STRESS INFO ----------------
disease_info = { 
    "Downy Mildew": { 
        "about": "Downy mildew is a serious fungal-like disease that affects the leaves of plants, causing yellow, angular patches on the upper leaf surface and greyish, downy growth underneath.", 
        "cause": "This disease thrives in conditions of high humidity, poor air circulation, and prolonged leaf wetness. Overcrowded plants and frequent overhead watering exacerbate infection.", 
        "symptoms": "Yellow or pale green patches on leaves, which eventually turn brown; white downy growth on the underside of leaves; stunted growth in severe cases.", 
        "solution": "Improve air circulation by proper spacing and pruning; avoid overhead irrigation; apply recommended fungicides such as mancozeb or copper-based sprays; remove and destroy infected leaves." }, 
    "Powdery Mildew": { 
        "about": "Powdery mildew is a fungal disease characterized by white, powder-like fungal growth on leaf surfaces, stems, and buds, affecting plant photosynthesis and growth.", 
        "cause": "Occurs in warm, dry days with cool nights and poor air movement. High nitrogen levels in soil can also promote susceptibility.", 
        "symptoms": "White powdery coating on leaves and stems; distorted or stunted growth; premature leaf drop.", 
        "solution": "Prune infected parts to improve airflow; apply sulfur or potassium bicarbonate fungicides; maintain balanced fertilization; avoid excessive nitrogen." }, 
    "Black Rot": { 
        "about": "Black rot is a bacterial or fungal disease that causes dark, rotting lesions on leaves, stems, and fruits, leading to significant yield loss.", 
        "cause": "Spread is facilitated by wet and humid conditions, poor sanitation, and use of infected tools.", 
        "symptoms": "Black or brown circular spots on leaves; lesions on stems and fruits; premature fruit drop.", 
        "solution": "Use sterilized tools; remove and destroy infected plant parts; apply copper-based sprays; ensure proper drainage and reduce leaf wetness." }, 
    "ESCA": { 
        "about": "ESCA, also called grapevine trunk disease, is a fungal infection that affects woody tissues of grapevines, reducing productivity and eventually killing the plant.", 
        "cause": "Caused by wood-decaying fungi entering through pruning wounds or trunk injuries; aggravated by water stress and poor vineyard hygiene.", 
        "symptoms": "Discolored wood, necrotic lesions, leaf chlorosis, and dieback of shoots; sometimes foliar symptoms appear suddenly.", 
        "solution": "Prune infected branches and disinfect tools; remove severely affected vines; apply fungicides to protect pruning wounds; maintain vineyard sanitation." }, 
    "Leaf Blight": { 
        "about": "Leaf blight is a fungal disease that rapidly causes browning and death of leaves, reducing photosynthesis and crop yield.", 
        "cause": "Fungal pathogens proliferate under humid conditions, poor crop rotation, and use of susceptible varieties.", 
        "symptoms": "Brown, water-soaked spots on leaves; rapid leaf necrosis; premature defoliation.", 
        "solution": "Use disease-resistant varieties; ensure crop rotation; apply recommended fungicides; remove and destroy affected leaves; improve plant spacing for airflow." }, 
    "Healthy Leaf": {
        "about": "A healthy leaf indicates that the plant is growing under suitable environmental and nutritional conditions. The leaf performs efficient photosynthesis, supporting strong plant growth and high yield.",
        "cause": "Proper irrigation, balanced nutrients, suitable soil pH, adequate sunlight, and absence of pests or diseases help maintain leaf health.",
        "symptoms": "Bright natural green color, firm leaf structure, smooth surface, no spots, no discoloration, no curling or drying at edges.",
        "solution": "Continue current plant care practices, maintain regular watering schedule, monitor nutrient levels, ensure good drainage, and perform routine inspection to prevent disease."
}
} 
        
stress_info = { 
    "Healthy": { 
        "reason": "The plant is in optimal condition with adequate water, nutrients, sunlight, and suitable environmental conditions.", 
        "solution": "Maintain current care practices; continue regular monitoring and preventive measures." }, 
    "Stress": { "reason": "The plant is experiencing suboptimal conditions such as drought, waterlogging, nutrient deficiency, pH imbalance, or extreme temperature stress.", 
        "symptoms": "Wilting, yellowing, leaf curl, stunted growth, and reduced yield.", 
        "solution": "Adjust irrigation to maintain proper soil moisture; test and correct soil pH; provide balanced nutrients; protect plants from extreme weather; monitor closely for early signs of disease." } }

# ---------------- HELPERS ----------------
def predict_disease(img_path):
    classes = ["Downy Mildew","Powdery Mildew","Black Rot","ESCA","Leaf Blight","Healthy Leaf"]
    img = Image.open(img_path).resize((224,224))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = disease_model.predict(arr)
    idx = int(np.argmax(pred))
    if idx >= len(classes):   # safety fix
        idx = len(classes)-1
    return classes[idx]

def calculate_severity(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15,50,50]), np.array([45,255,255]))
    percent = (np.sum(mask>0)/(mask.shape[0]*mask.shape[1]))*100
    level = "Mild" if percent<10 else "Moderate" if percent<30 else "Severe"
    return level, percent

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        name = request.form["name"]
        password = request.form["password"]

        if name=="admin" and password=="admin123":
            session["username"]="Admin"
            session["role"]="admin"
            return redirect("/admin")

        if os.path.exists("users.csv"):
            df = pd.read_csv("users.csv")
            user = df[(df["Name"]==name)&(df["Password"]==password)]
            if not user.empty:
                session["username"]=name
                session["role"]="user"
                return redirect("/dashboard")

    return render_template("login.html")

    # ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        # Save new user into users.csv
        if not os.path.exists("users.csv"):
            df = pd.DataFrame(columns=["Name", "Email", "Password"])
            df.to_csv("users.csv", index=False)

        df = pd.read_csv("users.csv")

        # Check if user already exists
        if email in df["Email"].values:
            return "User already registered! Please login."

        new_user = pd.DataFrame([[name, email, password]],
                                columns=["Name", "Email", "Password"])
        new_user.to_csv("users.csv", mode="a", header=False, index=False)

        return redirect("/")  # Go back to login page

    return render_template("register.html")

# ---------------- DASHBOARD ----------------
# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if session.get("role") != "user":
        return redirect("/")

    result = {}

    if request.method == "POST":

        file = request.files["leaf"]
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        plant_name = request.form["plant_name"]

        disease = predict_disease(path)

        img_arr = np.expand_dims(
            image.img_to_array(
                image.load_img(path, target_size=(224, 224))
            ) / 255.0,
            0
        )

        heatmap = make_gradcam_heatmap(img_arr, disease_model)
        gradcam_url = overlay_heatmap(path, heatmap)

        sev_level, sev_percent = calculate_severity(path)

        soil = float(request.form["soil"])
        hum = float(request.form["hum"])
        temp = float(request.form["temp"])
        ph = float(request.form["ph"])

        input_df = pd.DataFrame(
            [[soil, hum, temp, ph]],
            columns=["Soil Moisture", "Humidity", "Temperature", "PH level"]
        )

        pred = stress_model.predict(input_df)[0]
        stress_res = "Healthy" if pred == "Hydrated" else "Stress"

        conn = sqlite3.connect("crop.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions
            (username, plant_name, disease, stress, severity, severity_percent,
             soil, humidity, temperature, ph)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            session["username"],
            plant_name,
            disease,
            stress_res,
            sev_level,
            float(sev_percent),
            soil,
            hum,
            temp,
            ph
        ))

        conn.commit()
        conn.close()

        info = disease_info.get(disease, {})
        sinfo = stress_info.get(stress_res, {})

        result = {
            "disease": disease,
            "stress": stress_res,
            "gradcam": gradcam_url,
            "original_img": "/" + path.replace("\\", "/"),
            "severity_level": sev_level,
            "severity_percent": float(sev_percent),
            "disease_about": info.get("about", "No info"),
            "disease_cause": info.get("cause", "No cause"),
            "disease_symptoms": info.get("symptoms", "No symptoms data"),
            "disease_solution": info.get("solution", "No solution"),
            "stress_reason": sinfo.get("reason", "No reason"),
            "stress_solution": sinfo.get("solution", "No solution")
        }

    # ---------------- FETCH USER HISTORY ----------------
    conn = sqlite3.connect("crop.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT plant_name, severity_percent, date
        FROM predictions
        WHERE username = ?
        ORDER BY date ASC
    """, (session["username"],))

    records = cursor.fetchall()
    conn.close()

    timeline = {}

    for plant, severity, date in records:
        if severity is None:
            continue

        date_str = str(date)[:10]
        plant_lower = plant.lower().strip()

        if date_str not in timeline:
            timeline[date_str] = {
                "cucumber": None,
                "grapes": None
            }

        if "cucumber" in plant_lower:
            timeline[date_str]["cucumber"] = float(severity)
        elif "grape" in plant_lower:
            timeline[date_str]["grapes"] = float(severity)

    all_dates = sorted(timeline.keys())

    cucumber_severity = [
        timeline[d]["cucumber"] if timeline[d]["cucumber"] is not None else None
        for d in all_dates
    ]

    grapes_severity = [
        timeline[d]["grapes"] if timeline[d]["grapes"] is not None else None
        for d in all_dates
    ]

    return render_template(
        "user_dashboard.html",
        username=session["username"],
        **result,
        all_dates=all_dates,
        cucumber_severity=cucumber_severity,
        grapes_severity=grapes_severity
    )

# ---------------- USER CHART ----------------
@app.route("/user_chart")
def user_chart():

    if "username" not in session:
        return redirect("/")

    conn = sqlite3.connect("crop.db")

    df = pd.read_sql_query(
        "SELECT date, severity_percent FROM predictions WHERE username = ?",
        conn,
        params=(session["username"],)
    )

    conn.close()

    # If no records
    if df.empty:
        return render_template(
            "user_chart.html",
            username=session["username"],
            dates=[],
            severity=[]
        )

    # Convert and sort date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    severity = df["severity_percent"].tolist()

    return render_template(
        "user_chart.html",
        username=session["username"],
        dates=dates,
        severity=severity
    )

# ---------------- ADMIN ----------------
@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return redirect("/")

    conn = sqlite3.connect("crop.db")
    cursor = conn.cursor()

    # ✅ Total unique users who made predictions
    cursor.execute("SELECT COUNT(DISTINCT username) FROM predictions")
    user_count = cursor.fetchone()[0]

    # ✅ Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    pred_count = cursor.fetchone()[0]

    # ✅ Disease Frequency
    cursor.execute("""
        SELECT disease, COUNT(*)
        FROM predictions
        GROUP BY disease
    """)
    disease_data = cursor.fetchall()
    disease_labels = [row[0] for row in disease_data]
    disease_values = [row[1] for row in disease_data]

    # ✅ Stress Distribution
    cursor.execute("""
        SELECT stress, COUNT(*)
        FROM predictions
        GROUP BY stress
    """)
    stress_data = cursor.fetchall()
    stress_labels = [row[0] for row in stress_data]
    stress_values = [row[1] for row in stress_data]

    # ✅ Daily Prediction Count (Date vs Number of Predictions)
    # ✅ Daily Prediction Count (Timestamp → Only Date)
    cursor.execute("""
        SELECT DATE(date), COUNT(*)
        FROM predictions
        GROUP BY DATE(date)
        ORDER BY DATE(date)
    """)

    daily_data = cursor.fetchall()

    prediction_dates = [row[0] for row in daily_data]
    prediction_counts = [row[1] for row in daily_data]

    return render_template(
        "admin_user_dashboard.html",
        username=session["username"],
        user_count=user_count,
        pred_count=pred_count,
        disease_labels=disease_labels,
        disease_values=disease_values,
        stress_labels=stress_labels,
        stress_values=stress_values,
        prediction_dates=prediction_dates,
        prediction_counts=prediction_counts
    )
# ---------------- ADMIN TABLE ----------------
@app.route("/admin_table")
def table():
    if session.get("role") != "admin":
        return redirect("/")

    conn = sqlite3.connect("crop.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY date DESC", conn)
    conn.close()

    records = df.to_dict(orient="records")

    return render_template(
        "admin_table.html",
        username=session["username"],
        records=records
    )


# ---------------- CHARTS ----------------
@app.route("/charts")
def charts():
    # 🔐 Check login
    if "username" not in session:
        return redirect("/")

    # 📊 Connect to database
    conn = sqlite3.connect("crop.db")
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    # 🛑 If database is empty
    if df.empty:
        return render_template(
            "charts.html",
            username=session["username"],
            disease_labels=[],
            disease_values=[],
            stress_labels=[],
            stress_values=[],
            prediction_dates=[],
            prediction_counts=[]
        )

    # ---------------------------
    # 🦠 Disease Frequency
    # ---------------------------
    disease_counts = df["disease"].value_counts()
    disease_labels = disease_counts.index.tolist()
    disease_values = disease_counts.values.tolist()

    # ---------------------------
    # 🌡 Stress Distribution
    # ---------------------------
    stress_counts = df["stress"].value_counts()
    stress_labels = stress_counts.index.tolist()
    stress_values = stress_counts.values.tolist()

    # ---------------------------
    # 📅 Daily Prediction Count
    # ---------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    daily_counts = (
        df.groupby(df["date"].dt.date)
          .size()
          .reset_index(name="count")
    )

    prediction_dates = daily_counts["date"].astype(str).tolist()
    prediction_counts = daily_counts["count"].tolist()

    # ---------------------------
    # 📤 Send data to template
    # ---------------------------
    return render_template(
        "charts.html",
        username=session["username"],
        disease_labels=disease_labels,
        disease_values=disease_values,
        stress_labels=stress_labels,
        stress_values=stress_values,
        prediction_dates=prediction_dates,
        prediction_counts=prediction_counts
    )

# ----------------PREDICTS_TABLE ---------------
@app.route("/predict", methods=["POST"])
def predict():
    soil = float(request.form["soil"])
    hum = float(request.form["hum"])
    temp = float(request.form["temp"])
    ph = float(request.form["ph"])

    disease = disease_model.predict(img)[0]
    stress = "Healthy" if stress_model.predict([[soil,hum,temp,ph]])[0]==1 else "Stress"
    severity = calculate_severity(img)  # your % logic

    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO predictions(date, soil_moisture, humidity, temperature, ph, disease, severity, stress)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (datetime.now(), soil, hum, temp, ph, disease, severity, stress))
    mysql.connection.commit()

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__=="__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
