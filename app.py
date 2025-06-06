from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Mock user database
users = {
    'admin': 'password',  # username: password
    # Add more users as needed
}

# Load and preprocess data
data = pd.read_csv("C:\\Users\\rajuc\\OneDrive\\Documents\\CODING\\nnn\\coding\\diabetes-dataset.csv")
df = data.copy(deep=True)
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)

x = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]]
y = df["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
KNC = KNeighborsClassifier(n_neighbors=1)
KNC.fit(x_train, y_train)
y_pred_KNC = KNC.predict(x_test)
print("Test set Accuracy: ", accuracy_score(y_test, y_pred_KNC))

def prediction(g, b, s, i, bmi):
    y_pred_KNC = KNC.predict([[g, b, s, i, bmi]])
    return y_pred_KNC

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Authentication logic
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('start_prediction'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        # Handle registration logic (e.g., store user details)
        if username in users:
            flash('Username already exists', 'danger')
        else:
            users[username] = password
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        # Handle password recovery logic
        flash('Password recovery instructions sent', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/start_prediction', methods=['GET', 'POST'])
def start_prediction():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form['name']
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        
        p = prediction(Glucose, BloodPressure, SkinThickness, Insulin, BMI)
        result = 'Positive' if p[0] == 1 else 'Negative'
        
        return render_template('result.html', name=name, Glucose=Glucose, BloodPressure=BloodPressure, 
                               SkinThickness=SkinThickness, Insulin=Insulin, BMI=BMI, result=result)
    return render_template('start_prediction.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
