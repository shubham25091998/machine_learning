import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class DataProcessing:
  def __init__(self):
    self.salary_data = self.loadData()
    self.temp = self.salary_data.copy()
    self.outliers = self.outlierDetector(self.salary_data, ['Salary'])
    self.salary_data.drop(self.outliers.index, inplace=True)
    self.featureEngineering()
    self.encodeData()
    self.target = self.salary_data['Salary']
    self.features = self.salary_data.drop(['Salary'], axis = 1)
    self.features.drop(['Age'], axis = 1, inplace = True)
    self.normalizeData()
    self.splitData()
    self.scalar = None

  def featureEngineering(self):
    self.salary_data["Years of Experience"] = self.salary_data["Years of Experience"] **2
    self.salary_data["Experience per age"] = self.salary_data["Years of Experience"]/(self.salary_data["Age"] + 1)
    
  def encodingValueFinder(self):
    self.gender_encoding = {}
    self.job_title_encoding = {}
    self.education_encoding = {}

    for i in range(self.salary_data.shape[0]):
      if self.temp.iloc[i]['Gender'] not in self.gender_encoding:
        self.gender_encoding[self.temp.iloc[i]['Gender']] = self.salary_data.iloc[i]['Gender']

      if self.temp.iloc[i]['Job Title'] not in self.job_title_encoding:
        self.job_title_encoding[self.temp.iloc[i]['Job Title']] = self.salary_data.iloc[i]['Job Title']

      if self.temp.iloc[i]['Education Level'] not in self.education_encoding:
        self.education_encoding[self.temp.iloc[i]['Education Level']] = self.salary_data.iloc[i]['Education Level']

  def loadData(self):
      url = "https://drive.google.com/uc?expost=download&id=1xDgEgrsFdB52EVP6m7PQqU8B7CBJr6kQ"
      response = requests.get(url, allow_redirects = True)
      open('salary_data.csv',"wb").write(response.content)
      return pd.read_csv('salary_data.csv')

  def outlierDetector(self, data, feature):
     for f in feature:
      Q1 = data[f].quantile(0.25)
      Q3 = data[f].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 0.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      outliers = data[(data[f] < lower_bound) | (data[f] > upper_bound)]
      return outliers

  def encodeData(self):
    encoder = LabelEncoder()
    self.salary_data['Gender'] = encoder.fit_transform(self.salary_data['Gender'])
    self.salary_data['Education Level'] = encoder.fit_transform(self.salary_data['Education Level'])
    self.salary_data['Job Title'] = encoder.fit_transform(self.salary_data['Job Title'])

  def normalizeData(self):
    self.scaler = StandardScaler()
    self.scaler_y = StandardScaler()
    self.features = self.scaler.fit_transform(self.features)
    self.target = self.scaler_y.fit_transform(self.target.values.reshape(-1, 1))

  def splitData(self):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target, test_size = 0.2, random_state = 42)

class Neural_Network:
  def __init__(self):
    self.dataHandle = DataProcessing()
    self.dataHandle.encodingValueFinder()
    self.gender_encoding = self.dataHandle.gender_encoding
    self.job_title_encoding = self.dataHandle.job_title_encoding
    self.education_encoding = self.dataHandle.education_encoding
    self.input_layer_size = self.dataHandle.X_train.shape[1]
    self.hidden_layer_size = max(12, int((self.dataHandle.salary_data.shape[1] * (2/3)) + 1)) # select max(8 or (5*2/3)+1) which is going to select 8 for this dataset
    self.output_layer = 1
    self.learning_rate = 0.01

    self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size) * 2
    self.b1 = np.zeros((1, self.hidden_layer_size))
    self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer) * 2
    self.b2 = np.zeros((1, self.output_layer))

    #self.train(self.dataHandle.X_train, self.dataHandle.y_train, epochs=1000, batch_size=32)
    self.train(self.dataHandle.features, self.dataHandle.target, epochs=1000, batch_size=12)

  def findAccuracy(self):
    predictions = self.model.predict(self.dataHandle.X_test)
    accuracy = r2_score(self.dataHandle.y_test, predictions)
    print("accuracy:", accuracy)

  def ReLU(self, z):
    return np.maximum(0, z)

  def forward(self, X):
    self.Z1 = np.dot(X, self.W1) + self.b1
    self.A1 = self.ReLU(self.Z1)
    self.Z2 = np.dot(self.A1, self.W2) + self.b2

    #A2 = Z2 # No activation function required in regression
    return self.Z2

  def backward(self, train_set, target, prediction):
    # m is considered because of batch gradient descent method
    # m = 1 if stocastic gradient descent used

    m = train_set.shape[0]

    # output layer error
    target = target.reshape(-1, 1)
    dZ2 = prediction - target

    # gradient of weight in output layer
    dW2 = np.dot(self.A1.T, dZ2)/m

    # gradient of bias in output layer
    dB2 = np.sum(dZ2, axis=0, keepdims=True)/m

    # error at hidden layer
    dZ1 = np.dot(dZ2, self.W2.T) * (self.Z1 > 0)

    # gradient of weight in hidden layer
    dW1 = np.dot(train_set.T, dZ1)/m

    # gradient of bias in hidden layer
    dB1 = np.sum(dZ1, axis=0, keepdims=True)/m

    self.W1 -= self.learning_rate * dW1
    self.b1 -= self.learning_rate * dB1
    self.W2 -= self.learning_rate * dW2
    self.b2 -= self.learning_rate * dB2

  def train(self, train_set, target, epochs, batch_size):
    for epoch in range(epochs):
      indices = np.random.permutation(train_set.shape[0])
      train_set_shuffled = train_set[indices]
      target_shuffled = target[indices]

      for i in range(0, train_set.shape[0], batch_size):
        train_set_batch = train_set_shuffled[i:i+batch_size]
        target_batch = target_shuffled[i:i+batch_size]

        prediction = self.forward(train_set_batch)
        self.backward(train_set_batch, target_batch, prediction)

      prediction = self.forward(train_set)
      loss = np.mean(np.square(prediction - target))


  def predict(self, test_set):
      return self.forward(test_set)

  def predict_salary(self, age, gender, education, job_title, experience):
    gender_encoded = self.gender_encoding[gender]
    job_title_encoded = self.job_title_encoding[job_title]
    education_encoded = self.education_encoding[education]

    experience = experience ** 2
    experience_per_age = experience / age

    test_set = pd.DataFrame([[gender_encoded, education_encoded, job_title_encoded, experience, experience_per_age]], columns=["Gender","Education Level","Job Title","Years of Experience", "Experience per age"])

    test_set = self.dataHandle.scaler.transform(test_set)

    prediction = self.forward(test_set)
    prediction = self.dataHandle.scaler_y.inverse_transform(prediction)
    return prediction[0][0]

class Linear_Regression_Model:
    def __init__(self):
        self.dataHandle = DataProcessing()
        self.dataHandle.encodingValueFinder()
        self.gender_encoding = self.dataHandle.gender_encoding
        self.job_title_encoding = self.dataHandle.job_title_encoding
        self.education_encoding = self.dataHandle.education_encoding
        
        self.model = LinearRegression()
        self.model.fit(self.dataHandle.features, self.dataHandle.target)

        predictions = self.model.predict(self.dataHandle.features)
        
    def findAccuracy(self):
        predictions = self.model.predict(self.dataHandle.X_test)
        accuracy = r2_score(self.dataHandle.y_test, predictions)
        print("accuracy:", accuracy)

    def predict_salary(self, age, gender, education, job_title, experience):
        gender_encoded = self.gender_encoding[gender]
        job_title_encoded = self.job_title_encoding[job_title]
        education_encoded = self.education_encoding[education]

        experience = experience ** 2
        experience_per_age = experience / age

        test_set = pd.DataFrame([[gender_encoded, education_encoded, job_title_encoded, experience, experience_per_age]],
                                columns=["Gender", "Education Level", "Job Title", "Years of Experience","Experience per age"])

        test_set = self.dataHandle.scaler.transform(test_set)
        prediction = self.model.predict(test_set)
        prediction = self.dataHandle.scaler_y.inverse_transform(prediction.reshape(-1, 1))
        return prediction[0][0]

if "network" not in st.session_state:

  #Train neural network
  network = Neural_Network()

  # Train regression model
  regression = Linear_Regression_Model()

  st.session_state["network"] = network
  st.session_state["regression"] = regression

# salary_prediction_app.py
st.title("💼 Salary Predictor App")

st.subheader("Enter your details below:")

# Input: Age
age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)

# Input: Gender
gender = st.selectbox("Gender", options=st.session_state["network"].gender_encoding.keys())

# Input: Education Level
education = st.selectbox("Education Level", options= st.session_state["network"].education_encoding.keys())

# Input: Job Title
job_title = st.selectbox("Job Title", options= st.session_state["network"].job_title_encoding.keys())

# Input: Years of Experience
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# predict the salary using neural network and linear regression
if st.button("Predict Salary 💰"):

    if (int(experience) >= age):
      st.warning("Experience cannot be greater than age. You must be kidding 🤭")
    elif abs(int(experience) - age) < 14 :
      st.warning("Difference between age and experience is below 14 years. You must be kidding 🤭")
    else:
      # Calculate
      nn_salary = st.session_state["network"].predict_salary(age, gender, education, job_title, experience)
      lr_salary = st.session_state["regression"].predict_salary(age, gender, education, job_title, experience)

      salary = (nn_salary + lr_salary)/2
      st.success(f"🤑 Your Estimated Salary: ${salary:,.2f}")

