import streamlit as st
import pandas as pd
import pickle
import base64
import numpy as np

def get_binary_file_downloader_html(df):
    """
    Generates a link to download the prediction DataFrame as a CSV file.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href


# --- Page Configuration and Layout Configuration ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀Heart Disease Prediction Web App")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])
st.markdown("---")


# ==========================================
# --- TAB 1: INDIVIDUAL PATIENT ASSESSMENT ---
# ==========================================
with tab1:
    st.write("Enter the patient's clinical data below to check for heart disease risk.")
    
    # User Input Collection Form
    Age = st.number_input("Age(years)", min_value=0, max_value=150)
    Sex = st.selectbox("Sex,(1=Male,0=Female)", [1, 0])
    ChestPainType = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=220, value=50)
    Cholesterol = st.number_input("Serum Cholestrol (mg/dl)", min_value=0, max_value=600, value=150)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True,0 = False)", [0, 1])
    RestingECG = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    ExerciseAngina = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    Oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    ST_Slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])

    st.markdown("---")

    # Map inputs directly into a structure matching the training data feature schema
    input_data = pd.DataFrame({
        "Age": [Age],
        "Sex": [Sex],
        "ChestPainType": [ChestPainType],
        "RestingBP": [RestingBP],
        "Cholesterol": [Cholesterol],
        "FastingBS": [FastingBS],
        "RestingECG": [RestingECG],
        "MaxHR": [MaxHR],
        "ExerciseAngina": [ExerciseAngina],
        "Oldpeak": [Oldpeak],
        "ST_Slope": [ST_Slope],
    })

    algorithm_names = ['Decision Tree', 'Logistic Regression', 'Support Vector Machine']
    model_names = ['DecisionT.pkl', 'logisticR.pkl', 'SVM.pkl']
    predictions = []

    def predict_heart_disease(data):
        """
        Iterates over the trained classification model binaries to pull individual predictions.
        """
        for modelname in model_names:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    # Trigger model inference routines upon form validation
    if st.button("Submit"):
        st.subheader('Results....')
        st.markdown('------------------------------------------')

        result = predict_heart_disease(input_data)

        # Loop through diagnostic inferences and render outcomes dynamically
        for i in range(len(result)):
            st.subheader(algorithm_names[i])

            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
                st.markdown('------------------------------------')

# ==========================================
# --- TAB 2: BATCH DATA PROCESSING (CSV) ---
# ==========================================
with tab2:
    st.title("Upload CSV File")
    st.subheader('Instruction to note before uploading the file:')
    st.info("""
        **1. General File Constraints:**
        * No NaN or missing values are permitted in any rows.
        * File must contain exactly 11 specific features plus an optional true tracking column.
        * Check feature spellings carefully.
        
        **2. Strict Schema Ordering:**
        The file columns must explicitly run in this sequence:
        `('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope')`

        **3. Expected Variable Value Conventions:**
        * **Age:** Patient age in years (Numeric)
        * **Sex:** Patient gender [M: Male, F: Female]
        * **ChestPainType:** Diagnostic type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
        * **RestingBP:** Resting blood pressure metric measured in mm Hg
        * **Cholesterol:** Serum cholesterol level monitored in mg/dl
        * **FastingBS:** Fasting blood sugar scale [1: if FastingBS > 120 mg/dl, 0: otherwise]
        * **RestingECG:** Resting electrocardiogram measurements [Normal, ST, LVH]
        * **MaxHR:** Maximum heart rate achieved during clinical testing
        * **ExerciseAngina:** Exercise-induced angina manifestations [Y: Yes, N: No]
        * **Oldpeak:** Numeric value tracking ST segment depression levels
        * **ST_Slope:** The observed slope pattern of peak exercise ST segments [Up: Upsloping, Flat: Flat, Down: Downsloping]
         """)

    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
        
        # Create copy for feature preprocessing
        input_data = raw_data.copy()
        
        # Predefined mapping rules for categorical features
        mapping_rules = {
            'Sex': {'M': 1, 'F': 0},
            'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
            'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
            'ExerciseAngina': {'Y': 1, 'N': 0},
            'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
        }

        # Encode text columns using the mapping rules
        for col in input_data.columns:
            if col in mapping_rules and input_data[col].dtype == 'object':
                input_data[col] = input_data[col].map(mapping_rules[col])

        # Load the trained model binary
        model = pickle.load(open('logisticR.pkl', 'rb'))
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        
        if set(expected_columns).issubset(input_data.columns):
            raw_data['Prediction LR'] = ''

            # Generate predictions row by row
            for i in range(len(input_data)):
                # Extract features in the precise order expected by the model
                arr = input_data[expected_columns].iloc[i].values
                
                # Append predictions to the output dataframe
                raw_data.at[i, 'Prediction LR'] = int(model.predict([arr])[0])

            # Export processed predictions to CSV file
            raw_data.to_csv('PredictedHeartLR.csv', index=False)

            st.subheader("Predictions:")
            st.write(raw_data)
            st.markdown(get_binary_file_downloader_html(raw_data), unsafe_allow_html=True)

        else:
            st.warning("Please make sure the uploaded CSV file has the correct columns. ")

    else:
        st.info("Upload a CSV file to get predictions.")
# ==========================================
# --- TAB 3: MODEL CONFIGURATION PERFORMANCE ---
# ==========================================
with tab3:
    import matplotlib.pyplot as plt

    # Static baseline performance parameters captured from training pipelines
    data = {
        'Decision Tree': 80.97,
        'Logistic Regression': 85.86,
        'Support Vector Machine': 84.22
    }
    models = list(data.keys())
    accuracies = list(data.values())

    # Build the optimization visualization plot
    fig, ax = plt.subplots()
    
    bars = ax.bar(models, accuracies, color=['#4285F4', '#EA4335', '#FBBC05'])
    
    ax.set_ylabel('Accuracy Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 100)

    # Attach specific percentage values above coordinate distributions
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Stream rendering pipeline out safely to the dashboard canvas
    st.pyplot(fig)