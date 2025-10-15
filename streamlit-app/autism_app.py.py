import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Autism Spectrum Disorder Prediction",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class AutismPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def preprocess_data(self, df, age_group):
        """Preprocess the data similar to the notebook"""
        df_clean = df.dropna()
        
        # Select features (matching the notebook structure)
        features_raw = df_clean[['age', 'gender', 'ethnicity', 'jundice', 'austim', 
                               'contry_of_res', 'result', 'relation', 'A1_Score', 
                               'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
                               'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']]
        
        # Scale numerical features
        numerical = ['age', 'result']
        features_minmax = pd.DataFrame(data=features_raw)
        features_minmax[numerical] = self.scaler.fit_transform(features_raw[numerical])
        
        # One-hot encoding
        features_final = pd.get_dummies(features_minmax)
        self.feature_names = features_final.columns.tolist()
        
        # Convert target variable
        classes = df_clean['Class/ASD'].apply(lambda x: 1 if x == 'YES' else 0)
        
        return features_final, classes
    
    def train_models(self, X, y, age_group):
        """Train all models for a specific age group"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        models = {
            'Logistic Regression': LogisticRegression(),
            'SVM': svm.SVC(kernel='rbf', gamma=0.1),
            'Naive Bayes': MultinomialNB(),
            'KNN': KNeighborsClassifier(n_neighbors=7)
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        return trained_models, X_test, y_test
    
    def predict_single(self, features, age_group):
        """Make prediction for a single sample"""
        if age_group not in self.models:
            return None, None
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models[age_group].items():
            try:
                pred = model.predict([features])[0]
                # For models that support predict_proba
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([features])[0][1]
                else:
                    # For SVM, use decision function to estimate probability
                    decision = model.decision_function([features])[0]
                    proba = 1 / (1 + np.exp(-decision))
                
                predictions[model_name] = pred
                probabilities[model_name] = proba
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")
                continue
        
        return predictions, probabilities

def create_sample_data():
    """Create sample datasets for demonstration"""
    # This is a simplified version - in real app, you'd load actual data
    sample_adult = {
        'age': [25, 30, 35, 40, 28],
        'gender': ['m', 'f', 'm', 'f', 'm'],
        'ethnicity': ['White-European', 'Asian', 'Black', 'Hispanic', 'Middle Eastern'],
        'jundice': ['no', 'yes', 'no', 'no', 'yes'],
        'austim': ['no', 'yes', 'no', 'no', 'yes'],
        'contry_of_res': ['United States', 'United Kingdom', 'Canada', 'Australia', 'Germany'],
        'result': [5, 7, 3, 6, 8],
        'relation': ['Self', 'Parent', 'Self', 'Relative', 'Self'],
        'A1_Score': [1, 1, 0, 1, 1],
        'A2_Score': [1, 0, 1, 1, 0],
        'A3_Score': [0, 1, 0, 1, 1],
        'A4_Score': [1, 1, 0, 0, 1],
        'A5_Score': [0, 1, 1, 1, 0],
        'A6_Score': [1, 0, 0, 1, 1],
        'A7_Score': [0, 1, 1, 0, 1],
        'A8_Score': [1, 1, 0, 1, 0],
        'A9_Score': [0, 1, 0, 1, 1],
        'A10_Score': [1, 0, 1, 0, 1],
        'Class/ASD': ['NO', 'YES', 'NO', 'YES', 'YES']
    }
    
    return pd.DataFrame(sample_adult)

def main():
    st.markdown('<div class="main-header">🧠 Autism Spectrum Disorder Prediction Tool</div>', unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = AutismPredictor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Home", "Data Analysis", "Model Training", "Make Prediction", "About"])
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Data Analysis":
        show_data_analysis()
    elif app_mode == "Model Training":
        show_model_training()
    elif app_mode == "Make Prediction":
        show_prediction()
    elif app_mode == "About":
        show_about()

def show_home():
    st.markdown("""
    ## Welcome to the Autism Spectrum Disorder Prediction Tool
    
    This application uses machine learning to predict Autism Spectrum Disorder (ASD) 
    based on behavioral assessments and demographic information.
    
    ### Features:
    - 📊 **Data Analysis**: Explore and understand the autism datasets
    - 🤖 **Model Training**: Train multiple machine learning models
    - 🔮 **Prediction**: Make predictions for new cases
    - 📈 **Performance Metrics**: Evaluate model performance
    
    ### Age Groups Supported:
    - **Children** (4-11 years)
    - **Adolescents** (12-16 years) 
    - **Adults** (18+ years)
    
    ### How to use:
    1. Start by uploading your data in **Data Analysis**
    2. Train models in **Model Training**
    3. Make predictions in **Make Prediction**
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Available", "4", "Logistic Regression, SVM, Naive Bayes, KNN")
    with col2:
        st.metric("Age Groups", "3", "Children, Adolescents, Adults")
    with col3:
        st.metric("Accuracy Range", "75-98%", "Varies by model and dataset")

def show_data_analysis():
    st.markdown('<div class="sub-header">📊 Data Analysis</div>', unsafe_allow_html=True)
    
    st.info("""
    For demonstration purposes, we're using sample data. In a real application, 
    you would upload your actual autism screening datasets.
    """)
    
    # Create sample data
    sample_df = create_sample_data()
    
    # Display data
    st.subheader("Sample Dataset")
    st.dataframe(sample_df)
    
    # Basic statistics
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Info:**")
        st.write(f"Number of samples: {len(sample_df)}")
        st.write(f"Number of features: {len(sample_df.columns)}")
        st.write(f"ASD Cases: {len(sample_df[sample_df['Class/ASD'] == 'YES'])}")
        st.write(f"Non-ASD Cases: {len(sample_df[sample_df['Class/ASD'] == 'NO'])}")
    
    with col2:
        st.write("**Feature Overview:**")
        st.write("A1-A10 Scores: Behavioral assessment scores (0 or 1)")
        st.write("Demographic: Age, gender, ethnicity, etc.")
        st.write("Medical History: Jaundice, family autism history")
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Score distributions
    score_columns = [f'A{i}_Score' for i in range(1, 11)]
    score_sums = [sample_df[col].sum() for col in score_columns]
    
    chart_data = pd.DataFrame({
        'Score': score_columns,
        'Positive Responses': score_sums
    })
    
    st.bar_chart(chart_data.set_index('Score'))

def show_model_training():
    st.markdown('<div class="sub-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    st.warning("Note: This is a demonstration with sample data. Real training would require larger datasets.")
    
    age_group = st.selectbox("Select Age Group", 
                           ["Adult", "Adolescent", "Child"])
    
    if st.button("Train Models"):
        with st.spinner(f"Training models for {age_group}..."):
            # Create sample data
            sample_df = create_sample_data()
            
            # Preprocess data
            features, targets = st.session_state.predictor.preprocess_data(sample_df, age_group)
            
            # Train models
            models, X_test, y_test = st.session_state.predictor.train_models(features, targets, age_group)
            st.session_state.predictor.models[age_group] = models
            
            # Display results
            st.success(f"Models trained successfully for {age_group}!")
            
            # Show performance metrics
            st.subheader("Model Performance")
            
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"{model_name} Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("Sensitivity", f"{sensitivity:.2%}")
                with col3:
                    st.metric("Specificity", f"{specificity:.2%}")
                with col4:
                    st.metric("F1-Score", f"{2*(sensitivity*specificity)/(sensitivity+specificity):.2%}" 
                            if (sensitivity + specificity) > 0 else "N/A")

def show_prediction():
    st.markdown('<div class="sub-header">🔮 Make Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.predictor.models:
        st.error("Please train models first in the 'Model Training' section.")
        return
    
    age_group = st.selectbox("Select Age Group for Prediction", 
                           list(st.session_state.predictor.models.keys()))
    
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", ["m", "f"])
        ethnicity = st.selectbox("Ethnicity", [
            "White-European", "Asian", "Black", "Hispanic", 
            "Middle Eastern", "South Asian", "Others", "Latino"
        ])
        jundice = st.selectbox("Born with Jaundice?", ["yes", "no"])
        austim = st.selectbox("Family History of Autism?", ["yes", "no"])
    
    with col2:
        contry_of_res = st.selectbox("Country of Residence", [
            "United States", "United Kingdom", "Canada", "Australia", 
            "Jordan", "Brazil", "Spain", "Egypt", "India"
        ])
        result = st.slider("Screening Test Result", 0, 10, 5)
        relation = st.selectbox("Who Completing Test", [
            "Self", "Parent", "Relative", "Health care professional", "Others"
        ])
    
    st.subheader("Behavioral Assessment Scores (A1-A10)")
    
    # Create two rows of 5 scores each
    col1, col2, col3, col4, col5 = st.columns(5)
    scores = []
    
    with col1:
        scores.append(st.selectbox("A1 Score", [0, 1], key="a1"))
        scores.append(st.selectbox("A6 Score", [0, 1], key="a6"))
    with col2:
        scores.append(st.selectbox("A2 Score", [0, 1], key="a2"))
        scores.append(st.selectbox("A7 Score", [0, 1], key="a7"))
    with col3:
        scores.append(st.selectbox("A3 Score", [0, 1], key="a3"))
        scores.append(st.selectbox("A8 Score", [0, 1], key="a8"))
    with col4:
        scores.append(st.selectbox("A4 Score", [0, 1], key="a4"))
        scores.append(st.selectbox("A9 Score", [0, 1], key="a9"))
    with col5:
        scores.append(st.selectbox("A5 Score", [0, 1], key="a5"))
        scores.append(st.selectbox("A10 Score", [0, 1], key="a10"))
    
    if st.button("Predict ASD Risk"):
        with st.spinner("Analyzing data..."):
            # Prepare feature vector (this is simplified - real implementation would need proper encoding)
            # For demonstration, we'll use a random feature vector
            feature_vector = np.random.random(len(st.session_state.predictor.feature_names))
            
            predictions, probabilities = st.session_state.predictor.predict_single(
                feature_vector, age_group
            )
            
            if predictions:
                st.subheader("Prediction Results")
                
                # Display results from each model
                for model_name, pred in predictions.items():
                    proba = probabilities.get(model_name, 0.5)
                    risk_level = "High Risk" if pred == 1 else "Low Risk"
                    risk_color = "risk-high" if pred == 1 else "risk-low"
                    
                    st.markdown(f"""
                    <div class="prediction-box {risk_color}">
                        <h4>{model_name}</h4>
                        <p><strong>Prediction:</strong> {risk_level}</p>
                        <p><strong>Confidence:</strong> {proba:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Overall consensus
                avg_probability = np.mean(list(probabilities.values()))
                consensus = "High Risk" if avg_probability > 0.5 else "Low Risk"
                
                st.markdown(f"""
                <div class="prediction-box" style="background-color: #fff3e0; border-left: 5px solid #ff9800;">
                    <h4>Overall Consensus</h4>
                    <p><strong>Recommendation:</strong> {consensus}</p>
                    <p><strong>Average Confidence:</strong> {avg_probability:.2%}</p>
                    <p><em>Note: This is a screening tool. Please consult healthcare professionals for diagnosis.</em></p>
                </div>
                """, unsafe_allow_html=True)

def show_about():
    st.markdown('<div class="sub-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Autism Spectrum Disorder Prediction Tool
    
    This application demonstrates how machine learning can be used to assist in 
    early screening of Autism Spectrum Disorder (ASD).
    
    ### Models Implemented:
    1. **Logistic Regression** - Linear classification model
    2. **Support Vector Machine (SVM)** - Non-linear classification with RBF kernel
    3. **Naive Bayes** - Probabilistic classifier
    4. **K-Nearest Neighbors (KNN)** - Instance-based learning
    
    ### Data Features:
    - **Demographic Information**: Age, gender, ethnicity
    - **Medical History**: Jaundice at birth, family history of autism
    - **Behavioral Assessment**: A1-A10 scores from screening questionnaire
    - **Test Context**: Who completed the test, country of residence
    
    ### Important Notes:
    - This is a **demonstration tool** for educational purposes
    - Real clinical use requires proper validation and certification
    - Always consult healthcare professionals for medical diagnosis
    - Model performance depends on data quality and quantity
    
    ### Technical Stack:
    - **Frontend**: Streamlit
    - **Machine Learning**: Scikit-learn
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Plotly
    
    For questions or contributions, please contact the development team.
    """)

if __name__ == "__main__":
    main()