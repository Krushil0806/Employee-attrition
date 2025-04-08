import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/model(rf).pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Define feature names (must match model training order)
FEATURES = ['Age', 'Department', 'DistanceFromHome', 'EducationField',
            'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobRole',
            'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'OverTime', 'StockOptionLevel',
            'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Department and other dropdown mapping
DEPARTMENT_OPTIONS = {
    "Sales": 0,
    "Research & Development": 1,
    "Human Resources": 2
}

EDUCATION_FIELD_OPTIONS = {
    "Life Sciences": 0,
    "Other": 1,
    "Medical": 2,
    "Marketing": 3,
    "Technical Degree": 4,
    "Human Resources": 5
}

JOB_ROLE_OPTIONS = {
    "Sales Executive": 0,
    "Research Scientist": 1,
    "Laboratory Technician": 2,
    "Manufacturing Director": 3,
    "Healthcare Representative": 4,
    "Manager": 5,
    "Sales Representative": 6,
    "Research Director": 7,
    "Human Resources": 8
}

MARITAL_STATUS_OPTIONS = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

OVERTIME_OPTIONS = {
    "No": 0,
    "Yes": 1
}

# Main function
def main():
    st.title("Employee Attrition Prediction")
    
    # Create a form for input
    with st.form("prediction_form"):
        # Personal Information Section
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, value=30)
            marital_status = st.selectbox("Marital Status", options=list(MARITAL_STATUS_OPTIONS.keys()))
            education_field = st.selectbox("Education Field", options=list(EDUCATION_FIELD_OPTIONS.keys()))
            
        with col2:
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0, value=5)
            monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)
            monthly_rate = st.number_input("Monthly Rate", min_value=0, value=10000)
        
        # Job Details Section
        st.subheader("Job Details")
        col3, col4 = st.columns(2)
        
        with col3:
            department = st.selectbox("Department", options=list(DEPARTMENT_OPTIONS.keys()))
            job_role = st.selectbox("Job Role", options=list(JOB_ROLE_OPTIONS.keys()))
            job_level = st.slider("Job Level", min_value=1, max_value=5, value=2)
            
        with col4:
            overtime = st.selectbox("Overtime", options=list(OVERTIME_OPTIONS.keys()))
            stock_option_level = st.slider("Stock Option Level", min_value=0, max_value=3, value=0)
            num_companies_worked = st.slider("Number of Companies Worked", min_value=0, max_value=9, value=2)
        
        # Satisfaction Section
        st.subheader("Satisfaction & Involvement")
        col5, col6 = st.columns(2)
        
        with col5:
            env_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4, value=2)
            job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=2)
            
        with col6:
            job_involvement = st.slider("Job Involvement", min_value=1, max_value=4, value=2)
        
        # Experience Section
        st.subheader("Work Experience")
        col7, col8 = st.columns(2)
        
        with col7:
            total_working_years = st.number_input("Total Working Years", min_value=0, value=5)
            years_at_company = st.number_input("Years at Company", min_value=0, value=3)
            years_in_current_role = st.number_input("Years in Current Role", min_value=0, value=2)
            
        with col8:
            years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
            years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, value=2)
        
        # Submit button
        submit_button = st.form_submit_button("Predict Attrition")
        
    # Make prediction when form is submitted
    if submit_button:
        # Map selections to numeric values
        department_value = DEPARTMENT_OPTIONS[department]
        education_field_value = EDUCATION_FIELD_OPTIONS[education_field]
        job_role_value = JOB_ROLE_OPTIONS[job_role]
        marital_status_value = MARITAL_STATUS_OPTIONS[marital_status]
        overtime_value = OVERTIME_OPTIONS[overtime]
        
        # Create input array
        user_input = [
            age, department_value, distance_from_home, education_field_value,
            env_satisfaction, job_involvement, job_level, job_role_value,
            job_satisfaction, marital_status_value, monthly_income, monthly_rate,
            num_companies_worked, overtime_value, stock_option_level,
            total_working_years, years_at_company, years_in_current_role,
            years_since_last_promotion, years_with_curr_manager
        ]
        
        if model is not None:
            # Make prediction
            input_array = np.array(user_input).reshape(1, -1)
            try:
                prediction = model.predict(input_array)[0]
                probability = model.predict_proba(input_array)[0]
                
                # Display result
                st.markdown("---")
                
                if prediction == 1:
                    st.error("### Prediction: Employee is likely to leave")
                    likelihood = probability[1] * 100
                else:
                    st.success("### Prediction: Employee is likely to stay")
                    likelihood = probability[0] * 100
                
                # Display probability
                st.write(f"Confidence: {likelihood:.1f}%")
                
                # Display factors that might affect attrition
                st.subheader("Key Factors to Consider:")
                
                factors = []
                if overtime_value == 1:
                    factors.append("⚠️ Overtime might be causing stress")
                if job_satisfaction < 3:
                    factors.append("⚠️ Job satisfaction is low")
                if distance_from_home > 10:
                    factors.append("⚠️ Long commute distance")
                if years_since_last_promotion > 3:
                    factors.append("⚠️ No recent promotion")
                if env_satisfaction < 3:
                    factors.append("⚠️ Low environment satisfaction")
                if monthly_income < 3000:
                    factors.append("⚠️ Relatively low income")
                
                if factors:
                    for factor in factors:
                        st.write(factor)
                else:
                    st.write("No major risk factors identified.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
