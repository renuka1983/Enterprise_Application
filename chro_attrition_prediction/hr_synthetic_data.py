import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

def generate_synthetic_hr_data(num_employees=1000, seed=42):
    """Generate synthetic HR dataset with realistic attrition patterns."""
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Define HR parameters
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'IT', 'Legal']
    job_roles = {
        'Engineering': ['Software Engineer', 'Data Scientist', 'DevOps Engineer', 'QA Engineer'],
        'Sales': ['Sales Representative', 'Account Manager', 'Sales Manager'],
        'Marketing': ['Marketing Specialist', 'Content Creator', 'Digital Marketing'],
        'HR': ['HR Specialist', 'Recruiter', 'HR Manager'],
        'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager'],
        'Operations': ['Operations Manager', 'Project Manager', 'Process Analyst'],
        'IT': ['IT Support', 'System Administrator', 'Network Engineer'],
        'Legal': ['Legal Counsel', 'Compliance Officer', 'Contract Manager']
    }
    
    # Generate data
    employee_ids = [f"EMP{i:04d}" for i in range(1, num_employees + 1)]
    ages = np.clip(np.random.normal(35, 10, num_employees), 22, 65).astype(int)
    
    departments_list = []
    job_roles_list = []
    for _ in range(num_employees):
        dept = random.choice(departments)
        departments_list.append(dept)
        job_roles_list.append(random.choice(job_roles[dept]))
    
    # Generate other features
    base_salaries = {'Engineering': 85000, 'Sales': 70000, 'Marketing': 65000, 'HR': 60000,
                     'Finance': 75000, 'Operations': 70000, 'IT': 80000, 'Legal': 90000}
    
    salaries = []
    for dept, age in zip(departments_list, ages):
        base = base_salaries[dept]
        experience_factor = max(1, (age - 22) / 10)
        salary = base * experience_factor * np.random.uniform(0.8, 1.2)
        salaries.append(int(salary))
    
    tenure = np.clip(np.random.exponential(3, num_employees), 0, 20)
    promotion_last_2_years = np.clip(np.random.poisson(0.3, num_employees), 0, 3)
    training_hours = np.clip(np.random.exponential(15, num_employees), 0, 100)
    performance_rating = np.clip(np.random.normal(3.5, 0.8, num_employees), 1, 5).astype(int)
    satisfaction_score = np.clip(np.random.normal(6.5, 1.5, num_employees), 1, 10).astype(int)
    
    # Generate employee feedback
    feedback_templates = [
        "I feel {sentiment} about my role. {detail}",
        "The work environment is {environment}. {specific}",
        "My manager is {management}. {example}"
    ]
    
    employee_feedback = []
    for _ in range(num_employees):
        sentiment = random.choice(['satisfied', 'content', 'frustrated', 'excited', 'concerned'])
        detail = random.choice(['Good team dynamics.', 'Limited growth opportunities.', 'Challenging work.'])
        environment = random.choice(['collaborative', 'competitive', 'stressful', 'supportive'])
        specific = random.choice(['Good team dynamics.', 'High pressure deadlines.', 'Flexible arrangements.'])
        management = random.choice(['supportive', 'demanding', 'hands-off', 'micromanaging'])
        example = random.choice(['Provides clear guidance.', 'Sets high expectations.', 'Gives autonomy.'])
        
        template = random.choice(feedback_templates)
        feedback = template.format(sentiment=sentiment, detail=detail, environment=environment, 
                                specific=specific, management=management, example=example)
        employee_feedback.append(feedback)
    
    # Generate attrition based on multiple factors
    attrition_prob = np.zeros(num_employees)
    
    for i in range(num_employees):
        base_prob = 0.15
        
        # Age factor
        age_factor = 1.5 if ages[i] < 30 else 0.7 if ages[i] > 50 else 1.0
        
        # Tenure factor
        tenure_factor = 1.8 if tenure[i] < 1 else 1.3 if tenure[i] > 15 else 0.8
        
        # Promotion factor
        promotion_factor = 1.4 if promotion_last_2_years[i] == 0 else 0.6
        
        # Performance factor
        performance_factor = 1.6 if performance_rating[i] <= 2 else 0.7 if performance_rating[i] >= 4 else 1.0
        
        # Satisfaction factor
        satisfaction_factor = 2.0 if satisfaction_score[i] <= 3 else 0.5 if satisfaction_score[i] >= 8 else 1.0
        
        # Department factor
        dept_factors = {'Sales': 1.3, 'Engineering': 1.1, 'Marketing': 1.2, 'HR': 0.9,
                       'Finance': 0.8, 'Operations': 1.0, 'IT': 1.1, 'Legal': 0.7}
        dept_factor = dept_factors[departments_list[i]]
        
        attrition_prob[i] = min(base_prob * age_factor * tenure_factor * promotion_factor * 
                               performance_factor * satisfaction_factor * dept_factor, 0.8)
    
    attrition = np.random.binomial(1, attrition_prob)
    
    # Create DataFrame
    data = {
        'EmployeeID': employee_ids,
        'Age': ages,
        'Department': departments_list,
        'JobRole': job_roles_list,
        'Salary': salaries,
        'Tenure': np.round(tenure, 1),
        'PromotionLast2Years': promotion_last_2_years,
        'TrainingHours': np.round(training_hours, 1),
        'PerformanceRating': performance_rating,
        'SatisfactionScore': satisfaction_score,
        'EmployeeFeedback': employee_feedback,
        'Attrition': attrition
    }
    
    return pd.DataFrame(data)

def create_hr_dashboard(df):
    """Create an interactive HR dashboard using Plotly."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Attrition by Department', 'Tenure Distribution', 
                       'Satisfaction vs Attrition', 'Age Distribution by Attrition'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Attrition by Department
    dept_attrition = df.groupby('Department')['Attrition'].agg(['count', 'sum']).reset_index()
    dept_attrition['AttritionRate'] = (dept_attrition['sum'] / dept_attrition['count'] * 100).round(1)
    
    fig.add_trace(
        go.Bar(x=dept_attrition['Department'], y=dept_attrition['AttritionRate'], 
               name='Attrition Rate %', marker_color='red', opacity=0.7),
        row=1, col=1
    )
    
    # 2. Tenure Distribution
    fig.add_trace(
        go.Histogram(x=df['Tenure'], name='Tenure Distribution', 
                    marker_color='blue', opacity=0.7, nbinsx=20),
        row=1, col=2
    )
    
    # 3. Satisfaction vs Attrition
    fig.add_trace(
        go.Box(x=df['Attrition'], y=df['SatisfactionScore'], name='Satisfaction by Attrition',
               marker_color='green', opacity=0.7),
        row=2, col=1
    )
    
    # 4. Age Distribution by Attrition
    fig.add_trace(
        go.Histogram(x=df[df['Attrition'] == 0]['Age'], name='Stayed', 
                    marker_color='blue', opacity=0.7, nbinsx=15),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=df[df['Attrition'] == 1]['Age'], name='Left', 
                    marker_color='red', opacity=0.7, nbinsx=15),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="HR Attrition Dashboard - Key Insights", showlegend=True)
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_attrition_heatmap(df):
    """Create a correlation heatmap for attrition factors."""
    
    numerical_cols = ['Age', 'Salary', 'Tenure', 'PromotionLast2Years', 
                     'TrainingHours', 'PerformanceRating', 'SatisfactionScore']
    
    correlations = [df[col].corr(df['Attrition']) for col in numerical_cols]
    
    fig = go.Figure(data=go.Heatmap(
        z=[correlations],
        x=numerical_cols,
        y=['Attrition'],
        colorscale='RdBu',
        zmid=0,
        text=[[f'{corr:.3f}' for corr in correlations]],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(title="Attrition Correlation Heatmap", xaxis_title="Features", 
                     yaxis_title="Attrition", height=300)
    
    return fig

def save_to_csv(df, filename='hr_attrition_data.csv'):
    """Save the DataFrame to CSV file."""
    csv_data = df.to_csv(index=False)
    return csv_data, filename
