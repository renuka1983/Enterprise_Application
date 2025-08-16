import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def prepare_features_for_ml(df):
    """Prepare features for machine learning models."""
    
    # Create a copy to avoid modifying original data
    df_ml = df.copy()
    
    # Encode categorical variables
    le_dept = LabelEncoder()
    le_job = LabelEncoder()
    
    df_ml['Department_Encoded'] = le_dept.fit_transform(df_ml['Department'])
    df_ml['JobRole_Encoded'] = le_job.fit_transform(df_ml['JobRole'])
    
    # Select features for ML
    feature_cols = ['Age', 'Department_Encoded', 'JobRole_Encoded', 'Salary', 'Tenure', 
                   'PromotionLast2Years', 'TrainingHours', 'PerformanceRating', 'SatisfactionScore']
    
    X = df_ml[feature_cols]
    y = df_ml['Attrition']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, le_dept, le_job

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression model."""
    
    # Train model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'Feature': ['Age', 'Department', 'JobRole', 'Salary', 'Tenure', 
                   'PromotionLast2Years', 'TrainingHours', 'PerformanceRating', 'SatisfactionScore'],
        'Importance': np.abs(lr_model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    return lr_model, y_pred, y_pred_proba, accuracy, roc_auc, feature_importance

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model."""
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': ['Age', 'Department', 'JobRole', 'Salary', 'Tenure', 
                   'PromotionLast2Years', 'TrainingHours', 'PerformanceRating', 'SatisfactionScore'],
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return rf_model, y_pred, y_pred_proba, accuracy, roc_auc, feature_importance

def create_model_comparison(lr_results, rf_results):
    """Create comparison visualization of model performance."""
    
    models = ['Logistic Regression', 'Random Forest']
    accuracies = [lr_results['accuracy'], rf_results['accuracy']]
    roc_aucs = [lr_results['roc_auc'], rf_results['roc_auc']]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Model Accuracy Comparison', 'ROC AUC Comparison'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy comparison
    fig.add_trace(
        go.Bar(x=models, y=accuracies, name='Accuracy', marker_color=['blue', 'green']),
        row=1, col=1
    )
    
    # ROC AUC comparison
    fig.add_trace(
        go.Bar(x=models, y=roc_aucs, name='ROC AUC', marker_color=['red', 'orange']),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title_text="Model Performance Comparison",
        showlegend=False
    )
    
    return fig

def create_feature_importance_comparison(lr_importance, rf_importance):
    """Create feature importance comparison visualization."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Logistic Regression Feature Importance', 'Random Forest Feature Importance'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Logistic Regression importance
    fig.add_trace(
        go.Bar(x=lr_importance['Importance'], y=lr_importance['Feature'], 
               orientation='h', name='LR', marker_color='blue'),
        row=1, col=1
    )
    
    # Random Forest importance
    fig.add_trace(
        go.Bar(x=rf_importance['Importance'], y=rf_importance['Feature'], 
               orientation='h', name='RF', marker_color='green'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        title_text="Feature Importance Comparison",
        showlegend=False
    )
    
    return fig

def create_confusion_matrix_heatmap(y_true, y_pred, model_name):
    """Create confusion matrix heatmap."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Stay', 'Predicted Leave'],
        y=['Actual Stay', 'Actual Leave'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"{model_name} - Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    return fig

def predict_attrition_for_employee(model, scaler, le_dept, le_job, employee_data):
    """Predict attrition for a single employee."""
    
    # Prepare employee data
    features = np.array([
        employee_data['Age'],
        le_dept.transform([employee_data['Department']])[0],
        le_job.transform([employee_data['JobRole']])[0],
        employee_data['Salary'],
        employee_data['Tenure'],
        employee_data['PromotionLast2Years'],
        employee_data['TrainingHours'],
        employee_data['PerformanceRating'],
        employee_data['SatisfactionScore']
    ]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]  # Probability of leaving
    
    return prediction, probability

def generate_retention_recommendations(employee_data, prediction, probability):
    """Generate retention recommendations based on employee data and prediction."""
    
    recommendations = []
    
    if prediction == 1:  # High risk of leaving
        recommendations.append("🚨 **High Attrition Risk Detected**")
        
        if employee_data['SatisfactionScore'] <= 5:
            recommendations.append("💡 **Immediate Action**: Conduct satisfaction survey and address concerns")
        
        if employee_data['PromotionLast2Years'] == 0:
            recommendations.append("📈 **Career Growth**: Discuss promotion opportunities and career path")
        
        if employee_data['TrainingHours'] < 20:
            recommendations.append("🎓 **Development**: Increase training and development opportunities")
        
        if employee_data['Tenure'] < 2:
            recommendations.append("🤝 **Onboarding**: Strengthen onboarding and mentorship programs")
        
        if employee_data['PerformanceRating'] <= 3:
            recommendations.append("📊 **Performance**: Provide performance coaching and support")
    
    else:  # Low risk of leaving
        recommendations.append("✅ **Low Attrition Risk - Employee Likely to Stay**")
        
        if employee_data['SatisfactionScore'] >= 8:
            recommendations.append("🌟 **Recognition**: Consider this employee for recognition programs")
        
        if employee_data['PerformanceRating'] >= 4:
            recommendations.append("🏆 **Leadership**: Identify leadership development opportunities")
        
        if employee_data['TrainingHours'] >= 40:
            recommendations.append("🎯 **Specialization**: Consider advanced training or certifications")
    
    # General recommendations
    if employee_data['Age'] < 30:
        recommendations.append("👥 **Mentorship**: Pair with experienced team members")
    
    if employee_data['Salary'] < 70000:
        recommendations.append("💰 **Compensation**: Review salary competitiveness")
    
    return recommendations
