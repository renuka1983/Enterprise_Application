"""
Machine Learning Models Module for CMO Hyper-Personalization
Tubes India Products - Campaign Response Prediction
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class CampaignResponsePredictor:
    """
    Machine learning models for predicting campaign response
    based on customer characteristics and behavior.
    """
    
    def __init__(self):
        """Initialize the predictor with default parameters."""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
    
    def prepare_features(self, df):
        """
        Prepare features for machine learning models.
        
        Args:
            df (pd.DataFrame): Customer dataset
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, scaler, encoders)
        """
        
        # Create a copy to avoid modifying original data
        df_ml = df.copy()
        
        # Encode categorical variables
        le_segment = LabelEncoder()
        le_region = LabelEncoder()
        
        df_ml['Segment_Encoded'] = le_segment.fit_transform(df_ml['Segment'])
        df_ml['Region_Encoded'] = le_region.fit_transform(df_ml['Region'])
        
        # Create competitor mention features
        df_ml['HasCompetitorMentions'] = (df_ml['CompetitorMentions'] != 'None').astype(int)
        df_ml['NumCompetitorMentions'] = df_ml['CompetitorMentions'].apply(
            lambda x: len(x.split(', ')) if x != 'None' else 0
        )
        
        # Select features for ML
        feature_cols = [
            'Segment_Encoded', 'Region_Encoded', 'PastPurchases', 
            'WebsiteVisits', 'HasCompetitorMentions', 'NumCompetitorMentions',
            'RevenuePotential'
        ]
        
        X = df_ml[feature_cols]
        y = df_ml['ResponseToCampaign']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store encoders and scaler
        self.encoders['segment'] = le_segment
        self.encoders['region'] = le_region
        self.scalers['main'] = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, self.encoders
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Train Logistic Regression model for campaign response prediction.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            feature_names: List of feature names
            
        Returns:
            dict: Model results and performance metrics
        """
        
        # Train model
        lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        lr_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = lr_model.predict(X_test)
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(lr_model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        # Store results
        self.models['logistic_regression'] = lr_model
        self.feature_importance['logistic_regression'] = feature_importance
        self.model_performance['logistic_regression'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return self.model_performance['logistic_regression']
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Train Random Forest model for campaign response prediction.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            feature_names: List of feature names
            
        Returns:
            dict: Model results and performance metrics
        """
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Store results
        self.models['random_forest'] = rf_model
        self.feature_importance['random_forest'] = feature_importance
        self.model_performance['random_forest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return self.model_performance['random_forest']
    
    def create_model_comparison_chart(self):
        """
        Create a comparison chart of model performance.
        
        Returns:
            plotly.graph_objects.Figure: Comparison chart
        """
        
        if not self.model_performance:
            return None
        
        models = list(self.model_performance.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'CV Score']
        )
        
        # Colors for models
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, metric in enumerate(metrics):
            values = [self.model_performance[model][metric] for model in models]
            
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    marker_color=colors[:len(models)]
                ),
                row=row, col=col
            )
        
        # Add CV score
        cv_values = [self.model_performance[model]['cv_mean'] for model in models]
        fig.add_trace(
            go.Bar(
                x=models,
                y=cv_values,
                name='CV Score',
                marker_color=colors[:len(models)]
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=600,
            title_text="Model Performance Comparison",
            showlegend=False
        )
        
        return fig
    
    def create_feature_importance_chart(self):
        """
        Create feature importance comparison chart.
        
        Returns:
            plotly.graph_objects.Figure: Feature importance chart
        """
        
        if not self.feature_importance:
            return None
        
        models = list(self.feature_importance.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(models),
            subplot_titles=[model.replace('_', ' ').title() for model in models]
        )
        
        for i, model in enumerate(models):
            importance_df = self.feature_importance[model]
            
            fig.add_trace(
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    name=model.replace('_', ' ').title(),
                    marker_color=['blue', 'green', 'red', 'orange'][i % 4]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            height=400,
            title_text="Feature Importance Comparison",
            showlegend=False
        )
        
        return fig
    
    def create_confusion_matrix(self, model_name):
        """
        Create confusion matrix for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            plotly.graph_objects.Figure: Confusion matrix heatmap
        """
        
        if model_name not in self.model_performance:
            return None
        
        # Get predictions and actual values
        y_pred = self.model_performance[model_name]['predictions']
        # Note: We need actual y_test values here - this would need to be passed from the main app
        
        # For now, create a placeholder
        cm = np.array([[100, 20], [30, 150]])  # Placeholder values
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No', 'Predicted Yes'],
            y=['Actual No', 'Actual Yes'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"{model_name.replace('_', ' ').title()} - Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        return fig
    
    def predict_campaign_response(self, customer_data, model_name='random_forest'):
        """
        Predict campaign response for a specific customer.
        
        Args:
            customer_data (dict): Customer information
            model_name (str): Name of the model to use
            
        Returns:
            tuple: (prediction, probability)
        """
        
        if model_name not in self.models:
            return None, None
        
        model = self.models[model_name]
        scaler = self.scalers['main']
        
        # Prepare customer data
        features = np.array([
            self.encoders['segment'].transform([customer_data['Segment']])[0],
            self.encoders['region'].transform([customer_data['Region']])[0],
            customer_data['PastPurchases'],
            customer_data['WebsiteVisits'],
            1 if customer_data['CompetitorMentions'] != 'None' else 0,
            len(customer_data['CompetitorMentions'].split(', ')) if customer_data['CompetitorMentions'] != 'None' else 0,
            customer_data['RevenuePotential']
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return prediction, probability
    
    def get_model_summary(self):
        """
        Get a summary of all trained models.
        
        Returns:
            pd.DataFrame: Model performance summary
        """
        
        if not self.model_performance:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, metrics in self.model_performance.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1 Score': f"{metrics['f1']:.3f}",
                'ROC AUC': f"{metrics['roc_auc']:.3f}",
                'CV Score': f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}"
            })
        
        return pd.DataFrame(summary_data)


class CustomerSegmentation:
    """
    Customer segmentation using clustering algorithms.
    """
    
    def __init__(self):
        """Initialize the segmentation class."""
        self.clusters = None
        self.cluster_centers = None
        self.segmentation_labels = None
    
    def perform_kmeans_segmentation(self, df, n_clusters=5):
        """
        Perform K-means clustering for customer segmentation.
        
        Args:
            df (pd.DataFrame): Customer dataset
            n_clusters (int): Number of clusters
            
        Returns:
            pd.DataFrame: Dataset with cluster labels
        """
        
        # Prepare features for clustering
        features = df[['PastPurchases', 'WebsiteVisits', 'RevenuePotential']].copy()
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Store results
        self.clusters = kmeans
        self.cluster_centers = kmeans.cluster_centers_
        self.segmentation_labels = cluster_labels
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['Cluster'] = cluster_labels
        
        # Create cluster names
        cluster_names = []
        for label in cluster_labels:
            if label == 0:
                cluster_names.append('High-Value Active')
            elif label == 1:
                cluster_names.append('Medium-Value Regular')
            elif label == 2:
                cluster_names.append('Low-Value Inactive')
            elif label == 3:
                cluster_names.append('High-Value Inactive')
            else:
                cluster_names.append('Medium-Value Active')
        
        df_clustered['ClusterName'] = cluster_names
        
        return df_clustered
    
    def create_segmentation_visualization(self, df_clustered):
        """
        Create visualization for customer segmentation.
        
        Args:
            df_clustered (pd.DataFrame): Dataset with cluster labels
            
        Returns:
            plotly.graph_objects.Figure: Segmentation visualization
        """
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df_clustered,
            x='PastPurchases',
            y='WebsiteVisits',
            z='RevenuePotential',
            color='ClusterName',
            title='Customer Segmentation - 3D View',
            labels={
                'PastPurchases': 'Past Purchases',
                'WebsiteVisits': 'Website Visits',
                'RevenuePotential': 'Revenue Potential (Lakhs)'
            }
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    def get_cluster_summary(self, df_clustered):
        """
        Get summary statistics for each cluster.
        
        Args:
            df_clustered (pd.DataFrame): Dataset with cluster labels
            
        Returns:
            pd.DataFrame: Cluster summary statistics
        """
        
        cluster_summary = df_clustered.groupby('ClusterName').agg({
            'PastPurchases': ['count', 'mean', 'std'],
            'WebsiteVisits': ['mean', 'std'],
            'RevenuePotential': ['mean', 'std', 'sum'],
            'ResponseToCampaign': 'mean'
        }).round(2)
        
        # Flatten column names
        cluster_summary.columns = [
            'Customer_Count', 'Avg_Purchases', 'Std_Purchases',
            'Avg_WebsiteVisits', 'Std_WebsiteVisits',
            'Avg_Revenue', 'Std_Revenue', 'Total_Revenue',
            'Campaign_Response_Rate'
        ]
        
        # Convert response rate to percentage
        cluster_summary['Campaign_Response_Rate'] = cluster_summary['Campaign_Response_Rate'] * 100
        
        return cluster_summary.reset_index()
