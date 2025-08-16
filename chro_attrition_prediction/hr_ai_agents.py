import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from collections import Counter

class HRMetricsAgent:
    """Agent responsible for analyzing HR metrics and patterns."""
    
    def __init__(self):
        self.name = "HR Metrics Analyst Agent"
        self.role = "Analyze quantitative HR data and identify patterns"
    
    def analyze_attrition_patterns(self, df):
        """Analyze attrition patterns across different dimensions."""
        
        analysis = {}
        
        # Department analysis
        dept_analysis = df.groupby('Department')['Attrition'].agg(['count', 'sum']).reset_index()
        dept_analysis['AttritionRate'] = (dept_analysis['sum'] / dept_analysis['count'] * 100).round(1)
        dept_analysis = dept_analysis.sort_values('AttritionRate', ascending=False)
        
        analysis['department'] = dept_analysis
        
        # Age group analysis
        df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 70], labels=['20-30', '30-40', '40-50', '50+'])
        age_analysis = df.groupby('AgeGroup')['Attrition'].agg(['count', 'sum']).reset_index()
        age_analysis['AttritionRate'] = (age_analysis['sum'] / age_analysis['count'] * 100).round(1)
        
        analysis['age_group'] = age_analysis
        
        # Tenure analysis
        df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 2, 5, 10, 25], labels=['0-2', '2-5', '5-10', '10+'])
        tenure_analysis = df.groupby('TenureGroup')['Attrition'].agg(['count', 'sum']).reset_index()
        tenure_analysis['AttritionRate'] = (tenure_analysis['sum'] / tenure_analysis['count'] * 100).round(1)
        
        analysis['tenure_group'] = tenure_analysis
        
        # Performance analysis
        perf_analysis = df.groupby('PerformanceRating')['Attrition'].agg(['count', 'sum']).reset_index()
        perf_analysis['AttritionRate'] = (perf_analysis['sum'] / perf_analysis['count'] * 100).round(1)
        
        analysis['performance'] = perf_analysis
        
        # Satisfaction analysis
        sat_analysis = df.groupby('SatisfactionScore')['Attrition'].agg(['count', 'sum']).reset_index()
        sat_analysis['AttritionRate'] = (sat_analysis['sum'] / sat_analysis['count'] * 100).round(1)
        
        analysis['satisfaction'] = sat_analysis
        
        return analysis
    
    def identify_risk_factors(self, df):
        """Identify key risk factors for attrition."""
        
        risk_factors = []
        
        # Calculate correlation with attrition
        numerical_cols = ['Age', 'Salary', 'Tenure', 'PromotionLast2Years', 
                         'TrainingHours', 'PerformanceRating', 'SatisfactionScore']
        
        correlations = {}
        for col in numerical_cols:
            corr = df[col].corr(df['Attrition'])
            correlations[col] = abs(corr)
        
        # Sort by correlation strength
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        for factor, corr in sorted_correlations:
            if corr > 0.1:  # Only consider meaningful correlations
                risk_factors.append({
                    'factor': factor,
                    'correlation': corr,
                    'risk_level': 'High' if corr > 0.3 else 'Medium' if corr > 0.2 else 'Low'
                })
        
        return risk_factors
    
    def generate_metrics_insights(self, df, analysis):
        """Generate insights from HR metrics analysis."""
        
        insights = []
        
        # Department insights
        high_attrition_dept = analysis['department'].iloc[0]
        low_attrition_dept = analysis['department'].iloc[-1]
        
        insights.append(f"🏢 **Department Risk**: {high_attrition_dept['Department']} has the highest attrition rate at {high_attrition_dept['AttritionRate']}%")
        insights.append(f"✅ **Department Strength**: {low_attrition_dept['Department']} has the lowest attrition rate at {low_attrition_dept['AttritionRate']}%")
        
        # Age insights
        high_attrition_age = analysis['age_group'].iloc[0]
        insights.append(f"👥 **Age Risk**: {high_attrition_age['AgeGroup']} age group shows {high_attrition_age['AttritionRate']}% attrition rate")
        
        # Tenure insights
        high_attrition_tenure = analysis['tenure_group'].iloc[0]
        insights.append(f"⏰ **Tenure Risk**: {high_attrition_tenure['TenureGroup']} tenure group shows {high_attrition_tenure['AttritionRate']}% attrition rate")
        
        # Performance insights
        perf_correlation = df['PerformanceRating'].corr(df['Attrition'])
        if abs(perf_correlation) > 0.2:
            insights.append(f"📊 **Performance Impact**: Performance rating shows {'strong negative' if perf_correlation < 0 else 'strong positive'} correlation with attrition")
        
        # Satisfaction insights
        sat_correlation = df['SatisfactionScore'].corr(df['Attrition'])
        if abs(sat_correlation) > 0.2:
            insights.append(f"😊 **Satisfaction Impact**: Satisfaction score shows {'strong negative' if sat_correlation < 0 else 'strong positive'} correlation with attrition")
        
        return insights

class EmployeeFeedbackAgent:
    """Agent responsible for analyzing employee feedback and sentiment."""
    
    def __init__(self):
        self.name = "Employee Feedback Analyst Agent"
        self.role = "Analyze qualitative feedback and identify sentiment patterns"
    
    def analyze_feedback_sentiment(self, df):
        """Analyze sentiment in employee feedback."""
        
        # Define sentiment keywords
        positive_keywords = ['satisfied', 'excited', 'motivated', 'engaged', 'valued', 'good', 'great', 'excellent', 'supportive', 'collaborative']
        negative_keywords = ['frustrated', 'disappointed', 'concerned', 'unmotivated', 'overwhelmed', 'bad', 'poor', 'stressful', 'competitive', 'demanding']
        neutral_keywords = ['okay', 'content', 'stable', 'comfortable', 'adequate', 'routine', 'manageable']
        
        # Analyze each feedback
        sentiment_scores = []
        for feedback in df['EmployeeFeedback']:
            feedback_lower = feedback.lower()
            
            positive_count = sum(1 for word in positive_keywords if word in feedback_lower)
            negative_count = sum(1 for word in negative_keywords if word in feedback_lower)
            neutral_count = sum(1 for word in neutral_keywords if word in feedback_lower)
            
            # Calculate sentiment score (-1 to 1)
            if positive_count > negative_count:
                sentiment_scores.append(1)
            elif negative_count > positive_count:
                sentiment_scores.append(-1)
            else:
                sentiment_scores.append(0)
        
        # Add sentiment scores to dataframe
        df_with_sentiment = df.copy()
        df_with_sentiment['SentimentScore'] = sentiment_scores
        
        return df_with_sentiment
    
    def extract_feedback_themes(self, df):
        """Extract common themes from employee feedback."""
        
        # Define theme keywords
        themes = {
            'Career Growth': ['promotion', 'growth', 'advancement', 'career', 'development', 'opportunities'],
            'Work Environment': ['environment', 'culture', 'atmosphere', 'team', 'collaboration', 'support'],
            'Management': ['manager', 'leadership', 'guidance', 'expectations', 'autonomy', 'micromanaging'],
            'Workload': ['workload', 'pressure', 'deadlines', 'stress', 'balance', 'flexible'],
            'Compensation': ['salary', 'benefits', 'compensation', 'pay', 'rewards', 'recognition'],
            'Training': ['training', 'development', 'learning', 'skills', 'knowledge', 'education']
        }
        
        theme_counts = {theme: 0 for theme in themes}
        
        for feedback in df['EmployeeFeedback']:
            feedback_lower = feedback.lower()
            for theme, keywords in themes.items():
                if any(keyword in feedback_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Convert to DataFrame
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'MentionCount'])
        theme_df = theme_df.sort_values('MentionCount', ascending=False)
        
        return theme_df
    
    def identify_feedback_patterns(self, df):
        """Identify patterns in employee feedback."""
        
        patterns = []
        
        # Analyze feedback by department
        dept_feedback = df.groupby('Department')['SentimentScore'].mean().reset_index()
        dept_feedback = dept_feedback.sort_values('SentimentScore', ascending=False)
        
        patterns.append(f"🏢 **Department Sentiment**: {dept_feedback.iloc[0]['Department']} has the most positive sentiment")
        patterns.append(f"⚠️ **Department Concerns**: {dept_feedback.iloc[-1]['Department']} has the lowest sentiment score")
        
        # Analyze feedback by performance rating
        perf_feedback = df.groupby('PerformanceRating')['SentimentScore'].mean().reset_index()
        perf_correlation = perf_feedback['PerformanceRating'].corr(perf_feedback['SentimentScore'])
        
        if abs(perf_correlation) > 0.3:
            patterns.append(f"📊 **Performance-Sentiment Link**: Strong correlation between performance and sentiment")
        
        # Analyze feedback by satisfaction score
        sat_feedback = df.groupby('SatisfactionScore')['SentimentScore'].mean().reset_index()
        sat_correlation = sat_feedback['SatisfactionScore'].corr(sat_feedback['SentimentScore'])
        
        if abs(sat_correlation) > 0.3:
            patterns.append(f"😊 **Satisfaction-Sentiment Link**: Strong correlation between satisfaction and sentiment")
        
        return patterns

class MultiAgentSystem:
    """Multi-agent system that coordinates HR metrics and feedback analysis."""
    
    def __init__(self):
        self.metrics_agent = HRMetricsAgent()
        self.feedback_agent = EmployeeFeedbackAgent()
        self.name = "CHRO Multi-Agent AI System"
    
    def run_comprehensive_analysis(self, df):
        """Run comprehensive analysis using both agents."""
        
        st.info(f"🤖 **{self.name}** - Coordinating analysis between {self.metrics_agent.name} and {self.feedback_agent.name}")
        
        # Run metrics analysis
        with st.spinner(f"🔍 {self.metrics_agent.name} analyzing HR metrics..."):
            metrics_analysis = self.metrics_agent.analyze_attrition_patterns(df)
            risk_factors = self.metrics_agent.identify_risk_factors(df)
            metrics_insights = self.metrics_agent.generate_metrics_insights(df, metrics_analysis)
        
        # Run feedback analysis
        with st.spinner(f"💬 {self.feedback_agent.name} analyzing employee feedback..."):
            df_with_sentiment = self.feedback_agent.analyze_feedback_sentiment(df)
            feedback_themes = self.feedback_agent.extract_feedback_themes(df)
            feedback_patterns = self.feedback_agent.identify_feedback_patterns(df_with_sentiment)
        
        # Combine insights
        combined_insights = {
            'metrics_insights': metrics_insights,
            'feedback_patterns': feedback_patterns,
            'risk_factors': risk_factors,
            'sentiment_analysis': df_with_sentiment,
            'feedback_themes': feedback_themes
        }
        
        return combined_insights
    
    def create_agent_insights_dashboard(self, insights):
        """Create dashboard showing insights from both agents."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Risk Factors by Correlation', 'Feedback Themes', 
                           'Department Sentiment vs Attrition', 'Performance vs Sentiment'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Risk Factors
        risk_df = pd.DataFrame(insights['risk_factors'])
        fig.add_trace(
            go.Bar(x=risk_df['factor'], y=risk_df['correlation'], 
                   name='Risk Factor Correlation', marker_color='red'),
            row=1, col=1
        )
        
        # 2. Feedback Themes
        fig.add_trace(
            go.Bar(x=insights['feedback_themes']['Theme'], y=insights['feedback_themes']['MentionCount'],
                   name='Theme Mentions', marker_color='blue'),
            row=1, col=2
        )
        
        # 3. Department Sentiment vs Attrition
        dept_analysis = insights['sentiment_analysis'].groupby('Department').agg({
            'SentimentScore': 'mean',
            'Attrition': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(x=dept_analysis['SentimentScore'], y=dept_analysis['Attrition'],
                      mode='markers+text', text=dept_analysis['Department'],
                      name='Department Analysis', marker_color='green'),
            row=2, col=1
        )
        
        # 4. Performance vs Sentiment
        perf_sentiment = insights['sentiment_analysis'].groupby('PerformanceRating')['SentimentScore'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=perf_sentiment['PerformanceRating'], y=perf_sentiment['SentimentScore'],
                      mode='lines+markers', name='Performance vs Sentiment', marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Multi-Agent AI System - Combined Insights Dashboard",
            showlegend=True
        )
        
        return fig
    
    def generate_strategic_recommendations(self, insights):
        """Generate strategic recommendations based on combined agent insights."""
        
        recommendations = []
        
        # High-risk department recommendations
        high_risk_dept = insights['sentiment_analysis'].groupby('Department').agg({
            'Attrition': 'mean',
            'SentimentScore': 'mean'
        }).reset_index()
        high_risk_dept = high_risk_dept.sort_values('Attrition', ascending=False).iloc[0]
        
        if high_risk_dept['Attrition'] > 0.2:  # 20% attrition rate
            recommendations.append(f"🚨 **Immediate Action Required**: {high_risk_dept['Department']} shows {high_risk_dept['Attrition']*100:.1f}% attrition rate")
            recommendations.append(f"💡 **Strategy**: Conduct department-wide engagement survey and implement retention programs")
        
        # Low sentiment department recommendations
        low_sentiment_dept = high_risk_dept.sort_values('SentimentScore').iloc[0]
        if low_sentiment_dept['SentimentScore'] < -0.3:
            recommendations.append(f"😔 **Sentiment Crisis**: {low_sentiment_dept['Department']} has critically low sentiment")
            recommendations.append(f"🔄 **Action**: Implement immediate culture improvement initiatives and leadership training")
        
        # Top risk factor recommendations
        top_risk = insights['risk_factors'][0] if insights['risk_factors'] else None
        if top_risk and top_risk['risk_level'] == 'High':
            recommendations.append(f"⚠️ **Critical Risk Factor**: {top_risk['factor']} shows highest correlation with attrition")
            recommendations.append(f"🎯 **Focus**: Prioritize interventions targeting {top_risk['factor']}")
        
        # Feedback theme recommendations
        top_theme = insights['feedback_themes'].iloc[0] if not insights['feedback_themes'].empty else None
        if top_theme and top_theme['MentionCount'] > 100:
            recommendations.append(f"💬 **Top Concern**: {top_theme['Theme']} is mentioned {top_theme['MentionCount']} times")
            recommendations.append(f"📋 **Action**: Develop comprehensive strategy to address {top_theme['Theme']} concerns")
        
        return recommendations
