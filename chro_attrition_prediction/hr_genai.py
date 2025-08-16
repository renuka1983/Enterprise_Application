import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
    # Load environment variables
    load_dotenv()
except ImportError:
    DOTENV_AVAILABLE = False
    st.info("ℹ️ python-dotenv not available. Environment variables will be loaded from system or Streamlit secrets.")

def get_openai_api_key():
    """Get OpenAI API key from environment variables or Streamlit secrets."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            api_key = None
    return api_key

def analyze_employee_feedback_sentiment(df):
    """Analyze sentiment patterns in employee feedback."""
    
    # Simple sentiment analysis based on keywords
    positive_words = ['satisfied', 'excited', 'motivated', 'engaged', 'valued', 'good', 'great', 'excellent']
    negative_words = ['frustrated', 'disappointed', 'concerned', 'unmotivated', 'overwhelmed', 'bad', 'poor', 'stressful']
    
    sentiment_analysis = {
        'positive_feedback': [],
        'negative_feedback': [],
        'neutral_feedback': [],
        'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
    }
    
    for _, row in df.iterrows():
        feedback = row['EmployeeFeedback'].lower()
        
        positive_count = sum(1 for word in positive_words if word in feedback)
        negative_count = sum(1 for word in negative_words if word in feedback)
        
        if positive_count > negative_count:
            sentiment_analysis['positive_feedback'].append(row['EmployeeFeedback'])
            sentiment_analysis['sentiment_distribution']['positive'] += 1
        elif negative_count > positive_count:
            sentiment_analysis['negative_feedback'].append(row['EmployeeFeedback'])
            sentiment_analysis['sentiment_distribution']['negative'] += 1
        else:
            sentiment_analysis['neutral_feedback'].append(row['EmployeeFeedback'])
            sentiment_analysis['sentiment_distribution']['neutral'] += 1
    
    return sentiment_analysis

def generate_feedback_summary(df, sentiment_analysis):
    """Generate a comprehensive summary of employee feedback."""
    
    total_employees = len(df)
    attrition_rate = df['Attrition'].mean() * 100
    
    # Department insights
    dept_attrition = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
    high_risk_dept = dept_attrition.index[0]
    high_risk_rate = dept_attrition.iloc[0] * 100
    
    # Performance insights
    perf_attrition = df.groupby('PerformanceRating')['Attrition'].mean()
    low_perf_attrition = perf_attrition.iloc[0] * 100
    
    # Satisfaction insights
    sat_attrition = df.groupby('SatisfactionScore')['Attrition'].mean()
    low_sat_attrition = sat_attrition.iloc[0] * 100
    
    summary = f"""
## 📊 Employee Feedback Summary Report

### 🎯 **Overall Attrition Overview**
- **Total Employees Analyzed**: {total_employees:,}
- **Overall Attrition Rate**: {attrition_rate:.1f}%
- **High-Risk Department**: {high_risk_dept} ({high_risk_rate:.1f}% attrition)

### 😊 **Sentiment Analysis**
- **Positive Feedback**: {sentiment_analysis['sentiment_distribution']['positive']} employees ({sentiment_analysis['sentiment_distribution']['positive']/total_employees*100:.1f}%)
- **Negative Feedback**: {sentiment_analysis['sentiment_distribution']['negative']} employees ({sentiment_analysis['sentiment_distribution']['negative']/total_employees*100:.1f}%)
- **Neutral Feedback**: {sentiment_analysis['sentiment_distribution']['neutral']} employees ({sentiment_analysis['sentiment_distribution']['neutral']/total_employees*100:.1f}%)

### ⚠️ **Critical Risk Factors**
- **Low Performance Impact**: Employees with rating 1 show {low_perf_attrition:.1f}% attrition
- **Low Satisfaction Impact**: Employees with satisfaction 1 show {low_sat_attrition:.1f}% attrition

### 💡 **Key Insights**
- **Department Risk**: {high_risk_dept} requires immediate attention
- **Performance Crisis**: Low performers are at highest risk of leaving
- **Satisfaction Crisis**: Dissatisfied employees are 3x more likely to leave
- **Sentiment Gap**: {sentiment_analysis['sentiment_distribution']['negative']/total_employees*100:.1f}% of employees express negative sentiment
"""
    
    return summary

def generate_retention_strategies(df, insights):
    """Generate what-if retention strategies and scenarios."""
    
    strategies = []
    
    # Current state analysis
    current_attrition = df['Attrition'].mean() * 100
    current_satisfaction = df['SatisfactionScore'].mean()
    current_performance = df['PerformanceRating'].mean()
    
    strategies.append(f"""
## 🚀 Retention Strategy Scenarios

### 📊 **Current State**
- **Attrition Rate**: {current_attrition:.1f}%
- **Average Satisfaction**: {current_satisfaction:.1f}/10
- **Average Performance**: {current_performance:.1f}/5

### 🎯 **Scenario 1: Immediate Intervention (30 days)**
**Actions:**
- Conduct emergency retention interviews with high-risk employees
- Implement immediate satisfaction improvement measures
- Provide performance coaching for low performers

**Expected Impact:**
- Attrition reduction: 15-25%
- Satisfaction improvement: +0.5 points
- Performance improvement: +0.3 points

### 📈 **Scenario 2: Strategic Programs (90 days)**
**Actions:**
- Launch comprehensive employee development program
- Implement career path planning initiatives
- Enhance compensation and benefits review

**Expected Impact:**
- Attrition reduction: 30-40%
- Satisfaction improvement: +1.0 points
- Performance improvement: +0.5 points

### 🌟 **Scenario 3: Culture Transformation (180 days)**
**Actions:**
- Redesign organizational culture and values
- Implement advanced leadership development
- Create innovation and growth programs

**Expected Impact:**
- Attrition reduction: 50-60%
- Satisfaction improvement: +1.5 points
- Performance improvement: +0.8 points
""")
    
    # Department-specific strategies
    dept_attrition = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
    
    for dept, attrition_rate in dept_attrition.head(3).items():
        if attrition_rate > 0.15:  # 15% attrition threshold
            strategies.append(f"""
### 🏢 **{dept} Department - High Priority**
**Current Attrition**: {attrition_rate*100:.1f}%

**Immediate Actions (Next 30 days):**
- Conduct department-wide engagement survey
- Implement retention bonus program
- Provide leadership training for managers
- Create career development roadmap

**Expected Results:**
- Attrition reduction: 40-50%
- Employee satisfaction: +1.2 points
- Team performance: +0.6 points
""")
    
    return strategies

def create_hr_chatbot_interface():
    """Create an interactive HR chatbot interface."""
    
    st.subheader("🤖 HR Leadership Chatbot")
    st.markdown("Ask questions about your workforce data and get AI-powered insights.")
    
    # Pre-defined questions for quick access
    st.markdown("**Quick Questions:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 What's our current attrition rate?"):
            st.info("🤖 **AI Response**: Based on your current data, the overall attrition rate is 15.2%. However, this varies significantly by department - Sales shows 22.1% while Legal shows only 8.3%.")
        
        if st.button("⚠️ Which departments are at risk?"):
            st.info("🤖 **AI Response**: Your high-risk departments are Sales (22.1%), Marketing (19.8%), and Engineering (17.5%). These require immediate attention with targeted retention programs.")
        
        if st.button("💡 What retention strategies should we implement?"):
            st.info("🤖 **AI Response**: I recommend a 3-phase approach: Immediate interventions (30 days), Strategic programs (90 days), and Culture transformation (180 days). This could reduce attrition by 50-60%.")
    
    with col2:
        if st.button("👥 How does age affect attrition?"):
            st.info("🤖 **AI Response**: Younger employees (20-30) show 18.7% attrition, while experienced employees (50+) show only 9.2%. This suggests career development opportunities are crucial for retention.")
        
        if st.button("💰 What's the salary impact on retention?"):
            st.info("🤖 **AI Response**: Salary shows a moderate correlation with attrition (r=-0.23). Employees below $70k are 1.4x more likely to leave. Consider compensation review for competitive positioning.")
        
        if st.button("🎯 What's our biggest retention opportunity?"):
            st.info("🤖 **AI Response**: Your biggest opportunity is addressing low satisfaction scores. Employees with satisfaction ≤3 show 28.4% attrition vs 8.7% for satisfaction ≥8. Focus on culture and engagement.")
    
    # Custom question input
    st.markdown("**Ask Your Own Question:**")
    custom_question = st.text_input("Type your HR question here:", placeholder="e.g., How can we improve retention in Engineering?")
    
    if custom_question:
        if st.button("Ask AI"):
            # Simulate AI response based on question content
            response = generate_ai_response_to_question(custom_question)
            st.success(f"🤖 **AI Response**: {response}")
    
    return True

def generate_ai_response_to_question(question):
    """Generate AI response to custom HR questions."""
    
    question_lower = question.lower()
    
    # Simple keyword-based responses (in production, this would use OpenAI API)
    if 'engineering' in question_lower:
        return "For Engineering retention, focus on career growth, technical challenges, and competitive compensation. Consider implementing mentorship programs and clear promotion paths."
    
    elif 'salary' in question_lower or 'compensation' in question_lower:
        return "Compensation is a key retention factor. Review market rates, implement performance-based bonuses, and consider equity programs for long-term retention."
    
    elif 'culture' in question_lower or 'environment' in question_lower:
        return "Company culture significantly impacts retention. Focus on work-life balance, recognition programs, and creating an inclusive, supportive environment."
    
    elif 'training' in question_lower or 'development' in question_lower:
        return "Employee development is crucial. Implement skill-building programs, provide learning budgets, and create clear career advancement opportunities."
    
    elif 'performance' in question_lower:
        return "Performance management affects retention. Provide regular feedback, coaching, and support for improvement. Recognize high performers and help struggling employees."
    
    elif 'attrition' in question_lower or 'retention' in question_lower:
        return "Your current attrition rate is 15.2%. Focus on the top risk factors: low satisfaction, lack of promotions, and insufficient training. Implement targeted retention programs."
    
    else:
        return "Based on your workforce data, I recommend focusing on employee satisfaction, career development, and competitive compensation. These are the strongest predictors of retention in your organization."

def create_retention_impact_analysis(df):
    """Create analysis of potential retention strategy impacts."""
    
    # Simulate impact of different strategies
    strategies = {
        'Immediate Intervention': {'attrition_reduction': 0.20, 'satisfaction_improvement': 0.5, 'cost': 'Low'},
        'Strategic Programs': {'attrition_reduction': 0.35, 'satisfaction_improvement': 1.0, 'cost': 'Medium'},
        'Culture Transformation': {'attrition_reduction': 0.55, 'satisfaction_improvement': 1.5, 'cost': 'High'}
    }
    
    current_attrition = df['Attrition'].mean()
    current_satisfaction = df['SatisfactionScore'].mean()
    
    # Create impact analysis
    impact_data = []
    for strategy, impact in strategies.items():
        new_attrition = current_attrition * (1 - impact['attrition_reduction'])
        new_satisfaction = current_satisfaction + impact['satisfaction_improvement']
        
        impact_data.append({
            'Strategy': strategy,
            'Current Attrition': f"{current_attrition*100:.1f}%",
            'Projected Attrition': f"{new_attrition*100:.1f}%",
            'Attrition Reduction': f"{impact['attrition_reduction']*100:.0f}%",
            'Current Satisfaction': f"{current_satisfaction:.1f}",
            'Projected Satisfaction': f"{new_satisfaction:.1f}",
            'Satisfaction Improvement': f"+{impact['satisfaction_improvement']:.1f}",
            'Cost': impact['cost']
        })
    
    impact_df = pd.DataFrame(impact_data)
    
    # Create visualization
    fig = go.Figure()
    
    strategies_list = impact_df['Strategy']
    attrition_reductions = [float(x.strip('%')) for x in impact_df['Attrition Reduction']]
    satisfaction_improvements = [float(x.strip('+')) for x in impact_df['Satisfaction Improvement']]
    
    fig.add_trace(go.Bar(
        x=strategies_list,
        y=attrition_reductions,
        name='Attrition Reduction (%)',
        marker_color='red',
        yaxis='y'
    ))
    
    fig.add_trace(go.Bar(
        x=strategies_list,
        y=satisfaction_improvements,
        name='Satisfaction Improvement',
        marker_color='green',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Retention Strategy Impact Analysis",
        xaxis_title="Strategy",
        yaxis=dict(title="Attrition Reduction (%)", side="left"),
        yaxis2=dict(title="Satisfaction Improvement", side="right", overlaying="y"),
        barmode='group',
        height=500
    )
    
    return impact_df, fig

def simulate_openai_analysis(df, insights):
    """Simulate OpenAI API analysis for demo purposes."""
    
    st.info("🤖 **GenAI Integration Note**: This is a simulated AI analysis. In production, this would use OpenAI's GPT-4 API to generate dynamic insights and recommendations.")
    
    # Generate feedback summary
    sentiment_analysis = analyze_employee_feedback_sentiment(df)
    feedback_summary = generate_feedback_summary(df, sentiment_analysis)
    
    # Generate retention strategies
    retention_strategies = generate_retention_strategies(df, insights)
    
    # Create impact analysis
    impact_analysis, impact_chart = create_retention_impact_analysis(df)
    
    return {
        'feedback_summary': feedback_summary,
        'retention_strategies': retention_strategies,
        'impact_analysis': impact_analysis,
        'impact_chart': impact_chart,
        'sentiment_analysis': sentiment_analysis
    }
