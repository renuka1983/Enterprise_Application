# 👥 CHRO Attrition Prediction System

## Overview
The CHRO Attrition Prediction System is an AI-powered workforce retention analysis tool that demonstrates four different approaches to predicting and analyzing employee attrition. This module showcases how modern AI and machine learning techniques can be applied to human resources challenges.

## 🎯 Features

### 📈 Traditional Analysis
- **Statistical Analysis**: Comprehensive HR metrics and patterns
- **Visualization Dashboard**: Interactive charts showing attrition by department, tenure, age, and satisfaction
- **Correlation Analysis**: Heatmap showing relationships between factors and attrition
- **Summary Statistics**: Department-wise breakdown with key metrics

### 🤖 Machine Learning Prediction
- **Logistic Regression**: Linear model for attrition prediction
- **Random Forest**: Ensemble model for improved accuracy
- **Model Comparison**: Side-by-side performance metrics
- **Feature Importance**: Understanding what drives attrition decisions
- **Confusion Matrices**: Model performance visualization

### 🧠 Multi-Agent AI System
- **HR Metrics Agent**: Analyzes quantitative HR data and identifies patterns
- **Employee Feedback Agent**: Processes qualitative feedback and sentiment
- **Coordinated Analysis**: Combines insights from both agents
- **Strategic Recommendations**: AI-generated retention strategies

### 🌟 Generative AI Insights
- **Feedback Summarization**: AI-powered analysis of employee feedback
- **Retention Strategies**: What-if scenarios and strategic planning
- **Impact Analysis**: Projected outcomes of different interventions
- **HR Chatbot**: Interactive AI assistant for HR leaders

## 🏗️ Architecture

```
chro_attrition_prediction/
├── CHRO_Attrition_Prediction.py    # Main Streamlit application
├── hr_synthetic_data.py            # Synthetic data generation
├── hr_ml_prediction.py             # Machine learning models
├── hr_ai_agents.py                 # Multi-agent AI system
├── hr_genai.py                     # Generative AI integration
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   cd chro_attrition_prediction
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run CHRO_Attrition_Prediction.py
   ```

3. **Generate Data**: Use the sidebar to configure dataset size and click "Generate New Dataset"

4. **Explore Tabs**: Navigate through the four different analysis approaches

## 📊 Data Structure

The synthetic HR dataset includes the following columns:

- **EmployeeID**: Unique identifier for each employee
- **Age**: Employee age (22-65)
- **Department**: 8 different departments (Engineering, Sales, Marketing, HR, Finance, Operations, IT, Legal)
- **JobRole**: Specific job titles within each department
- **Salary**: Annual salary with department and experience-based variations
- **Tenure**: Years of service (0-20)
- **PromotionLast2Years**: Number of promotions in last 2 years (0-3)
- **TrainingHours**: Annual training hours (0-100)
- **PerformanceRating**: Performance score (1-5)
- **SatisfactionScore**: Job satisfaction rating (1-10)
- **EmployeeFeedback**: Simulated text feedback from employees
- **Attrition**: Binary outcome (0=Stay, 1=Leave)

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **AI Integration**: Simulated OpenAI API (easily replaceable with real API)
- **Environment Management**: python-dotenv

## 🎨 Key Visualizations

1. **HR Dashboard**: Multi-panel view of attrition patterns
2. **Correlation Heatmap**: Feature importance for attrition
3. **Model Performance**: Comparison charts for ML models
4. **Feature Importance**: Horizontal bar charts for model interpretability
5. **Agent Insights**: Multi-agent analysis dashboard
6. **Strategy Impact**: Retention strategy outcome projections

## 🧪 Use Cases

### For HR Professionals
- **Attrition Risk Assessment**: Identify high-risk employees and departments
- **Retention Strategy Planning**: Data-driven approach to retention programs
- **Performance Analysis**: Understand performance-attrition relationships
- **Department Benchmarking**: Compare attrition rates across teams

### For Data Scientists
- **Model Comparison**: Evaluate different ML approaches
- **Feature Engineering**: Understand what drives attrition
- **Multi-Agent Systems**: Learn about coordinated AI analysis
- **Explainable AI**: SHAP-style feature importance analysis

### For Business Leaders
- **Strategic Planning**: Long-term retention strategy development
- **Resource Allocation**: Target interventions where they'll have most impact
- **ROI Analysis**: Project costs and benefits of retention programs
- **Competitive Intelligence**: Benchmark against industry standards

## 🔮 Future Enhancements

- **Real-time Data Integration**: Connect to HRIS systems
- **Advanced ML Models**: Deep learning and ensemble methods
- **Predictive Analytics**: Forecast future attrition trends
- **Employee Journey Mapping**: Track retention factors over time
- **Integration APIs**: Connect with other business systems

## 📝 Configuration

### Environment Variables
Create a `.env` file in the module directory:
```env
OPENAI_API_KEY=your_api_key_here
```

### Streamlit Secrets
Alternatively, use Streamlit secrets:
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your_api_key_here"
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce dataset size for large employee counts
3. **Model Training**: Some ML models may take time to train
4. **Visualization**: Plotly charts may take time to render with large datasets

### Performance Tips

- Use smaller datasets for development (500-1000 employees)
- Generate new datasets sparingly
- Close unused browser tabs to free memory
- Use Streamlit's caching for expensive operations

## 📚 Learning Resources

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **Plotly Python**: https://plotly.com/python/
- **Pandas Documentation**: https://pandas.pydata.org/docs/

## 🤝 Contributing

This module is designed to be easily extensible. Key areas for contribution:

- **New ML Models**: Add different algorithms
- **Additional Agents**: Create specialized AI agents
- **Enhanced Visualizations**: Improve dashboard aesthetics
- **Real Data Connectors**: Integrate with HR systems
- **Advanced Analytics**: Add predictive capabilities

## 📄 License

This module is part of the Manufacturing Workshop App and follows the same licensing terms.

---

**Note**: This system uses synthetic data for demonstration purposes. In production, replace with real HR data and ensure compliance with data privacy regulations.
