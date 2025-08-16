import streamlit as st
import sys
import os

# Add the subdirectories to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'inventory'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'predictive_maintenance'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'product_design'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'QualityControl'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'chro_attrition_prediction'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'cmo_hyper_personalization'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'energy_esg_optimization'))

# Page configuration
st.set_page_config(
    page_title="Manufacturing Workshop App",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the main app
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .module-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .module-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .module-description {
        color: #666;
        margin-bottom: 1rem;
    }
    .feature-list {
        list-style: none;
        padding: 0;
    }
    .feature-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .feature-list li:before {
        content: "✅ ";
        color: #4CAF50;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
    .stat-item {
        text-align: center;
        padding: 0.5rem;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar configuration
    with st.sidebar:
        st.header("🏭 Manufacturing Workshop App")
        
        # Info button
        if st.button("ℹ️ App Information", type="secondary"):
            st.session_state.show_app_info = not st.session_state.get('show_app_info', False)
        
        st.markdown("---")
        
        # Quick navigation
        st.subheader("🚀 Quick Navigation")
        st.markdown("""
        **Available Modules:**
        - 📦 Inventory Management
        - 🔧 Predictive Maintenance
        - 🎨 Product Design
        - ✅ Quality Control
        - 💰 CFO Financial Case Study
        - 👥 CHRO Attrition Prediction
        - 🎯 CMO Hyper-Personalization
        - ⚡ Energy ESG Optimization
        """)
        
        # App information
        if st.session_state.get('show_app_info', False):
            st.subheader("ℹ️ About This Workshop")
            st.markdown("""
            This application demonstrates the power of AI and machine learning in modern manufacturing, finance, human resources, marketing, and sustainability. 
            Each module showcases different aspects of intelligent systems, from demand forecasting 
            to quality control, predictive maintenance, financial analysis, workforce retention, customer personalization, and energy optimization.
            """)
    
    st.markdown('<h1 class="main-header">🏭 Manufacturing Workshop App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            A comprehensive suite of AI-powered manufacturing tools for modern industry
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## 🎯 About This Workshop
    
    This application demonstrates the power of AI and machine learning in modern manufacturing, finance, human resources, marketing, and sustainability. 
    Each module showcases different aspects of intelligent systems, from demand forecasting 
    to quality control, predictive maintenance, financial analysis, workforce retention, customer personalization, and energy optimization.
    """)
    
    # Module showcase
    st.markdown("## 📊 Available Modules")
    
    # First row: 4 modules
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">📦 Inventory Management</div>
            <div class="module-description">
                AI-powered demand forecasting and production planning with dynamic adjustments.
            </div>
            <ul class="feature-list">
                <li>Multi-method demand forecasting (Traditional, ML, AI)</li>
                <li>Real-time external factor adjustments</li>
                <li>Risk analysis and inventory optimization</li>
                <li>Executive dashboard with key metrics</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">5</div>
                    <div class="stat-label">Products</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">90</div>
                    <div class="stat-label">Days Forecast</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">3</div>
                    <div class="stat-label">Methods</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🔧 Predictive Maintenance</div>
            <div class="module-description">
                Machine learning-based equipment health monitoring and maintenance scheduling.
            </div>
            <ul class="feature-list">
                <li>Real-time sensor data analysis</li>
                <li>Failure prediction models</li>
                <li>Maintenance optimization</li>
                <li>Equipment health scoring</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">5</div>
                    <div class="stat-label">Machines</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">6</div>
                    <div class="stat-label">Sensors</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">95%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🚴 Product Design Optimization</div>
            <div class="module-description">
                AI-driven design optimization with physics-informed modeling and generative design.
            </div>
            <ul class="feature-list">
                <li>Multi-objective optimization</li>
                <li>Physics-informed AI models</li>
                <li>Generative design capabilities</li>
                <li>Real-time simulation</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">4</div>
                    <div class="stat-label">Methods</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">92%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">75%</div>
                    <div class="stat-label">Time Saved</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">👥 CHRO Attrition Prediction</div>
            <div class="module-description">
                AI-powered workforce retention analysis with multi-agent systems and GenAI insights.
            </div>
            <ul class="feature-list">
                <li>Traditional statistical analysis</li>
                <li>ML prediction (Logistic Regression & Random Forest)</li>
                <li>Multi-agent AI system coordination</li>
                <li>GenAI-powered retention strategies</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">4</div>
                    <div class="stat-label">Approaches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">1000+</div>
                    <div class="stat-label">Employees</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">90%</div>
                    <div class="stat-label">ML Accuracy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row: 4 modules
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🔍 Quality Control & Defect Detection</div>
            <div class="module-description">
                Computer vision and ML-based quality inspection and defect detection system.
            </div>
            <ul class="feature-list">
                <li>Computer vision inspection</li>
                <li>Multiple defect type detection</li>
                <li>Real-time quality scoring</li>
                <li>Automated reporting</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">6</div>
                    <div class="stat-label">Defect Types</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">95%</div>
                    <div class="stat-label">Detection Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">5</div>
                    <div class="stat-label">Production Lines</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">💰 CFO Financial Case Study</div>
            <div class="module-description">
                Comprehensive financial forecasting and analysis: Traditional vs ML vs AI approaches.
            </div>
            <ul class="feature-list">
                <li>Traditional Excel-style calculations</li>
                <li>Machine Learning forecasting</li>
                <li>AI pipeline with SHAP explainability</li>
                <li>GenAI-powered insights and recommendations</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">24</div>
                    <div class="stat-label">Months Data</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">4</div>
                    <div class="stat-label">Methods</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">95%</div>
                    <div class="stat-label">ML Accuracy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🎯 CMO Hyper-Personalization</div>
            <div class="module-description">
                AI-powered customer personalization and market intelligence for manufacturing products.
            </div>
            <ul class="feature-list">
                <li>Traditional segmentation analysis</li>
                <li>ML campaign response prediction</li>
                <li>AI clustering and NLP analysis</li>
                <li>GenAI personalized product pitches</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">4</div>
                    <div class="stat-label">Approaches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">1000+</div>
                    <div class="stat-label">Customers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">90%</div>
                    <div class="stat-label">ML Accuracy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">⚡ Energy ESG Optimization</div>
            <div class="module-description">
                AI-powered energy optimization and ESG compliance for manufacturing sustainability.
            </div>
            <ul class="feature-list">
                <li>Traditional energy analytics</li>
                <li>ML energy consumption forecasting</li>
                <li>AI optimization & RL simulation</li>
                <li>GenAI ESG reporting & chatbot</li>
            </ul>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-number">4</div>
                    <div class="stat-label">Approaches</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">365</div>
                    <div class="stat-label">Days Data</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">25%</div>
                    <div class="stat-label">Savings Potential</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation instructions
    st.markdown("## 🚀 Getting Started")
    
    st.markdown("""
    To explore each module, use the sidebar navigation or click on the module cards above. 
    Each module is self-contained and demonstrates different AI/ML techniques applied to manufacturing challenges.
    
    ### Key Features:
    - **Interactive Dashboards**: Real-time data visualization and analysis
    - **AI/ML Models**: Sophisticated algorithms for prediction and optimization
    - **Synthetic Data**: Realistic manufacturing scenarios for demonstration
    - **Executive Insights**: Business-focused metrics and recommendations
    """)
    
    # Technology stack
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend & Visualization**
        - Streamlit
        - Plotly
        - Custom CSS
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning & AI**
        - Scikit-learn
        - Random Forest
        - Neural Networks
        - Physics-informed AI
        - SHAP explainability
        - Financial forecasting
        - Multi-agent systems
        - HR analytics
        - Customer personalization
        - Market intelligence
        - Energy optimization
        - ESG compliance
        """)
    
    with col3:
        st.markdown("""
        **Data Processing**
        - Pandas
        - NumPy
        - Synthetic data generation
        - Real-time analytics
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🏭 Manufacturing Workshop App | AI-Powered Manufacturing Solutions</p>
        <p>Built with Streamlit, Python, and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
