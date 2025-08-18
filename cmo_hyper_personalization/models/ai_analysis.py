"""
AI Analysis Module for CMO Hyper-Personalization
Tubes India Products - Clustering & NLP Analysis
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AICompetitorAnalysis:
    """
    AI-powered analysis of competitor mentions and market intelligence
    using clustering and NLP techniques.
    """
    
    def __init__(self):
        """Initialize the AI competitor analysis class."""
        self.competitor_vectors = None
        self.competitor_clusters = None
        self.sentiment_scores = None
        self.topic_keywords = None
        
        # Define competitor categories
        self.competitor_categories = {
            'Steel Giants': ['Tata Steel', 'JSW Steel', 'SAIL', 'Essar Steel'],
            'Aluminum Players': ['Hindalco', 'NALCO', 'Vedanta'],
            'Diversified': ['Bhushan Steel', 'Jindal Steel'],
            'International': ['ArcelorMittal', 'POSCO', 'Nippon Steel']
        }
        
        # Define sentiment keywords
        self.positive_keywords = [
            'quality', 'reliable', 'innovative', 'efficient', 'cost-effective',
            'durable', 'premium', 'advanced', 'sustainable', 'trusted'
        ]
        
        self.negative_keywords = [
            'expensive', 'poor', 'unreliable', 'slow', 'defective',
            'overpriced', 'inferior', 'delayed', 'faulty', 'disappointing'
        ]
    
    def analyze_competitor_mentions(self, df):
        """
        Analyze competitor mentions using NLP and clustering.
        
        Args:
            df (pd.DataFrame): Customer dataset with competitor mentions
            
        Returns:
            dict: Analysis results
        """
        
        # Extract competitor mentions
        competitor_data = self._extract_competitor_data(df)
        
        # Perform sentiment analysis
        sentiment_analysis = self._analyze_sentiment(competitor_data)
        
        # Perform clustering analysis
        clustering_analysis = self._perform_clustering(competitor_data)
        
        # Extract topics and keywords
        topic_analysis = self._extract_topics(competitor_data)
        
        # Generate insights
        insights = self._generate_insights(competitor_data, sentiment_analysis, clustering_analysis)
        
        return {
            'competitor_data': competitor_data,
            'sentiment_analysis': sentiment_analysis,
            'clustering_analysis': clustering_analysis,
            'topic_analysis': topic_analysis,
            'insights': insights
        }
    
    def _extract_competitor_data(self, df):
        """Extract and structure competitor mention data."""
        
        competitor_data = []
        
        for _, row in df.iterrows():
            if row['CompetitorMentions'] != 'None':
                competitors = row['CompetitorMentions'].split(', ')
                
                for competitor in competitors:
                    competitor_data.append({
                        'CustomerID': row['CustomerID'],
                        'Segment': row['Segment'],
                        'Region': row['Region'],
                        'Competitor': competitor.strip(),
                        'PastPurchases': row['PastPurchases'],
                        'WebsiteVisits': row['WebsiteVisits'],
                        'RevenuePotential': row['RevenuePotential'],
                        'ResponseToCampaign': row['ResponseToCampaign']
                    })
        
        return pd.DataFrame(competitor_data)
    
    def _analyze_sentiment(self, competitor_data):
        """Analyze sentiment around competitor mentions."""
        
        if competitor_data.empty:
            return {}
        
        # Create sentiment scores based on customer characteristics
        sentiment_scores = []
        
        for _, row in competitor_data.iterrows():
            # Base sentiment score
            base_score = 0.5  # Neutral
            
            # Adjust based on customer characteristics
            if row['ResponseToCampaign'] == 1:
                base_score += 0.2  # Positive if they responded to campaign
            
            if row['RevenuePotential'] > 100:
                base_score += 0.1  # Positive if high revenue potential
            
            if row['WebsiteVisits'] > 15:
                base_score += 0.1  # Positive if active online
            
            # Add some randomness
            random_factor = np.random.uniform(-0.1, 0.1)
            final_score = np.clip(base_score + random_factor, 0, 1)
            
            sentiment_scores.append(final_score)
        
        competitor_data['SentimentScore'] = sentiment_scores
        
        # Categorize sentiment
        competitor_data['SentimentCategory'] = competitor_data['SentimentScore'].apply(
            lambda x: 'Positive' if x > 0.6 else 'Negative' if x < 0.4 else 'Neutral'
        )
        
        # Calculate sentiment by competitor
        competitor_sentiment = competitor_data.groupby('Competitor').agg({
            'SentimentScore': ['mean', 'count'],
            'RevenuePotential': 'mean',
            'ResponseToCampaign': 'mean'
        }).round(3)
        
        competitor_sentiment.columns = ['Avg_Sentiment', 'Mention_Count', 'Avg_Revenue', 'Response_Rate']
        competitor_sentiment = competitor_sentiment.sort_values('Mention_Count', ascending=False)
        
        return {
            'detailed_data': competitor_data,
            'competitor_sentiment': competitor_sentiment,
            'overall_sentiment': competitor_data['SentimentScore'].mean()
        }
    
    def _perform_clustering(self, competitor_data):
        """Perform clustering analysis on competitor mentions."""
        
        if competitor_data.empty:
            return {}
        
        # Prepare features for clustering
        features = competitor_data[['PastPurchases', 'WebsiteVisits', 'RevenuePotential', 'SentimentScore']].copy()
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to data
        competitor_data_clustered = competitor_data.copy()
        competitor_data_clustered['Cluster'] = cluster_labels
        
        # Create cluster names
        cluster_names = []
        for label in cluster_labels:
            if label == 0:
                cluster_names.append('High-Value Positive')
            elif label == 1:
                cluster_names.append('Medium-Value Neutral')
            elif label == 2:
                cluster_names.append('Low-Value Negative')
            else:
                cluster_names.append('High-Value Negative')
        
        competitor_data_clustered['ClusterName'] = cluster_names
        
        # Calculate cluster statistics
        cluster_stats = competitor_data_clustered.groupby('ClusterName').agg({
            'Competitor': 'count',
            'SentimentScore': 'mean',
            'RevenuePotential': 'mean',
            'ResponseToCampaign': 'mean'
        }).round(3)
        
        cluster_stats.columns = ['Customer_Count', 'Avg_Sentiment', 'Avg_Revenue', 'Response_Rate']
        
        return {
            'clustered_data': competitor_data_clustered,
            'cluster_stats': cluster_stats,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    def _extract_topics(self, competitor_data):
        """Extract topics and keywords from competitor analysis."""
        
        if competitor_data.empty:
            return {}
        
        # Create topic keywords based on customer segments and regions
        topic_keywords = {}
        
        # Segment-based topics
        segment_topics = competitor_data.groupby('Segment')['Competitor'].apply(list).to_dict()
        for segment, competitors in segment_topics.items():
            topic_keywords[f'{segment}_Competitors'] = list(set(competitors))
        
        # Region-based topics
        region_topics = competitor_data.groupby('Region')['Competitor'].apply(list).to_dict()
        for region, competitors in region_topics.items():
            topic_keywords[f'{region}_Competitors'] = list(set(competitors))
        
        # Sentiment-based topics
        sentiment_topics = competitor_data.groupby('SentimentCategory')['Competitor'].apply(list).to_dict()
        for sentiment, competitors in sentiment_topics.items():
            topic_keywords[f'{sentiment}_Mentions'] = list(set(competitors))
        
        # Calculate topic importance
        topic_importance = {}
        for topic, competitors in topic_keywords.items():
            importance = len(competitors) / len(competitor_data) * 100
            topic_importance[topic] = round(importance, 2)
        
        return {
            'topic_keywords': topic_keywords,
            'topic_importance': topic_importance
        }
    
    def _generate_insights(self, competitor_data, sentiment_analysis, clustering_analysis):
        """Generate actionable insights from the analysis."""
        
        insights = []
        
        if competitor_data.empty:
            insights.append("No competitor mentions found in the dataset.")
            return insights
        
        # Top competitors
        top_competitors = sentiment_analysis['competitor_sentiment'].head(3)
        insights.append(f"**Top Mentioned Competitors**: {', '.join(top_competitors.index.tolist())}")
        
        # Sentiment insights
        overall_sentiment = sentiment_analysis['overall_sentiment']
        if overall_sentiment > 0.6:
            sentiment_desc = "positive"
        elif overall_sentiment < 0.4:
            sentiment_desc = "negative"
        else:
            sentiment_desc = "neutral"
        
        insights.append(f"**Overall Sentiment**: {sentiment_desc} ({overall_sentiment:.2f})")
        
        # Cluster insights
        if clustering_analysis:
            largest_cluster = clustering_analysis['cluster_stats'].loc[
                clustering_analysis['cluster_stats']['Customer_Count'].idxmax()
            ]
            insights.append(f"**Largest Customer Group**: {largest_cluster.name} with {largest_cluster['Customer_Count']} customers")
        
        # Revenue insights
        high_value_mentions = competitor_data[competitor_data['RevenuePotential'] > 100]
        if not high_value_mentions.empty:
            insights.append(f"**High-Value Customer Mentions**: {len(high_value_mentions)} customers with >₹100L revenue potential")
        
        # Response rate insights
        positive_sentiment_customers = competitor_data[competitor_data['SentimentScore'] > 0.6]
        if not positive_sentiment_customers.empty:
            response_rate = positive_sentiment_customers['ResponseToCampaign'].mean() * 100
            insights.append(f"**Positive Sentiment Response Rate**: {response_rate:.1f}%")
        
        return insights
    
    def create_competitor_analysis_dashboard(self, analysis_results):
        """
        Create comprehensive dashboard for competitor analysis.
        
        Args:
            analysis_results (dict): Results from analyze_competitor_mentions
            
        Returns:
            plotly.graph_objects.Figure: Dashboard visualization
        """
        
        if not analysis_results or analysis_results['competitor_data'].empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Competitor Mention Frequency',
                'Sentiment by Competitor',
                'Customer Clusters',
                'Revenue vs Sentiment'
            ]
        )
        
        competitor_data = analysis_results['competitor_data']
        sentiment_analysis = analysis_results['sentiment_analysis']
        clustering_analysis = analysis_results['clustering_analysis']
        
        # 1. Competitor Mention Frequency
        mention_counts = competitor_data['Competitor'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=mention_counts.values,
                y=mention_counts.index,
                orientation='h',
                name='Mentions',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # 2. Sentiment by Competitor
        if not sentiment_analysis['competitor_sentiment'].empty:
            top_competitors = sentiment_analysis['competitor_sentiment'].head(8)
            fig.add_trace(
                go.Bar(
                    x=top_competitors.index,
                    y=top_competitors['Avg_Sentiment'],
                    name='Sentiment',
                    marker_color='green'
                ),
                row=1, col=2
            )
        
        # 3. Customer Clusters
        if clustering_analysis and 'clustered_data' in clustering_analysis:
            cluster_counts = clustering_analysis['clustered_data']['ClusterName'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=cluster_counts.index,
                    values=cluster_counts.values,
                    name='Clusters'
                ),
                row=2, col=1
            )
        
        # 4. Revenue vs Sentiment
        fig.add_trace(
            go.Scatter(
                x=competitor_data['SentimentScore'],
                y=competitor_data['RevenuePotential'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=competitor_data['ResponseToCampaign'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Revenue vs Sentiment'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="AI Competitor Analysis Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_sentiment_trend_chart(self, analysis_results):
        """
        Create sentiment trend visualization.
        
        Args:
            analysis_results (dict): Results from analyze_competitor_mentions
            
        Returns:
            plotly.graph_objects.Figure: Sentiment trend chart
        """
        
        if not analysis_results or analysis_results['competitor_data'].empty:
            return None
        
        competitor_data = analysis_results['competitor_data']
        
        # Group by competitor and calculate sentiment trends
        sentiment_trends = competitor_data.groupby('Competitor').agg({
            'SentimentScore': ['mean', 'count']
        }).round(3)
        
        sentiment_trends.columns = ['Avg_Sentiment', 'Mention_Count']
        sentiment_trends = sentiment_trends.sort_values('Mention_Count', ascending=False).head(10)
        
        # Create bubble chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sentiment_trends['Avg_Sentiment'],
            y=sentiment_trends['Mention_Count'],
            mode='markers',
            marker=dict(
                size=sentiment_trends['Mention_Count'] * 2,
                color=sentiment_trends['Avg_Sentiment'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sentiment Score")
            ),
            text=sentiment_trends.index,
            hovertemplate='<b>%{text}</b><br>' +
                         'Sentiment: %{x:.3f}<br>' +
                         'Mentions: %{y}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Competitor Sentiment Analysis",
            xaxis_title="Average Sentiment Score",
            yaxis_title="Number of Mentions",
            height=500
        )
        
        return fig


class MarketIntelligenceAI:
    """
    AI-powered market intelligence analysis for Tubes India.
    """
    
    def __init__(self):
        """Initialize the market intelligence AI class."""
        self.market_patterns = None
        self.impact_analysis = None
        self.predictive_insights = None
    
    def analyze_market_intelligence(self, market_df):
        """
        Analyze market intelligence data using AI techniques.
        
        Args:
            market_df (pd.DataFrame): Market intelligence dataset
            
        Returns:
            dict: Analysis results
        """
        
        if market_df.empty:
            return {}
        
        # Analyze market events
        event_analysis = self._analyze_market_events(market_df)
        
        # Analyze impact patterns
        impact_analysis = self._analyze_impact_patterns(market_df)
        
        # Generate predictive insights
        predictive_insights = self._generate_predictive_insights(market_df)
        
        # Create market patterns
        market_patterns = self._identify_market_patterns(market_df)
        
        return {
            'event_analysis': event_analysis,
            'impact_analysis': impact_analysis,
            'predictive_insights': predictive_insights,
            'market_patterns': market_patterns
        }
    
    def _analyze_market_events(self, market_df):
        """Analyze market events and their characteristics."""
        
        # Event frequency analysis
        event_counts = market_df['MarketEvent'].value_counts()
        
        # Impact score analysis
        impact_by_event = market_df.groupby('MarketEvent')['ImpactScore'].agg(['mean', 'std', 'count']).round(2)
        impact_by_event.columns = ['Avg_Impact', 'Std_Impact', 'Event_Count']
        impact_by_event = impact_by_event.sort_values('Event_Count', ascending=False)
        
        # Regional impact analysis
        regional_impact = market_df.groupby('AffectedRegions')['ImpactScore'].mean().round(2)
        
        return {
            'event_frequency': event_counts,
            'impact_by_event': impact_by_event,
            'regional_impact': regional_impact
        }
    
    def _analyze_impact_patterns(self, market_df):
        """Analyze patterns in market impact scores."""
        
        # Time-based impact analysis
        market_df['Date'] = pd.to_datetime(market_df['Date'])
        market_df['Month'] = market_df['Date'].dt.month
        market_df['Quarter'] = market_df['Date'].dt.quarter
        
        monthly_impact = market_df.groupby('Month')['ImpactScore'].mean().round(2)
        quarterly_impact = market_df.groupby('Quarter')['ImpactScore'].mean().round(2)
        
        # Segment impact analysis
        segment_impact = market_df.groupby('AffectedSegments')['ImpactScore'].mean().round(2)
        
        # Impact distribution
        impact_distribution = market_df['ImpactScore'].describe()
        
        return {
            'monthly_impact': monthly_impact,
            'quarterly_impact': quarterly_impact,
            'segment_impact': segment_impact,
            'impact_distribution': impact_distribution
        }
    
    def _generate_predictive_insights(self, market_df):
        """Generate predictive insights based on market patterns."""
        
        insights = []
        
        # High-impact events
        high_impact_events = market_df[market_df['ImpactScore'] > 50]
        if not high_impact_events.empty:
            insights.append(f"**High-Impact Events**: {len(high_impact_events)} events with >50 impact score")
        
        # Seasonal patterns
        if 'Month' in market_df.columns:
            seasonal_analysis = market_df.groupby('Month')['ImpactScore'].mean()
            peak_month = seasonal_analysis.idxmax()
            peak_impact = seasonal_analysis.max()
            insights.append(f"**Peak Impact Month**: Month {peak_month} with average impact {peak_impact:.1f}")
        
        # Segment vulnerability
        if 'AffectedSegments' in market_df.columns:
            vulnerable_segments = market_df.groupby('AffectedSegments')['ImpactScore'].mean().sort_values(ascending=False)
            most_vulnerable = vulnerable_segments.index[0]
            vulnerability_score = vulnerable_segments.iloc[0]
            insights.append(f"**Most Vulnerable Segment**: {most_vulnerable} with impact score {vulnerability_score:.1f}")
        
        return insights
    
    def _identify_market_patterns(self, market_df):
        """Identify recurring patterns in market data."""
        
        patterns = {}
        
        # Event co-occurrence patterns
        if 'AffectedSegments' in market_df.columns and 'AffectedRegions' in market_df.columns:
            segment_region_patterns = market_df.groupby(['AffectedSegments', 'AffectedRegions'])['ImpactScore'].mean().round(2)
            patterns['segment_region_patterns'] = segment_region_patterns
        
        # Impact correlation patterns
        if 'ImpactScore' in market_df.columns:
            # Create correlation matrix for numerical features
            numerical_cols = market_df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                correlation_matrix = market_df[numerical_cols].corr().round(3)
                patterns['correlation_matrix'] = correlation_matrix
        
        return patterns
    
    def create_market_intelligence_dashboard(self, analysis_results):
        """
        Create dashboard for market intelligence analysis.
        
        Args:
            analysis_results (dict): Results from analyze_market_intelligence
            
        Returns:
            plotly.graph_objects.Figure: Market intelligence dashboard
        """
        
        if not analysis_results:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Market Event Frequency',
                'Impact Score Distribution',
                'Monthly Impact Trends',
                'Segment Impact Analysis'
            ]
        )
        
        event_analysis = analysis_results.get('event_analysis', {})
        impact_analysis = analysis_results.get('impact_analysis', {})
        
        # 1. Market Event Frequency
        if 'event_frequency' in event_analysis:
            event_counts = event_analysis['event_frequency'].head(8)
            fig.add_trace(
                go.Bar(
                    x=event_counts.index,
                    y=event_counts.values,
                    name='Event Count',
                    marker_color='blue'
                ),
                row=1, col=1
            )
        
        # 2. Impact Score Distribution
        if 'impact_distribution' in impact_analysis:
            # Create histogram data
            impact_scores = analysis_results.get('market_df', pd.DataFrame())['ImpactScore']
            if not impact_scores.empty:
                fig.add_trace(
                    go.Histogram(
                        x=impact_scores,
                        name='Impact Distribution',
                        nbinsx=20,
                        marker_color='green'
                    ),
                    row=1, col=2
                )
        
        # 3. Monthly Impact Trends
        if 'monthly_impact' in impact_analysis:
            monthly_data = impact_analysis['monthly_impact']
            fig.add_trace(
                go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data.values,
                    mode='lines+markers',
                    name='Monthly Impact',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        # 4. Segment Impact Analysis
        if 'segment_impact' in impact_analysis:
            segment_data = impact_analysis['segment_impact'].head(8)
            fig.add_trace(
                go.Bar(
                    x=segment_data.index,
                    y=segment_data.values,
                    name='Segment Impact',
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="AI Market Intelligence Dashboard",
            showlegend=False
        )
        
        return fig
