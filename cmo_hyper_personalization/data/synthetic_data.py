"""
Synthetic Data Generation Module for CMO Hyper-Personalization
Manufacturing Products - Market Intelligence & Customer Segmentation
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class ManufacturingDataGenerator:
    """
    Generates synthetic customer data for manufacturing products
    with realistic market intelligence and customer behavior patterns.
    """
    
    def __init__(self, seed=42):
        """Initialize the data generator with a random seed."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Define manufacturing product categories
        self.product_categories = [
            'Steel Tubes', 'Aluminum Tubes', 'Copper Tubes', 
            'Plastic Tubes', 'Composite Tubes', 'Specialty Tubes'
        ]
        
        # Define customer segments
        self.customer_segments = [
            'Enterprise', 'SMB', 'Startup', 'Government', 'Educational'
        ]
        
        # Define regions with realistic market presence
        self.regions = [
            'North India', 'South India', 'East India', 'West India', 
            'Central India', 'Northeast India'
        ]
        
        # Define competitor names for realistic market analysis
        self.competitors = [
            'Tata Steel', 'JSW Steel', 'SAIL', 'Essar Steel', 
            'Bhushan Steel', 'Vedanta', 'Hindalco', 'NALCO'
        ]
        
        # Define campaign types
        self.campaign_types = [
            'Product Launch', 'Seasonal Discount', 'Bulk Purchase', 
            'Loyalty Program', 'New Customer', 'Cross-sell'
        ]
    
    def generate_customer_data(self, num_customers=1000):
        """
        Generate comprehensive customer dataset for manufacturing.
        
        Args:
            num_customers (int): Number of customers to generate
            
        Returns:
            pd.DataFrame: Customer dataset with all required columns
        """
        
        # Generate customer IDs
        customer_ids = [f"CUST{i:06d}" for i in range(1, num_customers + 1)]
        
        # Generate customer segments with realistic distribution
        segments = np.random.choice(
            self.customer_segments, 
            size=num_customers, 
            p=[0.25, 0.35, 0.20, 0.15, 0.05]  # Enterprise and SMB dominate
        )
        
        # Generate regions with realistic distribution
        regions = np.random.choice(
            self.regions, 
            size=num_customers, 
            p=[0.30, 0.25, 0.20, 0.15, 0.08, 0.02]  # North and South dominate
        )
        
        # Generate past purchases (number of orders)
        past_purchases = np.random.poisson(3, num_customers)  # Most customers have 1-5 orders
        past_purchases = np.clip(past_purchases, 0, 20)
        
        # Generate website visits (monthly average)
        website_visits = np.random.exponential(8, num_customers)  # Exponential distribution
        website_visits = np.clip(website_visits, 0, 50)
        
        # Generate competitor mentions (social media and reviews)
        competitor_mentions = self._generate_competitor_mentions(num_customers)
        
        # Generate revenue potential (annual in lakhs)
        revenue_potential = self._generate_revenue_potential(segments, regions, past_purchases)
        
        # Generate campaign response (binary: 0=No, 1=Yes)
        response_to_campaign = self._generate_campaign_response(
            segments, regions, past_purchases, website_visits, revenue_potential
        )
        
        # Create the dataset
        data = {
            'CustomerID': customer_ids,
            'Segment': segments,
            'Region': regions,
            'PastPurchases': past_purchases,
            'WebsiteVisits': np.round(website_visits, 1),
            'CompetitorMentions': competitor_mentions,
            'RevenuePotential': np.round(revenue_potential, 2),
            'ResponseToCampaign': response_to_campaign
        }
        
        return pd.DataFrame(data)
    
    def _generate_competitor_mentions(self, num_customers):
        """Generate realistic competitor mentions based on market dynamics."""
        
        mentions = []
        for _ in range(num_customers):
            # Base probability of mentioning competitors
            base_prob = 0.3
            
            # Generate number of mentions (0-5)
            num_mentions = np.random.poisson(1.5)
            num_mentions = min(num_mentions, 5)
            
            if num_mentions == 0:
                mentions.append("None")
            else:
                # Select random competitors
                selected_competitors = random.sample(self.competitors, min(num_mentions, len(self.competitors)))
                mentions.append(", ".join(selected_competitors))
        
        return mentions
    
    def _generate_revenue_potential(self, segments, regions, past_purchases):
        """Generate revenue potential based on segment, region, and purchase history."""
        
        revenue_potential = []
        
        for i in range(len(segments)):
            # Base revenue by segment
            base_revenue = {
                'Enterprise': 150,    # 150 lakhs base
                'SMB': 75,           # 75 lakhs base
                'Startup': 25,       # 25 lakhs base
                'Government': 200,   # 200 lakhs base
                'Educational': 40    # 40 lakhs base
            }
            
            # Regional multiplier
            regional_multiplier = {
                'North India': 1.2,      # Higher purchasing power
                'South India': 1.1,      # Good market
                'East India': 0.9,       # Developing market
                'West India': 1.3,       # Industrial hub
                'Central India': 0.8,    # Developing market
                'Northeast India': 0.7   # Emerging market
            }
            
            # Purchase history multiplier
            purchase_multiplier = 1 + (past_purchases[i] * 0.1)
            
            # Calculate revenue potential
            base = base_revenue[segments[i]]
            regional = regional_multiplier[regions[i]]
            purchase = purchase_multiplier
            
            # Add some randomness
            random_factor = np.random.uniform(0.7, 1.3)
            
            revenue = base * regional * purchase * random_factor
            revenue_potential.append(revenue)
        
        return revenue_potential
    
    def _generate_campaign_response(self, segments, regions, past_purchases, website_visits, revenue_potential):
        """Generate campaign response based on customer characteristics."""
        
        response_probabilities = []
        
        for i in range(len(segments)):
            # Base response probability
            base_prob = 0.3
            
            # Segment-based adjustments
            segment_adjustments = {
                'Enterprise': 0.1,    # Lower response rate (more selective)
                'SMB': 0.2,           # Moderate response rate
                'Startup': 0.3,       # Higher response rate (seeking solutions)
                'Government': 0.05,   # Very low response rate
                'Educational': 0.25   # Moderate response rate
            }
            
            # Regional adjustments
            regional_adjustments = {
                'North India': 0.1,      # Higher response rate
                'South India': 0.05,     # Moderate response rate
                'East India': 0.15,      # Higher response rate (developing)
                'West India': 0.1,       # Higher response rate
                'Central India': 0.2,    # Higher response rate (developing)
                'Northeast India': 0.25  # Highest response rate (emerging)
            }
            
            # Behavioral adjustments
            purchase_adjustment = min(past_purchases[i] * 0.02, 0.2)  # More purchases = higher response
            website_adjustment = min(website_visits[i] * 0.01, 0.15)  # More visits = higher response
            revenue_adjustment = min(revenue_potential[i] / 1000, 0.1)  # Higher revenue = higher response
            
            # Calculate final probability
            final_prob = (base_prob + 
                         segment_adjustments[segments[i]] +
                         regional_adjustments[regions[i]] +
                         purchase_adjustment +
                         website_adjustment +
                         revenue_adjustment)
            
            # Ensure probability is between 0 and 1
            final_prob = np.clip(final_prob, 0.05, 0.95)
            response_probabilities.append(final_prob)
        
        # Generate binary responses
        responses = np.random.binomial(1, response_probabilities)
        return responses
    
    def generate_market_intelligence_data(self, num_records=500):
        """
        Generate additional market intelligence data for analysis.
        
        Args:
            num_records (int): Number of market intelligence records
            
        Returns:
            pd.DataFrame: Market intelligence dataset
        """
        
        # Generate dates for the last 12 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        dates = []
        for _ in range(num_records):
            random_date = start_date + timedelta(
                days=np.random.randint(0, 365)
            )
            dates.append(random_date.strftime('%Y-%m-%d'))
        
        # Generate market events
        market_events = [
            'Steel Price Fluctuation', 'New Government Policy', 'Infrastructure Project Announcement',
            'Competitor Product Launch', 'Raw Material Shortage', 'Export Policy Change',
            'Technology Innovation', 'Market Expansion', 'Regulatory Compliance Update'
        ]
        
        events = np.random.choice(market_events, size=num_records)
        
        # Generate impact scores (-100 to +100)
        impact_scores = np.random.normal(0, 25, num_records)
        impact_scores = np.clip(impact_scores, -100, 100)
        
        # Generate affected segments
        affected_segments = []
        for _ in range(num_records):
            num_segments = np.random.randint(1, 4)
            selected_segments = random.sample(self.customer_segments, num_segments)
            affected_segments.append(", ".join(selected_segments))
        
        # Generate affected regions
        affected_regions = []
        for _ in range(num_records):
            num_regions = np.random.randint(1, 4)
            selected_regions = random.sample(self.regions, num_regions)
            affected_regions.append(", ".join(selected_regions))
        
        # Create market intelligence dataset
        market_data = {
            'Date': dates,
            'MarketEvent': events,
            'ImpactScore': np.round(impact_scores, 1),
            'AffectedSegments': affected_segments,
            'AffectedRegions': affected_regions
        }
        
        return pd.DataFrame(market_data)
    
    def save_datasets(self, customer_df, market_df, base_path="./"):
        """
        Save generated datasets to CSV files.
        
        Args:
            customer_df (pd.DataFrame): Customer dataset
            market_df (pd.DataFrame): Market intelligence dataset
            base_path (str): Base path for saving files
        """
        
        # Save customer dataset
        customer_file = f"{base_path}/manufacturing_customers.csv"
        customer_df.to_csv(customer_file, index=False)
        
        # Save market intelligence dataset
        market_file = f"{base_path}/manufacturing_market_intelligence.csv"
        market_df.to_csv(market_file, index=False)
        
        return customer_file, market_file
    
    def get_data_summary(self, customer_df):
        """
        Generate a comprehensive summary of the customer dataset.
        
        Args:
            customer_df (pd.DataFrame): Customer dataset
            
        Returns:
            dict: Summary statistics
        """
        
        summary = {
            'total_customers': len(customer_df),
            'segments': customer_df['Segment'].value_counts().to_dict(),
            'regions': customer_df['Region'].value_counts().to_dict(),
            'avg_past_purchases': customer_df['PastPurchases'].mean(),
            'avg_website_visits': customer_df['WebsiteVisits'].mean(),
            'avg_revenue_potential': customer_df['RevenuePotential'].mean(),
            'campaign_response_rate': customer_df['ResponseToCampaign'].mean() * 100,
            'total_revenue_potential': customer_df['RevenuePotential'].sum(),
            'high_value_customers': len(customer_df[customer_df['RevenuePotential'] > 100]),
            'active_customers': len(customer_df[customer_df['WebsiteVisits'] > 10])
        }
        
        return summary


def generate_sample_data(num_customers=1000, num_market_records=500):
    """
    Convenience function to generate sample data for manufacturing.
    
    Args:
        num_customers (int): Number of customers to generate
        num_market_records (int): Number of market intelligence records
        
    Returns:
        tuple: (customer_df, market_df, summary)
    """
    
    # Initialize generator
    generator = ManufacturingDataGenerator(seed=42)
    
    # Generate datasets
    customer_df = generator.generate_customer_data(num_customers)
    market_df = generator.generate_market_intelligence_data(num_market_records)
    
    # Generate summary
    summary = generator.get_data_summary(customer_df)
    
    return customer_df, market_df, summary


if __name__ == "__main__":
    # Test the data generation
    print("Generating sample manufacturing customer data...")
    
    customer_df, market_df, summary = generate_sample_data(1000, 500)
    
    print(f"\nGenerated {len(customer_df)} customers and {len(market_df)} market records")
    print(f"\nData Summary:")
    print(f"Total Customers: {summary['total_customers']:,}")
    print(f"Campaign Response Rate: {summary['campaign_response_rate']:.1f}%")
    print(f"Total Revenue Potential: ₹{summary['total_revenue_potential']:.0f} lakhs")
    print(f"High Value Customers (>₹100L): {summary['high_value_customers']:,}")
    
    # Save datasets
    customer_file, market_file = generator.save_datasets(customer_df, market_df)
    print(f"\nDatasets saved to:")
    print(f"Customers: {customer_file}")
    print(f"Market Intelligence: {market_file}")
