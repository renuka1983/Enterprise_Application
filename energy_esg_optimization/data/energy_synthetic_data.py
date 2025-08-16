"""
Synthetic Data Generation Module for Energy Optimization & ESG Compliance
Manufacturing Energy Data - Consumption, Emissions, Production, Compliance
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class EnergyESGDataGenerator:
    """
    Generates synthetic energy and ESG data for manufacturing facilities
    with realistic patterns, seasonality, and correlations.
    """
    
    def __init__(self, seed=42):
        """Initialize the energy and ESG data generator."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Define manufacturing plant types
        self.plant_types = [
            'Steel Manufacturing', 'Automotive Assembly', 'Chemical Processing',
            'Electronics Manufacturing', 'Food Processing', 'Textile Manufacturing',
            'Pharmaceutical Production', 'Paper Manufacturing', 'Cement Production'
        ]
        
        # Define energy sources
        self.energy_sources = [
            'Grid Electricity', 'Natural Gas', 'Diesel', 'Coal', 'Biomass',
            'Solar PV', 'Wind Power', 'Hydroelectric', 'Nuclear'
        ]
        
        # Define compliance categories
        self.compliance_categories = [
            'ISO 50001', 'ISO 14001', 'LEED Certification', 'Carbon Disclosure Project',
            'RE100', 'Science Based Targets', 'GRI Standards', 'SASB Standards'
        ]
        
        # Define ESG factors
        self.esg_factors = [
            'Carbon Footprint', 'Water Usage', 'Waste Management', 'Air Quality',
            'Employee Safety', 'Community Impact', 'Supply Chain Ethics', 'Biodiversity'
        ]
    
    def generate_energy_data(self, num_days=365, num_plants=5):
        """
        Generate comprehensive energy consumption and ESG data.
        
        Args:
            num_days (int): Number of days to generate data for
            num_plants (int): Number of manufacturing plants
            
        Returns:
            tuple: (energy_df, production_df, compliance_df, summary)
        """
        
        # Generate date range
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(num_days)]
        
        # Generate plant data
        plants = [f"Plant_{i+1:02d}" for i in range(num_plants)]
        plant_types = np.random.choice(self.plant_types, num_plants, replace=False)
        
        # Initialize data containers
        energy_data = []
        production_data = []
        compliance_data = []
        
        for plant_idx, (plant, plant_type) in enumerate(zip(plants, plant_types)):
            # Base energy consumption by plant type
            base_consumption = self._get_base_consumption(plant_type)
            base_production = self._get_base_production(plant_type)
            
            for day_idx, date in enumerate(dates):
                # Add seasonality and trends
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_idx / 365)
                trend_factor = 1 + 0.001 * day_idx  # Slight upward trend
                
                # Generate energy consumption with realistic patterns
                daily_consumption = base_consumption * seasonal_factor * trend_factor
                daily_consumption += np.random.normal(0, daily_consumption * 0.1)  # Add noise
                
                # Generate production data
                daily_production = base_production * seasonal_factor * trend_factor
                daily_production += np.random.normal(0, daily_production * 0.15)
                
                # Calculate CO2 emissions (kg CO2)
                grid_electricity = daily_consumption * 0.6 * 0.5  # 60% grid, 0.5 kg CO2/kWh
                natural_gas = daily_consumption * 0.3 * 0.2  # 30% gas, 0.2 kg CO2/kWh
                renewable = daily_consumption * 0.1 * 0.0  # 10% renewable, 0 kg CO2
                daily_emissions = grid_electricity + natural_gas + renewable
                
                # Calculate renewable energy percentage
                renewable_percentage = 0.1 + 0.05 * np.sin(2 * np.pi * day_idx / 365)  # Seasonal variation
                renewable_percentage = min(max(renewable_percentage, 0.05), 0.25)  # 5-25% range
                
                # Generate downtime data
                downtime_hours = np.random.exponential(2)  # Exponential distribution
                downtime_hours = min(downtime_hours, 8)  # Max 8 hours per day
                
                # Calculate energy efficiency
                energy_efficiency = daily_production / daily_consumption
                energy_efficiency = max(energy_efficiency, 0.1)  # Minimum efficiency
                
                # Energy data
                energy_data.append({
                    'Date': date,
                    'Plant': plant,
                    'PlantType': plant_type,
                    'EnergyConsumption_kWh': round(daily_consumption, 2),
                    'CO2Emissions_kg': round(daily_emissions, 2),
                    'RenewableEnergy_Percentage': round(renewable_percentage * 100, 1),
                    'EnergyEfficiency': round(energy_efficiency, 3),
                    'PeakDemand_kW': round(daily_consumption / 24 * np.random.uniform(1.2, 1.8), 2),
                    'EnergyCost_USD': round(daily_consumption * np.random.uniform(0.08, 0.15), 2)
                })
                
                # Production data
                production_data.append({
                    'Date': date,
                    'Plant': plant,
                    'PlantType': plant_type,
                    'Production_Units': round(daily_production, 2),
                    'Downtime_Hours': round(downtime_hours, 2),
                    'ProductionEfficiency': round((24 - downtime_hours) / 24 * 100, 1),
                    'QualityScore': round(np.random.uniform(85, 98), 1),
                    'MaintenanceHours': round(np.random.exponential(1), 2)
                })
                
                # Compliance data (monthly updates)
                if day_idx % 30 == 0 or day_idx == 0:
                    compliance_score = np.random.uniform(70, 95)
                    compliance_data.append({
                        'Date': date,
                        'Plant': plant,
                        'PlantType': plant_type,
                        'ComplianceScore': round(compliance_score, 1),
                        'ISO50001_Status': 'Certified' if compliance_score > 85 else 'In Progress',
                        'ISO14001_Status': 'Certified' if compliance_score > 80 else 'In Progress',
                        'CarbonDisclosure_Score': round(compliance_score, 0),
                        'ESG_Rating': self._get_esg_rating(compliance_score),
                        'Audit_Status': 'Passed' if compliance_score > 80 else 'Action Required',
                        'NextAudit_Date': date + timedelta(days=365)
                    })
        
        # Create DataFrames
        energy_df = pd.DataFrame(energy_data)
        production_df = pd.DataFrame(production_data)
        compliance_df = pd.DataFrame(compliance_data)
        
        # Generate summary statistics
        summary = self._generate_summary(energy_df, production_df, compliance_df)
        
        return energy_df, production_df, compliance_df, summary
    
    def _get_base_consumption(self, plant_type):
        """Get base energy consumption by plant type (kWh/day)."""
        base_consumption = {
            'Steel Manufacturing': 50000,
            'Automotive Assembly': 25000,
            'Chemical Processing': 75000,
            'Electronics Manufacturing': 15000,
            'Food Processing': 20000,
            'Textile Manufacturing': 30000,
            'Pharmaceutical Production': 35000,
            'Paper Manufacturing': 40000,
            'Cement Production': 100000
        }
        return base_consumption.get(plant_type, 30000)
    
    def _get_base_production(self, plant_type):
        """Get base production by plant type (units/day)."""
        base_production = {
            'Steel Manufacturing': 1000,
            'Automotive Assembly': 500,
            'Chemical Processing': 2000,
            'Electronics Manufacturing': 10000,
            'Food Processing': 5000,
            'Textile Manufacturing': 3000,
            'Pharmaceutical Production': 1000,
            'Paper Manufacturing': 8000,
            'Cement Production': 5000
        }
        return base_production.get(plant_type, 2000)
    
    def _get_esg_rating(self, compliance_score):
        """Convert compliance score to ESG rating."""
        if compliance_score >= 90:
            return 'AAA'
        elif compliance_score >= 80:
            return 'AA'
        elif compliance_score >= 70:
            return 'A'
        elif compliance_score >= 60:
            return 'BBB'
        else:
            return 'BB'
    
    def _generate_summary(self, energy_df, production_df, compliance_df):
        """Generate comprehensive summary statistics."""
        
        summary = {
            'total_plants': energy_df['Plant'].nunique(),
            'total_days': len(energy_df),
            'avg_daily_consumption': energy_df['EnergyConsumption_kWh'].mean(),
            'total_energy_consumption': energy_df['EnergyConsumption_kWh'].sum(),
            'avg_daily_emissions': energy_df['CO2Emissions_kg'].mean(),
            'total_emissions': energy_df['CO2Emissions_kg'].sum(),
            'avg_renewable_percentage': energy_df['RenewableEnergy_Percentage'].mean(),
            'avg_energy_efficiency': energy_df['EnergyEfficiency'].mean(),
            'avg_production': production_df['Production_Units'].mean(),
            'total_production': production_df['Production_Units'].sum(),
            'avg_downtime': production_df['Downtime_Hours'].mean(),
            'avg_compliance_score': compliance_df['ComplianceScore'].mean(),
            'certified_plants': len(compliance_df[compliance_df['ISO50001_Status'] == 'Certified']),
            'energy_cost_savings_potential': energy_df['EnergyCost_USD'].sum() * 0.15,  # 15% savings potential
            'emissions_reduction_potential': energy_df['CO2Emissions_kg'].sum() * 0.25  # 25% reduction potential
        }
        
        return summary
    
    def save_datasets(self, energy_df, production_df, compliance_df, base_path="./"):
        """
        Save generated datasets to CSV files.
        
        Args:
            energy_df (pd.DataFrame): Energy consumption dataset
            production_df (pd.DataFrame): Production dataset
            compliance_df (pd.DataFrame): Compliance dataset
            base_path (str): Base path for saving files
            
        Returns:
            tuple: (energy_file, production_file, compliance_file)
        """
        
        # Save energy dataset
        energy_file = f"{base_path}/manufacturing_energy_data.csv"
        energy_df.to_csv(energy_file, index=False)
        
        # Save production dataset
        production_file = f"{base_path}/manufacturing_production_data.csv"
        production_df.to_csv(production_file, index=False)
        
        # Save compliance dataset
        compliance_file = f"{base_path}/manufacturing_compliance_data.csv"
        compliance_df.to_csv(compliance_file, index=False)
        
        return energy_file, production_file, compliance_file


def generate_sample_data(num_days=365, num_plants=5):
    """
    Convenience function to generate sample energy and ESG data.
    
    Args:
        num_days (int): Number of days to generate data for
        num_plants (int): Number of manufacturing plants
        
    Returns:
        tuple: (energy_df, production_df, compliance_df, summary)
    """
    
    # Initialize generator
    generator = EnergyESGDataGenerator(seed=42)
    
    # Generate datasets
    energy_df, production_df, compliance_df, summary = generator.generate_energy_data(num_days, num_plants)
    
    return energy_df, production_df, compliance_df, summary


if __name__ == "__main__":
    # Test the data generation
    print("Generating sample energy and ESG data...")
    
    energy_df, production_df, compliance_df, summary = generate_sample_data(365, 5)
    
    print(f"\nGenerated {len(energy_df)} energy records for {summary['total_plants']} plants")
    print(f"Data Summary:")
    print(f"Total Energy Consumption: {summary['total_energy_consumption']:,.0f} kWh")
    print(f"Total CO2 Emissions: {summary['total_emissions']:,.0f} kg")
    print(f"Average Renewable Energy: {summary['avg_renewable_percentage']:.1f}%")
    print(f"Total Production: {summary['total_production']:,.0f} units")
    print(f"Average Compliance Score: {summary['avg_compliance_score']:.1f}")
    
    # Save datasets
    generator = EnergyESGDataGenerator()
    energy_file, production_file, compliance_file = generator.save_datasets(energy_df, production_df, compliance_df)
    print(f"\nDatasets saved to:")
    print(f"Energy: {energy_file}")
    print(f"Production: {production_file}")
    print(f"Compliance: {compliance_file}")
