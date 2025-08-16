"""
AI Optimization Engine for Energy & ESG Compliance
Rule-based optimization and reinforcement learning simulation
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class EnergyOptimizationEngine:
    """
    AI-powered energy optimization engine using rule-based systems
    and simple reinforcement learning simulation.
    """
    
    def __init__(self):
        """Initialize the energy optimization engine."""
        self.optimization_rules = self._initialize_optimization_rules()
        self.optimization_history = []
        self.reinforcement_learning_state = {}
        
    def _initialize_optimization_rules(self):
        """Initialize optimization rules for different scenarios."""
        
        rules = {
            'peak_demand_management': {
                'condition': 'Peak demand > 80% of capacity',
                'action': 'Shift non-critical operations to off-peak hours',
                'expected_savings': '15-25% peak demand reduction',
                'priority': 'High'
            },
            'energy_efficiency': {
                'condition': 'Energy efficiency < 0.8',
                'action': 'Implement equipment maintenance and process optimization',
                'expected_savings': '10-20% energy consumption reduction',
                'priority': 'High'
            },
            'renewable_integration': {
                'condition': 'Renewable energy < 20%',
                'action': 'Increase solar/wind integration and energy storage',
                'expected_savings': '20-30% emissions reduction',
                'priority': 'Medium'
            },
            'production_scheduling': {
                'condition': 'Production efficiency < 85%',
                'action': 'Optimize production schedules and reduce downtime',
                'expected_savings': '15-25% production improvement',
                'priority': 'Medium'
            },
            'maintenance_optimization': {
                'condition': 'Maintenance hours > 2 per day',
                'action': 'Implement predictive maintenance and condition monitoring',
                'expected_savings': '10-15% maintenance cost reduction',
                'priority': 'Low'
            }
        }
        
        return rules
    
    def analyze_energy_patterns(self, energy_df, production_df):
        """
        Analyze energy consumption patterns and identify optimization opportunities.
        
        Args:
            energy_df (pd.DataFrame): Energy consumption data
            production_df (pd.DataFrame): Production data
            
        Returns:
            dict: Analysis results with optimization recommendations
        """
        
        # Merge datasets
        df = energy_df.merge(production_df, on=['Date', 'Plant', 'PlantType'])
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Calculate key metrics
        analysis_results = {
            'peak_demand_analysis': self._analyze_peak_demand(df),
            'efficiency_analysis': self._analyze_efficiency_trends(df),
            'seasonal_patterns': self._analyze_seasonal_patterns(df),
            'plant_comparison': self._analyze_plant_performance(df),
            'optimization_opportunities': self._identify_optimization_opportunities(df),
            'cost_analysis': self._analyze_cost_implications(df)
        }
        
        return analysis_results
    
    def _analyze_peak_demand(self, df):
        """Analyze peak demand patterns."""
        
        peak_analysis = df.groupby('Plant').agg({
            'PeakDemand_kW': ['mean', 'max', 'std'],
            'EnergyConsumption_kWh': 'mean'
        }).round(2)
        
        # Identify plants with high peak demand
        high_peak_plants = df[df['PeakDemand_kW'] > df['PeakDemand_kW'].quantile(0.8)]['Plant'].unique()
        
        return {
            'peak_demand_stats': peak_analysis,
            'high_peak_plants': high_peak_plants.tolist(),
            'peak_demand_reduction_potential': '15-25%',
            'recommendations': [
                'Implement demand response programs',
                'Shift non-critical operations to off-peak hours',
                'Install energy storage systems'
            ]
        }
    
    def _analyze_efficiency_trends(self, df):
        """Analyze energy efficiency trends."""
        
        efficiency_trends = df.groupby(['Plant', pd.Grouper(key='Date', freq='M')]).agg({
            'EnergyEfficiency': 'mean',
            'ProductionEfficiency': 'mean'
        }).reset_index()
        
        # Calculate efficiency improvement potential
        current_efficiency = df['EnergyEfficiency'].mean()
        target_efficiency = 0.9  # 90% target
        improvement_potential = ((target_efficiency - current_efficiency) / current_efficiency) * 100
        
        return {
            'efficiency_trends': efficiency_trends,
            'current_efficiency': round(current_efficiency, 3),
            'target_efficiency': target_efficiency,
            'improvement_potential': round(improvement_potential, 1),
            'recommendations': [
                'Implement energy management systems',
                'Optimize production processes',
                'Regular equipment maintenance'
            ]
        }
    
    def _analyze_seasonal_patterns(self, df):
        """Analyze seasonal energy consumption patterns."""
        
        seasonal_data = df.groupby(['Month', 'Plant']).agg({
            'EnergyConsumption_kWh': 'mean',
            'CO2Emissions_kg': 'mean',
            'RenewableEnergy_Percentage': 'mean'
        }).reset_index()
        
        # Identify seasonal trends
        monthly_consumption = df.groupby('Month')['EnergyConsumption_kWh'].mean()
        peak_month = monthly_consumption.idxmax()
        low_month = monthly_consumption.idxmin()
        
        return {
            'seasonal_data': seasonal_data,
            'peak_consumption_month': peak_month,
            'low_consumption_month': low_month,
            'seasonal_variation': round((monthly_consumption.max() - monthly_consumption.min()) / monthly_consumption.mean() * 100, 1),
            'recommendations': [
                'Implement seasonal load balancing',
                'Optimize HVAC systems for seasonal changes',
                'Plan maintenance during low-demand periods'
            ]
        }
    
    def _analyze_plant_performance(self, df):
        """Compare performance across different plants."""
        
        plant_performance = df.groupby('Plant').agg({
            'EnergyConsumption_kWh': 'mean',
            'CO2Emissions_kg': 'mean',
            'EnergyEfficiency': 'mean',
            'Production_Units': 'mean',
            'ProductionEfficiency': 'mean',
            'RenewableEnergy_Percentage': 'mean'
        }).round(3)
        
        # Calculate performance scores
        plant_performance['Performance_Score'] = (
            plant_performance['EnergyEfficiency'] * 0.3 +
            plant_performance['ProductionEfficiency'] * 0.3 +
            (plant_performance['RenewableEnergy_Percentage'] / 100) * 0.2 +
            (1 - plant_performance['CO2Emissions_kg'] / plant_performance['CO2Emissions_kg'].max()) * 0.2
        )
        
        # Rank plants by performance
        plant_performance = plant_performance.sort_values('Performance_Score', ascending=False)
        
        return {
            'plant_performance': plant_performance,
            'top_performer': plant_performance.index[0],
            'bottom_performer': plant_performance.index[-1],
            'performance_gap': round(plant_performance['Performance_Score'].max() - plant_performance['Performance_Score'].min(), 3),
            'recommendations': [
                'Share best practices from top performers',
                'Implement improvement programs for underperformers',
                'Standardize energy management processes'
            ]
        }
    
    def _identify_optimization_opportunities(self, df):
        """Identify specific optimization opportunities."""
        
        opportunities = []
        
        # Check each optimization rule
        for rule_name, rule in self.optimization_rules.items():
            if rule_name == 'peak_demand_management':
                high_peak_plants = df[df['PeakDemand_kW'] > df['PeakDemand_kW'].quantile(0.8)]['Plant'].unique()
                if len(high_peak_plants) > 0:
                    opportunities.append({
                        'rule': rule_name,
                        'condition': rule['condition'],
                        'action': rule['action'],
                        'affected_plants': high_peak_plants.tolist(),
                        'expected_savings': rule['expected_savings'],
                        'priority': rule['priority']
                    })
            
            elif rule_name == 'energy_efficiency':
                low_efficiency_plants = df[df['EnergyEfficiency'] < 0.8]['Plant'].unique()
                if len(low_efficiency_plants) > 0:
                    opportunities.append({
                        'rule': rule_name,
                        'condition': rule['condition'],
                        'action': rule['action'],
                        'affected_plants': low_efficiency_plants.tolist(),
                        'expected_savings': rule['expected_savings'],
                        'priority': rule['priority']
                    })
            
            elif rule_name == 'renewable_integration':
                low_renewable_plants = df[df['RenewableEnergy_Percentage'] < 20]['Plant'].unique()
                if len(low_renewable_plants) > 0:
                    opportunities.append({
                        'rule': rule_name,
                        'condition': rule['condition'],
                        'action': rule['action'],
                        'affected_plants': low_renewable_plants.tolist(),
                        'expected_savings': rule['expected_savings'],
                        'priority': rule['priority']
                    })
        
        return opportunities
    
    def _analyze_cost_implications(self, df):
        """Analyze cost implications of current energy usage."""
        
        total_energy_cost = df['EnergyCost_USD'].sum()
        avg_daily_cost = df.groupby('Plant')['EnergyCost_USD'].mean()
        
        # Calculate potential savings
        potential_savings = {
            'peak_demand_reduction': total_energy_cost * 0.15,  # 15% from peak demand management
            'efficiency_improvement': total_energy_cost * 0.20,  # 20% from efficiency improvements
            'renewable_integration': total_energy_cost * 0.10,  # 10% from renewable integration
            'maintenance_optimization': total_energy_cost * 0.05   # 5% from maintenance optimization
        }
        
        total_potential_savings = sum(potential_savings.values())
        
        return {
            'total_energy_cost': round(total_energy_cost, 2),
            'avg_daily_cost_by_plant': avg_daily_cost.round(2),
            'potential_savings': {k: round(v, 2) for k, v in potential_savings.items()},
            'total_potential_savings': round(total_potential_savings, 2),
            'roi_estimate': round(total_potential_savings / total_energy_cost * 100, 1)
        }
    
    def run_reinforcement_learning_simulation(self, energy_df, production_df, simulation_days=30):
        """
        Run a simple reinforcement learning simulation for energy optimization.
        
        Args:
            energy_df (pd.DataFrame): Historical energy data
            production_df (pd.DataFrame): Historical production data
            simulation_days (int): Number of days to simulate
            
        Returns:
            dict: Simulation results and optimization actions
        """
        
        # Initialize simulation environment
        current_state = self._initialize_simulation_state(energy_df, production_df)
        simulation_results = []
        
        # Run simulation
        for day in range(simulation_days):
            # Get current state
            state_features = self._extract_state_features(current_state)
            
            # Choose action using epsilon-greedy policy
            action = self._choose_optimization_action(state_features, epsilon=0.1)
            
            # Execute action and get reward
            reward, new_state = self._execute_action(action, current_state)
            
            # Update state
            current_state = new_state
            
            # Record results
            simulation_results.append({
                'day': day + 1,
                'action': action,
                'reward': reward,
                'energy_consumption': current_state['energy_consumption'],
                'energy_efficiency': current_state['energy_efficiency'],
                'renewable_percentage': current_state['renewable_percentage'],
                'total_cost': current_state['total_cost']
            })
        
        # Analyze simulation results
        simulation_analysis = self._analyze_simulation_results(simulation_results)
        
        return {
            'simulation_results': simulation_results,
            'simulation_analysis': simulation_analysis,
            'optimization_actions': self._extract_optimization_actions(simulation_results)
        }
    
    def _initialize_simulation_state(self, energy_df, production_df):
        """Initialize the simulation state."""
        
        # Get average values from historical data
        avg_energy = energy_df['EnergyConsumption_kWh'].mean()
        avg_efficiency = energy_df['EnergyEfficiency'].mean()
        avg_renewable = energy_df['RenewableEnergy_Percentage'].mean()
        avg_cost = energy_df['EnergyCost_USD'].mean()
        
        return {
            'energy_consumption': avg_energy,
            'energy_efficiency': avg_efficiency,
            'renewable_percentage': avg_renewable,
            'total_cost': avg_cost,
            'maintenance_schedule': 'normal',
            'production_level': 'normal'
        }
    
    def _extract_state_features(self, state):
        """Extract features from current state."""
        
        return [
            state['energy_consumption'] / 10000,  # Normalize energy consumption
            state['energy_efficiency'],
            state['renewable_percentage'] / 100,  # Normalize renewable percentage
            state['total_cost'] / 1000,  # Normalize cost
            1 if state['maintenance_schedule'] == 'aggressive' else 0,
            1 if state['production_level'] == 'high' else 0
        ]
    
    def _choose_optimization_action(self, state_features, epsilon=0.1):
        """Choose optimization action using epsilon-greedy policy."""
        
        if np.random.random() < epsilon:
            # Random action
            actions = ['maintain', 'optimize_efficiency', 'increase_renewable', 'schedule_maintenance', 'adjust_production']
            return np.random.choice(actions)
        else:
            # Greedy action based on simple heuristic
            if state_features[1] < 0.8:  # Low efficiency
                return 'optimize_efficiency'
            elif state_features[2] < 0.2:  # Low renewable
                return 'increase_renewable'
            elif state_features[4] == 0:  # No aggressive maintenance
                return 'schedule_maintenance'
            else:
                return 'maintain'
    
    def _execute_action(self, action, current_state):
        """Execute the chosen action and return reward and new state."""
        
        new_state = current_state.copy()
        reward = 0
        
        if action == 'optimize_efficiency':
            # Improve efficiency by 5-15%
            improvement = np.random.uniform(0.05, 0.15)
            new_state['energy_efficiency'] = min(1.0, current_state['energy_efficiency'] + improvement)
            new_state['energy_consumption'] *= (1 - improvement * 0.3)  # Reduce consumption
            reward = improvement * 100  # Higher reward for efficiency improvements
        
        elif action == 'increase_renewable':
            # Increase renewable energy by 5-10%
            increase = np.random.uniform(0.05, 0.10)
            new_state['renewable_percentage'] = min(50, current_state['renewable_percentage'] + increase)
            new_state['total_cost'] *= (1 - increase * 0.2)  # Reduce cost
            reward = increase * 50
        
        elif action == 'schedule_maintenance':
            # Schedule aggressive maintenance
            new_state['maintenance_schedule'] = 'aggressive'
            new_state['energy_efficiency'] = min(1.0, current_state['energy_efficiency'] + 0.1)
            new_state['total_cost'] *= 1.05  # Slight cost increase
            reward = 30
        
        elif action == 'adjust_production':
            # Adjust production level
            if current_state['production_level'] == 'high':
                new_state['production_level'] = 'normal'
                new_state['energy_consumption'] *= 0.9
                reward = 20
            else:
                new_state['production_level'] = 'high'
                new_state['energy_consumption'] *= 1.1
                reward = -10  # Penalty for high consumption
        
        else:  # maintain
            # Small random variations
            variation = np.random.normal(0, 0.02)
            new_state['energy_efficiency'] = max(0.1, current_state['energy_efficiency'] + variation)
            reward = 5
        
        # Update total cost based on new state
        new_state['total_cost'] = new_state['energy_consumption'] * np.random.uniform(0.08, 0.15)
        
        return reward, new_state
    
    def _analyze_simulation_results(self, simulation_results):
        """Analyze the results of the reinforcement learning simulation."""
        
        df = pd.DataFrame(simulation_results)
        
        analysis = {
            'total_reward': df['reward'].sum(),
            'avg_daily_reward': df['reward'].mean(),
            'energy_consumption_trend': 'Decreasing' if df['energy_consumption'].iloc[-1] < df['energy_consumption'].iloc[0] else 'Increasing',
            'efficiency_improvement': df['energy_efficiency'].iloc[-1] - df['energy_efficiency'].iloc[0],
            'renewable_increase': df['renewable_percentage'].iloc[-1] - df['renewable_percentage'].iloc[0],
            'cost_savings': df['total_cost'].iloc[0] - df['total_cost'].iloc[-1],
            'most_effective_action': df.groupby('action')['reward'].mean().idxmax(),
            'action_frequency': df['action'].value_counts().to_dict()
        }
        
        return analysis
    
    def _extract_optimization_actions(self, simulation_results):
        """Extract the most effective optimization actions from simulation."""
        
        df = pd.DataFrame(simulation_results)
        action_effectiveness = df.groupby('action').agg({
            'reward': ['mean', 'sum', 'count']
        }).round(2)
        
        # Get top 3 most effective actions
        top_actions = action_effectiveness['reward']['mean'].nlargest(3)
        
        recommendations = []
        for action, effectiveness in top_actions.items():
            if action == 'optimize_efficiency':
                recommendations.append({
                    'action': 'Optimize Energy Efficiency',
                    'description': 'Implement process improvements and equipment optimization',
                    'expected_impact': 'High',
                    'implementation_time': '2-4 weeks'
                })
            elif action == 'increase_renewable':
                recommendations.append({
                    'action': 'Increase Renewable Energy',
                    'description': 'Install solar panels and energy storage systems',
                    'expected_impact': 'Medium',
                    'implementation_time': '8-12 weeks'
                })
            elif action == 'schedule_maintenance':
                recommendations.append({
                    'action': 'Implement Predictive Maintenance',
                    'description': 'Use IoT sensors and condition monitoring',
                    'expected_impact': 'Medium',
                    'implementation_time': '4-6 weeks'
                })
        
        return recommendations
    
    def create_optimization_dashboard(self, analysis_results):
        """Create comprehensive optimization dashboard visualizations."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Peak Demand Analysis',
                'Energy Efficiency Trends',
                'Plant Performance Comparison',
                'Optimization Opportunities'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Peak Demand Analysis
        peak_data = analysis_results['peak_demand_analysis']['peak_demand_stats']
        plants = peak_data.index
        peak_demands = peak_data[('PeakDemand_kW', 'max')]
        
        fig.add_trace(
            go.Bar(
                x=plants,
                y=peak_demands,
                name='Peak Demand',
                marker_color='red',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Energy Efficiency Trends
        efficiency_data = analysis_results['efficiency_analysis']['efficiency_trends']
        fig.add_trace(
            go.Scatter(
                x=efficiency_data['Date'],
                y=efficiency_data['EnergyEfficiency'],
                mode='lines+markers',
                name='Energy Efficiency',
                line=dict(color='green'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Plant Performance Comparison
        plant_data = analysis_results['plant_comparison']['plant_performance']
        fig.add_trace(
            go.Bar(
                x=plant_data.index,
                y=plant_data['Performance_Score'],
                name='Performance Score',
                marker_color='blue',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Optimization Opportunities
        opportunities = analysis_results['optimization_opportunities']
        if opportunities:
            action_names = [opp['action'] for opp in opportunities]
            priorities = [opp['priority'] for opp in opportunities]
            priority_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            colors = [priority_colors[priority] for priority in priorities]
            
            fig.add_trace(
                go.Bar(
                    x=action_names,
                    y=[1] * len(action_names),
                    name='Optimization Actions',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Energy Optimization Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
