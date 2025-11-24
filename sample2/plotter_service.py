"""
Historical Feature Plotter Service
Flexible plotting system with support for basic, computed, and engineered features
Integrates with existing Flask application structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple, Union
import warnings
from scipy import stats
import re

warnings.filterwarnings('ignore')

# Professional color scheme
COLORS = {
    'primary': '#00395D',
    'secondary': '#0066CC',
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette([COLORS['primary'], COLORS['secondary'], COLORS['success'], 
                 COLORS['danger'], COLORS['warning'], COLORS['info']])


class FeaturePlotterService:
    """
    Main plotter service for historical feature analysis
    Supports basic, computed, and custom engineered features
    """
    
    def __init__(self, feature_config: Dict = None):
        """
        Initialize the plotter service
        
        Parameters:
        -----------
        feature_config : Dict
            Configuration for features to plot
        """
        self.feature_config = feature_config or self.get_default_config()
        self.df = None
        self.processed_df = None
        self.output_base_dir = 'out/PLOT'
        
    def get_default_config(self) -> Dict:
        """Get default feature configuration"""
        return {
            'basic_features': [
                'Long_market_value',
                'short_market_value',
                'Applied_req',
                'House_total_req',
                'Regulatory_Req',
                'Gross_Market_value'
            ],
            'computed_features': {
                'delta': True,
                'delta_pct': True,
                'z_score': ['Applied_req']
            },
            'engineered_features': {
                'applied_to_gross_ratio': {
                    'formula': 'Applied_req / (Gross_Market_value + 0.0001)',
                    'plot_type': 'ratio',
                    'y_label': 'Applied/Gross Ratio'
                },
                'applied_to_gross_pct_ratio': {
                    'formula': 'Applied_req_delta_pct / (Gross_Market_value_delta_pct + 0.0001)',
                    'plot_type': 'ratio',
                    'y_label': 'Applied/Gross Delta % Ratio'
                },
                'margin_efficiency': {
                    'formula': '(Applied_req - House_total_req) / (Gross_Market_value + 0.0001)',
                    'plot_type': 'percentage',
                    'y_label': 'Margin Efficiency'
                },
                'long_short_ratio': {
                    'formula': 'Long_market_value / (abs(short_market_value) + 1)',
                    'plot_type': 'ratio',
                    'y_label': 'Long/Short Ratio'
                },
                'total_exposure': {
                    'formula': 'Long_market_value + abs(short_market_value)',
                    'plot_type': 'value',
                    'y_label': 'Total Exposure'
                }
            }
        }
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Convert date column
        if 'Business_Date' in self.df.columns:
            self.df['Business_Date'] = pd.to_datetime(self.df['Business_Date'])
        elif 'business_date' in self.df.columns:
            self.df['Business_Date'] = pd.to_datetime(self.df['business_date'])
        
        # Sort by account and date
        if 'header_account_id' in self.df.columns:
            self.df = self.df.sort_values(['header_account_id', 'Business_Date'])
        
        print(f"Data loaded: {len(self.df)} records")
        print(f"Date range: {self.df['Business_Date'].min()} to {self.df['Business_Date'].max()}")
        
        # Check for margin center column
        if 'Margin_center' in self.df.columns:
            print(f"Margin centers found: {self.df['Margin_center'].nunique()}")
        
        return self.df
    
    def calculate_deltas(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Calculate delta and delta_pct for features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        features : List[str]
            List of features to calculate deltas for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with computed deltas
        """
        df = df.copy()
        
        for col in features:
            if col in df.columns:
                # Calculate delta
                df[f'{col}_delta'] = df[col].diff()
                
                # Calculate delta percentage
                df[f'{col}_delta_pct'] = df[col].pct_change() * 100
                
                # Handle infinite values
                df[f'{col}_delta_pct'] = df[f'{col}_delta_pct'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def calculate_robust_z_score(self, series: pd.Series) -> pd.Series:
        """
        Calculate robust Z-score using median and MAD
        
        Parameters:
        -----------
        series : pd.Series
            Series to calculate Z-score for
            
        Returns:
        --------
        pd.Series
            Robust Z-scores
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            return pd.Series(0, index=series.index)
        
        return (series - median) / (1.4826 * mad)
    
    def safe_eval_formula(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """
        Safely evaluate custom formula on dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        formula : str
            Formula to evaluate
            
        Returns:
        --------
        pd.Series
            Evaluated result
        """
        try:
            # Create a safe namespace with DataFrame columns
            namespace = {}
            
            # Add column data to namespace
            for col in df.columns:
                if col in formula:
                    namespace[col] = df[col]
            
            # Add safe mathematical functions
            namespace.update({
                'abs': np.abs,
                'log': np.log,
                'sqrt': np.sqrt,
                'exp': np.exp,
                'max': np.maximum,
                'min': np.minimum,
                'mean': np.mean,
                'std': np.std,
                'sum': np.sum
            })
            
            # Evaluate the formula
            result = pd.eval(formula, local_dict=namespace, engine='python')
            
            # Handle infinite values
            if isinstance(result, pd.Series):
                result = result.replace([np.inf, -np.inf], np.nan)
            
            return result
            
        except Exception as e:
            print(f"Error evaluating formula '{formula}': {str(e)}")
            return pd.Series(np.nan, index=df.index)
    
    def calculate_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all engineered features based on configuration
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with basic and computed features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features added
        """
        df = df.copy()
        
        for feature_name, feature_config in self.feature_config.get('engineered_features', {}).items():
            if isinstance(feature_config, dict):
                formula = feature_config.get('formula', '')
            else:
                formula = feature_config
            
            print(f"Calculating engineered feature: {feature_name}")
            df[feature_name] = self.safe_eval_formula(df, formula)
        
        return df
    
    def process_features(self, df: pd.DataFrame, 
                        account_id: str = None,
                        margin_center: str = None) -> pd.DataFrame:
        """
        Process all features (basic, computed, and engineered)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        account_id : str
            Filter for specific account
        margin_center : str
            Filter for specific margin center
            
        Returns:
        --------
        pd.DataFrame
            Processed dataframe with all features
        """
        # Filter data if needed
        if account_id:
            df = df[df['header_account_id'] == account_id].copy()
        elif margin_center:
            df = df[df['Margin_center'] == margin_center].copy()
        
        if df.empty:
            raise ValueError(f"No data found for filters: account_id={account_id}, margin_center={margin_center}")
        
        # Sort by date
        df = df.sort_values('Business_Date')
        
        # Calculate deltas for basic features
        if self.feature_config['computed_features'].get('delta') or \
           self.feature_config['computed_features'].get('delta_pct'):
            df = self.calculate_deltas(df, self.feature_config['basic_features'])
        
        # Calculate Z-scores for specified features
        z_score_features = self.feature_config['computed_features'].get('z_score', [])
        for feature in z_score_features:
            if f'{feature}_delta' in df.columns:
                df[f'{feature}_delta_zscore'] = self.calculate_robust_z_score(df[f'{feature}_delta'])
        
        # Calculate engineered features
        df = self.calculate_engineered_features(df)
        
        return df
    
    def create_output_directory(self, account_id: str = None, 
                              margin_center: str = None) -> str:
        """
        Create output directory structure
        
        Parameters:
        -----------
        account_id : str
            Account ID
        margin_center : str
            Margin center
            
        Returns:
        --------
        str
            Output directory path
        """
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        if account_id:
            output_dir = os.path.join(self.output_base_dir, 'ACCOUNT_LEVEL', 
                                     f'{account_id}_{current_date}')
        elif margin_center:
            output_dir = os.path.join(self.output_base_dir, 'MARGIN_CENTER_LEVEL', 
                                     f'{margin_center}_{current_date}')
        else:
            output_dir = os.path.join(self.output_base_dir, 'ALL_DATA', 
                                     f'all_{current_date}')
        
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def plot_combined_overview(self, df: pd.DataFrame, output_dir: str, 
                              identifier: str = "") -> str:
        """
        Create combined overview plot for all features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        output_dir : str
            Output directory path
        identifier : str
            Account or margin center identifier
            
        Returns:
        --------
        str
            Path to saved plot
        """
        features = self.feature_config['basic_features']
        n_features = len(features)
        
        fig, axes = plt.subplots(n_features, 3, figsize=(24, 4*n_features))
        
        if n_features == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Combined Feature Overview - {identifier}\n'
                    f'Period: {df["Business_Date"].min():%Y-%m-%d} to {df["Business_Date"].max():%Y-%m-%d}',
                    fontsize=16, y=1.02)
        
        for idx, feature in enumerate(features):
            if feature not in df.columns:
                continue
            
            # Plot 1: Time series
            ax1 = axes[idx, 0]
            ax1.plot(df['Business_Date'], df[feature], color=COLORS['primary'], 
                    linewidth=1, alpha=0.8)
            ax1.set_title(f'{feature} - Time Series', fontsize=12)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add moving average
            if len(df) > 7:
                ma7 = df[feature].rolling(window=7).mean()
                ax1.plot(df['Business_Date'], ma7, color=COLORS['secondary'], 
                        linewidth=1, alpha=0.6, label='7-day MA')
                ax1.legend()
            
            # Plot 2: Delta
            ax2 = axes[idx, 1]
            delta_col = f'{feature}_delta'
            if delta_col in df.columns:
                colors = [COLORS['danger'] if x < 0 else COLORS['success'] 
                         for x in df[delta_col].fillna(0)]
                ax2.bar(df['Business_Date'], df[delta_col], color=colors, alpha=0.6)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.set_title(f'{feature} - Daily Delta', fontsize=12)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Delta')
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Delta percentage
            ax3 = axes[idx, 2]
            delta_pct_col = f'{feature}_delta_pct'
            if delta_pct_col in df.columns:
                colors = [COLORS['danger'] if x < 0 else COLORS['success'] 
                         for x in df[delta_pct_col].fillna(0)]
                ax3.scatter(df['Business_Date'], df[delta_pct_col], 
                          c=colors, alpha=0.6, s=20)
                ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax3.set_title(f'{feature} - Daily % Change', fontsize=12)
                ax3.set_xlabel('Date')
                ax3.set_ylabel('% Change')
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, '01_combined_overview.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined overview to {output_path}")
        return output_path
    
    def plot_individual_feature(self, df: pd.DataFrame, feature: str, 
                              output_dir: str, plot_number: int) -> str:
        """
        Create detailed plot for individual feature
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        feature : str
            Feature name
        output_dir : str
            Output directory
        plot_number : int
            Plot number for naming
            
        Returns:
        --------
        str
            Path to saved plot
        """
        if feature not in df.columns:
            print(f"Feature {feature} not found in dataframe")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{feature} - Detailed Analysis', fontsize=14)
        
        # Plot 1: Time series with statistics
        ax1 = axes[0, 0]
        ax1.plot(df['Business_Date'], df[feature], color=COLORS['primary'], 
                linewidth=1, label='Original')
        
        # Add moving averages
        if len(df) > 7:
            ma7 = df[feature].rolling(window=7).mean()
            ma30 = df[feature].rolling(window=30).mean() if len(df) > 30 else None
            
            ax1.plot(df['Business_Date'], ma7, color=COLORS['secondary'], 
                    linewidth=1, alpha=0.7, label='7-day MA')
            if ma30 is not None:
                ax1.plot(df['Business_Date'], ma30, color=COLORS['info'], 
                        linewidth=1, alpha=0.7, label='30-day MA')
        
        ax1.set_title('Time Series with Moving Averages')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Distribution
        ax2 = axes[0, 1]
        ax2.hist(df[feature].dropna(), bins=50, color=COLORS['primary'], 
                alpha=0.7, edgecolor='black')
        ax2.axvline(df[feature].mean(), color=COLORS['danger'], 
                   linestyle='--', label=f'Mean: {df[feature].mean():.2f}')
        ax2.axvline(df[feature].median(), color=COLORS['success'], 
                   linestyle='--', label=f'Median: {df[feature].median():.2f}')
        ax2.set_title('Value Distribution')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plot
        ax3 = axes[0, 2]
        box_data = [df[feature].dropna()]
        bp = ax3.boxplot(box_data, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['primary'])
        bp['boxes'][0].set_alpha(0.7)
        ax3.set_title('Box Plot')
        ax3.set_ylabel('Value')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Delta analysis
        ax4 = axes[1, 0]
        delta_col = f'{feature}_delta'
        if delta_col in df.columns:
            ax4.plot(df['Business_Date'], df[delta_col], color=COLORS['primary'], 
                    linewidth=1, alpha=0.8)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add standard deviation bands
            mean_delta = df[delta_col].mean()
            std_delta = df[delta_col].std()
            ax4.axhline(y=mean_delta, color=COLORS['info'], linestyle='--', 
                       linewidth=1, alpha=0.5, label=f'Mean: {mean_delta:.2f}')
            ax4.axhline(y=mean_delta + 2*std_delta, color=COLORS['danger'], 
                       linestyle=':', linewidth=1, alpha=0.5, label=f'+2σ')
            ax4.axhline(y=mean_delta - 2*std_delta, color=COLORS['danger'], 
                       linestyle=':', linewidth=1, alpha=0.5, label=f'-2σ')
            
            ax4.set_title('Delta Analysis')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Delta')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Delta percentage
        ax5 = axes[1, 1]
        delta_pct_col = f'{feature}_delta_pct'
        if delta_pct_col in df.columns:
            valid_pct = df[delta_pct_col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_pct) > 0:
                ax5.hist(valid_pct, bins=50, color=COLORS['secondary'], 
                        alpha=0.7, edgecolor='black')
                
                # Add percentile lines
                p5 = np.percentile(valid_pct, 5)
                p95 = np.percentile(valid_pct, 95)
                ax5.axvline(p5, color=COLORS['warning'], linestyle='--', 
                          label=f'5th %ile: {p5:.2f}%')
                ax5.axvline(p95, color=COLORS['warning'], linestyle='--', 
                          label=f'95th %ile: {p95:.2f}%')
                
                ax5.set_title('Delta % Distribution')
                ax5.set_xlabel('% Change')
                ax5.set_ylabel('Frequency')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # Plot 6: Q-Q plot
        ax6 = axes[1, 2]
        try:
            stats.probplot(df[feature].dropna(), dist="norm", plot=ax6)
            ax6.set_title('Q-Q Plot (Normality Check)')
            ax6.grid(True, alpha=0.3)
        except:
            ax6.text(0.5, 0.5, 'Q-Q Plot unavailable', ha='center', va='center')
        
        plt.tight_layout()
        
        # Save plot
        feature_safe = feature.replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f'{plot_number:02d}_{feature_safe}_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {feature} analysis to {output_path}")
        return output_path
    
    def plot_engineered_features(self, df: pd.DataFrame, output_dir: str, 
                                start_number: int = 20) -> List[str]:
        """
        Plot engineered features with appropriate visualizations
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        output_dir : str
            Output directory
        start_number : int
            Starting number for plot naming
            
        Returns:
        --------
        List[str]
            List of saved plot paths
        """
        plot_paths = []
        
        for idx, (feature_name, feature_config) in enumerate(
            self.feature_config.get('engineered_features', {}).items()):
            
            if feature_name not in df.columns:
                continue
            
            plot_type = feature_config.get('plot_type', 'value') if isinstance(feature_config, dict) else 'value'
            y_label = feature_config.get('y_label', feature_name) if isinstance(feature_config, dict) else feature_name
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'{feature_name} - Engineered Feature Analysis', fontsize=14)
            
            # Plot 1: Time series
            ax1 = axes[0]
            ax1.plot(df['Business_Date'], df[feature_name], color=COLORS['primary'], 
                    linewidth=1, alpha=0.8)
            ax1.set_title(f'{feature_name} Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel(y_label)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add zero line for ratios
            if plot_type in ['ratio', 'percentage']:
                ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Plot 2: Distribution
            ax2 = axes[1]
            valid_data = df[feature_name].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_data) > 0:
                ax2.hist(valid_data, bins=50, color=COLORS['secondary'], 
                        alpha=0.7, edgecolor='black')
                ax2.axvline(valid_data.mean(), color=COLORS['danger'], 
                          linestyle='--', label=f'Mean: {valid_data.mean():.3f}')
                ax2.axvline(valid_data.median(), color=COLORS['success'], 
                          linestyle='--', label=f'Median: {valid_data.median():.3f}')
                ax2.set_title('Distribution')
                ax2.set_xlabel(y_label)
                ax2.set_ylabel('Frequency')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Rolling statistics
            ax3 = axes[2]
            if len(df) > 7:
                rolling_mean = df[feature_name].rolling(window=7).mean()
                rolling_std = df[feature_name].rolling(window=7).std()
                
                ax3.plot(df['Business_Date'], df[feature_name], 
                        color=COLORS['primary'], alpha=0.3, label='Original')
                ax3.plot(df['Business_Date'], rolling_mean, 
                        color=COLORS['secondary'], linewidth=2, label='7-day MA')
                ax3.fill_between(df['Business_Date'],
                                rolling_mean - rolling_std,
                                rolling_mean + rolling_std,
                                alpha=0.2, color=COLORS['info'], label='±1 Std')
                ax3.set_title('Rolling Statistics')
                ax3.set_xlabel('Date')
                ax3.set_ylabel(y_label)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_number = start_number + idx
            feature_safe = feature_name.replace('/', '_').replace(' ', '_')
            output_path = os.path.join(output_dir, f'{plot_number:02d}_{feature_safe}_engineered.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved engineered feature {feature_name} to {output_path}")
            plot_paths.append(output_path)
        
        return plot_paths
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, output_dir: str) -> str:
        """
        Create correlation heatmap for all features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        output_dir : str
            Output directory
            
        Returns:
        --------
        str
            Path to saved plot
        """
        # Select numeric columns for correlation
        feature_cols = []
        
        # Basic features
        for feat in self.feature_config['basic_features']:
            if feat in df.columns:
                feature_cols.append(feat)
                
                # Add deltas if they exist
                if f'{feat}_delta' in df.columns:
                    feature_cols.append(f'{feat}_delta')
        
        # Add engineered features
        for feat in self.feature_config.get('engineered_features', {}).keys():
            if feat in df.columns:
                feature_cols.append(feat)
        
        if len(feature_cols) < 2:
            print("Not enough features for correlation matrix")
            return None
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=1, cbar_kws={'label': 'Correlation'},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, '08_correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation heatmap to {output_path}")
        return output_path
    
    def plot_statistical_summary(self, df: pd.DataFrame, output_dir: str) -> str:
        """
        Create statistical summary plots
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        output_dir : str
            Output directory
            
        Returns:
        --------
        str
            Path to saved plot
        """
        features = self.feature_config['basic_features'][:6]  # Limit to 6 for visualization
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Summary', fontsize=14)
        
        # Plot 1: Box plots for all features
        ax1 = axes[0, 0]
        box_data = []
        labels = []
        for feat in features:
            if feat in df.columns:
                # Normalize data for comparison
                normalized = (df[feat] - df[feat].mean()) / df[feat].std()
                box_data.append(normalized.dropna())
                labels.append(feat.replace('_', ' '))
        
        if box_data:
            bp = ax1.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], sns.color_palette()):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax1.set_title('Normalized Feature Distributions')
            ax1.set_ylabel('Normalized Value (Z-score)')
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Violin plots for deltas
        ax2 = axes[0, 1]
        violin_data = []
        labels = []
        for feat in features[:4]:  # Limit to 4 for clarity
            delta_col = f'{feat}_delta'
            if delta_col in df.columns:
                valid_data = df[delta_col].dropna()
                if len(valid_data) > 0:
                    violin_data.append(valid_data)
                    labels.append(feat.replace('_', ' ')[:15])
        
        if violin_data:
            parts = ax2.violinplot(violin_data, positions=range(len(violin_data)), 
                                  widths=0.7, showmeans=True, showmedians=True)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.set_title('Delta Distributions')
            ax2.set_ylabel('Delta Value')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 3: Feature volatility (std over time)
        ax3 = axes[1, 0]
        window = 30 if len(df) > 30 else 7
        for feat in features[:4]:
            if feat in df.columns:
                rolling_std = df[feat].rolling(window=window).std()
                ax3.plot(df['Business_Date'], rolling_std, label=feat.replace('_', ' ')[:15], 
                        linewidth=1.5, alpha=0.8)
        
        ax3.set_title(f'Feature Volatility ({window}-day Rolling Std)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Standard Deviation')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary statistics
        summary_data = []
        for feat in features:
            if feat in df.columns:
                summary_data.append([
                    feat.replace('_', ' ')[:20],
                    f'{df[feat].mean():.2f}',
                    f'{df[feat].std():.2f}',
                    f'{df[feat].min():.2f}',
                    f'{df[feat].max():.2f}'
                ])
        
        if summary_data:
            table = ax4.table(cellText=summary_data,
                            colLabels=['Feature', 'Mean', 'Std', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the header
            for i in range(5):
                table[(0, i)].set_facecolor(COLORS['primary'])
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Summary Statistics', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, '09_statistical_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved statistical summary to {output_path}")
        return output_path
    
    def plot_margin_center_consolidated(self, df: pd.DataFrame, 
                                       margin_center: str,
                                       output_dir: str) -> str:
        """
        Create consolidated plot for all accounts in a margin center
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with all accounts in margin center
        margin_center : str
            Margin center identifier
        output_dir : str
            Output directory
            
        Returns:
        --------
        str
            Path to saved plot
        """
        # Get unique accounts
        accounts = df['header_account_id'].unique()
        n_accounts = len(accounts)
        print(f"Creating consolidated plot for {n_accounts} accounts in {margin_center}")
        
        features = self.feature_config['basic_features'][:3]  # Use first 3 features for clarity
        
        fig, axes = plt.subplots(len(features), 2, figsize=(20, 4*len(features)))
        
        if len(features) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Margin Center {margin_center} - Consolidated View ({n_accounts} accounts)\n'
                    f'Period: {df["Business_Date"].min():%Y-%m-%d} to {df["Business_Date"].max():%Y-%m-%d}',
                    fontsize=14)
        
        for idx, feature in enumerate(features):
            if feature not in df.columns:
                continue
            
            # Plot 1: Aggregated statistics
            ax1 = axes[idx, 0]
            
            # Calculate daily aggregates across all accounts
            daily_stats = df.groupby('Business_Date')[feature].agg(['mean', 'std', 'median'])
            
            ax1.plot(daily_stats.index, daily_stats['mean'], 
                    color=COLORS['primary'], linewidth=2, label='Mean')
            ax1.plot(daily_stats.index, daily_stats['median'], 
                    color=COLORS['secondary'], linewidth=1.5, label='Median')
            
            # Add confidence band
            ax1.fill_between(daily_stats.index,
                           daily_stats['mean'] - daily_stats['std'],
                           daily_stats['mean'] + daily_stats['std'],
                           alpha=0.2, color=COLORS['info'], label='±1 Std')
            
            ax1.set_title(f'{feature} - Aggregated Across Accounts')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Distribution across accounts
            ax2 = axes[idx, 1]
            
            # Sample dates for distribution plot
            sample_dates = pd.date_range(df['Business_Date'].min(), 
                                        df['Business_Date'].max(), 
                                        periods=min(10, len(df['Business_Date'].unique())))
            
            for date in sample_dates:
                date_data = df[df['Business_Date'] == date][feature].dropna()
                if len(date_data) > 0:
                    ax2.hist(date_data, alpha=0.3, bins=30, label=f'{date:%Y-%m-%d}'[:10])
            
            ax2.set_title(f'{feature} - Distribution Across Accounts')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.legend(fontsize='small', loc='best')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'consolidated_margin_center.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved margin center consolidated plot to {output_path}")
        return output_path
    
    def generate_metadata(self, df: pd.DataFrame, output_dir: str,
                         account_id: str = None, margin_center: str = None) -> str:
        """
        Generate metadata JSON file with statistics and information
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        output_dir : str
            Output directory
        account_id : str
            Account ID if applicable
        margin_center : str
            Margin center if applicable
            
        Returns:
        --------
        str
            Path to metadata file
        """
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'data_period': {
                'start': str(df['Business_Date'].min()),
                'end': str(df['Business_Date'].max()),
                'total_days': (df['Business_Date'].max() - df['Business_Date'].min()).days
            },
            'filters': {
                'account_id': account_id,
                'margin_center': margin_center
            },
            'data_stats': {
                'total_records': len(df),
                'unique_dates': df['Business_Date'].nunique()
            },
            'features_analyzed': {
                'basic': self.feature_config['basic_features'],
                'computed': list(df.filter(regex='_delta|_pct|_zscore').columns),
                'engineered': list(self.feature_config.get('engineered_features', {}).keys())
            },
            'summary_statistics': {}
        }
        
        # Add summary statistics for each feature
        for feature in self.feature_config['basic_features']:
            if feature in df.columns:
                metadata['summary_statistics'][feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'median': float(df[feature].median())
                }
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'report_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_path}")
        return metadata_path
    
    def generate_plots(self, csv_path: str, 
                      account_ids: List[str] = None,
                      margin_center: str = None,
                      generate_all: bool = True) -> Dict:
        """
        Main method to generate all plots
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file
        account_ids : List[str]
            List of account IDs to process
        margin_center : str
            Margin center to process
        generate_all : bool
            Whether to generate all types of plots
            
        Returns:
        --------
        Dict
            Dictionary with paths to generated files
        """
        # Load data
        self.load_data(csv_path)
        
        results = {
            'success': True,
            'outputs': []
        }
        
        try:
            # Process based on input parameters
            if margin_center:
                # Process margin center
                print(f"\nProcessing margin center: {margin_center}")
                
                # Filter and process data
                df_filtered = self.df[self.df['Margin_center'] == margin_center].copy()
                if df_filtered.empty:
                    raise ValueError(f"No data found for margin center {margin_center}")
                
                df_processed = self.process_features(df_filtered)
                
                # Create output directory
                output_dir = self.create_output_directory(margin_center=margin_center)
                
                # Generate consolidated plot
                self.plot_margin_center_consolidated(df_processed, margin_center, output_dir)
                
                # Generate metadata
                self.generate_metadata(df_processed, output_dir, margin_center=margin_center)
                
                results['outputs'].append({
                    'type': 'margin_center',
                    'identifier': margin_center,
                    'output_dir': output_dir
                })
                
            elif account_ids:
                # Process specific accounts
                for account_id in account_ids:
                    print(f"\nProcessing account: {account_id}")
                    
                    try:
                        # Process features for account
                        df_processed = self.process_features(self.df, account_id=account_id)
                        
                        # Create output directory
                        output_dir = self.create_output_directory(account_id=account_id)
                        
                        # Generate plots
                        self.plot_combined_overview(df_processed, output_dir, account_id)
                        
                        # Individual feature plots
                        plot_number = 2
                        for feature in self.feature_config['basic_features']:
                            self.plot_individual_feature(df_processed, feature, 
                                                        output_dir, plot_number)
                            plot_number += 1
                        
                        # Engineered features
                        self.plot_engineered_features(df_processed, output_dir)
                        
                        # Statistical plots
                        self.plot_correlation_heatmap(df_processed, output_dir)
                        self.plot_statistical_summary(df_processed, output_dir)
                        
                        # Generate metadata
                        self.generate_metadata(df_processed, output_dir, account_id=account_id)
                        
                        results['outputs'].append({
                            'type': 'account',
                            'identifier': account_id,
                            'output_dir': output_dir
                        })
                        
                    except Exception as e:
                        print(f"Error processing account {account_id}: {str(e)}")
                        results['outputs'].append({
                            'type': 'account',
                            'identifier': account_id,
                            'error': str(e)
                        })
            
            else:
                # Process all data
                print("\nProcessing all data...")
                
                df_processed = self.process_features(self.df)
                
                # Create output directory
                output_dir = self.create_output_directory()
                
                # Generate overview plots
                self.plot_combined_overview(df_processed, output_dir, "All Data")
                self.plot_correlation_heatmap(df_processed, output_dir)
                self.plot_statistical_summary(df_processed, output_dir)
                
                # Generate metadata
                self.generate_metadata(df_processed, output_dir)
                
                results['outputs'].append({
                    'type': 'all',
                    'identifier': 'all_data',
                    'output_dir': output_dir
                })
        
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            print(f"Error in plot generation: {str(e)}")
        
        return results
