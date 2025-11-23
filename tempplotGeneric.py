"""
Account-level Feature Plotting for Anomaly Detection Analysis
This script reads a CSV file with portfolio data and creates visualizations
of key features at the account level, including time series plots and
delta/delta_pct calculations used in the Isolation Forest model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AccountLevelPlotter:
    """Class to handle account-level plotting for anomaly detection features"""
    
    def __init__(self, csv_path, model_path=None):
        """
        Initialize the plotter with data and optionally the trained model
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing the data
        model_path : str, optional
            Path to the saved Isolation Forest model
        """
        self.csv_path = csv_path
        self.model_path = model_path
        self.df = None
        self.model = None
        self.feature_columns = [
            'Long_market_value',
            'short_market_value',
            'Applied_req',
            'House_total_req',
            'Regulatory_Req',
            'Gross_Market_value'
        ]
        
    def load_data(self):
        """Load and preprocess the CSV data"""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        # Convert Business_Date to datetime
        self.df['Business_Date'] = pd.to_datetime(self.df['Business_Date'])
        
        # Sort by account and date
        self.df = self.df.sort_values(['header_account_id', 'Business_Date'])
        
        print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"Date range: {self.df['Business_Date'].min()} to {self.df['Business_Date'].max()}")
        print(f"Number of unique accounts: {self.df['header_account_id'].nunique()}")
        
        return self.df
    
    def calculate_deltas(self, account_df):
        """
        Calculate delta and delta_pct for features (day-over-day changes)
        
        Parameters:
        -----------
        account_df : DataFrame
            DataFrame filtered for a specific account
        """
        account_df = account_df.copy()
        
        for col in self.feature_columns:
            if col in account_df.columns:
                # Calculate absolute delta
                account_df[f'{col}_delta'] = account_df[col].diff()
                
                # Calculate percentage delta
                account_df[f'{col}_delta_pct'] = account_df[col].pct_change() * 100
                
        return account_df
    
    def calculate_robust_z_score(self, series):
        """
        Calculate robust Z-score using median and MAD
        
        Parameters:
        -----------
        series : pd.Series
            Series to calculate Z-score for
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        # Avoid division by zero
        if mad == 0:
            return pd.Series(0, index=series.index)
        
        return (series - median) / (1.4826 * mad)
    
    def plot_account_features(self, account_id, save_path=None):
        """
        Create comprehensive plots for a specific account
        
        Parameters:
        -----------
        account_id : str/int
            Account ID to plot
        save_path : str, optional
            Path to save the plot
        """
        # Filter data for specific account
        account_df = self.df[self.df['header_account_id'] == account_id].copy()
        
        if account_df.empty:
            print(f"No data found for account {account_id}")
            return
        
        # Calculate deltas
        account_df = self.calculate_deltas(account_df)
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(self.feature_columns), 3, figsize=(20, 4*len(self.feature_columns)))
        fig.suptitle(f'Feature Analysis for Account: {account_id}', fontsize=16, y=1.02)
        
        for idx, col in enumerate(self.feature_columns):
            if col not in account_df.columns:
                continue
                
            # Plot 1: Time series of actual values
            axes[idx, 0].plot(account_df['Business_Date'], account_df[col], 
                            marker='o', markersize=3, linewidth=1)
            axes[idx, 0].set_title(f'{col} - Time Series')
            axes[idx, 0].set_xlabel('Date')
            axes[idx, 0].set_ylabel('Value')
            axes[idx, 0].tick_params(axis='x', rotation=45)
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Plot 2: Delta (day-over-day change)
            delta_col = f'{col}_delta'
            if delta_col in account_df.columns:
                axes[idx, 1].bar(account_df['Business_Date'], account_df[delta_col],
                               color=['red' if x < 0 else 'green' for x in account_df[delta_col]])
                axes[idx, 1].set_title(f'{col} - Daily Delta')
                axes[idx, 1].set_xlabel('Date')
                axes[idx, 1].set_ylabel('Delta')
                axes[idx, 1].tick_params(axis='x', rotation=45)
                axes[idx, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                axes[idx, 1].grid(True, alpha=0.3)
            
            # Plot 3: Delta percentage
            delta_pct_col = f'{col}_delta_pct'
            if delta_pct_col in account_df.columns:
                axes[idx, 2].bar(account_df['Business_Date'], account_df[delta_pct_col],
                               color=['red' if x < 0 else 'green' for x in account_df[delta_pct_col]])
                axes[idx, 2].set_title(f'{col} - Daily Delta %')
                axes[idx, 2].set_xlabel('Date')
                axes[idx, 2].set_ylabel('Delta %')
                axes[idx, 2].tick_params(axis='x', rotation=45)
                axes[idx, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                axes[idx, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return account_df
    
    def plot_interactive_account_features(self, account_id, save_html=None):
        """
        Create interactive Plotly plots for a specific account
        
        Parameters:
        -----------
        account_id : str/int
            Account ID to plot
        save_html : str, optional
            Path to save HTML file
        """
        # Filter data for specific account
        account_df = self.df[self.df['header_account_id'] == account_id].copy()
        
        if account_df.empty:
            print(f"No data found for account {account_id}")
            return
        
        # Calculate deltas
        account_df = self.calculate_deltas(account_df)
        
        # Create subplots
        fig = make_subplots(
            rows=len(self.feature_columns), 
            cols=3,
            subplot_titles=[f'{col} - {plot_type}' 
                          for col in self.feature_columns 
                          for plot_type in ['Time Series', 'Daily Delta', 'Daily Delta %']],
            vertical_spacing=0.05,
            horizontal_spacing=0.08
        )
        
        for idx, col in enumerate(self.feature_columns):
            if col not in account_df.columns:
                continue
            
            row = idx + 1
            
            # Time series plot
            fig.add_trace(
                go.Scatter(x=account_df['Business_Date'], 
                          y=account_df[col],
                          mode='lines+markers',
                          name=col,
                          marker=dict(size=4),
                          showlegend=False),
                row=row, col=1
            )
            
            # Delta plot
            delta_col = f'{col}_delta'
            if delta_col in account_df.columns:
                colors = ['red' if x < 0 else 'green' for x in account_df[delta_col]]
                fig.add_trace(
                    go.Bar(x=account_df['Business_Date'],
                          y=account_df[delta_col],
                          marker_color=colors,
                          name=f'{col} Delta',
                          showlegend=False),
                    row=row, col=2
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="solid", line_color="black", 
                            line_width=1, row=row, col=2)
            
            # Delta percentage plot
            delta_pct_col = f'{col}_delta_pct'
            if delta_pct_col in account_df.columns:
                colors = ['red' if x < 0 else 'green' for x in account_df[delta_pct_col]]
                fig.add_trace(
                    go.Bar(x=account_df['Business_Date'],
                          y=account_df[delta_pct_col],
                          marker_color=colors,
                          name=f'{col} Delta %',
                          showlegend=False),
                    row=row, col=3
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="solid", line_color="black", 
                            line_width=1, row=row, col=3)
        
        # Update layout
        fig.update_layout(
            title=f'Feature Analysis for Account: {account_id}',
            height=300*len(self.feature_columns),
            showlegend=False,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Value", col=1)
        fig.update_yaxes(title_text="Delta", col=2)
        fig.update_yaxes(title_text="Delta %", col=3)
        
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")
        
        fig.show()
        
        return account_df
    
    def plot_anomaly_scores(self, account_id, save_path=None):
        """
        Plot anomaly scores if model is available
        
        Parameters:
        -----------
        account_id : str/int
            Account ID to plot
        save_path : str, optional
            Path to save the plot
        """
        if self.model_path:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print("No model path provided, skipping anomaly score plotting")
            return
        
        # Filter and prepare data
        account_df = self.df[self.df['header_account_id'] == account_id].copy()
        account_df = self.calculate_deltas(account_df)
        
        # Calculate robust Z-score on Applied_req
        if 'Applied_req_delta' in account_df.columns:
            account_df['Applied_req_z_score'] = self.calculate_robust_z_score(
                account_df['Applied_req_delta'].dropna()
            )
        
        # Prepare features for model (excluding first row due to delta calculation)
        feature_list = []
        for col in self.feature_columns:
            if f'{col}_delta' in account_df.columns:
                feature_list.append(f'{col}_delta')
            if f'{col}_delta_pct' in account_df.columns:
                feature_list.append(f'{col}_delta_pct')
        
        # Add Z-score if available
        if 'Applied_req_z_score' in account_df.columns:
            feature_list.append('Applied_req_z_score')
        
        # Get valid rows (not NaN after delta calculation)
        valid_mask = account_df[feature_list].notna().all(axis=1)
        X = account_df.loc[valid_mask, feature_list]
        
        if len(X) > 0:
            # Get anomaly scores
            scores = self.model.decision_function(X)
            predictions = self.model.predict(X)
            
            # Add to dataframe
            account_df.loc[valid_mask, 'anomaly_score'] = scores
            account_df.loc[valid_mask, 'is_anomaly'] = predictions
            
            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Anomaly scores over time
            axes[0].plot(account_df.loc[valid_mask, 'Business_Date'], 
                        scores, marker='o', markersize=4)
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Threshold')
            axes[0].set_title(f'Anomaly Scores for Account {account_id}')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Anomaly Score')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Applied_req with anomalies highlighted
            axes[1].plot(account_df['Business_Date'], account_df['Applied_req'], 
                        marker='o', markersize=3, label='Applied_req')
            
            # Highlight anomalies
            anomalies = account_df[account_df['is_anomaly'] == -1]
            if not anomalies.empty:
                axes[1].scatter(anomalies['Business_Date'], anomalies['Applied_req'], 
                              color='red', s=100, marker='*', label='Anomalies', zorder=5)
            
            axes[1].set_title(f'Applied_req with Detected Anomalies - Account {account_id}')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Applied_req Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"Anomaly plot saved to {save_path}")
            
            plt.show()
        else:
            print(f"Not enough valid data for anomaly detection for account {account_id}")
    
    def create_summary_report(self, account_ids=None, output_dir='plots'):
        """
        Create summary plots for multiple accounts
        
        Parameters:
        -----------
        account_ids : list, optional
            List of account IDs to plot. If None, plot top N accounts by activity
        output_dir : str
            Directory to save plots
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if account_ids is None:
            # Get top 10 accounts by number of records
            top_accounts = self.df.groupby('header_account_id').size().nlargest(10)
            account_ids = top_accounts.index.tolist()
            print(f"Plotting top {len(account_ids)} accounts by activity")
        
        for account_id in account_ids:
            print(f"\nProcessing account {account_id}...")
            
            # Create static plots
            static_path = Path(output_dir) / f'account_{account_id}_features.png'
            self.plot_account_features(account_id, save_path=static_path)
            
            # Create interactive plots
            interactive_path = Path(output_dir) / f'account_{account_id}_interactive.html'
            self.plot_interactive_account_features(account_id, save_html=interactive_path)
            
            # Create anomaly plots if model is available
            if self.model_path:
                anomaly_path = Path(output_dir) / f'account_{account_id}_anomalies.png'
                self.plot_anomaly_scores(account_id, save_path=anomaly_path)
    
    def create_correlation_heatmap(self, account_id=None, save_path=None):
        """
        Create correlation heatmap for features
        
        Parameters:
        -----------
        account_id : str/int, optional
            Specific account ID. If None, use all data
        save_path : str, optional
            Path to save the plot
        """
        if account_id:
            data = self.df[self.df['header_account_id'] == account_id].copy()
            title = f'Feature Correlation Heatmap - Account {account_id}'
        else:
            data = self.df.copy()
            title = 'Feature Correlation Heatmap - All Accounts'
        
        # Calculate deltas for correlation
        if account_id:
            data = self.calculate_deltas(data)
        else:
            # Calculate deltas per account
            data = data.groupby('header_account_id').apply(self.calculate_deltas)
        
        # Select features for correlation
        corr_features = self.feature_columns + [f'{col}_delta' for col in self.feature_columns]
        corr_features = [f for f in corr_features if f in data.columns]
        
        # Calculate correlation matrix
        corr_matrix = data[corr_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Correlation heatmap saved to {save_path}")
        
        plt.show()


def main():
    """Main function to run the account-level plotting"""
    
    # Configuration
    CSV_PATH = 'your_data.csv'  # Update with your CSV path
    MODEL_PATH = 'model.joblib'  # Update with your model path (optional)
    OUTPUT_DIR = 'account_plots'
    
    # Initialize plotter
    plotter = AccountLevelPlotter(csv_path=CSV_PATH, model_path=MODEL_PATH)
    
    # Load data
    plotter.load_data()
    
    # Example: Plot specific account
    # Replace with an actual account ID from your data
    sample_account = plotter.df['header_account_id'].iloc[0]
    print(f"\nPlotting sample account: {sample_account}")
    
    # Create various plots for the sample account
    plotter.plot_account_features(sample_account, 
                                 save_path=f'{OUTPUT_DIR}/sample_account_features.png')
    
    plotter.plot_interactive_account_features(sample_account,
                                             save_html=f'{OUTPUT_DIR}/sample_account_interactive.html')
    
    if MODEL_PATH:
        plotter.plot_anomaly_scores(sample_account,
                                   save_path=f'{OUTPUT_DIR}/sample_account_anomalies.png')
    
    # Create correlation heatmap
    plotter.create_correlation_heatmap(save_path=f'{OUTPUT_DIR}/correlation_heatmap_all.png')
    
    # Create summary report for top accounts
    # plotter.create_summary_report(output_dir=OUTPUT_DIR)
    
    # Or specify specific accounts
    # specific_accounts = ['ACC001', 'ACC002', 'ACC003']  # Replace with your account IDs
    # plotter.create_summary_report(account_ids=specific_accounts, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
