"""
Simplified Account-Level Feature Plotting Script
Focused on essential plotting functionality for anomaly detection features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data(csv_path):
    """Load CSV and prepare data with proper date handling"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert Business_Date to datetime
    df['Business_Date'] = pd.to_datetime(df['Business_Date'])
    
    # Sort by account and date
    df = df.sort_values(['header_account_id', 'Business_Date'])
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Date range: {df['Business_Date'].min()} to {df['Business_Date'].max()}")
    print(f"Unique accounts: {df['header_account_id'].nunique()}")
    
    return df

def calculate_deltas(df, feature_columns):
    """Calculate day-over-day deltas and percentage changes"""
    df = df.copy()
    
    for col in feature_columns:
        if col in df.columns:
            # Absolute delta
            df[f'{col}_delta'] = df[col].diff()
            # Percentage delta
            df[f'{col}_delta_pct'] = df[col].pct_change() * 100
    
    return df

def plot_single_account(df, account_id, feature_columns, save_dir=None):
    """
    Create comprehensive plots for a single account
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    account_id : str/int
        Account ID to plot
    feature_columns : list
        List of features to plot
    save_dir : str
        Directory to save plots
    """
    # Filter for specific account
    account_df = df[df['header_account_id'] == account_id].copy()
    
    if account_df.empty:
        print(f"No data found for account {account_id}")
        return None
    
    # Calculate deltas
    account_df = calculate_deltas(account_df, feature_columns)
    
    # Create figure
    n_features = len(feature_columns)
    fig = plt.figure(figsize=(20, 4*n_features))
    fig.suptitle(f'Feature Analysis for Account: {account_id}', fontsize=16, y=1.00)
    
    for idx, col in enumerate(feature_columns):
        if col not in account_df.columns:
            continue
        
        # Create 3 subplots per feature
        # 1. Time series
        ax1 = plt.subplot(n_features, 3, idx*3 + 1)
        ax1.plot(account_df['Business_Date'], account_df[col], 
                marker='o', markersize=3, linewidth=1, alpha=0.8)
        ax1.set_title(f'{col} - Time Series')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Daily delta
        ax2 = plt.subplot(n_features, 3, idx*3 + 2)
        delta_col = f'{col}_delta'
        if delta_col in account_df.columns:
            colors = ['red' if x < 0 else 'green' for x in account_df[delta_col].fillna(0)]
            ax2.bar(account_df['Business_Date'], account_df[delta_col], 
                   color=colors, alpha=0.7)
            ax2.set_title(f'{col} - Daily Delta')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Delta')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        # 3. Percentage change
        ax3 = plt.subplot(n_features, 3, idx*3 + 3)
        delta_pct_col = f'{col}_delta_pct'
        if delta_pct_col in account_df.columns:
            colors = ['red' if x < 0 else 'green' for x in account_df[delta_pct_col].fillna(0)]
            ax3.bar(account_df['Business_Date'], account_df[delta_pct_col], 
                   color=colors, alpha=0.7)
            ax3.set_title(f'{col} - Daily % Change')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('% Change')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / f'account_{account_id}_features.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return account_df

def plot_feature_comparison(df, feature_columns, top_n_accounts=5, save_dir=None):
    """
    Create comparison plots across multiple accounts
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    feature_columns : list
        List of features to plot
    top_n_accounts : int
        Number of top accounts to include
    save_dir : str
        Directory to save plots
    """
    # Get top N accounts by number of records
    top_accounts = df.groupby('header_account_id').size().nlargest(top_n_accounts)
    account_ids = top_accounts.index.tolist()
    
    # Create comparison plots
    n_features = len(feature_columns)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    fig.suptitle(f'Feature Comparison Across Top {top_n_accounts} Accounts', fontsize=16)
    
    for idx, col in enumerate(feature_columns):
        if col not in df.columns:
            continue
        
        ax = axes[idx]
        
        # Plot each account
        for account_id in account_ids:
            account_df = df[df['header_account_id'] == account_id]
            if not account_df.empty:
                ax.plot(account_df['Business_Date'], account_df[col], 
                       marker='o', markersize=2, label=f'Account {account_id}', 
                       alpha=0.7, linewidth=1)
        
        ax.set_title(f'{col} - Time Series Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / 'feature_comparison.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

def create_statistical_summary(df, account_id, feature_columns, save_dir=None):
    """
    Create statistical summary plots for an account
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    account_id : str/int
        Account ID to analyze
    feature_columns : list
        List of features to analyze
    save_dir : str
        Directory to save plots
    """
    # Filter for specific account
    account_df = df[df['header_account_id'] == account_id].copy()
    
    if account_df.empty:
        print(f"No data found for account {account_id}")
        return None
    
    # Calculate deltas
    account_df = calculate_deltas(account_df, feature_columns)
    
    # Create figure for statistical summaries
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Statistical Summary for Account: {account_id}', fontsize=16)
    
    # 1. Distribution of values
    ax1 = axes[0, 0]
    for col in feature_columns[:3]:  # Limit to first 3 for clarity
        if col in account_df.columns:
            ax1.hist(account_df[col].dropna(), alpha=0.5, label=col, bins=30)
    ax1.set_title('Distribution of Feature Values')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation heatmap
    ax2 = axes[0, 1]
    corr_features = [col for col in feature_columns if col in account_df.columns]
    if len(corr_features) > 1:
        corr_matrix = account_df[corr_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax2, square=True)
        ax2.set_title('Feature Correlation Matrix')
    
    # 3. Box plot for outlier detection
    ax3 = axes[1, 0]
    delta_cols = [f'{col}_delta' for col in feature_columns if f'{col}_delta' in account_df.columns]
    if delta_cols:
        data_for_box = [account_df[col].dropna() for col in delta_cols[:3]]
        labels_for_box = [col.replace('_delta', '') for col in delta_cols[:3]]
        ax3.boxplot(data_for_box, labels=labels_for_box)
        ax3.set_title('Delta Distribution (Box Plot)')
        ax3.set_ylabel('Delta Value')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Rolling statistics
    ax4 = axes[1, 1]
    if 'Applied_req' in account_df.columns:
        account_df['Applied_req_rolling_mean'] = account_df['Applied_req'].rolling(window=7).mean()
        account_df['Applied_req_rolling_std'] = account_df['Applied_req'].rolling(window=7).std()
        
        ax4.plot(account_df['Business_Date'], account_df['Applied_req'], 
                label='Applied_req', alpha=0.5)
        ax4.plot(account_df['Business_Date'], account_df['Applied_req_rolling_mean'], 
                label='7-day Moving Average', linewidth=2)
        ax4.fill_between(account_df['Business_Date'],
                        account_df['Applied_req_rolling_mean'] - account_df['Applied_req_rolling_std'],
                        account_df['Applied_req_rolling_mean'] + account_df['Applied_req_rolling_std'],
                        alpha=0.2, label='±1 Std Dev')
        ax4.set_title('Applied_req with Rolling Statistics')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / f'account_{account_id}_statistics.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Statistical summary saved to {save_path}")
    
    plt.show()
    
    return account_df

def main():
    """Main execution function"""
    
    # Configuration - UPDATE THESE VALUES
    CSV_PATH = 'your_data.csv'  # Update with your actual CSV path
    OUTPUT_DIR = 'account_plots'
    
    # Feature columns from your description
    FEATURE_COLUMNS = [
        'Long_market_value',
        'short_market_value',
        'Applied_req',
        'House_total_req',
        'Regulatory_Req',
        'Gross_Market_value'
    ]
    
    # Load data
    df = load_and_prepare_data(CSV_PATH)
    
    # Get sample account IDs
    print("\nSample Account IDs:")
    sample_accounts = df['header_account_id'].value_counts().head(5)
    for acc_id, count in sample_accounts.items():
        print(f"  {acc_id}: {count} records")
    
    # Plot for a specific account (use the first one as example)
    sample_account_id = sample_accounts.index[0]
    print(f"\nGenerating plots for account: {sample_account_id}")
    
    # Generate different types of plots
    plot_single_account(df, sample_account_id, FEATURE_COLUMNS, OUTPUT_DIR)
    create_statistical_summary(df, sample_account_id, FEATURE_COLUMNS, OUTPUT_DIR)
    
    # Create comparison plots across top accounts
    plot_feature_comparison(df, FEATURE_COLUMNS, top_n_accounts=5, save_dir=OUTPUT_DIR)
    
    print(f"\nAll plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
