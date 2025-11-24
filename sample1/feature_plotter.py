"""
Historical Feature Plotter for Anomaly Detection
Plots computed features (deltas, delta_pct) used in Isolation Forest model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class HistoricalFeaturePlotter:
    """
    Class to plot historical computed features used in anomaly detection model
    """
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize plotter with either data path or DataFrame
        
        Parameters:
        -----------
        data_path : str, optional
            Path to CSV file
        df : pd.DataFrame, optional
            Pre-loaded DataFrame
        """
        self.data_path = data_path
        self.df = df
        
        # Default feature columns - can be overridden
        self.feature_columns = [
            'Long_market_value',
            'short_market_value',
            'Applied_req',
            'House_total_req',
            'Regulatory_Req',
            'Gross_Market_value'
        ]
        
        # Computed features that will be generated
        self.computed_features = []
        
        if self.data_path:
            self.load_data()
    
    def load_data(self):
        """Load data from CSV if path provided"""
        if self.data_path:
            self.df = pd.read_csv(self.data_path)
            self.df['Business_Date'] = pd.to_datetime(self.df['Business_Date'])
            self.df = self.df.sort_values(['header_account_id', 'Business_Date'])
            print(f"Data loaded: {len(self.df)} records")
            print(f"Date range: {self.df['Business_Date'].min()} to {self.df['Business_Date'].max()}")
            print(f"Total accounts: {self.df['header_account_id'].nunique()}")
    
    def compute_features_for_account(self, account_id: str, 
                                   features: List[str] = None) -> pd.DataFrame:
        """
        Compute delta and delta_pct features for a specific account
        
        Parameters:
        -----------
        account_id : str
            Account ID to process
        features : List[str], optional
            List of feature columns to compute deltas for
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with computed features
        """
        if features is None:
            features = self.feature_columns
        
        # Filter for account
        account_df = self.df[self.df['header_account_id'] == account_id].copy()
        
        if account_df.empty:
            raise ValueError(f"No data found for account {account_id}")
        
        # Sort by date
        account_df = account_df.sort_values('Business_Date')
        
        # Compute features for each column
        computed_cols = []
        for col in features:
            if col in account_df.columns:
                # Delta (day-over-day absolute change)
                delta_col = f'{col}_delta'
                account_df[delta_col] = account_df[col].diff()
                computed_cols.append(delta_col)
                
                # Delta percentage
                delta_pct_col = f'{col}_delta_pct'
                account_df[delta_pct_col] = account_df[col].pct_change() * 100
                computed_cols.append(delta_pct_col)
        
        # Calculate Robust Z-score for Applied_req if present
        if 'Applied_req' in features and 'Applied_req_delta' in account_df.columns:
            z_score_col = 'Applied_req_delta_zscore'
            account_df[z_score_col] = self.calculate_robust_z_score(
                account_df['Applied_req_delta']
            )
            computed_cols.append(z_score_col)
        
        self.computed_features = computed_cols
        return account_df
    
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
    
    def plot_historical_features(self, 
                               account_id: str,
                               features: List[str] = None,
                               date_range: Tuple[str, str] = None,
                               figsize: Tuple[int, int] = (20, 16),
                               return_base64: bool = False) -> Optional[str]:
        """
        Create comprehensive plot of historical computed features for an account
        
        Parameters:
        -----------
        account_id : str
            Account ID to plot
        features : List[str], optional
            Feature columns to plot
        date_range : Tuple[str, str], optional
            Start and end date for filtering
        figsize : Tuple[int, int]
            Figure size
        return_base64 : bool
            If True, return base64 encoded image instead of displaying
        
        Returns:
        --------
        str, optional
            Base64 encoded image if return_base64=True
        """
        # Compute features
        account_df = self.compute_features_for_account(account_id, features)
        
        # Apply date range filter if provided
        if date_range:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            account_df = account_df[(account_df['Business_Date'] >= start_date) & 
                                  (account_df['Business_Date'] <= end_date)]
        
        # Determine which computed features to plot
        delta_features = [f for f in self.computed_features if '_delta' in f and '_pct' not in f and '_zscore' not in f]
        delta_pct_features = [f for f in self.computed_features if '_delta_pct' in f]
        
        # Create subplots
        n_rows = len(delta_features)
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Historical Computed Features - Account: {account_id}\n'
                    f'Date Range: {account_df["Business_Date"].min():%Y-%m-%d} to '
                    f'{account_df["Business_Date"].max():%Y-%m-%d}',
                    fontsize=14, y=1.02)
        
        for idx, delta_feat in enumerate(delta_features):
            base_name = delta_feat.replace('_delta', '')
            
            # Plot 1: Delta time series
            ax1 = axes[idx, 0]
            valid_data = account_df[delta_feat].notna()
            dates = account_df.loc[valid_data, 'Business_Date']
            values = account_df.loc[valid_data, delta_feat]
            
            ax1.plot(dates, values, linewidth=1, alpha=0.8, marker='o', markersize=2)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add mean and std lines
            mean_val = values.mean()
            std_val = values.std()
            ax1.axhline(y=mean_val, color='blue', linestyle='--', linewidth=1, alpha=0.5, label=f'Mean: {mean_val:.2f}')
            ax1.axhline(y=mean_val + 2*std_val, color='red', linestyle='--', linewidth=1, alpha=0.3, label=f'+2σ: {mean_val + 2*std_val:.2f}')
            ax1.axhline(y=mean_val - 2*std_val, color='red', linestyle='--', linewidth=1, alpha=0.3, label=f'-2σ: {mean_val - 2*std_val:.2f}')
            
            ax1.set_title(f'{base_name} - Daily Delta (Absolute Change)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Delta Value')
            ax1.legend(fontsize='small', loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Delta percentage
            ax2 = axes[idx, 1]
            delta_pct_feat = f'{base_name}_delta_pct'
            if delta_pct_feat in account_df.columns:
                valid_data_pct = account_df[delta_pct_feat].notna()
                dates_pct = account_df.loc[valid_data_pct, 'Business_Date']
                values_pct = account_df.loc[valid_data_pct, delta_pct_feat]
                
                # Color based on positive/negative
                colors = ['red' if v < 0 else 'green' for v in values_pct]
                ax2.scatter(dates_pct, values_pct, c=colors, alpha=0.6, s=10)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                # Add percentile lines
                p95 = np.percentile(values_pct.dropna(), 95)
                p5 = np.percentile(values_pct.dropna(), 5)
                ax2.axhline(y=p95, color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'95th %ile: {p95:.2f}%')
                ax2.axhline(y=p5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'5th %ile: {p5:.2f}%')
                
                ax2.set_title(f'{base_name} - Daily Delta (Percentage Change)')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Delta %')
                ax2.legend(fontsize='small', loc='best')
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Distribution histogram
            ax3 = axes[idx, 2]
            
            # Plot distribution of delta values
            ax3.hist(values.dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax3.axvline(x=mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}')
            ax3.axvline(x=np.median(values), color='green', linestyle='--', linewidth=1, label=f'Median: {np.median(values):.2f}')
            
            # Mark outliers (beyond 3 sigma)
            outlier_threshold = 3 * std_val
            ax3.axvline(x=mean_val + outlier_threshold, color='red', linestyle=':', linewidth=1, alpha=0.5)
            ax3.axvline(x=mean_val - outlier_threshold, color='red', linestyle=':', linewidth=1, alpha=0.5)
            
            ax3.set_title(f'{base_name} Delta - Distribution')
            ax3.set_xlabel('Delta Value')
            ax3.set_ylabel('Frequency')
            ax3.legend(fontsize='small')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if return_base64:
            # Convert plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return image_base64
        else:
            plt.show()
            return None
    
    def plot_zscore_analysis(self,
                            account_id: str,
                            figsize: Tuple[int, int] = (15, 8),
                            return_base64: bool = False) -> Optional[str]:
        """
        Plot Z-score analysis for Applied_req
        
        Parameters:
        -----------
        account_id : str
            Account ID to plot
        figsize : Tuple[int, int]
            Figure size
        return_base64 : bool
            If True, return base64 encoded image
        
        Returns:
        --------
        str, optional
            Base64 encoded image if return_base64=True
        """
        # Compute features
        account_df = self.compute_features_for_account(account_id)
        
        if 'Applied_req_delta_zscore' not in account_df.columns:
            raise ValueError("Applied_req Z-score not computed")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Z-Score Analysis for Applied_req - Account: {account_id}', fontsize=14)
        
        # Plot 1: Z-score time series
        ax1 = axes[0, 0]
        valid_mask = account_df['Applied_req_delta_zscore'].notna()
        ax1.plot(account_df.loc[valid_mask, 'Business_Date'],
                account_df.loc[valid_mask, 'Applied_req_delta_zscore'],
                linewidth=1, marker='o', markersize=2)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='±3σ threshold')
        ax1.axhline(y=-3, color='red', linestyle='--', alpha=0.5)
        ax1.fill_between(account_df.loc[valid_mask, 'Business_Date'],
                         -3, 3, alpha=0.1, color='green', label='Normal range')
        ax1.set_title('Applied_req Delta - Robust Z-Score Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Z-Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Original vs Delta
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(account_df['Business_Date'], account_df['Applied_req'],
                        'b-', label='Applied_req', alpha=0.7)
        line2 = ax2_twin.plot(account_df['Business_Date'], account_df['Applied_req_delta'],
                             'r-', label='Delta', alpha=0.7)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Applied_req Value', color='b')
        ax2_twin.set_ylabel('Delta', color='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2.set_title('Applied_req: Original vs Delta')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best')
        
        # Plot 3: Z-score distribution
        ax3 = axes[1, 0]
        z_scores = account_df['Applied_req_delta_zscore'].dropna()
        ax3.hist(z_scores, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='±3σ')
        ax3.axvline(x=-3, color='red', linestyle='--', alpha=0.5)
        
        # Add statistics
        ax3.text(0.02, 0.98, f'Mean: {z_scores.mean():.3f}\n'
                            f'Std: {z_scores.std():.3f}\n'
                            f'Outliers (|z|>3): {(np.abs(z_scores) > 3).sum()}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_title('Z-Score Distribution')
        ax3.set_xlabel('Z-Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot for normality check
        ax4 = axes[1, 1]
        from scipy import stats
        stats.probplot(z_scores, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normality Check)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if return_base64:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return image_base64
        else:
            plt.show()
            return None
    
    def get_feature_statistics(self, account_id: str, features: List[str] = None) -> Dict:
        """
        Get statistical summary of computed features for an account
        
        Parameters:
        -----------
        account_id : str
            Account ID
        features : List[str], optional
            Feature columns
        
        Returns:
        --------
        Dict
            Statistical summary
        """
        account_df = self.compute_features_for_account(account_id, features)
        
        stats = {
            'account_id': account_id,
            'date_range': {
                'start': str(account_df['Business_Date'].min()),
                'end': str(account_df['Business_Date'].max())
            },
            'n_records': len(account_df),
            'features': {}
        }
        
        for feat in self.computed_features:
            if feat in account_df.columns:
                values = account_df[feat].dropna()
                stats['features'][feat] = {
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'outliers_3sigma': int((np.abs(values - values.mean()) > 3 * values.std()).sum()),
                    'missing': int(account_df[feat].isna().sum())
                }
        
        return stats
