"""
Example Client Script
Demonstrates how to use the Feature Plotting API
"""

import requests
import json
import base64
from PIL import Image
import io
from datetime import datetime, timedelta

# API base URL - adjust to match your deployment
BASE_URL = "http://localhost:5000/api/v1/plots"


def plot_historical_features(account_id, features=None, start_date=None, end_date=None):
    """
    Generate historical feature plots for an account
    
    Parameters:
    -----------
    account_id : str
        Account ID to plot
    features : list, optional
        List of features to plot
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    
    Returns:
    --------
    dict
        Response from API including base64 encoded image
    """
    url = f"{BASE_URL}/plot"
    
    payload = {
        "account_id": account_id,
        "plot_type": "historical_features"
    }
    
    if features:
        payload["features"] = features
    if start_date:
        payload["start_date"] = start_date
    if end_date:
        payload["end_date"] = end_date
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            # Save the image
            save_base64_image(data['image'], f"{account_id}_historical.png")
            print(f"Plot saved as {account_id}_historical.png")
        return data
    else:
        print(f"Error: {response.status_code}")
        return response.json()


def plot_zscore_analysis(account_id):
    """
    Generate Z-score analysis plot for an account
    
    Parameters:
    -----------
    account_id : str
        Account ID to plot
    
    Returns:
    --------
    dict
        Response from API
    """
    url = f"{BASE_URL}/plot"
    
    payload = {
        "account_id": account_id,
        "plot_type": "zscore_analysis"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            save_base64_image(data['image'], f"{account_id}_zscore.png")
            print(f"Z-score plot saved as {account_id}_zscore.png")
        return data
    else:
        print(f"Error: {response.status_code}")
        return response.json()


def get_feature_statistics(account_id, features=None):
    """
    Get statistical summary for an account
    
    Parameters:
    -----------
    account_id : str
        Account ID
    features : list, optional
        List of features
    
    Returns:
    --------
    dict
        Statistical summary
    """
    url = f"{BASE_URL}/statistics"
    
    payload = {
        "account_id": account_id
    }
    
    if features:
        payload["features"] = features
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print(f"\nStatistics for Account {account_id}:")
            print(json.dumps(data['statistics'], indent=2))
        return data
    else:
        print(f"Error: {response.status_code}")
        return response.json()


def get_quick_plot(account_id, plot_type='historical_features', save_path=None):
    """
    Get a quick plot using GET request
    
    Parameters:
    -----------
    account_id : str
        Account ID
    plot_type : str
        Type of plot
    save_path : str, optional
        Path to save the image
    
    Returns:
    --------
    bool
        Success status
    """
    url = f"{BASE_URL}/plot/{account_id}"
    params = {"plot_type": plot_type}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # Save the image
        save_path = save_path or f"{account_id}_{plot_type}_quick.png"
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Plot saved as {save_path}")
        return True
    else:
        print(f"Error: {response.status_code}")
        return False


def get_available_accounts():
    """
    Get list of available accounts
    
    Returns:
    --------
    list
        List of account IDs
    """
    url = f"{BASE_URL}/accounts"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print(f"Total accounts: {data['total_accounts']}")
            print(f"First 10 accounts: {data['accounts'][:10]}")
            return data['accounts']
    else:
        print(f"Error: {response.status_code}")
        return []


def get_date_range():
    """
    Get available date range in the data
    
    Returns:
    --------
    dict
        Date range information
    """
    url = f"{BASE_URL}/date-range"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print("Available date range:")
            print(json.dumps(data['date_range'], indent=2))
            return data['date_range']
    else:
        print(f"Error: {response.status_code}")
        return None


def get_available_features():
    """
    Get list of available features
    
    Returns:
    --------
    list
        List of feature names
    """
    url = f"{BASE_URL}/features"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print(f"Available features: {data['features']}")
            return data['features']
    else:
        print(f"Error: {response.status_code}")
        return []


def check_health():
    """
    Check API health status
    
    Returns:
    --------
    dict
        Health status
    """
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print("API Health Status:")
        print(json.dumps(data, indent=2))
        return data
    else:
        print(f"API is unhealthy: {response.status_code}")
        return None


def save_base64_image(base64_string, filename):
    """
    Save base64 encoded image to file
    
    Parameters:
    -----------
    base64_string : str
        Base64 encoded image
    filename : str
        Output filename
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(filename)


def main():
    """
    Example usage of the API
    """
    print("=" * 50)
    print("Feature Plotting API Client Example")
    print("=" * 50)
    
    # 1. Check health
    print("\n1. Checking API health...")
    check_health()
    
    # 2. Get available accounts
    print("\n2. Getting available accounts...")
    accounts = get_available_accounts()
    
    if not accounts:
        print("No accounts found. Please check if data is loaded.")
        return
    
    # Use first account for examples
    sample_account = accounts[0]
    print(f"\nUsing account {sample_account} for examples")
    
    # 3. Get date range
    print("\n3. Getting date range...")
    date_range = get_date_range()
    
    # 4. Get available features
    print("\n4. Getting available features...")
    features = get_available_features()
    
    # 5. Generate historical feature plot
    print(f"\n5. Generating historical feature plot for {sample_account}...")
    
    # Plot last 30 days if possible
    if date_range:
        end_date = datetime.strptime(date_range['max_date'], "%Y-%m-%d %H:%M:%S")
        start_date = end_date - timedelta(days=30)
        
        plot_historical_features(
            account_id=sample_account,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
    else:
        plot_historical_features(account_id=sample_account)
    
    # 6. Generate Z-score analysis
    print(f"\n6. Generating Z-score analysis for {sample_account}...")
    plot_zscore_analysis(sample_account)
    
    # 7. Get feature statistics
    print(f"\n7. Getting feature statistics for {sample_account}...")
    get_feature_statistics(sample_account)
    
    # 8. Quick plot example
    print(f"\n8. Generating quick plot for {sample_account}...")
    get_quick_plot(sample_account, plot_type='historical_features')
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("Check the generated PNG files in the current directory.")


if __name__ == "__main__":
    main()
