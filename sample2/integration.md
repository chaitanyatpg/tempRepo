 """
Integration Instructions for Feature Plotter Service
Add this to your existing Flask application
"""

# ==============================================================================
# STEP 1: Copy files to your app directory
# ==============================================================================
# Copy these files to your app/ directory:
# - plotter_service.py
# - plotter_routes.py  
# - report_generator.py

# ==============================================================================
# STEP 2: Update your app/__init__.py
# ==============================================================================
"""
Add to your app/__init__.py file:
"""

from flask import Flask
from flask_restx import Api
from app.plotter_routes import plotter_api  # Add this import

def create_app():
    app = Flask(__name__)
    
    # Your existing configuration
    api = Api(app, version='1.0', title='Your API Title')
    
    # Register existing namespaces
    # api.add_namespace(your_existing_namespace)
    
    # Add plotter namespace
    api.add_namespace(plotter_api, path='/api/v1/plotter')  # Add this line
    
    return app

# ==============================================================================
# STEP 3: Update your app/routes.py (if using single routes file)
# ==============================================================================
"""
If you have a single routes.py file, add this:
"""

from app.plotter_routes import plotter_api

# In your API initialization section:
api.add_namespace(plotter_api, path='/api/v1/plotter')

# ==============================================================================
# STEP 4: Install required dependencies
# ==============================================================================
"""
Add these to your requirements.txt:
"""
requirements = """
matplotlib>=3.5.0
seaborn>=0.12.0
scipy>=1.9.0
openpyxl>=3.0.0
pillow>=9.0.0
"""

# ==============================================================================
# STEP 5: Create output directory structure
# ==============================================================================
"""
Make sure your project has this directory structure:
"""
directory_structure = """
your_project/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── plotter_service.py      # NEW
│   ├── plotter_routes.py       # NEW
│   └── report_generator.py     # NEW
├── out/
│   └── PLOT/                    # Will be created automatically
│       ├── ACCOUNT_LEVEL/
│       ├── MARGIN_CENTER_LEVEL/
│       └── ALL_DATA/
├── run.py
└── requirements.txt
"""

# ==============================================================================
# STEP 6: Usage Examples
# ==============================================================================

# Example 1: Generate plots for a single account
example_single_account = """
import requests

url = "http://localhost:5000/api/v1/plotter/generate"
data = {
    "csv_path": "/path/to/your/data.csv",
    "account_ids": ["ACC12345"]
}

response = requests.post(url, json=data)
print(response.json())
"""

# Example 2: Generate plots for margin center
example_margin_center = """
import requests

url = "http://localhost:5000/api/v1/plotter/generate"
data = {
    "csv_path": "/path/to/your/data.csv",
    "margin_center": "MC001"
}

response = requests.post(url, json=data)
print(response.json())
"""

# Example 3: Batch process multiple accounts
example_batch = """
import requests

url = "http://localhost:5000/api/v1/plotter/batch"
data = {
    "csv_path": "/path/to/your/data.csv",
    "account_ids": ["ACC001", "ACC002", "ACC003", "ACC004", "ACC005"]
}

response = requests.post(url, json=data)
print(response.json())
"""

# Example 4: Custom engineered features
example_custom_features = """
import requests

url = "http://localhost:5000/api/v1/plotter/generate"
data = {
    "csv_path": "/path/to/your/data.csv",
    "account_ids": ["ACC12345"],
    "engineered_features": {
        "custom_ratio": {
            "formula": "Applied_req_delta_pct / Gross_Market_value_delta_pct",
            "plot_type": "ratio",
            "y_label": "Custom Ratio"
        },
        "new_metric": {
            "formula": "(Long_market_value - short_market_value) / Total_exposure",
            "plot_type": "percentage",
            "y_label": "New Metric %"
        }
    }
}

response = requests.post(url, json=data)
print(response.json())
"""

# ==============================================================================
# STEP 7: API Endpoints Available
# ==============================================================================
api_endpoints = """
POST   /api/v1/plotter/generate                     - Generate plots with custom parameters
POST   /api/v1/plotter/generate/account/<id>        - Generate plots for specific account
POST   /api/v1/plotter/generate/margin-center/<id>  - Generate plots for margin center
POST   /api/v1/plotter/batch                        - Batch process multiple accounts
GET    /api/v1/plotter/download/<path>              - Download plots as ZIP
GET    /api/v1/plotter/config                       - Get current configuration
POST   /api/v1/plotter/config                       - Update configuration
GET    /api/v1/plotter/list-outputs                 - List all generated outputs
"""

# ==============================================================================
# STEP 8: Swagger Documentation
# ==============================================================================
swagger_info = """
After integration, your Swagger UI will be available at:
http://localhost:5000/swagger

The plotter endpoints will appear under the 'plotter' namespace.
"""

# ==============================================================================
# STEP 9: Test the Integration
# ==============================================================================
test_script = """
# test_plotter.py
import requests
import json

BASE_URL = "http://localhost:5000/api/v1/plotter"

def test_plotter_integration():
    # Test 1: Check config endpoint
    response = requests.get(f"{BASE_URL}/config")
    print("Config endpoint:", response.status_code)
    
    # Test 2: List outputs
    response = requests.get(f"{BASE_URL}/list-outputs")
    print("List outputs:", response.status_code)
    
    # Test 3: Generate plot (replace with your CSV path)
    data = {
        "csv_path": "test_data.csv",
        "account_ids": ["TEST001"]
    }
    response = requests.post(f"{BASE_URL}/generate", json=data)
    print("Generate plot:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_plotter_integration()
"""

print("""
================================================================================
INTEGRATION COMPLETE!
================================================================================

Follow the steps above to integrate the plotter service into your Flask app.

Key Features:
✓ Flexible feature engineering with custom formulas
✓ Account-level and margin center-level plotting
✓ Batch processing support
✓ Excel and PDF report generation
✓ Professional plot styling with your color scheme
✓ Swagger documentation
✓ RESTful API endpoints

The service will automatically:
- Create directory structure in out/PLOT/
- Process basic, computed, and engineered features
- Generate comprehensive visualizations
- Create reports with statistics and analysis

For custom engineered features, simply add them to the configuration:
- Ratios: Applied_req / Gross_Market_value
- Combinations: (Feature1 - Feature2) / Feature3
- Any mathematical formula you need

All plots are saved with timestamps for historical tracking.
================================================================================
""")
