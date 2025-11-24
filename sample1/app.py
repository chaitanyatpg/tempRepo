"""
Main Flask Application Integration
Shows how to integrate the plotting routes into your existing Flask app
"""

from flask import Flask
from flask_restx import Api
from flask_cors import CORS
import os
import pandas as pd

# Import the routes and initialization function
from routes import api as plotting_namespace, initialize_plotter

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Configure Swagger UI
app.config['RESTX_MASK_SWAGGER'] = False
app.config['ERROR_404_HELP'] = False

# Create API instance
api = Api(
    app,
    version='1.0',
    title='Anomaly Detection Feature Plotter API',
    description='API for plotting historical computed features used in anomaly detection',
    doc='/swagger'  # Swagger UI available at /swagger
)

# Add the plotting namespace to your API
api.add_namespace(plotting_namespace, path='/api/v1/plots')


def load_data_and_initialize():
    """
    Load your data and initialize the plotter
    This should be called when the app starts
    """
    # Option 1: Load from CSV file
    data_path = os.environ.get('DATA_PATH', 'your_data.csv')
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        initialize_plotter(data_path=data_path)
        print("Plotter initialized successfully!")
    else:
        print(f"Warning: Data file {data_path} not found")
        
    # Option 2: Load from database or other source
    # df = load_from_database()  # Your function to load data
    # initialize_plotter(dataframe=df)


# Initialize data when app starts
with app.app_context():
    load_data_and_initialize()


# Add any additional routes here
@app.route('/')
def home():
    """Home endpoint"""
    return {
        'message': 'Anomaly Detection Feature Plotter API',
        'swagger_ui': '/swagger',
        'endpoints': {
            'plot': '/api/v1/plots/plot',
            'statistics': '/api/v1/plots/statistics',
            'accounts': '/api/v1/plots/accounts',
            'quick_plot': '/api/v1/plots/plot/<account_id>',
            'date_range': '/api/v1/plots/date-range',
            'features': '/api/v1/plots/features',
            'health': '/api/v1/plots/health'
        }
    }


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
