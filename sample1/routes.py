"""
Flask API Routes for Historical Feature Plotting
Integrates with Swagger/OpenAPI for documentation
"""

from flask import Blueprint, request, jsonify, send_file
from flask_restx import Namespace, Resource, fields, reqparse
import pandas as pd
import io
import base64
from datetime import datetime
from typing import Optional
import os

# Import the plotter class (adjust path as needed for your project structure)
from feature_plotter import HistoricalFeaturePlotter

# Create namespace for Swagger documentation
api = Namespace('feature_plots', description='Historical feature plotting for anomaly detection')

# Define models for Swagger documentation
plot_request_model = api.model('PlotRequest', {
    'account_id': fields.String(required=True, description='Account ID to plot'),
    'features': fields.List(fields.String, description='List of feature columns to plot', 
                           default=['Long_market_value', 'short_market_value', 'Applied_req', 
                                   'House_total_req', 'Regulatory_Req', 'Gross_Market_value']),
    'start_date': fields.String(description='Start date (YYYY-MM-DD format)'),
    'end_date': fields.String(description='End date (YYYY-MM-DD format)'),
    'plot_type': fields.String(description='Type of plot', 
                               enum=['historical_features', 'zscore_analysis', 'both'],
                               default='historical_features')
})

statistics_request_model = api.model('StatisticsRequest', {
    'account_id': fields.String(required=True, description='Account ID'),
    'features': fields.List(fields.String, description='List of feature columns')
})

# Response models
plot_response_model = api.model('PlotResponse', {
    'success': fields.Boolean(description='Success status'),
    'account_id': fields.String(description='Account ID'),
    'plot_type': fields.String(description='Type of plot generated'),
    'image': fields.String(description='Base64 encoded image'),
    'message': fields.String(description='Response message')
})

statistics_response_model = api.model('StatisticsResponse', {
    'success': fields.Boolean(description='Success status'),
    'account_id': fields.String(description='Account ID'),
    'statistics': fields.Raw(description='Statistical summary of features'),
    'message': fields.String(description='Response message')
})


# Initialize plotter (configure with your data path)
DATA_PATH = os.environ.get('DATA_PATH', 'data.csv')  # Set via environment variable or config
plotter = None


def initialize_plotter(data_path: str = None, dataframe: pd.DataFrame = None):
    """
    Initialize the plotter with data
    Call this from your main app initialization
    """
    global plotter
    if data_path:
        plotter = HistoricalFeaturePlotter(data_path=data_path)
    elif dataframe is not None:
        plotter = HistoricalFeaturePlotter(df=dataframe)
    else:
        raise ValueError("Either data_path or dataframe must be provided")


@api.route('/plot')
class FeaturePlot(Resource):
    @api.expect(plot_request_model)
    @api.marshal_with(plot_response_model)
    @api.doc(
        description='Generate historical feature plots for a specific account',
        responses={
            200: 'Success',
            400: 'Bad Request',
            404: 'Account not found',
            500: 'Internal Server Error'
        }
    )
    def post(self):
        """Generate historical feature plots"""
        try:
            # Parse request data
            data = request.json
            account_id = data.get('account_id')
            features = data.get('features')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            plot_type = data.get('plot_type', 'historical_features')
            
            # Validate required parameters
            if not account_id:
                return {
                    'success': False,
                    'message': 'account_id is required'
                }, 400
            
            # Check if plotter is initialized
            if plotter is None:
                return {
                    'success': False,
                    'message': 'Plotter not initialized. Please contact administrator.'
                }, 500
            
            # Prepare date range if provided
            date_range = None
            if start_date and end_date:
                try:
                    date_range = (start_date, end_date)
                except ValueError as e:
                    return {
                        'success': False,
                        'message': f'Invalid date format: {str(e)}'
                    }, 400
            
            # Generate plots based on type
            image_base64 = None
            
            if plot_type in ['historical_features', 'both']:
                try:
                    image_base64 = plotter.plot_historical_features(
                        account_id=account_id,
                        features=features,
                        date_range=date_range,
                        return_base64=True
                    )
                except ValueError as e:
                    return {
                        'success': False,
                        'account_id': account_id,
                        'message': str(e)
                    }, 404
            
            if plot_type in ['zscore_analysis', 'both']:
                try:
                    zscore_image = plotter.plot_zscore_analysis(
                        account_id=account_id,
                        return_base64=True
                    )
                    
                    # If both plots requested, combine them (or return separately)
                    if plot_type == 'both' and image_base64:
                        # For simplicity, we'll return the zscore as a separate field
                        return {
                            'success': True,
                            'account_id': account_id,
                            'plot_type': plot_type,
                            'image': image_base64,
                            'zscore_image': zscore_image,
                            'message': 'Both plots generated successfully'
                        }
                    else:
                        image_base64 = zscore_image
                        
                except ValueError as e:
                    if plot_type == 'zscore_analysis':
                        return {
                            'success': False,
                            'account_id': account_id,
                            'message': str(e)
                        }, 400
            
            return {
                'success': True,
                'account_id': account_id,
                'plot_type': plot_type,
                'image': image_base64,
                'message': 'Plot generated successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Internal error: {str(e)}'
            }, 500


@api.route('/statistics')
class FeatureStatistics(Resource):
    @api.expect(statistics_request_model)
    @api.marshal_with(statistics_response_model)
    @api.doc(
        description='Get statistical summary of computed features for an account',
        responses={
            200: 'Success',
            400: 'Bad Request',
            404: 'Account not found',
            500: 'Internal Server Error'
        }
    )
    def post(self):
        """Get feature statistics for an account"""
        try:
            # Parse request data
            data = request.json
            account_id = data.get('account_id')
            features = data.get('features')
            
            # Validate required parameters
            if not account_id:
                return {
                    'success': False,
                    'message': 'account_id is required'
                }, 400
            
            # Check if plotter is initialized
            if plotter is None:
                return {
                    'success': False,
                    'message': 'Plotter not initialized. Please contact administrator.'
                }, 500
            
            # Get statistics
            try:
                stats = plotter.get_feature_statistics(
                    account_id=account_id,
                    features=features
                )
                
                return {
                    'success': True,
                    'account_id': account_id,
                    'statistics': stats,
                    'message': 'Statistics computed successfully'
                }
                
            except ValueError as e:
                return {
                    'success': False,
                    'account_id': account_id,
                    'message': str(e)
                }, 404
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Internal error: {str(e)}'
            }, 500


@api.route('/accounts')
class AccountList(Resource):
    @api.doc(
        description='Get list of available accounts',
        responses={
            200: 'Success',
            500: 'Internal Server Error'
        }
    )
    def get(self):
        """Get list of all available account IDs"""
        try:
            if plotter is None or plotter.df is None:
                return {
                    'success': False,
                    'message': 'Data not loaded'
                }, 500
            
            accounts = plotter.df['header_account_id'].unique().tolist()
            
            return {
                'success': True,
                'total_accounts': len(accounts),
                'accounts': accounts[:100],  # Return first 100 for safety
                'message': f'Found {len(accounts)} accounts'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Internal error: {str(e)}'
            }, 500


@api.route('/plot/<string:account_id>')
@api.param('account_id', 'The account identifier')
class QuickPlot(Resource):
    @api.doc(
        description='Quick plot generation with default settings',
        params={
            'plot_type': {'description': 'Type of plot', 'enum': ['historical_features', 'zscore_analysis']},
            'start_date': {'description': 'Start date (YYYY-MM-DD)'},
            'end_date': {'description': 'End date (YYYY-MM-DD)'}
        },
        responses={
            200: 'Success - returns PNG image',
            404: 'Account not found',
            500: 'Internal Server Error'
        }
    )
    def get(self, account_id):
        """Generate and return plot as PNG image"""
        try:
            # Get query parameters
            plot_type = request.args.get('plot_type', 'historical_features')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            
            # Check if plotter is initialized
            if plotter is None:
                return {'error': 'Plotter not initialized'}, 500
            
            # Prepare date range
            date_range = None
            if start_date and end_date:
                date_range = (start_date, end_date)
            
            # Generate plot
            if plot_type == 'zscore_analysis':
                image_base64 = plotter.plot_zscore_analysis(
                    account_id=account_id,
                    return_base64=True
                )
            else:
                image_base64 = plotter.plot_historical_features(
                    account_id=account_id,
                    date_range=date_range,
                    return_base64=True
                )
            
            # Convert base64 to bytes
            image_bytes = base64.b64decode(image_base64)
            
            # Return as image file
            return send_file(
                io.BytesIO(image_bytes),
                mimetype='image/png',
                as_attachment=True,
                download_name=f'{account_id}_{plot_type}.png'
            )
            
        except ValueError as e:
            return {'error': str(e)}, 404
        except Exception as e:
            return {'error': f'Internal error: {str(e)}'}, 500


# Additional utility routes

@api.route('/date-range')
class DateRange(Resource):
    @api.doc(description='Get available date range in the data')
    def get(self):
        """Get the date range available in the data"""
        try:
            if plotter is None or plotter.df is None:
                return {
                    'success': False,
                    'message': 'Data not loaded'
                }, 500
            
            date_range = {
                'min_date': str(plotter.df['Business_Date'].min()),
                'max_date': str(plotter.df['Business_Date'].max()),
                'total_days': (plotter.df['Business_Date'].max() - plotter.df['Business_Date'].min()).days
            }
            
            return {
                'success': True,
                'date_range': date_range
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Internal error: {str(e)}'
            }, 500


@api.route('/features')
class FeatureList(Resource):
    @api.doc(description='Get list of available feature columns')
    def get(self):
        """Get list of available feature columns"""
        try:
            if plotter is None:
                return {
                    'success': False,
                    'message': 'Plotter not initialized'
                }, 500
            
            return {
                'success': True,
                'features': plotter.feature_columns,
                'total_features': len(plotter.feature_columns)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Internal error: {str(e)}'
            }, 500


# Health check endpoint
@api.route('/health')
class HealthCheck(Resource):
    @api.doc(description='Health check endpoint')
    def get(self):
        """Check if the plotting service is healthy"""
        try:
            is_healthy = plotter is not None and plotter.df is not None
            
            if is_healthy:
                return {
                    'status': 'healthy',
                    'data_loaded': True,
                    'total_records': len(plotter.df),
                    'total_accounts': plotter.df['header_account_id'].nunique()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'data_loaded': False,
                    'message': 'Data not loaded'
                }, 503
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }, 503
