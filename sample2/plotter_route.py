"""
Flask Routes for Feature Plotter Service
Add these routes to your existing app/routes.py file
"""

from flask import Blueprint, request, jsonify, send_file, send_from_directory
from flask_restx import Namespace, Resource, fields, reqparse
import os
import json
from datetime import datetime
from pathlib import Path
import zipfile
import io

# Import the plotter service - adjust import based on your structure
from app.plotter_service import FeaturePlotterService

# Create namespace for Swagger documentation
plotter_api = Namespace('plotter', description='Historical feature plotting operations')

# Define Swagger models
plot_request_model = plotter_api.model('PlotRequest', {
    'csv_path': fields.String(required=True, description='Path to CSV file'),
    'account_ids': fields.List(fields.String, description='List of account IDs to plot'),
    'margin_center': fields.String(description='Margin center to plot'),
    'features': fields.List(fields.String, description='Custom list of features'),
    'engineered_features': fields.Raw(description='Custom engineered features configuration')
})

plot_response_model = plotter_api.model('PlotResponse', {
    'success': fields.Boolean(description='Success status'),
    'outputs': fields.List(fields.Raw, description='List of output directories and files'),
    'message': fields.String(description='Response message'),
    'error': fields.String(description='Error message if any')
})

# Initialize plotter service
plotter_service = FeaturePlotterService()


@plotter_api.route('/generate')
class GeneratePlots(Resource):
    @plotter_api.expect(plot_request_model)
    @plotter_api.marshal_with(plot_response_model)
    @plotter_api.doc(
        description='Generate plots for specified accounts or margin centers',
        responses={
            200: 'Success',
            400: 'Bad Request',
            500: 'Internal Server Error'
        }
    )
    def post(self):
        """Generate historical feature plots"""
        try:
            data = request.json
            csv_path = data.get('csv_path')
            account_ids = data.get('account_ids')
            margin_center = data.get('margin_center')
            features = data.get('features')
            engineered_features = data.get('engineered_features')
            
            # Validate input
            if not csv_path:
                return {
                    'success': False,
                    'message': 'csv_path is required',
                    'error': 'Missing required parameter'
                }, 400
            
            if not os.path.exists(csv_path):
                return {
                    'success': False,
                    'message': f'CSV file not found: {csv_path}',
                    'error': 'File not found'
                }, 400
            
            # Update configuration if custom features provided
            if features or engineered_features:
                config = plotter_service.feature_config
                if features:
                    config['basic_features'] = features
                if engineered_features:
                    config['engineered_features'] = engineered_features
                plotter_service.feature_config = config
            
            # Generate plots
            results = plotter_service.generate_plots(
                csv_path=csv_path,
                account_ids=account_ids,
                margin_center=margin_center
            )
            
            return {
                'success': results.get('success', False),
                'outputs': results.get('outputs', []),
                'message': 'Plots generated successfully' if results.get('success') else 'Plot generation failed',
                'error': results.get('error')
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': 'Internal server error',
                'error': str(e)
            }, 500


@plotter_api.route('/generate/account/<string:account_id>')
@plotter_api.param('account_id', 'The account identifier')
class GenerateAccountPlots(Resource):
    @plotter_api.doc(
        description='Generate plots for a specific account',
        params={
            'csv_path': {'description': 'Path to CSV file', 'required': True}
        },
        responses={
            200: 'Success',
            400: 'Bad Request',
            500: 'Internal Server Error'
        }
    )
    def post(self, account_id):
        """Generate plots for specific account"""
        try:
            data = request.json
            csv_path = data.get('csv_path')
            
            if not csv_path:
                return {
                    'success': False,
                    'message': 'csv_path is required'
                }, 400
            
            # Generate plots for single account
            results = plotter_service.generate_plots(
                csv_path=csv_path,
                account_ids=[account_id]
            )
            
            return {
                'success': results.get('success', False),
                'outputs': results.get('outputs', []),
                'message': f'Plots generated for account {account_id}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': 'Internal server error',
                'error': str(e)
            }, 500


@plotter_api.route('/generate/margin-center/<string:margin_center>')
@plotter_api.param('margin_center', 'The margin center identifier')
class GenerateMarginCenterPlots(Resource):
    @plotter_api.doc(
        description='Generate consolidated plots for a margin center',
        params={
            'csv_path': {'description': 'Path to CSV file', 'required': True}
        },
        responses={
            200: 'Success',
            400: 'Bad Request',
            500: 'Internal Server Error'
        }
    )
    def post(self, margin_center):
        """Generate plots for margin center"""
        try:
            data = request.json
            csv_path = data.get('csv_path')
            
            if not csv_path:
                return {
                    'success': False,
                    'message': 'csv_path is required'
                }, 400
            
            # Generate plots for margin center
            results = plotter_service.generate_plots(
                csv_path=csv_path,
                margin_center=margin_center
            )
            
            return {
                'success': results.get('success', False),
                'outputs': results.get('outputs', []),
                'message': f'Plots generated for margin center {margin_center}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': 'Internal server error',
                'error': str(e)
            }, 500


@plotter_api.route('/batch')
class BatchGeneratePlots(Resource):
    @plotter_api.doc(
        description='Generate plots for multiple accounts in batch',
        responses={
            200: 'Success',
            400: 'Bad Request',
            500: 'Internal Server Error'
        }
    )
    def post(self):
        """Batch generate plots for multiple accounts"""
        try:
            data = request.json
            csv_path = data.get('csv_path')
            account_ids = data.get('account_ids', [])
            
            if not csv_path:
                return {
                    'success': False,
                    'message': 'csv_path is required'
                }, 400
            
            if not account_ids:
                return {
                    'success': False,
                    'message': 'account_ids list is required'
                }, 400
            
            # Process in batches
            batch_results = []
            batch_size = 10  # Process 10 accounts at a time
            
            for i in range(0, len(account_ids), batch_size):
                batch = account_ids[i:i+batch_size]
                results = plotter_service.generate_plots(
                    csv_path=csv_path,
                    account_ids=batch
                )
                batch_results.extend(results.get('outputs', []))
            
            return {
                'success': True,
                'outputs': batch_results,
                'message': f'Processed {len(account_ids)} accounts in batches',
                'total_accounts': len(account_ids)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': 'Internal server error',
                'error': str(e)
            }, 500


@plotter_api.route('/download/<path:output_path>')
class DownloadPlots(Resource):
    @plotter_api.doc(
        description='Download generated plots as ZIP file',
        responses={
            200: 'Success - returns ZIP file',
            404: 'Directory not found'
        }
    )
    def get(self, output_path):
        """Download plots as ZIP file"""
        try:
            full_path = os.path.join('out/PLOT', output_path)
            
            if not os.path.exists(full_path):
                return {'error': 'Output directory not found'}, 404
            
            # Create ZIP file in memory
            memory_file = io.BytesIO()
            
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(full_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, full_path)
                        zipf.write(file_path, arcname)
            
            memory_file.seek(0)
            
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'{os.path.basename(output_path)}.zip'
            )
            
        except Exception as e:
            return {'error': str(e)}, 500


@plotter_api.route('/config')
class PlotterConfig(Resource):
    @plotter_api.doc(description='Get current plotter configuration')
    def get(self):
        """Get current feature configuration"""
        return {
            'success': True,
            'config': plotter_service.feature_config
        }
    
    @plotter_api.doc(description='Update plotter configuration')
    def post(self):
        """Update feature configuration"""
        try:
            data = request.json
            plotter_service.feature_config = data
            
            return {
                'success': True,
                'message': 'Configuration updated',
                'config': plotter_service.feature_config
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }, 500


@plotter_api.route('/list-outputs')
class ListOutputs(Resource):
    @plotter_api.doc(description='List all available output directories')
    def get(self):
        """List all generated output directories"""
        try:
            base_dir = 'out/PLOT'
            outputs = {
                'account_level': [],
                'margin_center_level': [],
                'all_data': []
            }
            
            # List account level outputs
            account_dir = os.path.join(base_dir, 'ACCOUNT_LEVEL')
            if os.path.exists(account_dir):
                outputs['account_level'] = sorted(os.listdir(account_dir))
            
            # List margin center outputs
            mc_dir = os.path.join(base_dir, 'MARGIN_CENTER_LEVEL')
            if os.path.exists(mc_dir):
                outputs['margin_center_level'] = sorted(os.listdir(mc_dir))
            
            # List all data outputs
            all_dir = os.path.join(base_dir, 'ALL_DATA')
            if os.path.exists(all_dir):
                outputs['all_data'] = sorted(os.listdir(all_dir))
            
            return {
                'success': True,
                'outputs': outputs,
                'base_directory': base_dir
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }, 500


# Integration function to add to your existing routes.py
def register_plotter_routes(api_instance):
    """
    Register plotter routes with your existing API instance
    Call this from your main routes.py file
    
    Example:
        from app.plotter_routes import register_plotter_routes
        register_plotter_routes(api)
    """
    api_instance.add_namespace(plotter_api, path='/api/v1/plotter')
