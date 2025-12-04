"""
Flask API Service for Plant Readiness Prediction
Loads models exported from Google Colab

Usage:
    python app/Services/ML/flask_api.py

Models loaded:
    - rf_readiness_model.pkl (Readiness classifier)
    - label_encoder_seed.pkl (Seed type encoder)
    - label_encoder_soil.pkl (Soil type encoder)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables for model and encoders
rf_model = None
le_seed = None
le_soil = None
dataset_df = None
model_loaded = False

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR


def load_models():
    """Load the exported models and encoders from Colab"""
    global rf_model, le_seed, le_soil, dataset_df, model_loaded
    
    try:
        # Debug: Print current directory and files
        print(f"BASE_DIR: {BASE_DIR}")
        print(f"MODEL_DIR: {MODEL_DIR}")
        print(f"Files in directory: {os.listdir(BASE_DIR)}")
        
        # Model file paths
        readiness_model_path = os.path.join(MODEL_DIR, 'rf_readiness_model.pkl')
        seed_encoder_path = os.path.join(MODEL_DIR, 'label_encoder_seed.pkl')
        soil_encoder_path = os.path.join(MODEL_DIR, 'label_encoder_soil.pkl')
        # Try dataset in same directory first (for Railway), then Laravel path (for local)
        dataset_path = os.path.join(BASE_DIR, 'plant_readiness_dataset.csv')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(BASE_DIR, '../../../storage/app/plant_readiness_dataset.csv')
        
        # Check if files exist
        if not os.path.exists(readiness_model_path):
            raise FileNotFoundError(f"Readiness model not found: {readiness_model_path}")
        if not os.path.exists(seed_encoder_path):
            raise FileNotFoundError(f"Seed encoder not found: {seed_encoder_path}")
        if not os.path.exists(soil_encoder_path):
            raise FileNotFoundError(f"Soil encoder not found: {soil_encoder_path}")
        
        print("Loading models from Colab exports...")
        
        # Load readiness model
        rf_model = joblib.load(readiness_model_path)
        print(f"✓ Readiness model loaded: {type(rf_model).__name__}")
        
        # Load encoders
        le_seed = joblib.load(seed_encoder_path)
        print(f"✓ Seed encoder loaded ({len(le_seed.classes_)} types)")
        
        le_soil = joblib.load(soil_encoder_path)
        print(f"✓ Soil encoder loaded ({len(le_soil.classes_)} types)")
        
        # Load dataset for stage/care/fertilizer lookup
        if os.path.exists(dataset_path):
            dataset_df = pd.read_csv(dataset_path)
            print(f"✓ Dataset loaded ({len(dataset_df)} records)")
        else:
            print("⚠️ Dataset not found, will use fallback values")
            dataset_df = None
        
        model_loaded = True
        print("\n✅ All models loaded successfully!")
        print(f"   Seed types: {list(le_seed.classes_)}")
        print(f"   Soil types: {list(le_soil.classes_)}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        model_loaded = False
        return False


def get_stage_info(seed_type, age_weeks, soil_type):
    """Get stage description, care instructions, and fertilizer from dataset"""
    if dataset_df is None:
        return {
            'stage': 'Growing',
            'care': 'Maintain proper care and monitoring.',
            'fertilizer': 'None'
        }
    
    try:
        # Try exact match first
        matching = dataset_df[
            (dataset_df['Seed_Type'] == seed_type) &
            (dataset_df['Age_Weeks'] == age_weeks) &
            (dataset_df['Soil_Type'] == soil_type)
        ]
        
        if not matching.empty:
            record = matching.iloc[0]
            return {
                'stage': record['Stage_Description'],
                'care': record['Care_Instructions'],
                'fertilizer': record['Fertilizer'] if pd.notna(record['Fertilizer']) and record['Fertilizer'] != '' else 'None'
            }
        
        # Find closest match by age
        closest = dataset_df[
            (dataset_df['Seed_Type'] == seed_type) &
            (dataset_df['Soil_Type'] == soil_type)
        ]
        
        if not closest.empty:
            closest = closest.iloc[(closest['Age_Weeks'] - age_weeks).abs().argsort()[:1]]
            record = closest.iloc[0]
            return {
                'stage': record['Stage_Description'],
                'care': record['Care_Instructions'],
                'fertilizer': record['Fertilizer'] if pd.notna(record['Fertilizer']) and record['Fertilizer'] != '' else 'None'
            }
        
        # Fallback: just match seed type
        fallback = dataset_df[dataset_df['Seed_Type'] == seed_type]
        if not fallback.empty:
            fallback = fallback.iloc[(fallback['Age_Weeks'] - age_weeks).abs().argsort()[:1]]
            record = fallback.iloc[0]
            return {
                'stage': record['Stage_Description'],
                'care': record['Care_Instructions'],
                'fertilizer': record['Fertilizer'] if pd.notna(record['Fertilizer']) and record['Fertilizer'] != '' else 'None'
            }
        
    except Exception as e:
        print(f"Error getting stage info: {str(e)}")
    
    return {
        'stage': 'Growing',
        'care': 'Maintain proper care and monitoring.',
        'fertilizer': 'None'
    }


def calculate_days_until_ready(seed_type, age_weeks):
    """Calculate days until plant is ready to sell"""
    if dataset_df is None:
        return 0
    
    try:
        ready_records = dataset_df[
            (dataset_df['Seed_Type'] == seed_type) &
            (dataset_df['Ready_to_Sell'] == 1)
        ]
        
        if not ready_records.empty:
            min_ready_age = ready_records['Age_Weeks'].min()
            days = max(0, (min_ready_age - age_weeks) * 7)
            return int(days)
    except Exception as e:
        print(f"Error calculating days until ready: {str(e)}")
    
    return 0


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - shows API is running"""
    return jsonify({
        'message': 'Plant Readiness Prediction API is running!',
        'status': 'online',
        'model_loaded': model_loaded,
        'endpoints': {
            'health': '/health',
            'model_info': '/model-info',
            'predict': '/predict (POST)'
        },
        'version': '1.0.0'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_ready': model_loaded,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Models not loaded'
        }), 503
    
    try:
        info = {
            'algorithm': 'Random Forest (Scikit-learn)',
            'model_type': 'RandomForestClassifier',
            'n_estimators': rf_model.n_estimators if hasattr(rf_model, 'n_estimators') else 100,
            'max_depth': rf_model.max_depth if hasattr(rf_model, 'max_depth') else 10,
            'features': ['Age (Weeks)', 'Seed Type', 'Soil Type', 'Height (inches)'],
            'seed_types': list(le_seed.classes_),
            'soil_types': list(le_soil.classes_),
            'model_loaded': True,
            'dataset_size': len(dataset_df) if dataset_df is not None else 0,
            'note': 'Model uses ACTUAL height from progress records'
        }
        
        return jsonify({
            'success': True,
            'model_info': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict plant readiness"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please ensure .pkl files are in app/Services/ML/'
        }), 503
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['seed_type', 'age_weeks', 'soil_type', 'avg_height_in']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        seed_type = str(data['seed_type'])
        age_weeks = float(data['age_weeks'])
        soil_type = str(data['soil_type'])
        height_in = float(data['avg_height_in'])
        
        # Validate inputs
        if height_in < 0:
            return jsonify({
                'success': False,
                'error': 'Height cannot be negative.'
            }), 400
        
        # Validate seed type and soil type
        if seed_type not in le_seed.classes_:
            return jsonify({
                'success': False,
                'error': f'Invalid seed_type. Must be one of: {list(le_seed.classes_)}'
            }), 400
        
        if soil_type not in le_soil.classes_:
            return jsonify({
                'success': False,
                'error': f'Invalid soil_type. Must be one of: {list(le_soil.classes_)}'
            }), 400
        
        # Encode inputs
        seed_encoded = le_seed.transform([seed_type])[0]
        soil_encoded = le_soil.transform([soil_type])[0]
        
        # Create feature vector: [Age, Seed, Soil, Height]
        X_input = np.array([[age_weeks, seed_encoded, soil_encoded, height_in]])
        
        # Predict readiness
        readiness_prob = rf_model.predict_proba(X_input)[0]
        is_ready = bool(rf_model.predict(X_input)[0])
        confidence = float(max(readiness_prob) * 100)
        
        # Get stage information from dataset
        stage_info = get_stage_info(seed_type, age_weeks, soil_type)
        
        # Calculate days until ready
        days_until_ready = 0 if is_ready else calculate_days_until_ready(seed_type, age_weeks)
        
        # Weekly growth rate
        weekly_growth = height_in / max(1, age_weeks)
        
        # Build response
        result = {
            'ready_to_sell': is_ready,
            'confidence': round(confidence, 2),
            'readiness_probability': {
                'not_ready': round(readiness_prob[0] * 100, 2),
                'ready': round(readiness_prob[1] * 100, 2)
            },
            'status': stage_info['stage'],
            'stage_description': stage_info['stage'],
            'care_instructions': stage_info['care'],
            'recommended_fertilizer': stage_info['fertilizer'],
            'current_fertilizer': data.get('fertilizer', stage_info['fertilizer']),
            'estimated_days_until_ready': days_until_ready,
            'predicted_height': round(height_in, 2),  # Using actual height
            'weekly_growth_rate': {
                'height': round(weekly_growth, 2)
            },
            'age_weeks': age_weeks,
            'seed_type': seed_type,
            'soil_type': soil_type
        }
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


# Load models when the module is imported (for gunicorn)
load_models()

if __name__ == '__main__':
    import os
    
    print("=" * 70)
    print("Plant Readiness Prediction API")
    print("Loading models exported from Google Colab...")
    print("=" * 70)
    
    port = int(os.environ.get('PORT', 5000))
    is_production = os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RENDER')
    
    print("\n🚀 Starting Flask API server...")
    if is_production:
        print(f"📍 Production mode - Port: {port}")
    else:
        print(f"📍 Development mode - API at: http://127.0.0.1:{port}")
    
    print("📡 Endpoints:")
    print("   - GET  /         - API info")
    print("   - GET  /health      - Health check")
    print("   - GET  /model-info  - Model information")
    print("   - POST /predict     - Make prediction")
    
    if not is_production:
        print(f"\n⚠️  Make sure Laravel .env has: PYTHON_API_URL=http://127.0.0.1:{port}")
    
    if not model_loaded:
        print("\n⚠️  WARNING: Models not loaded, API will return errors")
        print("   Check logs above for details")
    
    print("=" * 70)
    
    # Production: bind to 0.0.0.0, Development: bind to 127.0.0.1
    host = '0.0.0.0' if is_production else '127.0.0.1'
    debug = not is_production
    
    # Always start the server, even if models didn't load
    app.run(host=host, port=port, debug=debug)

