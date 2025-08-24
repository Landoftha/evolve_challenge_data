#!/usr/bin/env python3
"""
Automated Prediction Pipeline
Loads MLflow models, fetches data from PostgreSQL Silver,
applies feature engineering, and saves predictions to PostgreSQL Gold
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import mlflow
import pickle
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self, config_path: str = "prediction_config.json"):
        """Initialize prediction pipeline with configuration"""
        self.config = self.load_config(config_path)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.models = {}
        self.scalers = {}

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
        "database": {
        "silver": {
            "host": "localhost",
            "port": 5432,
            "database": "evolve_database_silver",
            "user": "postgres",
            "password": "24111999"
},
        "gold": {
            "host": "localhost",
            "port": 5432,
            "database": "evolve_database_gold",
            "user": "postgres",
            "password": "24111999"
    }
},
        "mlflow": {
            "tracking_uri": "sqlite:///mlflow.db",
            "model_names": {
                "churn": "churn_prediction_model",
                "clustering": "customer_clustering_model"
            }
},
        "prediction": {
            "batch_size": 1000,
            "retention_days": 30
}
}

    def connect_database(self, db_type: str) -> psycopg2.extensions.connection:
        """Connect to PostgreSQL database"""
        try:
            db_config = self.config["database"][db_type]
            conn = psycopg2.connect(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"]
            )
            logger.info(f"Connected to {db_type} database")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to {db_type} database: {e}")
            raise

    def load_mlflow_models(self) -> None:
        """Load models from local files"""
        try:
            # Load churn model
            churn_model_path = os.path.join("..", "models", "churn_model.pkl")
            if os.path.exists(churn_model_path):
                with open(churn_model_path, 'rb') as f:
                    self.models['churn'] = pickle.load(f)
                logger.info("Loaded churn model from local file")
            else:
                logger.warning("Churn model file not found")
        except Exception as e:
            logger.error(f"Erro: {e}")
            
            # Load clustering model
            clustering_model_path = os.path.join("..", "models", "cluster_model.pkl")
            if os.path.exists(clustering_model_path):
                with open(clustering_model_path, 'rb') as f:
                    self.models['clustering'] = pickle.load(f)
                logger.info("Loaded clustering model from local file")
            else:
                logger.warning("Clustering model file not found")

            # Try to load scalers if they exist
            scaler_paths = {
                'churn': 'mlflow_artifacts/churn_prediction_model/scaler.pkl',
                'clustering': 'mlflow_artifacts/customer_clustering_model/scaler.pkl'
            }

            for model_type, scaler_path in scaler_paths.items():
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers[model_type] = pickle.load(f)
                    logger.info(f"Loaded {model_type} scaler")
                else:
                    logger.warning(f"Scaler not found for {model_type} model")

            logger.info(f"Loaded {len(self.models)} models from local files")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

def fetch_silver_data(self) -> pd.DataFrame:
    """Fetch data from PostgreSQL Silver database"""
    try:
        query = """
        SELECT 
            c.customer_id,
            c.customer_since,
            c.customer_type,
            c.customer_status,
            cf.features,
            f.mrr_12m,
            f.contract_quantity_12m,
            f.contract_value_12m,
            f.total_contracts_12m,
            n.nps_score_12m,
            n.nps_category_12m,
            n.nps_trend_12m,
            cl.cluster_id,
            cl.cluster_label,
            cl.cluster_confidence
        FROM dim_customers c
        LEFT JOIN customer_features cf ON c.customer_id = cf.customer_id
        LEFT JOIN fact_customer_financials f ON c.customer_id = f.customer_id
        LEFT JOIN fact_customer_nps n ON c.customer_id = n.customer_id
        LEFT JOIN fact_customer_clusters cl ON c.customer_id = cl.customer_id
        WHERE c.customer_status = 'active'
        ORDER BY c.customer_id
        """

        from sqlalchemy import create_engine

        db_config = self.config["database"]["silver"]
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}",
            connect_args={'client_encoding': 'latin1'}
        )
    
        df = pd.read_sql_query(query, engine)
    
        logger.info(f"Fetched {len(df)} records from Silver database")
        return df
    
    except Exception as e:
        logger.error(f"Failed to fetch data from Silver: {e}")
        raise

def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to the dataset"""
    try:
        df_processed = df.copy()
    
        # Convert customer_since to datetime if it's not
        if df_processed['customer_since'].dtype == 'object':
            df_processed['customer_since'] = pd.to_datetime(df_processed['customer_since'])
        
        # Calculate customer age in days
        df_processed['customer_age_days'] = (datetime.now() - df_processed['customer_since']).dt.days
        
        # Extract features from JSON
        if 'features' in df_processed.columns:
            features_df = pd.json_normalize(df_processed['features'].fillna('{}'))
            df_processed = pd.concat([df_processed, features_df], axis=1)
            df_processed = df_processed.drop('features', axis=1)
    
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(0)
        
        # Convert categorical to numeric
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'customer_since':
                df_processed[col] = df_processed[col].astype('category').cat.codes
    
        # Ensure all columns are numeric for ML models
        df_processed = df_processed.select_dtypes(include=[np.number])
        
        logger.info(f"Feature engineering completed. Shape: {df_processed.shape}")
        return df_processed
    
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

def prepare_churn_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features specifically for churn prediction"""
    try:
        churn_features = [
            'customer_age_days', 'mrr_12m', 'contract_quantity_12m',
            'contract_value_12m', 'total_contracts_12m', 'nps_score_12m',
            'nps_category_12m', 'nps_trend_12m', 'cluster_id'
        ]
    
        # Select only available features
        available_features = [col for col in churn_features if col in df.columns]
        df_churn = df[available_features].copy()
        
        # Fill any remaining missing values
        df_churn = df_churn.fillna(0)
        
        logger.info(f"Churn features prepared. Shape: {df_churn.shape}")
        return df_churn
        
    except Exception as e:
        logger.error(f"Churn feature preparation failed: {e}")
        raise

def prepare_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features specifically for clustering"""
    try:
        clustering_features = [
            'customer_age_days', 'mrr_12m', 'contract_quantity_12m',
            'contract_value_12m', 'total_contracts_12m', 'nps_score_12m'
        ]
        
        # Select only available features
        available_features = [col for col in clustering_features if col in df.columns]
        df_clustering = df[available_features].copy()
        
        # Fill any remaining missing values
        df_clustering = df_clustering.fillna(0)
        
        logger.info(f"Clustering features prepared. Shape: {df_clustering.shape}")
        return df_clustering
        
    except Exception as e:
        logger.error(f"Clustering feature preparation failed: {e}")
        raise

def make_predictions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Make predictions using loaded models"""
    try:
        predictions = {}
        
        # Churn prediction
        if 'churn' in self.models:
            df_churn = self.prepare_churn_features(df)
            if len(df_churn.columns) > 0:
                # Scale features if scaler exists
                if 'churn' in self.scalers:
                    df_churn_scaled = self.scalers['churn'].transform(df_churn)
                else:
                    df_churn_scaled = df_churn
                
                churn_proba = self.models['churn'].predict_proba(df_churn_scaled)
                churn_pred = self.models['churn'].predict(df_churn_scaled)
                
                predictions['churn'] = {
                    'predictions': churn_pred,
                    'probabilities': churn_proba[:, 1] if churn_proba.shape[1] > 1 else churn_proba[:, 0]
                }
                logger.info(f"Churn predictions made for {len(churn_pred)} customers")
    
            # Clustering prediction
            if 'clustering' in self.models:
                df_clustering = self.prepare_clustering_features(df)
                if len(df_clustering.columns) > 0:
                    # Scale features if scaler exists
                    if 'clustering' in self.scalers:
                        df_clustering_scaled = self.scalers['clustering'].transform(df_clustering)
                    else:
                        df_clustering_scaled = df_clustering
                    
                    cluster_pred = self.models['clustering'].predict(df_clustering_scaled)
                    
                    predictions['clustering'] = {
                        'predictions': cluster_pred
                    }
                    logger.info(f"Clustering predictions made for {len(cluster_pred)} customers")
            
            return predictions
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def create_gold_database(self) -> None:
    """Create Gold database tables if they don't exist"""
    try:
        conn = self.connect_database("gold")
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_predictions (
                id BIGSERIAL PRIMARY KEY,
                customer_id VARCHAR(50) NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                churn_prediction BOOLEAN,
                churn_probability DECIMAL(5,4),
                cluster_prediction INTEGER,
                model_version VARCHAR(100),
                confidence_score DECIMAL(5,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create prediction history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id BIGSERIAL PRIMARY KEY,
                customer_id VARCHAR(50) NOT NULL,
                prediction_date TIMESTAMP NOT NULL,
                churn_prediction BOOLEAN,
                churn_probability DECIMAL(5,4),
                cluster_prediction INTEGER,
                model_version VARCHAR(100),
                confidence_score DECIMAL(5,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_customer_id 
            ON fact_predictions(customer_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_date 
            ON fact_predictions(prediction_date)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Gold database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create Gold database: {e}")
        raise

def save_predictions_to_gold(self, df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> None:
    """Save predictions to PostgreSQL Gold database"""
    try:
        conn = self.connect_database("gold")
        cursor = conn.cursor()
        
        # Prepare data for insertion
        data_to_insert = []
        current_time = datetime.now()
        
        for idx, customer_id in enumerate(df['customer_id']):
            row_data = {
                'customer_id': str(customer_id),
                'prediction_date': current_time,
                'churn_prediction': None,
                'churn_probability': None,
                'cluster_prediction': None,
                'model_version': 'latest',
                'confidence_score': 0.0,
                'created_at': current_time
            }
            
            # Add churn predictions
            if 'churn' in predictions:
                row_data['churn_prediction'] = bool(predictions['churn']['predictions'][idx])
                row_data['churn_probability'] = float(predictions['churn']['probabilities'][idx])
                row_data['confidence_score'] = float(predictions['churn']['probabilities'][idx])
            
            # Add clustering predictions
            if 'clustering' in predictions:
                row_data['cluster_prediction'] = int(predictions['clustering']['predictions'][idx])
            
            data_to_insert.append(row_data)
        
        # Insert predictions
        insert_query = """
            INSERT INTO fact_predictions 
            (customer_id, prediction_date, churn_prediction, churn_probability, 
            cluster_prediction, model_version, confidence_score, created_at)
            VALUES (%(customer_id)s, %(prediction_date)s, %(churn_prediction)s, 
                    %(churn_probability)s, %(cluster_prediction)s, %(model_version)s, 
                    %(confidence_score)s, %(created_at)s)
        """
        
        cursor.executemany(insert_query, data_to_insert)
        
        # Also insert into history table
        history_query = """
            INSERT INTO prediction_history 
            (customer_id, prediction_date, churn_prediction, churn_probability, 
            cluster_prediction, model_version, confidence_score, created_at)
            VALUES (%(customer_id)s, %(prediction_date)s, %(churn_prediction)s, 
                    %(churn_probability)s, %(cluster_prediction)s, %(model_version)s, 
                    %(confidence_score)s, %(created_at)s)
        """
        
        cursor.executemany(history_query, data_to_insert)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Saved {len(data_to_insert)} predictions to Gold database")
        
    except Exception as e:
        logger.error(f"Failed to save predictions to Gold: {e}")
        raise

def cleanup_old_predictions(self) -> None:
    """Clean up old predictions based on retention policy"""
    try:
        conn = self.connect_database("gold")
        cursor = conn.cursor()
        
        retention_days = self.config["prediction"]["retention_days"]
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Delete old predictions from main table
        cursor.execute("""
            DELETE FROM fact_predictions 
            WHERE prediction_date < %s
        """, (cutoff_date,))
        
        deleted_predictions = cursor.rowcount
        
        # Delete old history records
        cursor.execute("""
            DELETE FROM prediction_history 
            WHERE prediction_date < %s
        """, (cutoff_date,))
        
        deleted_history = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Cleaned up {deleted_predictions} old predictions and {deleted_history} history records")
        
    except Exception as e:
        logger.error(f"Failed to cleanup old predictions: {e}")

def run_pipeline(self) -> None:
    """Run the complete prediction pipeline"""
    try:
        logger.info("Starting prediction pipeline...")
        
        # Load MLflow models
        self.load_mlflow_models()
        
        # Create Gold database if needed
        self.create_gold_database()
        
        # Fetch data from Silver
        df = self.fetch_silver_data()
        
        if df.empty:
            logger.warning("No data found in Silver database")
            return
        
        # Apply feature engineering
        df_processed = self.apply_feature_engineering(df)
        
        # Make predictions
        predictions = self.make_predictions(df_processed)
        
        if not predictions:
            logger.warning("No predictions generated")
            return
        
        # Save predictions to Gold
        self.save_predictions_to_gold(df, predictions)
        
        # Cleanup old predictions
        self.cleanup_old_predictions()
        
        logger.info("Prediction pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}")
        raise

def main():
    """Main function to run the prediction pipeline"""
    try:
        pipeline = PredictionPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

    if __name__ == "__main__":
        main()