# MLflow Integration for Evolve Challenge 2025
# TOTVS Challenge - Churn Prediction & Clustering

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLflowChurnExperiment:
    def __init__(self, experiment_name="churn_prediction_evolve"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Configure MLflow tracking"""
        mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Local SQLite
        mlflow.set_experiment(self.experiment_name)
        
        # Alternative: Remote tracking server
        # mlflow.set_tracking_uri("http://localhost:5000")
        
    def load_data(self, csv_file='dataset_unificado.csv'):
        """Load and preprocess data"""
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file, sep=';')
        
        # Preprocessing
        df['CLIENTE_DESDE'] = pd.to_datetime(df['CLIENTE_DESDE'], errors='coerce')
        df['data_ultima_resposta_nps'] = pd.to_datetime(df['data_ultima_resposta_nps'], errors='coerce')
        
        df['ano_cliente'] = df['CLIENTE_DESDE'].dt.year
        df['mes_cliente'] = df['CLIENTE_DESDE'].dt.month
        df['dias_cliente'] = (pd.Timestamp.now() - df['CLIENTE_DESDE']).dt.days
        df['dias_ultima_resposta'] = (pd.Timestamp.now() - df['data_ultima_resposta_nps']).dt.days
        
        # Convert numeric columns - handle string values first
        numeric_columns = ['MRR_12M', 'QTD_CONTRATACOES_12M', 'VLR_CONTRATACOES_12M', 'resposta_NPS']
        for col in numeric_columns:
            if col in df.columns:
                # Replace common string representations
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # NPS scores
        nps_columns = [col for col in df.columns if 'Nota_' in col]
        for col in nps_columns:
            if col in df.columns:
                # Replace common string representations
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['media_notas_nps'] = df[nps_columns].mean(axis=1)
        df['std_notas_nps'] = df[nps_columns].std(axis=1)
        df['min_nota_nps'] = df[nps_columns].min(axis=1)
        df['max_notas_nps'] = df[nps_columns].max(axis=1)
        
        df['tem_nps'] = df['resposta_NPS'].notna().astype(int)
        df['tem_contratacoes'] = df['QTD_CONTRATACOES_12M'].notna().astype(int)
        
        print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def create_churn_target(self, df):
        """Create churn target variable"""
        df['churn'] = 0
        
        # More balanced churn definition - only the most critical indicators
        churn_conditions = (
            (df['MRR_12M'].isna()) |
            (df['MRR_12M'] == 0) |
            (df['QTD_CONTRATACOES_12M'].isna()) |
            (df['QTD_CONTRATACOES_12M'] == 0)
        )
        df.loc[churn_conditions, 'churn'] = 1
        
        # Print class distribution
        churn_rate = df['churn'].mean()
        print(f"✓ Churn rate: {churn_rate:.2%}")
        print(f"✓ Total samples: {len(df)}")
        print(f"✓ Churn samples: {df['churn'].sum()}")
        print(f"✓ Non-churn samples: {(df['churn'] == 0).sum()}")
        
        # If still too imbalanced, adjust further
        if churn_rate > 0.8:
            print("⚠️  Churn rate too high, adjusting criteria...")
            # Use only the most severe cases - missing MRR
            df['churn'] = 0
            severe_churn = (df['MRR_12M'].isna())
            df.loc[severe_churn, 'churn'] = 1
            churn_rate = df['churn'].mean()
            print(f"✓ Adjusted churn rate: {churn_rate:.2%}")
            
            # If still too high, use a threshold approach
            if churn_rate > 0.8:
                print("⚠️  Still too high, using threshold approach...")
                df['churn'] = 0
                # Use bottom 20% of MRR values as churn
                mrr_threshold = df['MRR_12M'].quantile(0.2)
                df.loc[df['MRR_12M'] <= mrr_threshold, 'churn'] = 1
                churn_rate = df['churn'].mean()
                print(f"✓ Final churn rate: {churn_rate:.2%}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        feature_columns = [
            'ano_cliente', 'mes_cliente', 'dias_cliente', 'dias_ultima_resposta',
            'MRR_12M', 'QTD_CONTRATACOES_12M', 'VLR_CONTRATACOES_12M',
            'resposta_NPS', 'media_notas_nps', 'std_notas_nps', 'min_nota_nps', 'max_notas_nps',
            'tem_nps', 'tem_contratacoes'
        ]
        
        # Add NPS individual scores
        nps_columns = [col for col in df.columns if 'Nota_' in col]
        feature_columns.extend(nps_columns)
        
        # Select features and handle missing values
        X = df[feature_columns].copy()
        X = X.fillna(X.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, X.columns.tolist(), scaler
    
    def run_churn_experiment(self, run_name="churn_rf_experiment"):
        """Run churn prediction experiment with MLflow tracking"""
        
        with mlflow.start_run(run_name=run_name):
            print(f"=== Starting MLflow Experiment: {run_name} ===")
            
            # Load and prepare data
            df = self.load_data()
            df = self.create_churn_target(df)
            X_scaled, feature_names, scaler = self.prepare_features(df)
            y = df['churn']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Log parameters
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("churn_rate", y.mean())
            
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            # Grid search with better handling of imbalanced classes
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='recall',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)
            
            # Handle case where model only predicts one class
            if y_pred_proba.shape[1] == 1:
                # Only one class predicted, use the probability of that class
                y_pred_proba_positive = y_pred_proba[:, 0]
                print("⚠️  Warning: Model only predicted one class")
            else:
                # Two classes predicted, use probability of positive class
                y_pred_proba_positive = y_pred_proba[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate AUC only if we have multiple classes
            if y_pred_proba.shape[1] > 1:
                auc = roc_auc_score(y_test, y_pred_proba_positive)
            else:
                auc = 0.5  # Default value when only one class
                print("⚠️  AUC set to 0.5 (only one class predicted)")
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            
            # Log feature importance
            feature_importance = dict(zip(feature_names, best_model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            # Log model
            mlflow.sklearn.log_model(best_model, "churn_model")
            
            # Log scaler
            mlflow.sklearn.log_model(scaler, "scaler")
            
            # Log feature names
            mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")
            
            # Save model locally
            model_data = {
                'model': best_model,
                'scaler': scaler,
                'feature_names': feature_names,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': auc
                }
            }
            
            with open('churn_model_mlflow.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            mlflow.log_artifact('churn_model_mlflow.pkl')
            
            print(f"✓ Experiment completed successfully!")
            print(f"✓ Best Recall: {recall:.4f}")
            print(f"✓ Best Parameters: {grid_search.best_params_}")
            
            return best_model, scaler, feature_names

class MLflowClusteringExperiment:
    def __init__(self, experiment_name="customer_clustering_evolve"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Configure MLflow tracking"""
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.experiment_name)
    
    def prepare_clustering_features(self, df):
        """Prepare features for clustering"""
        # Use features that are already processed in the main data loading
        clustering_features = [
            'dias_cliente', 'MRR_12M', 'QTD_CONTRATACOES_12M', 'VLR_CONTRATACOES_12M',
            'media_notas_nps', 'tem_nps', 'tem_contratacoes'
        ]
        
        # Check which features are available
        available_features = [col for col in clustering_features if col in df.columns]
        print(f"✓ Available clustering features: {available_features}")
        
        if len(available_features) < 3:
            # Fallback to basic features
            available_features = ['dias_cliente', 'MRR_12M', 'media_notas_nps']
            print(f"⚠️  Using fallback features: {available_features}")
        
        X = df[available_features].copy()
        
        # Better handling of missing values
        for col in available_features:
            if col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    # For numeric columns, fill with median
                    X[col] = X[col].fillna(X[col].median())
                else:
                    # For categorical columns, fill with mode
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
        
        # Remove any remaining NaN values
        X = X.dropna()
        
        if len(X) == 0:
            raise ValueError("No valid data after removing NaN values")
        
        print(f"✓ Clustering features prepared: {len(X)} samples, {len(available_features)} features")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, available_features, scaler
    
    def run_clustering_experiment(self, run_name="clustering_kmeans_experiment"):
        """Run customer clustering experiment with MLflow tracking"""
        
        with mlflow.start_run(run_name=run_name):
            print(f"=== Starting MLflow Clustering Experiment: {run_name} ===")
            
            # Load data
            df = self.load_data()
            X_scaled, feature_names, scaler = self.prepare_clustering_features(df)
            
            # Log parameters
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("n_samples", len(df))
            
            # Test different numbers of clusters
            n_clusters_range = range(3, 11)
            silhouette_scores = []
            inertias = []
            
            for n_clusters in n_clusters_range:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Calculate metrics
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                inertia = kmeans.inertia_
                
                silhouette_scores.append(silhouette_avg)
                inertias.append(inertia)
                
                # Log metrics for each cluster number
                mlflow.log_metric(f"silhouette_score_{n_clusters}", silhouette_avg)
                mlflow.log_metric(f"inertia_{n_clusters}", inertia)
            
            # Find best number of clusters
            best_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
            
            # Train final model with best number of clusters
            final_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            final_kmeans.fit(X_scaled)
            
            # Log best parameters
            mlflow.log_param("best_n_clusters", best_n_clusters)
            mlflow.log_metric("best_silhouette_score", max(silhouette_scores))
            mlflow.log_metric("best_inertia", inertias[np.argmax(silhouette_scores)])
            
            # Log model
            mlflow.sklearn.log_model(final_kmeans, "clustering_model")
            mlflow.sklearn.log_model(scaler, "clustering_scaler")
            
            # Log feature names
            mlflow.log_dict({"feature_names": feature_names}, "clustering_feature_names.json")
            
            # Save model locally
            clustering_data = {
                'model': final_kmeans,
                'scaler': scaler,
                'feature_names': feature_names,
                'n_clusters': best_n_clusters,
                'silhouette_score': max(silhouette_scores),
                'inertia': inertias[np.argmax(silhouette_scores)]
            }
            
            with open('clustering_model_mlflow.pkl', 'wb') as f:
                pickle.dump(clustering_data, f)
            
            mlflow.log_artifact('clustering_model_mlflow.pkl')
            
            print(f"✓ Clustering experiment completed successfully!")
            print(f"✓ Best number of clusters: {best_n_clusters}")
            print(f"✓ Best Silhouette Score: {max(silhouette_scores):.4f}")
            
            return final_kmeans, scaler, feature_names
    
    def load_data(self, csv_file='dataset_unificado.csv'):
        """Load data (same as churn experiment)"""
        df = pd.read_csv(csv_file, sep=';')
        
        # Preprocessing (same as churn experiment)
        df['CLIENTE_DESDE'] = pd.to_datetime(df['CLIENTE_DESDE'], errors='coerce')
        df['data_ultima_resposta_nps'] = pd.to_datetime(df['data_ultima_resposta_nps'], errors='coerce')
        
        df['ano_cliente'] = df['CLIENTE_DESDE'].dt.year
        df['mes_cliente'] = df['CLIENTE_DESDE'].dt.month
        df['dias_cliente'] = (pd.Timestamp.now() - df['CLIENTE_DESDE']).dt.days
        df['dias_ultima_resposta'] = (pd.Timestamp.now() - df['data_ultima_resposta_nps']).dt.days
        
        # Convert numeric columns - handle string values first
        numeric_columns = ['MRR_12M', 'QTD_CONTRATACOES_12M', 'VLR_CONTRATACOES_12M', 'resposta_NPS']
        for col in numeric_columns:
            if col in df.columns:
                # Replace common string representations
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # NPS scores
        nps_columns = [col for col in df.columns if 'Nota_' in col]
        for col in nps_columns:
            if col in df.columns:
                # Replace common string representations
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['media_notas_nps'] = df[nps_columns].mean(axis=1)
        df['std_notas_nps'] = df[nps_columns].std(axis=1)
        df['min_nota_nps'] = df[nps_columns].min(axis=1)
        df['max_notas_nps'] = df[nps_columns].max(axis=1)
        
        df['tem_nps'] = df['resposta_NPS'].notna().astype(int)
        df['tem_contratacoes'] = df['QTD_CONTRATACOES_12M'].notna().astype(int)
        
        return df

class MLflowModelRegistry:
    def __init__(self):
        self.client = mlflow.tracking.MlflowClient()
    
    def register_churn_model(self, run_id, model_name="churn_prediction_model"):
        """Register churn model in MLflow Model Registry"""
        try:
            model_uri = f"runs:/{run_id}/churn_model"
            model_version = mlflow.register_model(model_uri, model_name)
            print(f"✓ Churn model registered: {model_name} v{model_version.version}")
            return model_version
        except Exception as e:
            print(f"✗ Error registering churn model: {e}")
            return None
    
    def register_clustering_model(self, run_id, model_name="customer_clustering_model"):
        """Register clustering model in MLflow Model Registry"""
        try:
            model_uri = f"runs:/{run_id}/clustering_model"
            model_version = mlflow.register_model(model_uri, model_name)
            print(f"✓ Clustering model registered: {model_name} v{model_version.version}")
            return model_version
        except Exception as e:
            print(f"✗ Error registering clustering model: {e}")
            return None
    
    def list_models(self):
        """List all registered models"""
        try:
            models = self.client.search_registered_models()
            print("\n=== Registered Models ===")
            for model in models:
                print(f"Model: {model.name}")
                print(f"Latest Version: {model.latest_versions[0].version if model.latest_versions else 'None'}")
                print(f"Status: {model.latest_versions[0].status if model.latest_versions else 'None'}")
                print("-" * 40)
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
            print("✓ Models were trained and logged successfully")
    
    def load_model(self, model_name, version=None):
        """Load a registered model"""
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✓ Model loaded: {model_uri}")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None

def main():
    """Main function to run MLflow experiments"""
    print("=== MLflow Integration for Evolve Challenge 2025 ===\n")
    
    # Run Churn Prediction Experiment
    print("1. Running Churn Prediction Experiment...")
    churn_exp = MLflowChurnExperiment()
    churn_model, churn_scaler, churn_features = churn_exp.run_churn_experiment()
    
    # Run Clustering Experiment
    print("\n2. Running Customer Clustering Experiment...")
    clustering_exp = MLflowClusteringExperiment()
    clustering_model, clustering_scaler, clustering_features = clustering_exp.run_clustering_experiment()
    
    # Register models
    print("\n3. Registering Models in MLflow Registry...")
    registry = MLflowModelRegistry()
    
    # Get run IDs from active runs
    churn_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    clustering_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    
    if churn_run_id:
        registry.register_churn_model(churn_run_id)
    
    if clustering_run_id:
        registry.register_clustering_model(clustering_run_id)
    
    # List registered models
    registry.list_models()
    
    print("\n=== MLflow Integration Complete! ===")
    print("Check the MLflow UI: mlflow ui")
    print("Or view experiments: mlflow experiments list")

if __name__ == "__main__":
    main()
