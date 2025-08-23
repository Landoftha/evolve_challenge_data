# MLflow Configuration and Utilities
# Evolve Challenge 2025 - TOTVS

import mlflow
import os
from typing import Dict, Any, Optional
import json

class MLflowConfig:
    """MLflow configuration manager"""
    
    def __init__(self, config_file: str = "mlflow_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_mlflow()
    
    def load_config(self) -> Dict[str, Any]:
        """Load MLflow configuration from file"""
        default_config = {
            "tracking_uri": "sqlite:///mlflow.db",
            "experiment_name": "evolve_challenge_2025",
            "artifact_location": "./mlflow_artifacts",
            "registry_uri": None,
            "model_registry": {
                "churn_model_name": "churn_prediction_model",
                "clustering_model_name": "customer_clustering_model"
            },
            "experiments": {
                "churn": "churn_prediction_evolve",
                "clustering": "customer_clustering_evolve"
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    print(f"✓ Loaded MLflow config from {self.config_file}")
            except Exception as e:
                print(f"⚠️  Error loading config file: {e}, using defaults")
        else:
            # Create default config file
            self.save_config(default_config)
            print(f"✓ Created default MLflow config: {self.config_file}")
        
        return default_config
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"✗ Error saving config: {e}")
    
    def setup_mlflow(self):
        """Setup MLflow tracking and experiment"""
        # Set tracking URI
        mlflow.set_tracking_uri(self.config["tracking_uri"])
        
        # Set experiment
        mlflow.set_experiment(self.config["experiment_name"])
        
        # Set registry URI if specified
        if self.config.get("registry_uri"):
            mlflow.set_registry_uri(self.config["registry_uri"])
        
        print(f"✓ MLflow configured:")
        print(f"  Tracking URI: {self.config['tracking_uri']}")
        print(f"  Experiment: {self.config['experiment_name']}")
        if self.config.get("registry_uri"):
            print(f"  Registry URI: {self.config['registry_uri']}")
    
    def get_experiment_name(self, experiment_type: str) -> str:
        """Get experiment name for specific type"""
        return self.config["experiments"].get(experiment_type, f"{experiment_type}_experiment")
    
    def get_model_name(self, model_type: str) -> str:
        """Get model name for specific type"""
        return self.config["model_registry"].get(f"{model_type}_model_name", f"{model_type}_model")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self.config.update(updates)
        self.save_config(self.config)
        self.setup_mlflow()
        print("✓ Configuration updated and MLflow reconfigured")

class MLflowUtils:
    """Utility functions for MLflow operations"""
    
    @staticmethod
    def start_experiment_run(experiment_name: str, run_name: str = None):
        """Start a new MLflow experiment run"""
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name)
    
    @staticmethod
    def log_model_metadata(model_info: Dict[str, Any], run_name: str = None):
        """Log model metadata to MLflow"""
        with mlflow.start_run(run_name=run_name):
            # Log model parameters
            if "parameters" in model_info:
                mlflow.log_params(model_info["parameters"])
            
            # Log model metrics
            if "metrics" in model_info:
                for metric_name, metric_value in model_info["metrics"].items():
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log model tags
            if "tags" in model_info:
                mlflow.set_tags(model_info["tags"])
            
            # Log model description
            if "description" in model_info:
                mlflow.log_text(model_info["description"], "model_description.txt")
    
    @staticmethod
    def compare_experiments(experiment_names: list, metric: str = "recall"):
        """Compare experiments based on a specific metric"""
        client = mlflow.tracking.MlflowClient()
        
        comparison_data = []
        for exp_name in experiment_names:
            try:
                experiment = client.get_experiment_by_name(exp_name)
                if experiment:
                    runs = client.search_runs(experiment.experiment_id)
                    for run in runs:
                        if metric in run.data.metrics:
                            comparison_data.append({
                                'experiment': exp_name,
                                'run_id': run.info.run_id,
                                'run_name': run.info.run_name,
                                'metric': run.data.metrics[metric],
                                'timestamp': run.info.start_time
                            })
            except Exception as e:
                print(f"⚠️  Error accessing experiment {exp_name}: {e}")
        
        # Sort by metric value
        comparison_data.sort(key=lambda x: x['metric'], reverse=True)
        
        print(f"\n=== Experiment Comparison ({metric}) ===")
        for i, data in enumerate(comparison_data[:5]):  # Top 5
            print(f"{i+1}. {data['experiment']} - {data['run_name']}")
            print(f"   {metric}: {data['metric']:.4f}")
            print(f"   Run ID: {data['run_id']}")
            print("-" * 50)
        
        return comparison_data
    
    @staticmethod
    def export_model_to_database(model_name: str, version: int, db_config: Dict[str, Any]):
        """Export MLflow model metadata to database"""
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version(model_name, version)
            
            # Extract model information
            model_info = {
                'model_name': model_name,
                'version': version,
                'status': model_version.status,
                'run_id': model_version.run_id,
                'created_timestamp': model_version.creation_timestamp,
                'last_updated_timestamp': model_version.last_updated_timestamp,
                'description': model_version.description or ''
            }
            
            # Here you would insert into your database
            # This is a placeholder for the actual database insertion
            print(f"✓ Model metadata extracted for {model_name} v{version}")
            print(f"  Run ID: {model_info['run_id']}")
            print(f"  Status: {model_info['status']}")
            
            return model_info
            
        except Exception as e:
            print(f"✗ Error exporting model metadata: {e}")
            return None

def create_mlflow_project():
    """Create MLflow project structure"""
    project_structure = {
        "MLproject": {
            "name": "evolve_challenge_2025",
            "conda_env": "conda.yaml",
            "entry_points": {
                "churn_training": "mlflow_integration.py:MLflowChurnExperiment",
                "clustering_training": "mlflow_integration.py:MLflowClusteringExperiment"
            }
        },
        "conda.yaml": {
            "name": "evolve_challenge_2025",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.8",
                "pip",
                {"pip": [
                    "mlflow",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "psycopg2-binary"
                ]}
            ]
        }
    }
    
    # Create MLproject file
    mlproject_content = """name: evolve_challenge_2025
conda_env: conda.yaml
entry_points:
  churn_training:
    command: "python mlflow_integration.py --experiment churn"
  clustering_training:
    command: "python mlflow_integration.py --experiment clustering"
"""
    
    with open("MLproject", "w") as f:
        f.write(mlproject_content)
    
    # Create conda.yaml
    conda_content = """name: evolve_challenge_2025
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - pip:
    - mlflow
    - scikit-learn
    - pandas
    - numpy
    - psycopg2-binary
"""
    
    with open("conda.yaml", "w") as f:
        f.write(conda_content)
    
    print("✓ MLflow project structure created:")
    print("  - MLproject")
    print("  - conda.yaml")

def main():
    """Main function to demonstrate MLflow configuration"""
    print("=== MLflow Configuration Manager ===\n")
    
    # Initialize configuration
    config = MLflowConfig()
    
    # Show current configuration
    print("Current Configuration:")
    print(json.dumps(config.config, indent=2))
    
    # Create MLflow project structure
    print("\nCreating MLflow project structure...")
    create_mlflow_project()
    
    # Example of updating configuration
    print("\nExample: Updating tracking URI to remote server...")
    updates = {
        "tracking_uri": "sqlite:///mlflow.db",
        "registry_uri": None
    }
    config.update_config(updates)
    
    print("\n=== Configuration Complete ===")
    print("You can now:")
    print("1. Run experiments: python mlflow_integration.py")
    print("2. View UI: mlflow ui")
    print("3. List experiments: mlflow experiments list")
    print("4. Run MLflow project: mlflow run .")

if __name__ == "__main__":
    main()
