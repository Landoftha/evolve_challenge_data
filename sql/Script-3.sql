
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Customer dimension table
CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_since DATE NOT NULL,
    customer_since_year INTEGER,
    customer_since_month INTEGER,
    customer_age_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NPS dimension table
CREATE TABLE IF NOT EXISTS dim_nps_categories (
    nps_category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    category_description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Clustering dimension table
CREATE TABLE IF NOT EXISTS dim_clustering_segments (
    segment_id SERIAL PRIMARY KEY,
    segment_name VARCHAR(100) NOT NULL,
    segment_description TEXT,
    segment_type VARCHAR(50) NOT NULL, -- 'churn_risk', 'value_tier', 'behavior_pattern'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- FACT TABLES
-- =====================================================

-- Customer financial metrics fact table
CREATE TABLE IF NOT EXISTS fact_customer_financials (
    fact_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    mrr_12m DECIMAL(15,2),
    contract_quantity_12m INTEGER,
    contract_value_12m DECIMAL(15,2),
    measurement_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id)
);

-- Customer NPS scores fact table
CREATE TABLE IF NOT EXISTS fact_customer_nps (
    fact_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    nps_response_score DECIMAL(3,1),
    nps_support_agility_score DECIMAL(3,1),
    nps_support_service_score DECIMAL(3,1),
    nps_commercial_score DECIMAL(3,1),
    nps_costs_score DECIMAL(3,1),
    nps_admin_fin_service_score DECIMAL(3,1),
    nps_software_score DECIMAL(3,1),
    nps_software_update_score DECIMAL(3,1),
    nps_response_date TIMESTAMP,
    days_since_last_nps_response INTEGER,
    nps_mean_score DECIMAL(3,2),
    nps_std_score DECIMAL(3,2),
    nps_min_score DECIMAL(3,1),
    nps_max_score DECIMAL(3,1),
    has_nps_response BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id)
);

-- Customer churn predictions fact table
CREATE TABLE IF NOT EXISTS fact_churn_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    churn_probability DECIMAL(5,4) NOT NULL,
    churn_prediction BOOLEAN NOT NULL,
    actual_churn_status BOOLEAN,
    prediction_date DATE NOT NULL,
    model_version VARCHAR(50),
    model_performance_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id)
);

-- Customer clustering assignments fact table
CREATE TABLE IF NOT EXISTS fact_customer_clusters (
    cluster_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    segment_id INTEGER NOT NULL,
    cluster_probability DECIMAL(5,4),
    cluster_confidence DECIMAL(5,4),
    assignment_date DATE NOT NULL,
    model_version VARCHAR(50),
    cluster_features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id),
    FOREIGN KEY (segment_id) REFERENCES dim_clustering_segments(segment_id)
);

-- =====================================================
-- FEATURE STORE TABLES
-- =====================================================

-- Customer features table for model training
CREATE TABLE IF NOT EXISTS customer_features (
    feature_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    feature_vector JSONB NOT NULL,
    feature_names JSONB NOT NULL,
    feature_extraction_date DATE NOT NULL,
    feature_type VARCHAR(50) NOT NULL, -- 'churn', 'clustering', 'general'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id)
);

-- Model metadata table
CREATE TABLE IF NOT EXISTS model_metadata (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- 'churn_prediction', 'clustering', 'classification'
    training_date TIMESTAMP NOT NULL,
    model_file_path VARCHAR(500),
    model_performance JSONB,
    hyperparameters JSONB,
    feature_importance JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- AUDIT AND MONITORING TABLES
-- =====================================================

-- Model prediction logs
CREATE TABLE IF NOT EXISTS prediction_logs (
    log_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    model_type VARCHAR(50) NOT NULL, -- 'churn', 'clustering'
    prediction_request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_response_time_ms INTEGER,
    model_version_used VARCHAR(50),
    input_features_hash VARCHAR(64),
    prediction_result JSONB,
    error_message TEXT
);

-- Data quality metrics
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- STAGING TABLES
-- =====================================================

-- Staging table for raw data ingestion
CREATE TABLE IF NOT EXISTS staging_raw_data (
    staging_id BIGSERIAL PRIMARY KEY,
    raw_data JSONB NOT NULL,
    source_file VARCHAR(255),
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Customer dimension indexes
CREATE INDEX IF NOT EXISTS idx_customers_since ON dim_customers(customer_since);
CREATE INDEX IF NOT EXISTS idx_customers_age ON dim_customers(customer_age_days);

-- Financial fact indexes
CREATE INDEX IF NOT EXISTS idx_financials_customer ON fact_customer_financials(customer_id);
CREATE INDEX IF NOT EXISTS idx_financials_date ON fact_customer_financials(measurement_date);
CREATE INDEX IF NOT EXISTS idx_financials_mrr ON fact_customer_financials(mrr_12m);
CREATE INDEX IF NOT EXISTS idx_financials_contracts ON fact_customer_financials(contract_quantity_12m);
CREATE INDEX IF NOT EXISTS idx_financials_customer_date ON fact_customer_financials(customer_id, measurement_date);

-- NPS fact indexes
CREATE INDEX IF NOT EXISTS idx_nps_customer ON fact_customer_nps(customer_id);
CREATE INDEX IF NOT EXISTS idx_nps_date ON fact_customer_nps(nps_response_date);
CREATE INDEX IF NOT EXISTS idx_nps_scores ON fact_customer_nps(nps_response_score);
CREATE INDEX IF NOT EXISTS idx_nps_customer_date ON fact_customer_nps(customer_id, nps_response_date);

-- Churn prediction indexes
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON fact_churn_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON fact_churn_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_probability ON fact_churn_predictions(churn_probability);
CREATE INDEX IF NOT EXISTS idx_predictions_customer_date ON fact_churn_predictions(customer_id, prediction_date);

-- Clustering indexes
CREATE INDEX IF NOT EXISTS idx_clusters_customer ON fact_customer_clusters(customer_id);
CREATE INDEX IF NOT EXISTS idx_clusters_segment ON fact_customer_clusters(segment_id);
CREATE INDEX IF NOT EXISTS idx_clusters_date ON fact_customer_clusters(assignment_date);
CREATE INDEX IF NOT EXISTS idx_clusters_customer_date ON fact_customer_clusters(customer_id, assignment_date);

-- Feature store indexes
CREATE INDEX IF NOT EXISTS idx_features_customer ON customer_features(customer_id);
CREATE INDEX IF NOT EXISTS idx_features_date ON customer_features(feature_extraction_date);
CREATE INDEX IF NOT EXISTS idx_features_type ON customer_features(feature_type);
CREATE INDEX IF NOT EXISTS idx_features_customer_date ON customer_features(customer_id, feature_extraction_date);

-- Model metadata indexes
CREATE INDEX IF NOT EXISTS idx_model_version ON model_metadata(model_version);
CREATE INDEX IF NOT EXISTS idx_model_type ON model_metadata(model_type);
CREATE INDEX IF NOT EXISTS idx_model_training_date ON model_metadata(training_date);

-- Staging indexes
CREATE INDEX IF NOT EXISTS idx_staging_status ON staging_raw_data(processing_status);
CREATE INDEX IF NOT EXISTS idx_staging_timestamp ON staging_raw_data(ingestion_timestamp);

-- =====================================================
-- VIEWS FOR ANALYSIS
-- =====================================================

-- Customer churn risk view
CREATE OR REPLACE VIEW v_customer_churn_risk AS
SELECT 
    c.customer_id,
    c.customer_since,
    c.customer_age_days,
    f.mrr_12m,
    f.contract_quantity_12m,
    n.nps_mean_score,
    n.nps_response_date,
    cp.churn_probability,
    cp.churn_prediction,
    CASE 
        WHEN cp.churn_probability >= 0.8 THEN 'Very High Risk'
        WHEN cp.churn_probability >= 0.6 THEN 'High Risk'
        WHEN cp.churn_probability >= 0.4 THEN 'Medium Risk'
        WHEN cp.churn_probability >= 0.2 THEN 'Low Risk'
        ELSE 'Very Low Risk'
    END AS risk_category
FROM dim_customers c
LEFT JOIN fact_customer_financials f ON c.customer_id = f.customer_id
LEFT JOIN fact_customer_nps n ON c.customer_id = n.customer_id
LEFT JOIN fact_churn_predictions cp ON c.customer_id = cp.customer_id
WHERE cp.prediction_date = (SELECT MAX(prediction_date) FROM fact_churn_predictions);

-- Customer clustering view
CREATE OR REPLACE VIEW v_customer_clusters AS
SELECT 
    c.customer_id,
    c.customer_since,
    c.customer_age_days,
    f.mrr_12m,
    n.nps_mean_score,
    cs.segment_name,
    cs.segment_type,
    cc.cluster_probability,
    cc.cluster_confidence,
    cc.assignment_date
FROM dim_customers c
LEFT JOIN fact_customer_financials f ON c.customer_id = f.customer_id
LEFT JOIN fact_customer_nps n ON c.customer_id = n.customer_id
LEFT JOIN fact_customer_clusters cc ON c.customer_id = cc.customer_id
LEFT JOIN dim_clustering_segments cs ON cc.segment_id = cs.segment_id
WHERE cc.assignment_date = (SELECT MAX(assignment_date) FROM fact_customer_clusters);

-- Model performance monitoring view
CREATE OR REPLACE VIEW v_model_performance AS
SELECT 
    model_type,
    model_version,
    DATE(training_date) as training_date,
    COUNT(*) as total_models,
    AVG(CAST(model_performance->>'accuracy' AS DECIMAL)) as avg_accuracy,
    AVG(CAST(model_performance->>'precision' AS DECIMAL)) as avg_precision,
    AVG(CAST(model_performance->>'recall' AS DECIMAL)) as avg_recall
FROM model_metadata
WHERE is_active = TRUE
GROUP BY model_type, model_version, DATE(training_date)
ORDER BY model_type, training_date DESC;

-- =====================================================
-- FUNCTIONS AND PROCEDURES
-- =====================================================

-- Function to update customer age
CREATE OR REPLACE FUNCTION update_customer_age()
RETURNS TRIGGER AS $$
BEGIN
    NEW.customer_age_days = EXTRACT(EPOCH FROM (CURRENT_DATE - NEW.customer_since)) / 86400;
    NEW.customer_since_year = EXTRACT(YEAR FROM NEW.customer_since);
    NEW.customer_since_month = EXTRACT(MONTH FROM NEW.customer_since);
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to insert customer data
CREATE OR REPLACE FUNCTION insert_customer_data(
    p_customer_id VARCHAR(50),
    p_customer_since DATE,
    p_mrr_12m DECIMAL(15,2),
    p_contract_quantity INTEGER,
    p_contract_value DECIMAL(15,2)
)
RETURNS TEXT AS $$
DECLARE
    v_customer_age_days INTEGER;
    v_customer_since_year INTEGER;
    v_customer_since_month INTEGER;
BEGIN
    v_customer_age_days = EXTRACT(EPOCH FROM (CURRENT_DATE - p_customer_since)) / 86400;
    v_customer_since_year = EXTRACT(YEAR FROM p_customer_since);
    v_customer_since_month = EXTRACT(MONTH FROM p_customer_since);
    
    -- Insert or update customer dimension
    INSERT INTO dim_customers (customer_id, customer_since, customer_since_year, customer_since_month, customer_age_days)
    VALUES (p_customer_id, p_customer_since, v_customer_since_year, v_customer_since_month, v_customer_age_days)
    ON CONFLICT (customer_id) DO UPDATE SET
        customer_since = EXCLUDED.customer_since,
        customer_since_year = EXCLUDED.customer_since_year,
        customer_since_month = EXCLUDED.customer_since_month,
        customer_age_days = EXCLUDED.customer_age_days,
        updated_at = CURRENT_TIMESTAMP;
    
    -- Insert financial fact
    INSERT INTO fact_customer_financials (customer_id, mrr_12m, contract_quantity_12m, contract_value_12m, measurement_date)
    VALUES (p_customer_id, p_mrr_12m, p_contract_quantity, p_contract_value, CURRENT_DATE);
    
    RETURN 'Customer data inserted successfully';
END;
$$ LANGUAGE plpgsql;

-- Function to calculate churn metrics
CREATE OR REPLACE FUNCTION calculate_churn_metrics(p_date DATE)
RETURNS TABLE(
    date DATE,
    total_customers BIGINT,
    churned_customers BIGINT,
    churn_rate DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p_date as date,
        COUNT(*)::BIGINT as total_customers,
        SUM(CASE WHEN cp.churn_prediction = TRUE THEN 1 ELSE 0 END)::BIGINT as churned_customers,
        ROUND(
            SUM(CASE WHEN cp.churn_prediction = TRUE THEN 1 ELSE 0 END)::DECIMAL / 
            COUNT(*)::DECIMAL, 4
        ) as churn_rate
    FROM dim_customers c
    LEFT JOIN fact_churn_predictions cp ON c.customer_id = cp.customer_id
    WHERE c.customer_since <= p_date;
END;
$$ LANGUAGE plpgsql;

-- Function to get customer insights
CREATE OR REPLACE FUNCTION get_customer_insights(p_customer_id VARCHAR(50))
RETURNS TABLE(
    customer_id VARCHAR(50),
    customer_age_days INTEGER,
    mrr_12m DECIMAL(15,2),
    nps_mean_score DECIMAL(3,2),
    churn_probability DECIMAL(5,4),
    segment_name VARCHAR(100),
    segment_type VARCHAR(50)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.customer_id,
        c.customer_age_days,
        f.mrr_12m,
        n.nps_mean_score,
        cp.churn_probability,
        cs.segment_name,
        cs.segment_type
    FROM dim_customers c
    LEFT JOIN fact_customer_financials f ON c.customer_id = f.customer_id
    LEFT JOIN fact_customer_nps n ON c.customer_id = n.customer_id
    LEFT JOIN fact_churn_predictions cp ON c.customer_id = cp.customer_id
    LEFT JOIN fact_customer_clusters cc ON c.customer_id = cc.customer_id
    LEFT JOIN dim_clustering_segments cs ON cc.segment_id = cs.segment_id
    WHERE c.customer_id = p_customer_id;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- TRIGGERS
-- =====================================================

-- Trigger to update customer age when customer_since changes
CREATE TRIGGER tr_update_customer_age
    BEFORE INSERT OR UPDATE ON dim_customers
    FOR EACH ROW
    EXECUTE FUNCTION update_customer_age();

-- =====================================================
-- INITIAL DATA INSERTS
-- =====================================================

-- Insert NPS categories
INSERT INTO dim_nps_categories (category_name, category_description) VALUES
('Support Technical Agility', 'NPS score for technical support agility'),
('Support Technical Service', 'NPS score for technical support service quality'),
('Commercial', 'NPS score for commercial relationship'),
('Costs', 'NPS score for cost satisfaction'),
('Administrative Financial Service', 'NPS score for administrative and financial services'),
('Software', 'NPS score for software quality'),
('Software Updates', 'NPS score for software update process')
ON CONFLICT DO NOTHING;

-- Insert clustering segments
INSERT INTO dim_clustering_segments (segment_name, segment_description, segment_type) VALUES
('High Value Loyal', 'High MRR customers with long tenure and high NPS', 'value_tier'),
('Medium Value Growing', 'Medium MRR customers showing growth potential', 'value_tier'),
('Low Value At Risk', 'Low MRR customers at risk of churning', 'value_tier'),
('New Customer', 'Recently acquired customers', 'behavior_pattern'),
('Established Customer', 'Long-term stable customers', 'behavior_pattern'),
('Churn Risk High', 'Customers with high churn probability', 'churn_risk'),
('Churn Risk Medium', 'Customers with medium churn probability', 'churn_risk'),
('Churn Risk Low', 'Customers with low churn probability', 'churn_risk')
ON CONFLICT DO NOTHING;

-- Insert sample model metadata
INSERT INTO model_metadata (model_name, model_version, model_type, training_date, model_performance, hyperparameters) VALUES
('Churn Prediction RandomForest', 'v1.0', 'churn_prediction', CURRENT_TIMESTAMP, 
 '{"accuracy": 0.6387, "precision": 0.7282, "recall": 0.7368, "auc": 0.6698}'::jsonb,
 '{"n_estimators": 300, "max_depth": 20, "min_samples_split": 2, "min_samples_leaf": 1, "class_weight": "balanced_subsample"}'::jsonb),
('Customer Clustering KMeans', 'v1.0', 'clustering', CURRENT_TIMESTAMP,
 '{"silhouette_score": 0.45, "inertia": 1250.5, "n_clusters": 8}'::jsonb,
 '{"n_clusters": 8, "random_state": 42, "n_init": 10}'::jsonb)
ON CONFLICT DO NOTHING;

-- =====================================================
-- COMMENTS AND DOCUMENTATION
-- =====================================================

-- Add table comments for documentation
COMMENT ON TABLE dim_customers IS 'Customer dimension table with demographic and temporal information';
COMMENT ON TABLE dim_clustering_segments IS 'Clustering segments and categories for customer segmentation';
COMMENT ON TABLE fact_customer_financials IS 'Customer financial metrics fact table';
COMMENT ON TABLE fact_customer_nps IS 'Customer NPS scores and feedback fact table';
COMMENT ON TABLE fact_churn_predictions IS 'Churn prediction results and actual outcomes';
COMMENT ON TABLE fact_customer_clusters IS 'Customer clustering assignments and segment information';
COMMENT ON TABLE customer_features IS 'Feature vectors for machine learning model training';
COMMENT ON TABLE model_metadata IS 'Machine learning model metadata and performance tracking';

-- =====================================================
-- FINAL STATUS
-- =====================================================

SELECT 'Silver database tables created successfully in evolve_database_silver!' as status;
SELECT COUNT(*) as total_tables FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
