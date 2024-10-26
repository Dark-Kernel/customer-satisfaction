import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from decimal import Decimal
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)


class PostgresConnection:
    def __init__(self, **kwargs):
        """Initialize PostgreSQL connection parameters"""
        self.conn_params = {
            'host': kwargs.get('host', 'localhost'),
            'database': kwargs.get('database'),
            'user': kwargs.get('user'),
            'password': kwargs.get('password'),
            'port': kwargs.get('port', 5432)
        }
        self.conn = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                **self.conn_params, cursor_factory=RealDictCursor)
            logging.info("Successfully connected to PostgreSQL database")
            return self.conn
        except Exception as e:
            logging.error(f"Database connection error: {str(e)}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("Disconnected from PostgreSQL database")


class CustomerSatisfactionModel:
    def __init__(self, db_connection):
        """Initialize the model with database connection"""
        self.db = db_connection
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()

    def extract_features(self):
        """Extract features for all organizations"""
        query = """
            WITH support_metrics AS (
                SELECT 
                    organization_id,
                    CAST(AVG(satisfaction_score) AS FLOAT) as avg_support_satisfaction,
                    CAST(AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))/3600) AS FLOAT) as avg_resolution_time,
                    CAST(COUNT(CASE WHEN sla_violated = true THEN 1 END)::float / 
                        NULLIF(COUNT(*), 0) AS FLOAT) as sla_violation_rate
                FROM support_cases
                WHERE created_at >= NOW() - INTERVAL '90 days'
                GROUP BY organization_id
            ),
            feedback_metrics AS (
                SELECT 
                    organization_id,
                    CAST(AVG(CASE WHEN survey_type = 'CSAT' THEN score END) AS FLOAT) as avg_csat,
                    CAST(AVG(CASE WHEN survey_type = 'NPS' THEN score END) AS FLOAT) as avg_nps,
                    CAST(AVG(sentiment_score) AS FLOAT) as avg_sentiment
                FROM feedback
                WHERE created_at >= NOW() - INTERVAL '90 days'
                GROUP BY organization_id
            ),
            usage_metrics_agg AS (
                SELECT 
                    organization_id,
                    CAST(AVG(engagement_score) AS FLOAT) as avg_engagement,
                    CAST(AVG(adoption_rate) AS FLOAT) as avg_adoption,
                    CAST(AVG(health_score) AS FLOAT) as avg_health,
                    CAST(AVG(churn_risk_score) AS FLOAT) as avg_churn_risk
                FROM usage_metrics
                WHERE metric_date >= NOW() - INTERVAL '90 days'
                GROUP BY organization_id
            ),
            interaction_metrics AS (
                SELECT 
                    organization_id,
                    CAST(AVG(sentiment_score) AS FLOAT) as avg_interaction_sentiment,
                    CAST(COUNT(*)::float / 90 AS FLOAT) as interaction_frequency
                FROM interactions
                WHERE created_at >= NOW() - INTERVAL '90 days'
                GROUP BY organization_id
            ),
            opportunity_metrics AS (
                SELECT 
                    organization_id,
                    CAST(COUNT(CASE WHEN status = 'won' THEN 1 END)::float / 
                        NULLIF(COUNT(*), 0) AS FLOAT) as opportunity_win_rate,
                    CAST(AVG(value_amount) AS FLOAT) as avg_opportunity_value
                FROM opportunities
                WHERE created_at >= NOW() - INTERVAL '180 days'
                GROUP BY organization_id
            )
            SELECT 
                o.id as organization_id,
                o.name as organization_name,
                COALESCE(sm.avg_support_satisfaction, 0) as avg_support_satisfaction,
                COALESCE(sm.avg_resolution_time, 0) as avg_resolution_time,
                COALESCE(sm.sla_violation_rate, 0) as sla_violation_rate,
                COALESCE(fm.avg_csat, 0) as avg_csat,
                COALESCE(fm.avg_nps, 0) as avg_nps,
                COALESCE(fm.avg_sentiment, 0) as avg_sentiment,
                COALESCE(um.avg_engagement, 0) as avg_engagement,
                COALESCE(um.avg_adoption, 0) as avg_adoption,
                COALESCE(um.avg_health, 0) as avg_health,
                COALESCE(um.avg_churn_risk, 0) as avg_churn_risk,
                COALESCE(im.avg_interaction_sentiment, 0) as avg_interaction_sentiment,
                COALESCE(im.interaction_frequency, 0) as interaction_frequency,
                COALESCE(om.opportunity_win_rate, 0) as opportunity_win_rate,
                COALESCE(om.avg_opportunity_value, 0) as avg_opportunity_value
            FROM organizations o
            LEFT JOIN support_metrics sm ON o.id = sm.organization_id
            LEFT JOIN feedback_metrics fm ON o.id = fm.organization_id
            LEFT JOIN usage_metrics_agg um ON o.id = um.organization_id
            LEFT JOIN interaction_metrics im ON o.id = im.organization_id
            LEFT JOIN opportunity_metrics om ON o.id = om.organization_id
            WHERE o.organization_type = 'customer'
        """

        with self.db.conn.cursor() as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
            df = pd.DataFrame(data)

            # Convert all numeric columns to float
            numeric_columns = df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                df[col] = df[col].astype(float)

            return df

    def calculate_satisfaction_scores(self):
        """Calculate satisfaction scores for all organizations"""
        try:
            # Get features for all organizations
            features_df = self.extract_features()

            if features_df.empty:
                logging.warning("No organization data found")
                return pd.DataFrame()

            # Store organization IDs and names
            org_ids = features_df['organization_id']
            org_names = features_df['organization_name']

            # Prepare features for model
            feature_columns = [col for col in features_df.columns
                               if col not in ['organization_id', 'organization_name']]
            X = features_df[feature_columns]

            # Ensure all values are float
            X = X.astype(float)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Calculate base satisfaction score
            satisfaction_scores = pd.DataFrame({
                'organization_id': org_ids,
                'organization_name': org_names,
                'satisfaction_score': np.zeros(len(org_ids))
            })

            # Define feature weights
            weights = {
                'avg_support_satisfaction': 0.25,
                'avg_csat': 0.20,
                'avg_nps': 0.15,
                'avg_sentiment': 0.10,
                'avg_health': 0.10,
                'sla_violation_rate': -0.10,
                'avg_engagement': 0.10
            }

            # Calculate weighted score
            for i, row in X.iterrows():
                score = 0.0
                for feature, weight in weights.items():
                    if feature in row:
                        value = float(row[feature])
                        if feature == 'avg_nps':
                            value = (value + 100) / 40
                        elif feature == 'sla_violation_rate':
                            value = 5 * (1 - value)
                        elif feature == 'avg_sentiment':
                            value = (value + 1) * 2.5

                        score += value * weight

                satisfaction_scores.loc[i, 'satisfaction_score'] = min(
                    max(score, 0), 5)

            # Save scores to database
            self.save_satisfaction_scores(satisfaction_scores)

            return satisfaction_scores
        except:
            print("Some err")

    def save_satisfaction_scores(self, satisfaction_scores):
        """Save satisfaction scores to database"""
        current_time = datetime.now()

        create_table_query = """
            CREATE TABLE IF NOT EXISTS organization_satisfaction (
                organization_id INTEGER REFERENCES organizations(id),
                satisfaction_score DECIMAL(3,2),
                calculated_at TIMESTAMP,
                PRIMARY KEY (organization_id, calculated_at)
            )
        """

        insert_query = """
            INSERT INTO organization_satisfaction 
            (organization_id, satisfaction_score, calculated_at)
            VALUES (%s, %s, %s)
        """

        with self.db.conn.cursor() as cursor:
            cursor.execute(create_table_query)

            for _, row in satisfaction_scores.iterrows():
                cursor.execute(insert_query, (
                    int(row['organization_id']),
                    float(row['satisfaction_score']),
                    current_time
                ))

            self.db.conn.commit()


def main():
    # Initialize database connection
    db = PostgresConnection(
        host='10.1.133.239',
        database='data_query_db',
        user='user',
        password='password'
    )

    try:
        # Connect to database
        db.connect()

        # Initialize and run satisfaction model
        model = CustomerSatisfactionModel(db)
        satisfaction_scores = model.calculate_satisfaction_scores()

        # Print results
        print("\nCustomer Satisfaction Scores:")
        print("-----------------------------")
        for _, row in satisfaction_scores.iterrows():
            print(
                f"{row['organization_name']}: {row['satisfaction_score']:.2f}/5.00")

        # Print summary statistics
        if not satisfaction_scores.empty:
            print("\nSummary Statistics:")
            print("------------------")
            print(
                f"Average Satisfaction: {satisfaction_scores['satisfaction_score'].mean():.2f}")
            print(
                f"Highest Satisfaction: {satisfaction_scores['satisfaction_score'].max():.2f}")
            print(
                f"Lowest Satisfaction: {satisfaction_scores['satisfaction_score'].min():.2f}")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
