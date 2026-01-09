"""
Data Pipeline Module
Handles CSV loading, cleaning, preprocessing, and feature engineering.
"""
import os
import io
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.parse
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logger = logging.getLogger(__name__)

class DataPipeline:
    """Comprehensive data pipeline for burnout prediction"""
    
    def __init__(self):
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_columns = []
    
    @staticmethod
    def create_requests_session():
        """Create a robust requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Burnout-Training-Service/1.0',
            'Accept': 'text/csv, application/csv, */*'
        })
        
        return session
    
    @staticmethod
    def validate_csv_source(csv_source, default_path="data/burnout_data.csv"):
        """
        Validate and normalize the CSV source input.
        """
        data_path = Path(default_path)
        
        if csv_source is None:
            return {
                'type': 'default',
                'path': str(data_path),
                'valid': data_path.exists()
            }
        
        csv_source = str(csv_source)
        
        if csv_source.startswith(('http://', 'https://')):
            return {
                'type': 'url',
                'path': csv_source,
                'valid': True
            }
        elif csv_source.startswith('gs://') or 'firebasestorage.googleapis.com' in csv_source:
            return {
                'type': 'firebase',
                'path': csv_source,
                'valid': True
            }
        else:
            file_path = Path(csv_source)
            return {
                'type': 'local',
                'path': str(file_path),
                'valid': file_path.exists()
            }
    
    def load_csv(self, source):
        """
        Load CSV from URL, Firebase, or local file.
        """
        source_info = self.validate_csv_source(source)
        
        logger.info(f"[INPUT] Loading data from: {source_info['path']} (type: {source_info['type']})")
        
        try:
            if source_info['type'] == 'url':
                session = self.create_requests_session()
                response = session.get(source_info['path'], timeout=30)
                response.raise_for_status()
                csv_content = response.text
                
            elif source_info['type'] == 'firebase':
                csv_content = self.download_from_firebase_storage(source_info['path'])
                
            elif source_info['type'] in ['local', 'default']:
                if not source_info['valid']:
                    raise FileNotFoundError(f"File not found: {source_info['path']}")
                with open(source_info['path'], 'r', encoding='utf-8') as f:
                    csv_content = f.read()
            
            else:
                raise ValueError(f"Unsupported source type: {source_info['type']}")
            
            # Parse CSV content
            df = pd.read_csv(io.StringIO(csv_content))
            df.columns = [str(col) for col in df.columns]
            
            logger.info(f"[SUCCESS] Successfully loaded CSV: {len(df)} rows × {df.shape[1]} columns")
            return df, source_info
            
        except Exception as e:
            logger.error(f"[ERROR] CSV loading failed: {e}")
            raise
    
    def download_from_firebase_storage(self, firebase_url):
        """Download CSV from Firebase Storage URL."""
        try:
            logger.info(f"[FIRE] Downloading from Firebase Storage: {firebase_url}")
            
            session = self.create_requests_session()
            
            if 'alt=media' not in firebase_url:
                parsed_url = urllib.parse.urlparse(firebase_url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                query_params['alt'] = ['media']
                new_query = urllib.parse.urlencode(query_params, doseq=True)
                firebase_url = urllib.parse.urlunparse((
                    parsed_url.scheme,
                    parsed_url.netloc,
                    parsed_url.path,
                    parsed_url.params,
                    new_query,
                    parsed_url.fragment
                ))
            
            response = session.get(firebase_url, timeout=30)
            response.raise_for_status()
            
            return response.text
            
        except requests.RequestException as e:
            logger.error(f"[ERROR] Firebase Storage download failed: {e}")
            raise ValueError(f"Failed to download from Firebase Storage: {str(e)}")
    
    def clean_and_prepare_data(self, df):
        """Clean and prepare data for training."""
        logger.info("Starting data cleaning and preparation...")
        
        original_count = len(df)
        
        # Normalize column names
        df.columns = [
            c.strip().lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace(":", "")
            .replace(",", "")
            .replace("'", "")
            .replace('"', "")
            for c in df.columns
        ]
        
        # Remove metadata columns
        metadata_columns = [
            'timestamp', 'name', 'institution', 'gender', 
            'year_level', 'latest_general_weighted_average_gwa',
            'how_far_is_your_home_from_school_one_way',
            'what_type_of_learning_modality_do_you_currently_attend'
        ]
        
        cols_to_drop = []
        for col in df.columns:
            for meta in metadata_columns:
                if meta in col:
                    cols_to_drop.append(col)
                    break
        
        logger.info(f"[LIST] Removing metadata columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Enhanced empty value cleaning
        empty_values = ["", " ", "nan", "NaN", "NA", "N/A", "null", "None", "#N/A", "?", "--"]
        df.replace(empty_values, np.nan, inplace=True)
        
        # Remove completely empty rows
        df.dropna(how='all', inplace=True)
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        cleaned_count = len(df)
        logger.info(f"[SUCCESS] Data cleaned: {cleaned_count} rows, {df.shape[1]} columns")
        
        return df
    
    def map_likert_responses(self, df):
        """Map Likert scale responses to numerical values."""
        logger.info("Mapping Likert scale responses to numerical values...")
        
        likert_map = {
            "strongly disagree": 1,
            "disagree": 2,
            "neutral": 3,
            "agree": 4,
            "strongly agree": 5,
            "strongly_disagree": 1,
            "strongly_agree": 5,
            "argee": 4,
            "agre": 4,
            "neural": 3,
            "nuetral": 3,
            "disargee": 2,
            "disagre": 2,
            "never": 1,
            "rarely": 2,
            "sometimes": 3,
            "often": 4,
            "always": 5,
            "no": 1,
            "yes": 5,
        }
        
        columns_mapped = 0
        for col in df.select_dtypes(include=["object"]).columns:
            original_values = df[col].copy()
            df[col] = df[col].apply(
                lambda v: likert_map.get(str(v).strip().lower(), v) if pd.notna(v) else v
            )
            if not df[col].equals(original_values):
                columns_mapped += 1
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        logger.info(f"[SUCCESS] Likert mapping applied to {columns_mapped} columns")
        return df
    
    def derive_burnout_labels(self, df):
        """Derive burnout labels using multi-dimensional analysis."""
        logger.info("Deriving burnout labels using multi-dimensional analysis...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for burnout analysis!")
        
        logger.info(f"[DATA] Using {len(numeric_cols)} survey response columns")
        
        # Calculate composite burnout score
        burnout_index = df[numeric_cols].mean(axis=1)
        
        # Statistical thresholding
        q25, q50, q75 = burnout_index.quantile([0.25, 0.50, 0.75])
        low_threshold = q25
        high_threshold = q75
        
        # Create burnout level categories
        conditions = [
            (burnout_index <= low_threshold),
            (burnout_index > low_threshold) & (burnout_index <= high_threshold),
            (burnout_index > high_threshold)
        ]
        choices = ["Low", "Moderate", "High"]
        
        df["burnout_level"] = np.select(conditions, choices, default="Moderate")
        
        # Log distribution
        distribution = df["burnout_level"].value_counts().to_dict()
        logger.info(f"[SUCCESS] Burnout distribution: {distribution}")
        logger.info(f"[CHART] Thresholds: Low ≤ {low_threshold:.2f}, High > {high_threshold:.2f}")
        
        return df, "burnout_level"
    
    def preprocess_features(self, X, y):
        """Preprocess features: encoding, imputation, scaling with robust index handling."""
        logger.info("Preprocessing features...")
        
        # Ensure X and y are pandas Series/DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Store original indices for reference
        original_x_indices = X.index.tolist()
        original_y_indices = y.index.tolist()
        
        logger.info(f"[DEBUG] Original X indices: {len(original_x_indices)}, Y indices: {len(original_y_indices)}")
        
        # Align X and y indices by resetting both
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        logger.info(f"[DEBUG] After reset: X shape: {X.shape}, Y shape: {y.shape}")
        
        # Handle categorical variables
        self.categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
        self.label_encoders = {}
        
        logger.info(f"[DEBUG] Categorical columns found: {len(self.categorical_columns)}")
        
        for col in self.categorical_columns:
            try:
                le = LabelEncoder()
                # Convert to string, handle NaN
                col_data = X[col].astype(str).fillna('unknown')
                X[col] = le.fit_transform(col_data)
                self.label_encoders[col] = le
                logger.debug(f"[DEBUG] Encoded column: {col}")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to encode column {col}: {e}")
                # Drop the column if encoding fails
                X = X.drop(columns=[col])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        logger.info(f"[DEBUG] Feature names: {len(self.feature_names)} features")
        
        # Convert to numpy arrays for imputation and scaling
        X_np = X.values
        
        # Imputation
        try:
            X_imputed_np = self.imputer.fit_transform(X_np)
            logger.debug(f"[DEBUG] Imputation successful: {X_imputed_np.shape}")
        except Exception as e:
            logger.error(f"[ERROR] Imputation failed: {e}")
            # Use simple mean imputation as fallback
            from sklearn.impute import SimpleImputer
            fallback_imputer = SimpleImputer(strategy='mean')
            X_imputed_np = fallback_imputer.fit_transform(X_np)
        
        # Scaling
        try:
            X_scaled_np = self.scaler.fit_transform(X_imputed_np)
            logger.debug(f"[DEBUG] Scaling successful: {X_scaled_np.shape}")
        except Exception as e:
            logger.error(f"[ERROR] Scaling failed: {e}")
            # Skip scaling as fallback
            X_scaled_np = X_imputed_np
        
        # Create DataFrame with proper indices
        X_scaled = pd.DataFrame(X_scaled_np, columns=self.feature_names)
        
        # Remove any invalid labels from y
        y_clean = y.copy()
        
        # Define invalid values
        invalid_values = ['', 'nan', 'NaN', 'None', 'null', 'NA', 'N/A', '?', '--']
        mask = ~y_clean.isin(invalid_values) & y_clean.notna()
        
        # Apply the mask to both X and y
        X_scaled = X_scaled[mask.values]
        y_clean = y_clean[mask.values]
        
        # Reset indices for both
        X_scaled = X_scaled.reset_index(drop=True)
        y_clean = y_clean.reset_index(drop=True)
        
        # Final validation
        if len(X_scaled) != len(y_clean):
            logger.error(f"[ERROR] Mismatch after cleaning: X={len(X_scaled)}, y={len(y_clean)}")
            # Align by taking intersection
            min_len = min(len(X_scaled), len(y_clean))
            X_scaled = X_scaled.iloc[:min_len]
            y_clean = y_clean.iloc[:min_len]
        
        logger.info(f"[SUCCESS] Features preprocessed: {X_scaled.shape[1]} features, {len(X_scaled)} samples")
        logger.info(f"[DATA] Class distribution: {dict(y_clean.value_counts())}")
        
        return X_scaled, y_clean
    
    def get_preprocessor(self):
        """Return preprocessor for saving."""
        return {
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'categorical_columns': self.categorical_columns
        }