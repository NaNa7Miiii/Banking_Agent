import joblib
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


class FraudPredictor:
    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            model_dir = Path(__file__).parent

        self.model_dir = Path(model_dir)
        self.model = None
        self.categorical_categories = None
        self.numeric_medians = None
        self.feature_info = None
        self.decision_threshold = 0.6
        self._load_model()

    def _load_model(self):
        # Core model files
        model_path = self.model_dir / "fraud_detection_model.pkl"
        categories_path = self.model_dir / "categorical_categories.pkl"
        medians_path = self.model_dir / "numeric_medians.pkl"
        info_path = self.model_dir / "model_info.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load trained model
        self.model = joblib.load(model_path)

        # Load categorical encodings (training categories)
        if categories_path.exists():
            self.categorical_categories = joblib.load(categories_path)
        else:
            self.categorical_categories = {}

        # Load numeric medians (for missing values)
        if medians_path.exists():
            self.numeric_medians = joblib.load(medians_path)
        else:
            self.numeric_medians = {}

        # Load feature configuration and metadata
        if not info_path.exists():
            raise FileNotFoundError(f"Model info file not found: {info_path}")

        with open(info_path, "r") as f:
            self.feature_info = json.load(f)

        # Use the same decision threshold as in training
        self.decision_threshold = float(self.feature_info.get("decision_threshold", 0.6))

    def _preprocess_transaction(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([transaction])

        # Time-based features (must match training logic)
        if "transaction_datetime" in df.columns:
            df["transaction_datetime"] = pd.to_datetime(
                df["transaction_datetime"], errors="coerce"
            )
            df["trans_hour"] = df["transaction_datetime"].dt.hour
            df["trans_dayofweek"] = df["transaction_datetime"].dt.dayofweek
            df["trans_day"] = df["transaction_datetime"].dt.day
            df["trans_month"] = df["transaction_datetime"].dt.month
            df["is_night"] = df["trans_hour"].isin([0, 1, 2, 3, 4, 5, 6]).astype(int)

        # Age feature from date of birth
        if "customer_dob" in df.columns:
            today = pd.Timestamp("today")
            df["customer_dob"] = pd.to_datetime(df["customer_dob"], errors="coerce")
            df["customer_age"] = (
                (today - df["customer_dob"]).dt.days / 365.25
            ).astype("float32")

        # Feature config from training
        feature_cols = self.feature_info["feature_cols"]
        numeric_cols = self.feature_info["numeric_cols"]
        derived_numeric_cols = self.feature_info["derived_numeric_cols"]
        categorical_cols = self.feature_info["categorical_cols"]

        # Categorical encoding must follow training categories
        for col in categorical_cols:
            if col not in self.categorical_categories:
                # If we have a categorical col but no categories for it, config is broken
                raise ValueError(
                    f"Categorical column '{col}' not found in categorical_categories. "
                    f"Check training and inference feature configs."
                )

            categories = self.categorical_categories[col]

            if col in df.columns:
                df[col] = df[col].fillna("Missing").astype(str)
                df[col] = pd.Categorical(df[col], categories=categories)
                # Unseen categories become -1 via codes
                df[col] = df[col].cat.codes
            else:
                # Field not provided → treat as "missing category"
                df[col] = -1

        # Numeric features: fill missing with training medians
        for col in numeric_cols + derived_numeric_cols:
            median_val = self.numeric_medians.get(col, 0)

            if col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(median_val)
            else:
                # Column missing in request → fill whole column with median
                df[col] = median_val

        # Make sure every feature used in training exists here
        for col in feature_cols:
            if col not in df.columns:
                # Categorical → -1, numeric → 0 (safe fallback)
                if col in categorical_cols:
                    df[col] = -1
                else:
                    df[col] = 0

        # Final feature matrix in the exact training order
        X = df[feature_cols].copy()
        return X

    def _preprocess_batch(self, transactions: list) -> pd.DataFrame:
        if not transactions:
            return pd.DataFrame()

        df = pd.DataFrame(transactions)

        # Time-based features (must match training logic)
        if "transaction_datetime" in df.columns:
            df["transaction_datetime"] = pd.to_datetime(
                df["transaction_datetime"], errors="coerce"
            )
            df["trans_hour"] = df["transaction_datetime"].dt.hour
            df["trans_dayofweek"] = df["transaction_datetime"].dt.dayofweek
            df["trans_day"] = df["transaction_datetime"].dt.day
            df["trans_month"] = df["transaction_datetime"].dt.month
            df["is_night"] = df["trans_hour"].isin([0, 1, 2, 3, 4, 5, 6]).astype(int)

        # Age feature from date of birth
        if "customer_dob" in df.columns:
            today = pd.Timestamp("today")
            df["customer_dob"] = pd.to_datetime(df["customer_dob"], errors="coerce")
            df["customer_age"] = (
                (today - df["customer_dob"]).dt.days / 365.25
            ).astype("float32")

        # Feature config from training
        feature_cols = self.feature_info["feature_cols"]
        numeric_cols = self.feature_info["numeric_cols"]
        derived_numeric_cols = self.feature_info["derived_numeric_cols"]
        categorical_cols = self.feature_info["categorical_cols"]

        # Categorical encoding must follow training categories
        for col in categorical_cols:
            if col not in self.categorical_categories:
                raise ValueError(
                    f"Categorical column '{col}' not found in categorical_categories. "
                    f"Check training and inference feature configs."
                )

            categories = self.categorical_categories[col]

            if col in df.columns:
                df[col] = df[col].fillna("Missing").astype(str)
                df[col] = pd.Categorical(df[col], categories=categories)
                # Unseen categories become -1 via codes
                df[col] = df[col].cat.codes
            else:
                # Field not provided → treat as "missing category"
                df[col] = -1

        # Numeric features: fill missing with training medians
        for col in numeric_cols + derived_numeric_cols:
            median_val = self.numeric_medians.get(col, 0)

            if col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(median_val)
            else:
                # Column missing in request → fill whole column with median
                df[col] = median_val

        # Make sure every feature used in training exists here
        for col in feature_cols:
            if col not in df.columns:
                # Categorical → -1, numeric → 0 (safe fallback)
                if col in categorical_cols:
                    df[col] = -1
                else:
                    df[col] = 0

        # Final feature matrix in the exact training order
        X = df[feature_cols].copy()
        return X

    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud probability for a transaction.

        Returns:
            {
                "is_fraud": bool,
                "fraud_probability": float,
                "confidence": "low" | "medium" | "high"
            }
        """
        try:
            X = self._preprocess_transaction(transaction)

            # Model output: probability of class 1 (fraud)
            fraud_proba = float(self.model.predict_proba(X)[0, 1])

            # Use global decision threshold (same as training)
            threshold = self.decision_threshold
            is_fraud = fraud_proba >= threshold

            # Confidence based on how far score is from threshold
            high_conf_boundary = max(0.9, threshold + 0.2)

            if fraud_proba >= high_conf_boundary:
                confidence = "high"
            elif fraud_proba >= threshold:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "is_fraud": bool(is_fraud),
                "fraud_probability": fraud_proba,
                "confidence": confidence,
            }
        except Exception as e:
            # Fail safe: do not block caller, return error info
            return {
                "is_fraud": False,
                "fraud_probability": 0.0,
                "confidence": "error",
                "error": str(e),
            }

    def predict_batch(self, transactions: list) -> list:
        """
        Batch prediction for multiple transactions - much faster than individual predictions.
        Processes all transactions at once instead of one-by-one.
        This is optimized for large batches (e.g., 100k+ transactions).
        """
        if not transactions:
            return []

        try:
            # Batch preprocess all transactions at once
            X = self._preprocess_batch(transactions)

            # Batch prediction - much faster than individual calls
            # Get probability of class 1 (fraud) for all transactions at once
            fraud_probas = self.model.predict_proba(X)[:, 1]

            # Use global decision threshold
            threshold = self.decision_threshold
            high_conf_boundary = max(0.9, threshold + 0.2)

            # Convert to list of prediction dictionaries
            predictions = []
            for fraud_proba in fraud_probas:
                is_fraud = fraud_proba >= threshold

                # Confidence based on how far score is from threshold
                if fraud_proba >= high_conf_boundary:
                    confidence = "high"
                elif fraud_proba >= threshold:
                    confidence = "medium"
                else:
                    confidence = "low"

                predictions.append({
                    "is_fraud": bool(is_fraud),
                    "fraud_probability": float(fraud_proba),
                    "confidence": confidence,
                })

            return predictions
        except Exception as e:
            # Fallback: if batch processing fails, try individual predictions
            # (but this should rarely happen)
            print(f"[FraudPredictor] Warning: Batch prediction failed, falling back to individual predictions: {e}")
            import traceback
            traceback.print_exc()
            return [self.predict(t) for t in transactions]


# Global predictor instance (lazy-loaded singleton)
_predictor_instance = None


def get_fraud_predictor(model_dir: Optional[Path] = None) -> FraudPredictor:
    # Reuse one global instance to avoid reloading model each time
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor(model_dir)
    return _predictor_instance
