"""
Parking Occupancy Prediction - Complete Self-Contained Script
For processing 9000+ images with comprehensive statistics
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import warnings
import time
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# VEHICLE DETECTOR
# ============================================================================

class VehicleDetector:
    """Vehicle detector using image processing techniques"""
    
    def __init__(self):
        pass
        
    def detect_vehicles(self, image_path):
        """Detect vehicles in an image"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to identify vehicles
        vehicle_count = 0
        min_area = (width * height) * 0.0003
        max_area = (width * height) * 0.25
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    if len(approx) >= 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.5 < aspect_ratio < 2.5:
                            vehicle_count += 1
        
        # Estimate parking spots
        estimated_spots = max(10, int((width * height) / 5000))
        occupancy_percentage = min(100, (vehicle_count / max(estimated_spots, 1)) * 100)
        
        # Image features
        brightness = np.mean(gray)
        contrast = np.std(gray)
        edge_density = np.sum(edges) / edges.size if edges.size > 0 else 0
        
        # Texture variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        return {
            'vehicle_count': vehicle_count,
            'occupancy_percentage': occupancy_percentage,
            'estimated_spots': estimated_spots,
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'texture_variance': texture_variance,
            'has_vehicles': 1 if vehicle_count > 0 else 0
        }


# ============================================================================
# DATA PREPROCESSOR
# ============================================================================

class DataPreprocessor:
    """Handle data preprocessing"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = None
        
    def extract_time_features(self, df, datetime_column):
        """Extract time-based features"""
        df = df.copy()
        
        if datetime_column in df.columns:
            df['timestamp'] = pd.to_datetime(df[datetime_column])
            
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Boolean features
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                                  (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
            
            df = df.drop('timestamp', axis=1)
        
        return df
    
    def prepare_features(self, df, target_column):
        """Prepare features for training"""
        df = df.copy()
        df = df.dropna(subset=[target_column])
        
        # Feature columns
        feature_cols = ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
                       'day_sin', 'day_cos', 'is_weekend', 'is_rush_hour',
                       'vehicle_count', 'estimated_spots', 'brightness', 
                       'contrast', 'edge_density', 'texture_variance', 'has_vehicles']
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Remove NaN rows
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        self.feature_columns = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def transform_features(self, df, target_column):
        """Transform features for prediction"""
        df = df.copy()
        
        feature_cols = ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
                       'day_sin', 'day_cos', 'is_weekend', 'is_rush_hour',
                       'vehicle_count', 'estimated_spots', 'brightness', 
                       'contrast', 'edge_density', 'texture_variance', 'has_vehicles']
        
        feature_cols = [col for col in feature_cols if col in df.columns and col in self.feature_columns]
        
        if not feature_cols:
            feature_cols = [col for col in self.feature_columns if col in df.columns]
        
        X = df[feature_cols].copy()
        X = X.fillna(0)
        
        # Ensure all required features
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled


# ============================================================================
# PARKING PREDICTOR
# ============================================================================

class ParkingPredictor:
    """Main parking occupancy prediction model"""
    
    def __init__(self):
        self.vehicle_detector = VehicleDetector()
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.training_stats = {}
        
    def process_images_to_dataset(self, image_folder):
        """Process all images and create dataset"""
        print("📸 Processing images...")
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No images found in {image_folder}")
            return pd.DataFrame()
        
        print(f"Found {len(image_files)} images")
        
        data = []
        start_time = time.time()
        
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(image_folder, img_name)
            features = self.vehicle_detector.detect_vehicles(img_path)
            
            if features is None:
                continue
            
            # Extract timestamp from filename
            try:
                name_without_ext = img_name.split('.')[0]
                if '_' in name_without_ext:
                    timestamp = datetime.strptime(name_without_ext, '%Y%m%d_%H%M')
                else:
                    timestamp = datetime.fromtimestamp(os.path.getmtime(img_path))
            except:
                timestamp = datetime.now()
            
            data.append({
                'timestamp': timestamp,
                **features
            })
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(image_files) - (i + 1)) / rate
                print(f"  Processed {i + 1}/{len(image_files)} images | "
                      f"Speed: {rate:.1f} img/sec | ETA: {remaining/60:.1f} min")
        
        if not data:
            print("❌ No valid images processed")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        
        print(f"✅ Created dataset with {len(df)} samples")
        return df
    
    def train(self, df, test_size=0.2):
        """Train the model"""
        
        print("=" * 70)
        print("PARKING OCCUPANCY PREDICTION TRAINING")
        print("=" * 70)
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Occupancy range: {df['occupancy_percentage'].min():.1f}% - {df['occupancy_percentage'].max():.1f}%")
        print(f"   Mean occupancy: {df['occupancy_percentage'].mean():.1f}%")
        print(f"   Std occupancy: {df['occupancy_percentage'].std():.1f}%")
        
        # Extract time features
        print(f"\n⏰ Extracting time features...")
        df = self.preprocessor.extract_time_features(df, 'timestamp')
        
        # Prepare features
        print(f"🔧 Preparing features...")
        X, y = self.preprocessor.prepare_features(df, 'occupancy_percentage')
        
        print(f"   Feature dimensions: {X.shape}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        print(f"\n📊 Split Statistics:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")
        
        # Train model
        print(f"\n🚀 Training Gradient Boosting model...")
        start_time = time.time()
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"   Training completed in {training_time:.2f} seconds")
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_explained_var = explained_variance_score(y_test, y_test_pred)
        
        # Cross-validation
        print(f"\n📊 Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                    cv=5, scoring='neg_mean_absolute_error')
        
        # Store statistics
        self.training_stats = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_explained_var': test_explained_var,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'training_time': training_time
        }
        
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE")
        print("=" * 70)
        print(f"\n📈 Training Set:")
        print(f"   MAE: {train_mae:.2f}% | RMSE: {train_rmse:.2f}% | R²: {train_r2:.4f}")
        
        print(f"\n📈 Testing Set:")
        print(f"   MAE: {test_mae:.2f}% | RMSE: {test_rmse:.2f}% | R²: {test_r2:.4f}")
        print(f"   Explained Variance: {test_explained_var:.4f}")
        
        print(f"\n📊 Cross-Validation (5-fold):")
        print(f"   Mean MAE: {-cv_scores.mean():.2f}% (±{cv_scores.std():.2f}%)")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n🔑 Top 10 Feature Importance:")
            importance_df = pd.DataFrame({
                'feature': self.preprocessor.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return y_test, y_test_pred
    
    def save_model(self, filepath):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'preprocessor': self.preprocessor,
                'training_stats': self.training_stats
            }, f)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.preprocessor = data['preprocessor']
            self.training_stats = data['training_stats']
        print(f"✅ Model loaded from {filepath}")
    
    def predict_from_image(self, image_path):
        """Predict from a single image"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        features = self.vehicle_detector.detect_vehicles(image_path)
        if features is None:
            return {'error': 'Could not process image'}
        
        df = pd.DataFrame([{
            'timestamp': datetime.now(),
            **features
        }])
        
        df = self.preprocessor.extract_time_features(df, 'timestamp')
        X = self.preprocessor.transform_features(df, 'occupancy_percentage')
        prediction = self.model.predict(X)[0]
        
        return {
            'detected_vehicles': features['vehicle_count'],
            'detected_occupancy': features['occupancy_percentage'],
            'predicted_occupancy': max(0, min(100, prediction))
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(y_test, y_pred, save_path=None):
    """Plot prediction results"""
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel("Actual Occupancy (%)")
    axes[0, 0].set_ylabel("Predicted Occupancy (%)")
    axes[0, 0].set_title("Actual vs Predicted")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals Distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel("Residuals (%)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(f"Prediction Errors\nMean: {residuals.mean():.2f}%, Std: {residuals.std():.2f}%")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time Series
    n_samples = min(200, len(y_test))
    axes[1, 0].plot(y_test[:n_samples], label='Actual', alpha=0.7)
    axes[1, 0].plot(y_pred[:n_samples], label='Predicted', alpha=0.7)
    axes[1, 0].set_xlabel("Time Steps")
    axes[1, 0].set_ylabel("Occupancy (%)")
    axes[1, 0].set_title(f"Time Series (First {n_samples} samples)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[1, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel("Predicted Occupancy (%)")
    axes[1, 1].set_ylabel("Residuals (%)")
    axes[1, 1].set_title("Residuals vs Predicted")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Figure saved to {save_path}")
    
    plt.show()


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_model_on_test_set(model_path, test_folder):
    """Test trained model on test dataset"""
    print("=" * 70)
    print("TESTING MODEL ON TEST DATASET")
    print("=" * 70)
    
    # Load model
    predictor = ParkingPredictor()
    predictor.load_model(model_path)
    
    # Process test images
    print(f"\n📸 Processing test images from: {test_folder}")
    df_test = predictor.process_images_to_dataset(test_folder)
    
    if df_test.empty:
        print("❌ No valid test images found")
        return
    
    # Prepare test data
    df_test = predictor.preprocessor.extract_time_features(df_test, 'timestamp')
    X_test, y_test = predictor.preprocessor.prepare_features(df_test, 'occupancy_percentage')
    
    # Make predictions
    y_pred = predictor.model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100
    
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"\n📊 Test samples: {len(y_test):,}")
    print(f"   MAE: {mae:.2f}%")
    print(f"   RMSE: {rmse:.2f}%")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    print(f"\n📊 Accuracy Metrics:")
    print(f"   Within ±5%: {np.mean(np.abs(y_test - y_pred) <= 5) * 100:.1f}%")
    print(f"   Within ±10%: {np.mean(np.abs(y_test - y_pred) <= 10) * 100:.1f}%")
    print(f"   Within ±20%: {np.mean(np.abs(y_test - y_pred) <= 20) * 100:.1f}%")
    
    # Plot results
    plot_results(y_test, y_pred, save_path='test_results.png')
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function"""
    
    print("=" * 70)
    print("PARKING OCCUPANCY PREDICTION SYSTEM")
    print("=" * 70)
    
    # Paths
    train_folder = "/Users/kaustavmanideka/Desktop/parkingg/train"
    test_folder = "/Users/kaustavmanideka/Desktop/parkingg/test"
    model_file = "parking_model.pkl"
    
    # Check if model exists
    if os.path.exists(model_file):
        print(f"\n✅ Found existing model: {model_file}")
        response = input("Do you want to retrain? (y/n): ").lower()
        
        if response != 'y':
            # Test existing model
            if os.path.exists(test_folder):
                test_model_on_test_set(model_file, test_folder)
            return
    
    # Train new model
    if not os.path.exists(train_folder):
        print(f"❌ Training folder not found: {train_folder}")
        return
    
    predictor = ParkingPredictor()
    
    # Process training images
    df_train = predictor.process_images_to_dataset(train_folder)
    
    if len(df_train) < 10:
        print(f"❌ Not enough samples: {len(df_train)} (need at least 10)")
        return
    
    # Train model
    y_test, y_pred = predictor.train(df_train, test_size=0.2)
    
    # Plot results
    plot_results(y_test, y_pred, save_path='training_results.png')
    
    # Save model
    predictor.save_model(model_file)
    
    # Test on test set if available
    if os.path.exists(test_folder):
        print("\n" + "=" * 70)
        print("EVALUATING ON TEST SET")
        print("=" * 70)
        test_model_on_test_set(model_file, test_folder)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()