import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lime.lime_tabular
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignatureFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'Ratio',
            'Centroid_Y',
            'Centroid_X',
            'Eccentricity',
            'Solidity',
            'Skewness_X',
            'Skewness_Y',
            'Kurtosis_X',
            'Kurtosis_Y'
        ]
        self.scaler = StandardScaler()
        
    def rgbgrey(self, img):
        """Convert RGB image to greyscale"""
        if len(img.shape) == 2:  # Already grayscale
            return img
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    def greybin(self, img):
        """Convert greyscale to binary image"""
        blur_radius = 0.8
        img = ndimage.gaussian_filter(img, blur_radius)
        thres = threshold_otsu(img)
        binimg = img > thres
        binimg = np.logical_not(binimg)
        return binimg

    def preproc(self, path, img=None):
        """Preprocess the signature image"""
        try:
            if img is None:
                img = mpimg.imread(path)
            grey = self.rgbgrey(img)
            binimg = self.greybin(grey)
            r, c = np.where(binimg==1)
            if len(r) == 0 or len(c) == 0:
                raise ValueError("No signature content found in image")
            signimg = binimg[r.min(): r.max(), c.min(): c.max()]
            return signimg
        except Exception as e:
            logging.error(f"Error in preprocessing image: {str(e)}")
            raise

    def extract_features(self, path, img=None):
        """Extract features from the signature image"""
        try:
            if img is None:
                img = mpimg.imread(path)
            processed_img = self.preproc(path, img)
            
            features = {}
            
            # Calculate ratio
            features['Ratio'] = self.calculate_ratio(processed_img)
            
            # Calculate centroid
            cent_y, cent_x = self.calculate_centroid(processed_img)
            features['Centroid_Y'] = cent_y
            features['Centroid_X'] = cent_x
            
            # Calculate eccentricity and solidity
            ecc, sol = self.calculate_eccentricity_solidity(processed_img)
            features['Eccentricity'] = ecc
            features['Solidity'] = sol
            
            # Calculate skewness and kurtosis
            skew, kurt = self.calculate_skew_kurtosis(processed_img)
            features['Skewness_X'] = skew[0]
            features['Skewness_Y'] = skew[1]
            features['Kurtosis_X'] = kurt[0]
            features['Kurtosis_Y'] = kurt[1]
            
            return features
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            raise

    def calculate_ratio(self, img):
        """Calculate black pixel ratio"""
        return np.sum(img) / (img.shape[0] * img.shape[1])

    def calculate_centroid(self, img):
        """Calculate normalized centroid coordinates"""
        y, x = np.nonzero(img)
        return np.mean(y) / img.shape[0], np.mean(x) / img.shape[1]

    def calculate_eccentricity_solidity(self, img):
        """Calculate eccentricity and solidity"""
        regions = regionprops(img.astype("int8"))
        if not regions:
            return 0, 0
        return regions[0].eccentricity, regions[0].solidity

    def calculate_skew_kurtosis(self, img):
        """Calculate skewness and kurtosis for both axes"""
        h, w = img.shape
        x = np.arange(w)
        y = np.arange(h)
        
        xp = np.sum(img, axis=0)
        yp = np.sum(img, axis=1)
        
        # Prevent division by zero
        if np.sum(xp) == 0 or np.sum(yp) == 0:
            return (0, 0), (0, 0)
        
        cx = np.sum(x*xp)/np.sum(xp)
        cy = np.sum(y*yp)/np.sum(yp)
        
        sx = np.sqrt(np.sum((x-cx)**2 * xp)/np.sum(img))
        sy = np.sqrt(np.sum((y-cy)**2 * yp)/np.sum(img))
        
        # Prevent division by zero
        sx = sx if sx != 0 else 1
        sy = sy if sy != 0 else 1
        
        skewx = np.sum(xp*(x-cx)**3)/(np.sum(img) * sx**3)
        skewy = np.sum(yp*(y-cy)**3)/(np.sum(img) * sy**3)
        
        kurtx = np.sum(xp*(x-cx)**4)/(np.sum(img) * sx**4) - 3
        kurty = np.sum(yp*(y-cy)**4)/(np.sum(img) * sy**4) - 3
        
        return (skewx, skewy), (kurtx, kurty)

class SignatureVerifier:
    def __init__(self, learning_rate=0.01, training_epochs=200, batch_size=32):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.feature_extractor = SignatureFeatureExtractor()
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(9,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X_train, y_train, validation_split=0.2):
        """Train the model with the provided data"""
        logging.info("Starting model training...")
        
        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, 2)
        
        # Training the model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.training_epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        logging.info("Model training completed")
        return history

    def predict(self, features):
        """Make prediction for given features"""
        if isinstance(features, dict):
            feature_array = np.array([[v for v in features.values()]])
        else:
            feature_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(feature_array)
        return prediction

    def explain_prediction(self, features, train_data):
        """Explain the model's prediction"""
        if isinstance(features, dict):
            feature_array = np.array([[v for v in features.values()]])
        else:
            feature_array = np.array(features).reshape(1, -1)
            
        prediction = self.predict(feature_array)
        
        # Explainable AI with LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=train_data,
            feature_names=self.feature_extractor.feature_names,
            class_names=['Forged', 'Genuine'],
            mode='classification'
        )
        
        # explanation
        exp = explainer.explain_instance(
            feature_array[0],
            self.model.predict,
            num_features=len(self.feature_extractor.feature_names)
        )
        
        # Get feature importance
        feature_importance = []
        weights = self.model.layers[0].get_weights()[0]
        for idx, name in enumerate(self.feature_extractor.feature_names):
            importance = np.mean(np.abs(weights[idx]))
            feature_importance.append((name, importance))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Generate explanation
        is_genuine = prediction[0][1] > prediction[0][0]
        confidence = float(max(prediction[0]) * 100)
        
        explanation = {
            'prediction': 'Genuine' if is_genuine else 'Forged',
            'confidence': confidence,
            'feature_importance': feature_importance,
            'lime_explanation': exp
        }
        
        self._print_explanation(explanation, features)
        self._plot_feature_importance(feature_importance)
        
        return explanation
    
    def _print_explanation(self, explanation, features):
        """Print detailed explanation of the prediction"""
        print(f"\nSignature Analysis Results:")
        print(f"Prediction: {explanation['prediction']} Signature")
        print(f"Confidence: {explanation['confidence']:.2f}%\n")
        
        print("Key Features Contributing to Decision:")
        for feature, importance in explanation['feature_importance'][:3]:
            if isinstance(features, dict):
                feature_value = features[feature]
            else:
                feature_value = features[0][self.feature_extractor.feature_names.index(feature)]
            print(f"- {feature}: {feature_value:.4f} (importance: {importance:.4f})")
    
    def _plot_feature_importance(self, feature_importance):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        features, importance = zip(*feature_importance)
        sns.barplot(x=list(importance), y=list(features))
        plt.title('Feature Importance in Signature Analysis')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

def analyze_signature(image_path, train_path):
    """Main function to analyze a signature image"""
    try:
        # Initialize components
        verifier = SignatureVerifier()
        feature_extractor = SignatureFeatureExtractor()
        
        # Load and prepare training data
        logging.info("Loading training data...")
        train_data = pd.read_csv(train_path)
        X_train = train_data.iloc[:, :9].values 
        y_train = train_data.iloc[:, 9].values  
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Train the model
        logging.info("Training model...")
        verifier.train(X_train, y_train)
        
        # Extract features from test image
        logging.info("Extracting features from test image...")
        features = feature_extractor.extract_features(image_path)
        
        # Scale features
        features_array = np.array([[v for v in features.values()]])
        features_array_scaled = scaler.transform(features_array)
        features_scaled = dict(zip(features.keys(), features_array_scaled[0]))
        
        # Get explanation
        logging.info("Generating explanation...")
        explanation = verifier.explain_prediction(features_scaled, X_train)
        
        return explanation
        
    except Exception as e:
        logging.error(f"Error in signature analysis: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_person_id = input("Enter person's id: ")
        test_image_path = input("Enter path of signature image: ")
        train_path = f'Features/Training/training_{train_person_id}.csv'
        
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"Image file not found: {test_image_path}")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        
        explanation = analyze_signature(test_image_path, train_path)
        
    except Exception as e:
        logging.error(f"Program error: {str(e)}")
        print(f"An error occurred: {str(e)}")


