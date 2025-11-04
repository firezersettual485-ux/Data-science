import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from scipy import stats
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

class DiabetesDataPreprocessor:
    def __init__(self, data_path=None):
        """
        Initialize the Diabetes Data Preprocessor
        
        Parameters:
        data_path (str): Path to the diabetes dataset CSV file
        """
        self.data = None
        self.data_path = data_path
        self.cleaned_data = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        
    def load_data(self):
        """
        Load the diabetes dataset
        """
        try:
            # Try to load from provided path
            if self.data_path:
                self.data = pd.read_csv(self.data_path)
            else:
                # Load built-in dataset (example structure)
                from sklearn.datasets import make_classification
                X, y = make_classification(n_samples=1000, n_features=8, n_classes=2, 
                                         random_state=42)
                feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                self.data = pd.DataFrame(X, columns=feature_names)
                self.data['Outcome'] = y
                
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample dataset for demonstration
            self.create_sample_dataset()
            return self.data
    
    def create_sample_dataset(self):
        """Create a sample diabetes dataset for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Pregnancies': np.random.randint(0, 15, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples),
            'BloodPressure': np.random.normal(70, 12, n_samples),
            'SkinThickness': np.random.normal(20, 10, n_samples),
            'Insulin': np.random.normal(80, 50, n_samples),
            'BMI': np.random.normal(25, 5, n_samples),
            'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
            'Age': np.random.randint(21, 80, n_samples),
            'Outcome': np.random.randint(0, 2, n_samples)
        }
        
        self.data = pd.DataFrame(data)
        print("Sample dataset created for demonstration")
    
    def initial_exploration(self):
        """
        Phase 1: Perform initial data exploration and identify data quality issues
        """
        print("=" * 50)
        print("PHASE 1: DATA COLLECTION & UNDERSTANDING")
        print("=" * 50)
        
        # Basic information
        print("\n1. DATASET INFORMATION:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        print("\n2. DATA TYPES:")
        print(self.data.dtypes)
        
        print("\n3. FIRST 5 ROWS:")
        print(self.data.head())
        
        print("\n4. BASIC STATISTICS:")
        print(self.data.describe())
        
        # Identify data quality issues
        print("\n5. DATA QUALITY ISSUES:")
        
        # Biological impossibilities - features where zero is impossible
        zero_sensitive_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        print("\nBiological Impossibilities (Zeros):")
        for feature in zero_sensitive_features:
            zero_count = (self.data[feature] == 0).sum()
            if zero_count > 0:
                print(f"{feature}: {zero_count} zeros ({zero_count/len(self.data)*100:.2f}%)")
        
        # Data type inconsistencies
        print(f"\nData Type Issues: {self.data.dtypes.unique()}")
        
        # Potential outliers using IQR method
        print("\nPotential Outliers (using IQR method):")
        numerical_features = self.data.select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            if feature != 'Outcome':  # Skip target variable
                Q1 = self.data[feature].quantile(0.25)
                Q3 = self.data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.data[(self.data[feature] < lower_bound) | (self.data[feature] > upper_bound)]
                print(f"{feature}: {len(outliers)} outliers")
    
    def clean_data(self):
        """
        Phase 2: Data Cleaning - Handle missing values and outliers
        """
        print("\n" + "=" * 50)
        print("PHASE 2: DATA CLEANING")
        print("=" * 50)
        
        # Create a copy for cleaning
        self.cleaned_data = self.data.copy()
        
        # 1. Missing Value Analysis
        print("\n1. MISSING VALUE ANALYSIS:")
        
        # Identify zeros in biologically impossible features
        zero_sensitive_features = {
            'Glucose': (0, "Glucose cannot be 0"),
            'BloodPressure': (0, "Blood pressure cannot be 0"),
            'SkinThickness': (0, "Skin thickness cannot be 0"),
            'Insulin': (0, "Insulin cannot be 0"),
            'BMI': (0, "BMI cannot be 0")
        }
        
        print("\nReplacing impossible zeros with NaN:")
        for feature, (threshold, reason) in zero_sensitive_features.items():
            zero_count = (self.cleaned_data[feature] == threshold).sum()
            if zero_count > 0:
                print(f"{feature}: Replacing {zero_count} zeros with NaN")
                self.cleaned_data[feature] = self.cleaned_data[feature].replace(threshold, np.nan)
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        msno.matrix(self.cleaned_data)
        plt.title('Missing Values Matrix')
        plt.tight_layout()
        plt.show()
        
        # Missing value percentages
        missing_percent = self.cleaned_data.isnull().sum() / len(self.cleaned_data) * 100
        print("\nMissing Value Percentages:")
        for col, percent in missing_percent.items():
            if percent > 0:
                print(f"{col}: {percent:.2f}%")
        
        # 2. Imputation Strategy
        print("\n2. IMPUTATION STRATEGY:")
        
        # Use KNN Imputer for better results
        imputer = KNNImputer(n_neighbors=5)
        features_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        self.cleaned_data[features_to_impute] = imputer.fit_transform(
            self.cleaned_data[features_to_impute]
        )
        print("Applied KNN Imputer for missing values")
        
        # 3. Outlier Detection & Treatment
        print("\n3. OUTLIER DETECTION & TREATMENT:")
        
        numerical_features = self.cleaned_data.select_dtypes(include=[np.number]).columns
        numerical_features = numerical_features.drop('Outcome', errors='ignore')
        
        for feature in numerical_features:
            Q1 = self.cleaned_data[feature].quantile(0.25)
            Q3 = self.cleaned_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            outliers_before = ((self.cleaned_data[feature] < lower_bound) | 
                             (self.cleaned_data[feature] > upper_bound)).sum()
            
            if outliers_before > 0:
                self.cleaned_data[feature] = np.clip(self.cleaned_data[feature], lower_bound, upper_bound)
                outliers_after = ((self.cleaned_data[feature] < lower_bound) | 
                                (self.cleaned_data[feature] > upper_bound)).sum()
                print(f"{feature}: Capped {outliers_before} outliers")
        
        print("\nData cleaning completed!")
        return self.cleaned_data
    
    def transform_data(self):
        """
        Phase 3: Data Transformation - Feature engineering, encoding, and scaling
        """
        print("\n" + "=" * 50)
        print("PHASE 3: DATA TRANSFORMATION")
        print("=" * 50)
        
        if self.cleaned_data is None:
            self.cleaned_data = self.data.copy()
        
        # 1. Feature Engineering
        print("\n1. FEATURE ENGINEERING:")
        
        # Create age groups
        bins_age = [20, 35, 50, 65, 100]
        labels_age = ['Young', 'Middle-aged', 'Senior', 'Elderly']
        self.cleaned_data['AgeGroup'] = pd.cut(self.cleaned_data['Age'], bins=bins_age, 
                                             labels=labels_age, right=False)
        print("Created AgeGroup feature")
        
        # Calculate BMI categories
        bins_bmi = [0, 18.5, 25, 30, 100]
        labels_bmi = ['Underweight', 'Normal', 'Overweight', 'Obese']
        self.cleaned_data['BMICategory'] = pd.cut(self.cleaned_data['BMI'], bins=bins_bmi, 
                                                labels=labels_bmi, right=False)
        print("Created BMICategory feature")
        
        # Create glucose level categories
        bins_glucose = [0, 70, 100, 125, 200, 300]
        labels_glucose = ['Low', 'Normal', 'Prediabetes', 'Diabetes', 'Severe Diabetes']
        self.cleaned_data['GlucoseCategory'] = pd.cut(self.cleaned_data['Glucose'], bins=bins_glucose, 
                                                    labels=labels_glucose, right=False)
        print("Created GlucoseCategory feature")
        
        # 2. Encoding
        print("\n2. ENCODING:")
        
        # Label encoding for ordinal categories
        label_encoder = LabelEncoder()
        ordinal_features = ['AgeGroup', 'BMICategory', 'GlucoseCategory']
        
        for feature in ordinal_features:
            self.cleaned_data[f'{feature}_encoded'] = label_encoder.fit_transform(
                self.cleaned_data[feature]
            )
            print(f"Label encoded {feature}")
        
        # 3. Scaling
        print("\n3. SCALING:")
        
        # Select numerical features for scaling
        numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Compare StandardScaler vs MinMaxScaler
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        
        data_standard = standard_scaler.fit_transform(self.cleaned_data[numerical_features])
        data_minmax = minmax_scaler.fit_transform(self.cleaned_data[numerical_features])
        
        print("StandardScaler - Mean ~ 0, Std ~ 1")
        print(f"MinMaxScaler - Range: [{data_minmax.min():.2f}, {data_minmax.max():.2f}]")
        
        # Choose StandardScaler (better for algorithms assuming normal distribution)
        self.scaler = StandardScaler()
        self.cleaned_data[numerical_features] = self.scaler.fit_transform(
            self.cleaned_data[numerical_features]
        )
        print("Applied StandardScaler to numerical features")
        
        return self.cleaned_data
    
    def reduce_data(self):
        """
        Phase 4: Data Reduction - Feature selection and dimensionality reduction
        """
        print("\n" + "=" * 50)
        print("PHASE 4: DATA REDUCTION")
        print("=" * 50)
        
        # Prepare features and target
        feature_columns = [col for col in self.cleaned_data.columns if col not in 
                         ['Outcome', 'AgeGroup', 'BMICategory', 'GlucoseCategory'] 
                         and not col.endswith('_encoded')]
        
        X = self.cleaned_data[feature_columns]
        y = self.cleaned_data['Outcome']
        
        # 1. Feature Selection
        print("\n1. FEATURE SELECTION:")
        
        # Correlation matrix
        plt.figure(figsize=(12, 8))
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # SelectKBest with mutual information
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        self.feature_selector.fit(X, y)
        
        feature_scores = pd.DataFrame({
            'Feature': feature_columns,
            'Score': self.feature_selector.scores_
        }).sort_values('Score', ascending=False)
        
        print("\nFeature Importance Scores (Mutual Information):")
        print(feature_scores)
        
        # 2. Dimensionality Reduction
        print("\n2. DIMENSIONALITY REDUCTION (PCA):")
        
        # Perform PCA
        self.pca = PCA()
        X_pca = self.pca.fit_transform(X)
        
        # Visualize explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                np.cumsum(self.pca.explained_variance_ratio_), marker='o')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA - Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Determine optimal number of components (95% variance)
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"Optimal number of components for 95% variance: {optimal_components}")
        print("Explained variance by component:", self.pca.explained_variance_ratio_)
        
        return X, y, feature_scores
    
    def handle_imbalance(self):
        """
        Phase 5: Handle class imbalance
        """
        print("\n" + "=" * 50)
        print("PHASE 5: DATA IMBALANCE HANDLING")
        print("=" * 50)
        
        # 1. Class Distribution Analysis
        target_counts = self.cleaned_data['Outcome'].value_counts()
        imbalance_ratio = target_counts[0] / target_counts[1]
        
        print(f"Class Distribution: {dict(target_counts)}")
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%')
        plt.title('Class Distribution (%)')
        plt.tight_layout()
        plt.show()
        
        # 2. Balancing Techniques
        print("\n2. BALANCING TECHNIQUES:")
        
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN
        
        # Prepare data
        feature_columns = [col for col in self.cleaned_data.columns if col != 'Outcome']
        X = self.cleaned_data[feature_columns]
        y = self.cleaned_data['Outcome']
        
        print("Available balancing techniques:")
        print("1. SMOTE (Synthetic Minority Over-sampling Technique)")
        print("2. Random UnderSampling")
        print("3. SMOTEENN (SMOTE + Edited Nearest Neighbors)")
        
        # Apply SMOTE (generally recommended)
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        balanced_counts = pd.Series(y_balanced).value_counts()
        print(f"\nAfter SMOTE - Class Distribution: {dict(balanced_counts)}")
        
        return X_balanced, y_balanced
    
    def save_processed_data(self, filename='diabetes_processed.csv'):
        """
        Save the processed data and create data dictionary
        """
        if self.cleaned_data is not None:
            # Save processed data
            self.cleaned_data.to_csv(filename, index=False)
            print(f"\nProcessed data saved as: {filename}")
            
            # Create data dictionary
            data_dict = {
                'Feature': [],
                'Description': [],
                'Type': [],
                'Range/Values': []
            }
            
            # Original features
            original_features = {
                'Pregnancies': 'Number of times pregnant',
                'Glucose': 'Plasma glucose concentration (mg/dL)',
                'BloodPressure': 'Diastolic blood pressure (mm Hg)',
                'SkinThickness': 'Triceps skin fold thickness (mm)',
                'Insulin': '2-Hour serum insulin (mu U/ml)',
                'BMI': 'Body mass index (kg/m²)',
                'DiabetesPedigreeFunction': 'Diabetes pedigree function',
                'Age': 'Age in years',
                'Outcome': 'Target variable (0: No diabetes, 1: Diabetes)'
            }
            
            # Engineered features
            engineered_features = {
                'AgeGroup': 'Age categories (Young, Middle-aged, Senior, Elderly)',
                'BMICategory': 'BMI categories (Underweight, Normal, Overweight, Obese)',
                'GlucoseCategory': 'Glucose level categories',
                'AgeGroup_encoded': 'Encoded age groups',
                'BMICategory_encoded': 'Encoded BMI categories',
                'GlucoseCategory_encoded': 'Encoded glucose categories'
            }
            
            # Add original features to dictionary
            for feature, description in original_features.items():
                if feature in self.cleaned_data.columns:
                    data_dict['Feature'].append(feature)
                    data_dict['Description'].append(description)
                    data_dict['Type'].append('Numerical')
                    if feature == 'Outcome':
                        data_dict['Range/Values'].append('0, 1')
                    else:
                        data_dict['Range/Values'].append(f"{self.cleaned_data[feature].min():.2f} - {self.cleaned_data[feature].max():.2f}")
            
            # Add engineered features to dictionary
            for feature, description in engineered_features.items():
                if feature in self.cleaned_data.columns:
                    data_dict['Feature'].append(feature)
                    data_dict['Description'].append(description)
                    if '_encoded' in feature:
                        data_dict['Type'].append('Numerical (Encoded)')
                        data_dict['Range/Values'].append(f"{self.cleaned_data[feature].min():.0f} - {self.cleaned_data[feature].max():.0f}")
                    else:
                        data_dict['Type'].append('Categorical')
                        unique_vals = self.cleaned_data[feature].unique()
                        data_dict['Range/Values'].append(str(list(unique_vals)))
            
            # Save data dictionary
            data_dict_df = pd.DataFrame(data_dict)
            data_dict_df.to_csv('data_dictionary.csv', index=False)
            print("Data dictionary saved as: data_dictionary.csv")
            
            return data_dict_df
        else:
            print("No processed data available. Run cleaning and transformation first.")
            return None
    
    def run_complete_pipeline(self):
        """
        Run the complete data preprocessing pipeline
        """
        print("STARTING COMPLETE DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Phase 1: Data Collection & Understanding
        self.load_data()
        self.initial_exploration()
        
        # Phase 2: Data Cleaning
        self.clean_data()
        
        # Phase 3: Data Transformation
        self.transform_data()
        
        # Phase 4: Data Reduction
        X, y, feature_scores = self.reduce_data()
        
        # Phase 5: Data Imbalance Handling
        X_balanced, y_balanced = self.handle_imbalance()
        
        # Save results
        data_dict = self.save_processed_data()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return {
            'cleaned_data': self.cleaned_data,
            'feature_scores': feature_scores,
            'X_balanced': X_balanced,
            'y_balanced': y_balanced,
            'data_dictionary': data_dict
        }

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DiabetesDataPreprocessor('diabetes.csv')  # Replace with your file path
    
    # Run complete pipeline
    results = preprocessor.run_complete_pipeline()