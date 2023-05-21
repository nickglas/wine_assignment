from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np

# Load the wine dataset
wine_dataset = load_wine()

# Access the features (X) and target(y)
X = wine_dataset.data
y = wine_dataset.target

# Access the existing features (columns)
feature_names = wine_dataset.feature_names

print(feature_names)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Get the indices of the relevant features
        color_intensity_index = feature_names.index('color_intensity')
        total_phenols_index = feature_names.index('total_phenols')
        flavanoids_index = feature_names.index('flavanoids')

        # Create the new feature - Color Intensity Category where 1 is low, 2 is medium and 3 is high
        color_intensity = X[:, color_intensity_index]
        color_intensity_category = np.where(color_intensity < 5.0, 1, np.where(color_intensity < 10.0, 2, 3))

        # Create the new feature - Flavanoid Proportion
        total_phenols = X[:, total_phenols_index]
        flavanoids = X[:, flavanoids_index]
        flavanoid_proportion = flavanoids / total_phenols

        # Append the new features to the dataset
        new_features_dataset = np.column_stack((X, color_intensity_category, flavanoid_proportion))
        print("['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', \n'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',\n 'color_intensity', 'hue', \n'od280/od315_of_diluted_wines', 'proline', 'color_intensity_category', 'flavanoid_proportion']")
        #print(new_features_dataset)

        return new_features_dataset

def univariate_feature_selection(k=3):
    global X
    global y
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    top_features = selector.get_support(indices=True)
    return top_features

def recursive_feature_elimination(estimator, n_features=3):
    global X
    global y
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X, y)
    top_features = selector.get_support(indices=True)
    return top_features

def feature_importance(n_features=3):
    global X
    global y
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    top_features = np.argsort(importances)[-n_features:]
    return top_features

def main():
    
    # Define the column transformer for the pipeline
    numeric_features = slice(0, 11)  # Assuming the first 11 columns are numeric

    column_transformer_scaler = ColumnTransformer([
        ('numeric_scaler', StandardScaler(), numeric_features),
    ])

    # Create the pipeline with preprocessing and any desired machine learning model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('preprocessing', column_transformer_scaler),
        ('feature_engineering', FeatureCreator()),
        ('classifier', LogisticRegression())
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Predict the cultivator for the test data
    y_pred = pipeline.predict(X_test)

    # Compare the predicted cultivator with the actual cultivator
    for pred, actual in zip(y_pred, y_test):
        if pred == actual:
            print(f"RIGHTFULLY Predicted: {pred}, Actual: {actual}")
        else:
            print(f"WRONGFULLY predicted: {pred}, Actual: {actual}")


    print("TOP FEATURES FROM THE UNIVARIATE FEATURE METHOD")
    univariateFeatures = univariate_feature_selection(k=3)
    for i, featureIndex in enumerate(univariateFeatures):
        print("Top " + str(i+1) + " feature is: " + feature_names[featureIndex])
 
    print("TOP FEATURES FROM THE FEATURE IMPORTANCE METHOD")
    importanceFeatures = feature_importance(n_features=3)
    for i, featureIndex in enumerate(importanceFeatures):
        print("Top " + str(i+1) + " feature is: " + feature_names[featureIndex])

    print("TOP FEATURES FROM THE RECURSIVE FEATURE ELIMINATION METHOD")
    estimator = LogisticRegression(max_iter=5000)
    eliminationFeatures = recursive_feature_elimination(estimator=estimator, n_features=3)
    for i, featureIndex in enumerate(eliminationFeatures):
        print("Top " + str(i+1) + " feature is: " + feature_names[featureIndex])

    # Evaluate the model on the testing data    
    print("model score: %.3f" % pipeline.score(X_test, y_test))

if __name__ == "__main__":
    main()
