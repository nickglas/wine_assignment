from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the wine dataset
wine_dataset = load_wine()

# Access the features (X)
X = wine_dataset.data

# Access the existing features (columns)
feature_names = wine_dataset.feature_names

#applied feature scaling to specified dataset
def applyFeatureScaling(dataset):
    # Initialize the scaler
    scaler = StandardScaler()

    # Perform feature scaling on the dataset
    scaled_dataset = scaler.fit_transform(dataset)

    #return scaled dataset
    return scaled_dataset

def outlierRemoval(dataset):
    # Define the outlier removal method
    outlier_removal = OneClassSVM(nu=0.05)

    # Define the column transformer for the pipeline
    column_transformer = ColumnTransformer([('scaler', RobustScaler(), slice(0, -1))])

    # create a pipeline with the column transformer
    pipeline = Pipeline([('preprocessing', column_transformer), ('outlier_removal', outlier_removal)])

    # Fit the pipeline on the dataset and remove outliers
    processed_dataset = pipeline.fit_predict(dataset)
    processed_dataset = dataset[processed_dataset == 1]

    return processed_dataset

# Main function for preprocessing
def preProcess(dataset):
    #dataset = applyFeatureScaling(dataset)
    dataset = outlierRemoval(dataset)
    return dataset

def predictCultivator(dataset):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)


#Create 2 more features based on the other features
#We are creating a new color intensity category that categorizes the color intensity into low, medium, and high categories. THis provides a new represenation of the color intensity levels 
#The other feature is the flavanoid proportion. This feature can give insights into the relative concentration of flavanoids in each sample
def createFeatures():

    # Get the indices of the relevant features
    color_intensity_index = feature_names.index('color_intensity')
    total_phenols_index = feature_names.index('total_phenols')
    flavanoids_index = feature_names.index('flavanoids')

    # Create the new feature - Color Intensity Category
    color_intensity = X[:, color_intensity_index]
    color_intensity_category = np.where(color_intensity < 5.0, 'Low', np.where(color_intensity < 10.0, 'Medium', 'High'))

    # Create the new feature - Flavanoid Proportion
    total_phenols = X[:, total_phenols_index]
    flavanoids = X[:, flavanoids_index]
    flavanoid_proportion = flavanoids / total_phenols

    # Append the new features to the dataset
    new_features_dataset = np.column_stack((X, color_intensity_category, flavanoid_proportion))
    # Update the feature names
    new_feature_names = np.append(feature_names, ['color_intensity_category', 'flavanoid_proportion'])
    return new_features_dataset, new_feature_names

#update original dataset with new data or features
def updateDataSet(newDataSet: None, newFeatureNames: None):
    
    global feature_names
    global X

    if newDataSet is not None:
        X = newDataSet

    if newFeatureNames is not None:
        feature_names = newFeatureNames


def printData():
    
    for sample in X:
        print(sample)


def main():
    updateDataSet(preProcess(X), None)
    newFeaturesDataset, newFeatureNames = createFeatures()
    updateDataSet(newFeaturesDataset, newFeatureNames)
    predictCultivator()
    printData()

if __name__ == "__main__":
    main()





# Split the data into features (X) and target (y)
# X = wine_dataset.data
# y = wine_dataset.target

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the preprocessing steps
# numeric_features = [0, 5, 9]  # Indices of numeric features in X
# categorical_features = [6]  # Indices of categorical features in X

# # Uses the mean of the other values to fix the missing values
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])

# # handlews missing value bvy using the most frequent strategy fotr for imputation and encodes categorical feature using the one Hot encodign method
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Column transformer combines these transformers for the respective numeric and categorical features.
# preprocessor = ColumnTransformer(transformers=[
#     ('numeric', numeric_transformer, numeric_features),
#     ('categorical', categorical_transformer, categorical_features)
# ])

# # Define the feature selection step, in this case we select the 
# feature_selector = SelectKBest(score_func=f_classif, k=6)  # Select the top 6 features

# # Define the ML model
# model = RandomForestClassifier()

# # Create the pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('feature_selector', feature_selector),
#     ('model', model)
# ])

# # Fit the pipeline on the training data
# pipeline.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = pipeline.predict(X_test)

# # Evaluate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)