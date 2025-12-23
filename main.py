import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    # For numerical columns
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    # For Categorical column
    cat_pipeline = Pipeline([ 
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), 
        ('cat', cat_pipeline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # load Dataset
    mobile = pd.read_csv("mobile_cleaned_data.csv")

    # Create a Stratified Test set
    mobile["Battery_new"] = pd.cut(mobile["Battery"],
                                   bins = [0, 3000, 4500, 5500, np.inf],
                                   labels = [1, 2, 3, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(mobile, mobile["Battery_new"]):
        mobile.loc[test_index].drop("Battery_new", axis=1).to_csv("input.csv", index=False)
        mobile = mobile.loc[train_index].drop("Battery_new", axis=1)

    # Separate features and labels
    mobile_labels = mobile["Price"].copy()
    mobile_features = mobile.drop("Price", axis=1)

    
    
    num_attribs = mobile_features.drop("Brand", axis=1).columns.tolist()
    cat_attribs = ["Brand"]
    

    pipeline = build_pipeline(num_attribs, cat_attribs)
    mobile_prepared = pipeline.fit_transform(mobile_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(mobile_prepared, mobile_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is Trained Congratulations!")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    prediction = model.predict(transformed_input)
    
    # Saving the predictions
    input_data["predicted_price"] = prediction
    input_data.to_csv("Output.csv", index=False)
    print('Inference Successfully completed, result saved to Output.csv')