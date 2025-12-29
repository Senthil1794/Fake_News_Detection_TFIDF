# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:06:21 2025

@author: sea
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data_processing_and_features import(
    fetch_data_from_db,
    text_data_processing,
    tf_idf_features_fit,
    tf_idf_features_transform,
    train_test_split,
    fit_and_evaluate_model,
    get_important_features
    )

data = fetch_data_from_db("Classification.db", "fake_news_classification")

# Drop unnecessary columns
data.drop(["Unnamed: 0.1", "Unnamed: 0"],axis=1,inplace=True)
# Drop entire row which has missinig feilds
data = data.dropna(subset=["text", "label"])

print(data.info())

data = text_data_processing(data)
print("Hellow")

print(data["text"][0])
print()
print()
print()
print(data["text_original"][0])
print(data["label"].value_counts())
data.reset_index(drop=True,inplace=True)
data_train = data[0:55031]
data_test = data[55031:70754]
tfidf, data_train_matrix = tf_idf_features_fit(data_train)
features = tfidf.get_feature_names_out()
print(len(features))
data_test_matrix = tf_idf_features_transform(tfidf, data_test)
x_train,x_test,y_train,y_test = train_test_split(data_train, data_test, data_train_matrix, data_test_matrix)
model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)
feature_importance = get_important_features(model, features)
print(feature_importance.head(10))
