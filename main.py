import numpy as np
from ucimlrepo import fetch_ucirepo

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split



def main():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    # metadata
    print(adult.metadata)
    # variable information
    print(adult.variables)
    # .head
    print(X.head())
    # .info
    print(X.info())
    # .describe
    print(X.describe())
    # .shape
    print(X.shape)

    print((X == '?').sum())
    X_Features = X.copy()
    X_Features.replace('?', np.NaN, inplace=True)
    print(X.info())
    prepared_data = preprocess_pipline(X_Features)
    Y_Target = y.copy()
    Y_Target.replace('>50K.', '>50K',inplace=True)
    Y_Target.replace('<=50K.', '<=50K', inplace=True)
    value_counts = Y_Target.value_counts()
    print(value_counts)
    X_train, X_test, y_train, y_test = train_test_split(prepared_data, Y_Target, test_size=0.2,random_state=42)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)



def preprocess_pipline(features):
    print((features == '?').sum())
    # For extract the columns of number
    num_col = features.select_dtypes(include='number').columns.to_list()
    # For extract the columns of non-number
    cat_col = features.select_dtypes(exclude='number').columns.to_list()
    #make a pipeline
    num_pipline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    cat_pipline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(sparse_output=False))
    preprocessing = ColumnTransformer([('num', num_pipline, num_col),
                                       ('cat', cat_pipline, cat_col)],
                                      remainder='passthrough')
    adult_prepared = preprocessing.fit_transform(features)
    feature_names = preprocessing.get_feature_names_out()
    adult_prepared = pd.DataFrame(data=adult_prepared, columns=feature_names)
    return adult_prepared


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
