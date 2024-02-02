import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import random
import missingno as msno

if __name__=="__main__":
    pd.set_option('display.max_columns', None)
    data = pd.read_csv("data/Real estate.csv")

    print(data.head())
    print(data.describe())

    msno.matrix(data)
    plt.show()

    sns.heatmap(data.corr(), annot=True)
    plt.show()
    data.drop(columns=['No', 'transaction date'], inplace=True)

    sns.heatmap(data.corr(), annot=True)
    plt.show() #multicolinearity detected
    data.drop(columns=['longitude', ], inplace=True)

    sns.heatmap(data.corr(), annot=True)
    plt.show()# pretty much good

    #dispersion diagram
    sns.pairplot(data)
    plt.show()

    models={}

    #split data
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['house price of unit area'], test_size=0.2,random_state=42)

    # linear single-factor regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train['distance to the nearest MRT station'].values.reshape(-1, 1), y_train)

    models["Linear single-factor"] = (lin_reg, ["distance to the nearest MRT station"], X_test['distance to the nearest MRT station'].values.reshape(-1, 1))

    #linear multi-factor and polynomial of 2 degree
    for degr in range(1, 3):
        poly = PolynomialFeatures(degree=degr)
        predictors_train_polyfeat = poly.fit_transform(X_train)

        poly_reg = LinearRegression()
        poly_reg.fit(predictors_train_polyfeat, y_train)

        models[f"polynomial regression of {degr} degree"]=(poly_reg, poly.get_feature_names(X_test.columns), poly.transform(X_test))

    print("Checking out coefficients:")
    for name, tuple in models.items():
        print(f"{name}:")
        print(f"\tBase constant value: {tuple[0].intercept_}")
        for feature, coef in zip(tuple[1], tuple[0].coef_):
            print(f"\tCoefficient for {feature}: {coef}")

    print("\nChecking models quality:")
    for name, tuple in models.items():
        model, features, test_data = tuple

        y_pred = model.predict(test_data)  # make predictions

        # print the model name
        print(f"\nModel: {name}")

        # print model efficiency metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\tMSE: {mse}")
        print(f"\tR-squared: {r2}\n")

    #Perform random experiment:
    print("\nPreforming experiments:")
    for i in range(3):
        rec_index=random.randint(0,len(X_test) - 1)

        record_features= X_test.iloc[rec_index]
        record_res=y_test.iloc[rec_index]

        print(f"Test number {i+1}:")
        print(f"\tHouse age: {record_features['house age']}")
        print(f"\tDistance to the nearest MRT station: {record_features['distance to the nearest MRT station']}")
        print(f"\tNumber of convenience stores: {record_features['number of convenience stores']}")
        print(f"\tLatitude: {record_features['latitude']}")

        print(f"\tActual price: {record_res}")
        print(f"\tLinear single-factor regression says: {lin_reg.predict(np.array([[record_features['distance to the nearest MRT station']]]))[0]}")

        poly = PolynomialFeatures(degree=1)
        predictors_test_polyfeat = poly.fit_transform(record_features.values.reshape(1, -1))

        print(f"\tLinear multifactor regression says: {models['polynomial regression of 1 degree'][0].predict(predictors_test_polyfeat)[0]}")

        poly = PolynomialFeatures(degree=2)
        predictors_test_polyfeat = poly.fit_transform(record_features.values.reshape(1, -1))

        print(f"\tPolynomial multifactor of 2 degree regression says: {models['polynomial regression of 2 degree'][0].predict(predictors_test_polyfeat)[0]}")