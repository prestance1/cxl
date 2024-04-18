from sklearn.linear_model import LinearRegression


def prepare_data():
    pass


future_X = None


def main():
    X, y = prepare_data()
    regressor = LinearRegression()
    regressor.fit(X, y)
    predictions = regressor.predict(future_X)


if __name__ == "__main__":
    main()
