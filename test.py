from ucimlrepo import fetch_ucirepo

def main():
    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets

    # metadata
    print(adult.metadata)

    # variable information
    print(adult.variables)

if __name__ == '__main__':
    main()