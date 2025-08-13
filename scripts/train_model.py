from catboost import Pool

def catboost_pools(x_train, x_val, x_test, y_train, y_val, y_test, categorical_cols):
    """
    Cria pools do CatBoost

    Arguments:
        x_train, x_val, x_test: DataFrames de features
        y_train, y_val, y_test: Series de targets
        categorical_cols: Lista de colunas categóricas

    Returns:
        tuple: train_pool, val_pool, test_pool
    """

    logger.info(f"A criar as pools do CatBoost")

    train_pool = Pool(x_train, y_train, cat_features=categorical_cols)
    val_pool = Pool(x_val, y_val, cat_features=categorical_cols)
    test_pool = Pool(x_test, y_test, cat_features=categorical_cols)

    logger.info(f"Pools criadas sem problemas!")
    return train_pool, val_pool, test_pool