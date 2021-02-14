import xgboost as xgb


class XGBoost(xgb.XGBRegressor):
    def __init__(self, max_depth):
        super().__init__(max_depth=max_depth, learning_rate=0.1, silent=True, objective='reg:squarederror', nthread=4)