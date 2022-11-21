import joblib
import pandas as pd
import numpy as np

def predict(target, fee_tier, price_lower, price_upper, age):
    mod = joblib.load('mainnet-wbtc-weth-xgbmod-{}.joblib'.format(target))
    Xnew = pd.DataFrame({
        'fee_tier': fee_tier,
        'price_lower': price_lower, 'price_upper': price_upper,
        'price_rng_width': price_upper - price_lower, 'age': age
        }, index=range(1))
    pred = mod.predict(Xnew)
    if target == 'log1p_fee_apr':
        pred = np.exp(pred) - 1
    return pred
