# Machine Learning Predictions of ROI and Fee APR of Univ3 WBTC-WETH positions on Ethereum

Using historical Univ3 pool positions data obtained from Revert Finance, we built
XGBOOST regression models to predict ROI and Fee APR using fee tier, 
lower limit price, upper limit price, price range, and duration.

The dashboard compares the predictions with the actuals on a set of data that weren't
used at all during model training and selection via cross-validation. It also 
allows users to enter their own numbers and get back model predicted results. 

![screen](https://github.com/coindataschool/univ3-roi-prediction-wbtc-weth/blob/main/screen.png)

The models are limited by the data used for training. Use with caution.

[Dashboard Link](https://coindataschool-univ3-roi-prediction-wbtc-weth-main-oufzxi.streamlit.app/)
