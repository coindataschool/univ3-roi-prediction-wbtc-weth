import pandas as pd
from plot_preds_vs_actuals import plot_preds_vs_actuals
from mk_predictions import predict
import streamlit as st

# set_page_config() can only be called once per app, and must be called as the 
# first Streamlit command in your script.
st.set_page_config(page_title='ROI-Prediction-Univ3-WBTC-WETH', 
    layout='wide', page_icon='📈')

st.title('Machine Learning Predictions of ROI and Fee APR of UniV3 WBTC-WETH positions on Ethereum')

# plot actual vs. predictions on test set
st.header('Model performance on a set of data not used in training')
c1, c2 = st.columns(2)
with c1:
    fig = plot_preds_vs_actuals('roi')
    st.plotly_chart(fig, use_container_width=True)
    st.write('ROI is the total PnL divided by initial deposits. It accounts for gas and impermanent loss.')
with c2:
    fig = plot_preds_vs_actuals('fee_apr')
    st.plotly_chart(fig, use_container_width=True)
    st.write('Fee APR is the annualized rate of fees earned. It does NOT include gas or impermanent loss.')

st.markdown("""---""")

# allow user to input values and get predictions
st.header('Predict by entering your own values')

# user input
c1, c2, c3, c4 = st.columns(4)
with c1:
    fee_tier = st.selectbox("Select fee-tier:", ("0.05%", "0.3%"))
with c2:
    price_lower = st.number_input(
        "Set Price Range - Min", value=12.0, 
        min_value=5.0, max_value=17.0,
        step=0.001, format="%.3f")
with c3:
    price_upper = st.number_input(
        "Set Price Range - Max", value=14.0,
        # min_value=max(12.0, price_lower)+0.2, 
        min_value=12.0, 
        max_value=50.0,
        step=0.001, format="%.3f")
with c4:
    age = st.number_input(
        "How many days will you provide liquidity for?", value=7.0, 
        min_value=0.5, max_value=564.0, step=0.1, format="%.1f")

# predict using ML models
if price_upper <= price_lower:
    st.write('Make sure your upper price limit > lower price limit!')
else:
    row_pred = pd.DataFrame({
        'ROI': predict('roi', fee_tier, price_lower, price_upper, age),
        'Fee APR': predict('log1p_fee_apr', fee_tier, price_lower, price_upper, age), 
    }, index=['Prediction'])
    
    def color(s):
        # color the cell value of a data frame green if it's > 0 
        # and red otherwise
        res = []
        for val in s:
            if val > 0:
                res.append('color: #8db600') 
            else: 
                res.append('color: #ff2052')
        return res 
    st.header('Output')
    st.table(row_pred.style.apply(color, axis=1).format('{:.2%}')\
        .set_properties(**{'font-size': '25px'}))

st.markdown("""---""")

# method description
st.header('Methodology')
c1, c2 = st.columns(2)
with c1:
    st.subheader('Data')
    st.markdown("Ideally, we want to work with a dataset that has only\n1. positions that never had liquidity partially removed,\n2. positions that never had additional liquidity added,\n3. positions that ran its course and had 100% liquidity removed and no new liquidity added afterwards.\nA dataset of 1 and 2 was downloaded from Revert Finance on Nov 20, 2022 at 00:01:13 UTC. The data-pulling script was written by [0x1egolas](https://twitter.com/0x1egolas). Unfortunately, data of 3 cannot be easily obtained.")
    st.markdown("We then took a subset of data that meet all of the following criteria:\n - initial deposit value is between \$1,000 and \$1M US Dollars,\n- age of position is at least 0.5 days (or 12 hours),\n- lower BTCETH limit is set at least 5,\n- upper BTCETH limit is set at most 60.")
    st.markdown("Ranges of Important Variables on this subset:\n- Age ranges from 0.5 to 564.0 days,\n- Deposit ranges from 1005.9 to 925109.0 USD,\n- Price lower limit ranges from 5.0 to 17.0,\n- Price upper limit ranges from 12.0 to 50.0,\n- ROI ranges from -25.59% to 34.80%,\n- Fee APR ranges from 0.05% to 167.84%.")
    st.markdown("Finally, we split this subset into a training set (80%, 673 records) and a test set (20%, 169 records). We trained ML models on the training set with cross-validation (cv). Throughout the training, we did not touch the test set, nor did we once peek at model performance on the test set. We chose the best model according to cv error. Only after we finalized the model, we used it on the test set to get the predictions displayed at the top of this page.")
with c2:
    st.subheader('Machine Learning')
    st.markdown("Targets:\n- ROI \n- Fee APR")
    st.markdown("Features:\n- Fee Tier \n- Age \n- Price Lower Limit \n- Price Upper Limit \n- Price Range (Price Upper Limit - Price Lower Limit).")
    st.markdown("Transformations:\n- Applied `log1p()` to Fee APR, Age, Price Upper Limit and Price Range because they had long right tails.\n- Standardized Price Lower Limit.\n- One-Hot Encoded Fee Tier.")
    st.markdown("Model selection:\n- XGBOOST regressor was trained, where a single parameter, learning rate, was tuned for optimal performance via 5-fold cross-validation, with negative mean absolute error as the scoring metric.\n- The best model for ROI prediction had a CV error of 1.2%.\n- The best model for Fee APR prediction had a CV error of 3.05%.")
    st.markdown("Test Set Performance:\n- The best model for ROI prediction had a test error of 1.3%.\n- The best model for Fee APR prediction had a test error of 3.15%.")

st.markdown("""---""")
    
# about
c1, c2 = st.columns(2)
with c1:
    st.subheader('Code')
    st.markdown('- [Download data](https://github.com/coindataschool/univ3lp/blob/main/scripts/01-pull-data.py)')
    st.markdown('- [Prepare data](https://github.com/coindataschool/univ3lp/blob/main/scripts/02-prep-data.py)')
    st.markdown('- [Train Models](https://github.com/coindataschool/univ3lp/blob/main/scripts/03-model-roi-n-fee-apr.py)')
    st.markdown('- [This Dashboard](https://github.com/coindataschool/univ3-roi-prediction-wbtc-weth)')
with c2:
    st.subheader('Support my work')
    st.markdown("- Subscribe to my [newsletter](https://coindataschool.substack.com/about)")
    st.markdown("- Follow me on twitter: [@coindataschool](https://twitter.com/coindataschool)")
    st.markdown("- Follow me on github: [@coindataschool](https://github.com/coindataschool)")
    st.markdown("- Find me on Dune: [@coindataschool](https://dune.com/coindataschool)")    
    st.markdown("- Buy me a coffee with ETH: `0x783c5546c863f65481bd05fd0e3fd5f26724604e`")
    st.markdown("- [Tip me sat](https://tippin.me/@coindataschool)")