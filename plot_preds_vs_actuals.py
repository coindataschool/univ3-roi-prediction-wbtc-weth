import plotly.express as px
import pandas as pd

def plot_preds_vs_actuals(target):
    cleaned_target = target.replace('_', ' ').upper()
    df = pd.read_pickle('mainnet-wbtc-weth-xgbpred-{}.pkl'.format(target))
    df['actual_minus_prediction'] = df[target] - df['xgb_pred']
    fig = px.scatter(
        data_frame=df,
        x=target,
        y='xgb_pred',
        custom_data=['actual_minus_prediction'],
        color='actual_minus_prediction', 
        color_continuous_midpoint=0, 
        opacity=0.5, 
        color_continuous_scale='IceFire', 
        # trendline='lowess', 
        # trendline_color_override='black'
    ).update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor='#0E1117', 
        yaxis_tickformat = '.0%',
        xaxis_tickformat = '.0%',
        title_text=f'{cleaned_target} Predictions vs. Actuals',
        font=dict(size=18),
        autosize=False,
        width=600,
        height=600,        
    )
    fig.update_traces(
        marker=dict(size=15),
        hovertemplate="<br>".join([
        f'{cleaned_target}'+": %{x:.2%}",
        "Prediction: %{y:.2%}",
        "Error: %{customdata[0]:.2%}",])
    ) 
    fig.update_xaxes(showgrid=False, title_text=cleaned_target)
    fig.update_yaxes(showgrid=False, title_text='Prediction')
    return fig    
