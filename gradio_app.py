import os
import pandas as pd
import plotly.express as px
import gradio as gr

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'final_scored_customers.csv')
df = pd.read_csv(data_path)

# Metrics calculation
total_rev = f"${df['monetary_value'].sum():,.2f}"
churn_rate = f"{(df['is_churned'].mean()) * 100:.1f}%"
high_risk_count = f"{len(df[df['risk_level'] == 'High Risk']):,}"
at_risk_rev_val = df[df['risk_level'] == 'High Risk']['monetary_value'].sum()
at_risk_rev = f"${at_risk_rev_val:,.2f}"

def plot_pie():
    fig = px.pie(
        df, names='risk_level', color='risk_level',
        title="Customer Distribution by Risk Tier",
        color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'},
        hole=0.4
    )
    return fig

def plot_scatter():
    fig = px.scatter(
        df, x='customer_tenure_days', y='monetary_value',
        color='risk_level', size='frequency',
        title="Total Spend vs. Customer Tenure",
        hover_data=['CustomerID'],
        color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
    )
    return fig

def run_simulator(discount, conversion):
    saved_rev = at_risk_rev_val * (conversion / 100)
    campaign_cost = saved_rev * (discount / 100)
    net_profit = saved_rev - campaign_cost
    return f"${net_profit:,.2f}"

def get_high_risk_table():
    return df[df['risk_level'] == 'High Risk'].sort_values('monetary_value', ascending=False)

# Build Gradio Interface
with gr.Blocks(title="Retail Churn Intelligence") as demo:
    gr.Markdown("# 🛒 Retail Intelligence & Churn Prediction System")
    gr.Markdown("### End-to-End Customer Risk Analytics Dashboard")
    
    with gr.Row():
        gr.Textbox(label="Total Revenue", value=total_rev, interactive=False)
        gr.Textbox(label="Overall Churn Rate", value=churn_rate, interactive=False)
        gr.Textbox(label="High Risk Customers", value=high_risk_count, interactive=False)
        gr.Textbox(label="Revenue at Risk", value=at_risk_rev, interactive=False)

    with gr.Row():
        gr.Plot(value=plot_pie())
        gr.Plot(value=plot_scatter())

    gr.Markdown("### 💡 What-If Retention Campaign Simulator")
    with gr.Row():
        discount_slider = gr.Slider(5, 30, value=15, step=1, label="Targeted Discount Offer (%)")
        conversion_slider = gr.Slider(10, 50, value=25, step=1, label="Expected Retention Success Rate (%)")
        net_savings_output = gr.Textbox(label="Projected Net Campaign Savings", value=run_simulator(15, 25), interactive=False)
    
    discount_slider.change(run_simulator, inputs=[discount_slider, conversion_slider], outputs=net_savings_output)
    conversion_slider.change(run_simulator, inputs=[discount_slider, conversion_slider], outputs=net_savings_output)

    gr.Markdown("### 🚨 High Risk Customer Action List")
    gr.Dataframe(value=get_high_risk_table())

demo.launch(inbrowser=True)