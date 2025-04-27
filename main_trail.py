import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Layoff Forecast Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Layoffs.fyi-1128.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date Added'] = pd.to_datetime(df['Date Added'])
    df['Layoff_Percentage'] = df['%'].str.replace('%', '', regex=False).astype(float)
    df['Raised_Millions'] = df['$ Raised (mm)'].replace('[\$,]', '', regex=True).replace('', np.nan).astype(float)
    df['Laid_Off'] = pd.to_numeric(df['# Laid Off'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    return df

def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained_models[name] = model
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })
    return pd.DataFrame(results), trained_models

def plot_forecast(forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name="Forecast", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], name="Upper Bound", line=dict(dash='dot', color='lightblue')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], name="Lower Bound", line=dict(dash='dot', color='lightblue')))
    fig.update_layout(title="Layoffs Forecast", xaxis_title="Date", yaxis_title="Layoffs")
    return fig

def plot_model_comparison(df_results):
    fig = go.Figure()
    for metric in ['MAE', 'RMSE', 'R2']:
        fig.add_trace(go.Bar(x=df_results['Model'], y=df_results[metric], name=metric))
    fig.update_layout(barmode='group', title="Model Performance Comparison")
    return fig

# Sidebar navigation
with st.sidebar:
    page = option_menu(
        menu_title="\U0001F4CA Layoff Forecast Dashboard",
        options=["Overview", "Company Data", "Trends", "EDA", "Forecast", "Model Comparison"],
        icons=["house", "building", "bar-chart-line", "pie-chart", "graph-up-arrow", "activity"],
        default_index=0
    )

# Load data
df = load_data()

industry_filter = st.sidebar.multiselect("Filter by Industry", options=sorted(df['Industry'].dropna().unique()))
country_filter = st.sidebar.multiselect("Filter by Country", options=sorted(df['Country'].dropna().unique()))

# Session state setup
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()
    st.session_state.forecast = None
    st.session_state.model_results = None
    st.session_state.trained_models = {}
    st.session_state.selected_model_name = "Random Forest"

if st.sidebar.button("Run Analysis"):
    filtered_df = df.copy()
    if industry_filter:
        filtered_df = filtered_df[filtered_df['Industry'].isin(industry_filter)]
    if country_filter:
        filtered_df = filtered_df[filtered_df['Country'].isin(country_filter)]

    if not filtered_df.empty:
        model_df = filtered_df.dropna(subset=['Layoff_Percentage'])
        X = model_df[['Industry', 'Stage', 'Country', 'Raised_Millions', 'Laid_Off', 'Year', 'Month', 'Quarter']]
        y = model_df['Layoff_Percentage']

        for col in ['Industry', 'Stage', 'Country']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        for col in ['Raised_Millions', 'Laid_Off']:
            X[col] = X[col].fillna(X[col].median())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_results, trained_models = evaluate_models(X_train, X_test, y_train, y_test)

        best_model = model_results.sort_values("RMSE").iloc[0]['Model']
        st.session_state.selected_model_name = best_model

        selected_model = trained_models[best_model]
        predicted_percent = selected_model.predict(X)
        risk_score = 10 - np.clip(predicted_percent / 10, 0, 10)
        filtered_df['Layoff_Risk_Score'] = np.nan
        filtered_df.loc[model_df.index, 'Layoff_Risk_Score'] = risk_score.round(1)

        time_df = filtered_df.dropna(subset=['Laid_Off', 'Date'])
        time_series = time_df.groupby('Date').agg({'Laid_Off': 'sum'}).resample('D').sum().fillna(0).reset_index()
        time_series.columns = ['ds', 'y']
        prophet = Prophet()
        prophet.fit(time_series)
        future = prophet.make_future_dataframe(periods=90)
        forecast = prophet.predict(future)

        st.session_state.filtered_df = filtered_df
        st.session_state.forecast = forecast
        st.session_state.model_results = model_results
        st.session_state.trained_models = trained_models
    else:
        st.warning("No data available after filtering.")

# Page Routing
if page == "Overview":
    st.title("Overview")
    st.markdown("This dashboard analyzes layoffs and predicts future trends.")
    st.markdown("""
    Analyzing corporate layoffs is traditionally time-consuming due to scattered data and inconsistent reporting across industries. 
    To address this challenge, we developed an AI-driven layoff risk prediction and visualization system called **Job Shield**.

    Our system processes historical layoff data, financial indicators, and industry trends to predict layoff risk scores for companies. 
    Using machine learning models such as **Random Forest**, **Gradient Boosting**, **Linear Regression**, and **XGBoost**, 
    we trained and validated predictive models that assess key factors like layoff frequency, funding stability, and workforce reductions across different sectors.

    The results are visualized through an interactive **Streamlit dashboard**, allowing users to:
    - Filter by industry and country,
    - Analyze real-time risk predictions,
    - Explore historical layoff trends,
    - Compare model performance, and
    - View comprehensive forecasts of future layoffs.

    Through exploratory data analysis (EDA) and predictive modeling, our dashboard highlights significant variations in layoff patterns across countries, industries, and time periods. 
    This empowers job seekers, professionals, and analysts to make more informed career and business decisions.

    By integrating machine learning, interactive forecasting (using Prophet), and dynamic visualizations, 
    our platform simplifies layoff risk assessment, enhances workforce trend transparency, and promotes data-driven decision-making.
    """)

    # Summary Visuals - Overview Page

    st.subheader("📊 Data Summary Visualizations")

    col1, col2 = st.columns(2)

# 1. Top 5 Industries by Total Layoffs (3D Pie Chart)
    top_industries = df.groupby('Industry')['Laid_Off'].sum().sort_values(ascending=False).head(5).reset_index()

    fig_industry_pie = go.Figure(
        data=[go.Pie(
            labels=top_industries['Industry'],
            values=top_industries['Laid_Off'],
            hole=0.3,
            pull=[0.05]*5,
        )]
    )

    fig_industry_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
    fig_industry_pie.update_layout(
        title_text="Top 5 Industries by Layoffs",
        showlegend=True,
        height=400,
        width=400
    )

    col1.plotly_chart(fig_industry_pie, use_container_width=True)

# 2. Top 5 Countries by Total Layoffs (3D Pie Chart)
    top_countries = df.groupby('Country')['Laid_Off'].sum().sort_values(ascending=False).head(5).reset_index()

    fig_country_pie = go.Figure(
        data=[go.Pie(
            labels=top_countries['Country'],
            values=top_countries['Laid_Off'],
            hole=0.3,
            pull=[0.05]*5,
        )]
    )

    fig_country_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
    fig_country_pie.update_layout(
        title_text="Top 5 Countries by Layoffs",
        showlegend=True,
        height=400,
        width=400
    )

    col2.plotly_chart(fig_country_pie, use_container_width=True)

    
    total_layoffs = df['Laid_Off'].sum(skipna=True)

    if not st.session_state.filtered_df.empty:
        filtered_layoffs = st.session_state.filtered_df['Laid_Off'].sum(skipna=True)
        layoff_percentage = (filtered_layoffs / total_layoffs) * 100
        layoff_percentage = round(layoff_percentage, 2)
    else:
        filtered_layoffs = 0
        layoff_percentage = 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Layoffs", f"{int(total_layoffs):,}")
    col2.metric("Total Layoffs based on selections", f"{int(filtered_layoffs):,}")
    col3.metric("Layoff %", f"{layoff_percentage}%")

    st.metric("Companies Affected", df['Company'].nunique())


elif page == "Company Data" and not st.session_state.filtered_df.empty:
    st.title("Company Layoff Details")
    st.dataframe(st.session_state.filtered_df[['Company', 'Industry', 'Country', 'Raised_Millions', 'Laid_Off',
                                               'Layoff_Percentage', 'Layoff_Risk_Score']].dropna())

elif page == "Trends" and not st.session_state.filtered_df.empty:
    st.title("Layoff Trends")
    by_industry = st.session_state.filtered_df.groupby('Industry')['Laid_Off'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(by_industry, x='Industry', y='Laid_Off'), use_container_width=True)
    by_country = st.session_state.filtered_df.groupby('Country')['Laid_Off'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(by_country, x='Country', y='Laid_Off'), use_container_width=True)

elif page == "EDA" and not st.session_state.filtered_df.empty:
    st.title("Exploratory Data Analysis (EDA)")

    # 1. Boxplot: Laid Off across Industries
    st.header("Distribution of Layoffs Across Industries")
    fig_box = px.box(
        st.session_state.filtered_df, 
        x='Industry', 
        y='Laid_Off', 
        title="Layoffs by Industry",
        labels={"Industry": "Industry", "Laid_Off": "Number of Employees Laid Off"},
        color='Industry'
    )
    fig_box.update_layout(xaxis_title="Industry", yaxis_title="Number of Layoffs", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # 2. Scatter Plot: Fund Raised vs. Layoffs
    st.header("Relation between Funds Raised and Layoffs")
    fig_scatter = px.scatter(
        st.session_state.filtered_df,
        x='Raised_Millions',
        y='Laid_Off',
        color='Stage',
        title="Funding vs Layoffs",
        labels={"Raised_Millions": "Funds Raised (in Millions USD)", "Laid_Off": "Number of Employees Laid Off", "Stage": "Company Stage"},
        trendline="ols"
    )
    fig_scatter.update_layout(xaxis_title="Funds Raised (Millions USD)", yaxis_title="Number of Layoffs")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Histogram: Layoff Percentage
    st.header("Distribution of Layoff Percentage")
    fig_hist = px.histogram(
        st.session_state.filtered_df,
        x='Layoff_Percentage',
        nbins=30,
        title="Layoff Percentage Distribution",
        labels={"Layoff_Percentage": "Layoff Percentage (%)"},
        color_discrete_sequence=['indianred']
    )
    fig_hist.update_layout(xaxis_title="Layoff Percentage (%)", yaxis_title="Number of Companies")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 4. Violin Plot: Layoff Percentage by Country
    st.header("Layoff Percentage Across Countries")
    fig_violin = px.violin(
        st.session_state.filtered_df,
        y='Layoff_Percentage',
        x='Country',
        box=True,
        points="all",
        title="Layoff Percentage by Country",
        labels={"Country": "Country", "Layoff_Percentage": "Layoff Percentage (%)"},
        color='Country'
    )
    fig_violin.update_layout(xaxis_title="Country", yaxis_title="Layoff Percentage (%)", showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True)

    # --- New Charts ---

    # 5. Line Chart: Timeline of Layoffs
    st.header("Layoffs Over Time")
    timeline_df = st.session_state.filtered_df.dropna(subset=['Date', 'Laid_Off'])
    layoffs_over_time = timeline_df.groupby('Date').agg({'Laid_Off': 'sum'}).reset_index()

    fig_time = px.line(
        layoffs_over_time,
        x='Date',
        y='Laid_Off',
        title="Layoffs Timeline",
        labels={"Date": "Date", "Laid_Off": "Number of Employees Laid Off"},
        markers=True
    )
    fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Layoffs", hovermode="x unified")
    st.plotly_chart(fig_time, use_container_width=True)

    # 6. Heatmap: Industry vs Country Layoffs
    st.header("Layoffs by Industry and Country")
    heatmap_df = st.session_state.filtered_df.pivot_table(
        index='Industry',
        columns='Country',
        values='Laid_Off',
        aggfunc='sum'
    ).fillna(0)

    fig_heatmap = px.imshow(
        heatmap_df,
        labels=dict(x="Country", y="Industry", color="Number of Layoffs"),
        title="Industry vs Country Layoffs Heatmap",
        color_continuous_scale='Reds'
    )
    fig_heatmap.update_layout(xaxis_title="Country", yaxis_title="Industry")
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == "Forecast" and st.session_state.forecast is not None:
    st.title("Layoff Forecast")
    st.plotly_chart(plot_forecast(st.session_state.forecast), use_container_width=True)
    st.plotly_chart(px.line(st.session_state.forecast, x='ds', y=['yhat', 'yhat_upper', 'yhat_lower'], title='Forecast Range'), use_container_width=True)
    st.plotly_chart(px.area(st.session_state.forecast, x='ds', y='yhat', title='Forecasted Layoffs Area'), use_container_width=True)
    st.plotly_chart(px.scatter(st.session_state.forecast, x='ds', y='yhat', title='Forecast Scatter'), use_container_width=True)
    st.download_button("Download Forecast CSV", st.session_state.forecast.to_csv(index=False), "forecast.csv")

elif page == "Model Comparison" and st.session_state.model_results is not None:
    st.title("Model Performance Comparison")
    st.dataframe(st.session_state.model_results)
    st.plotly_chart(plot_model_comparison(st.session_state.model_results), use_container_width=True)
    st.success(f"Recommended Model Based on RMSE: {st.session_state.selected_model_name}")
    st.session_state.selected_model_name = st.selectbox("Select Model to Use for Risk Scoring", st.session_state.model_results['Model'])