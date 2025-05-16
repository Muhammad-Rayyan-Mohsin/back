import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform, randint
import warnings
import json
import io
import base64
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI application
app = FastAPI(
    title="Marketing CPI Forecast API",
    description="API for forecasting CPI metrics with business-friendly insights",
    version="1.0.0"
)

# Add CORS middleware with enhanced configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"],  # Expose all headers to the browser
    max_age=86400  # Cache preflight requests for 24 hours
)

# Set random seed for reproducibility
np.random.seed(42)

# Define available datasets
DATASETS = {
    "cards": {
        "file_path": "SN003-Cards_AM_DE_B_IMG.csv",
        "display_name": "Cards"
    },
    "cutelook": {
        "file_path": "SN029-CuteLook_AM_DE_G_IMG.csv", 
        "display_name": "CuteLook"
    }
}

# Model and forecast data storage for each dataset
MODEL_DATA = {dataset_id: {
    "model": None,
    "scaler": None,
    "important_features": None,
    "forecast": None,
    "last_trained": None,
    "rmse": None
} for dataset_id in DATASETS}

# Pydantic models for API
class ChartData(BaseModel):
    image_base64: str
    chart_data: Dict[str, Any]

class ForecastResponse(BaseModel):
    summary: Dict[str, Any]
    daily_forecast: List[Dict[str, Any]]
    day_of_week_insights: List[Dict[str, Any]]
    recommendations: List[str]
    charts: Dict[str, ChartData]

# Enhanced feature engineering for time series
def create_features(df):
    """Creates time series features from datetime index"""
    df = df.copy()
    # Basic date features
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day'] = df.index.day
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    df['is_weekend'] = df.index.dayofweek >= 5
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    
    # Lag features with different time windows
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f'lag_{lag}'] = df['cpi'].shift(lag)
    
    # Rolling window features with multiple windows
    for window in [3, 7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['cpi'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['cpi'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['cpi'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['cpi'].rolling(window=window).max()
        df[f'rolling_median_{window}'] = df['cpi'].rolling(window=window).median()
    
    # Expanding window features
    df['expanding_mean'] = df['cpi'].expanding().mean()
    df['expanding_std'] = df['cpi'].expanding().std()
    
    # Cyclical features
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofweek/7)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofweek/7)
    df['sin_month'] = np.sin(2 * np.pi * df.index.month/12)
    df['cos_month'] = np.cos(2 * np.pi * df.index.month/12)
    df['sin_quarter'] = np.sin(2 * np.pi * df.index.quarter/4)
    df['cos_quarter'] = np.cos(2 * np.pi * df.index.quarter/4)
    
    # Add differencing features
    df['diff_1'] = df['cpi'].diff(1)
    df['diff_7'] = df['cpi'].diff(7)
    
    # Add percentage change features
    df['pct_change_1'] = df['cpi'].pct_change(1)
    df['pct_change_7'] = df['cpi'].pct_change(7)
    
    # Growth rate features
    df['growth_rate'] = df['cpi'].pct_change() * 100
    df['growth_acceleration'] = df['growth_rate'].diff()
    
    # Enhanced volatility features to better capture price movements
    df['volatility_7'] = df['cpi'].rolling(7).std() / df['cpi'].rolling(7).mean()
    df['volatility_14'] = df['cpi'].rolling(14).std() / df['cpi'].rolling(14).mean()
    df['volatility_diff'] = df['volatility_7'] - df['volatility_14']
    
    # Recent momentum features
    df['momentum_3'] = df['cpi'].pct_change(3) * 100
    df['momentum_5'] = df['cpi'].pct_change(5) * 100
    df['momentum_7'] = df['cpi'].pct_change(7) * 100
    
    # Add day of week interaction with volatility
    for day in range(7):
        df[f'day_{day}_effect'] = (df['dayofweek'] == day).astype(int) * df['volatility_7']
    
    return df

def preprocess_data(dataset_id="cards"):
    """Preprocess the CPI data for a specific dataset"""
    # Get the file path for the selected dataset
    data_path = DATASETS[dataset_id]["file_path"]
    
    # Load data
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    
    # Sort by date
    data = data.sort_index()
    
    # Handle missing/zero values
    data.loc[data['cpi'] == 0, 'cpi'] = np.nan
    data['cpi'] = data['cpi'].fillna(data['cpi'].rolling(window=7, min_periods=1).mean())
    
    # Handle outliers using IQR method
    Q1 = data['cpi'].quantile(0.25)
    Q3 = data['cpi'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    data['cpi_original'] = data['cpi'].copy()
    data.loc[data['cpi'] < lower_bound, 'cpi'] = lower_bound
    data.loc[data['cpi'] > upper_bound, 'cpi'] = upper_bound
    
    return data

def train_model(data, dataset_id="cards"):
    """Train the CPI forecast model for a specific dataset"""
    # Create features
    data_features = create_features(data)
    data_features = data_features.dropna()
    
    # Calculate historical volatility to guide model parameters
    historical_volatility = data['cpi'].std() / data['cpi'].mean()
    
    # Split into train/validation/test
    test_size = int(0.1 * len(data_features))
    validation_size = int(0.1 * len(data_features))
    train_size = len(data_features) - test_size - validation_size
    
    train_df = data_features.iloc[:train_size]
    validation_df = data_features.iloc[train_size:train_size+validation_size]
    test_df = data_features.iloc[train_size+validation_size:]
    
    # Define feature columns
    feature_cols = [col for col in data_features.columns if col != 'cpi' and col != 'cpi_original']
    
    # Feature selection
    simple_model = XGBRegressor(n_estimators=100, random_state=42)
    simple_model.fit(train_df[feature_cols], train_df['cpi'])
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': simple_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top features (top 70%)
    top_features_threshold = 0.7 * feature_importance['importance'].sum()
    cum_importance = 0
    important_features = []
    
    for i, feature in enumerate(feature_importance['feature']):
        cum_importance += feature_importance['importance'].iloc[i]
        important_features.append(feature)
        if cum_importance >= top_features_threshold:
            break
    
    # Scale features
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    validation_scaled = validation_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[important_features] = scaler.fit_transform(train_df[important_features])
    validation_scaled[important_features] = scaler.transform(validation_df[important_features])
    test_scaled[important_features] = scaler.transform(test_df[important_features])
    
    # Train with parameters adjusted based on volatility
    if historical_volatility > 0.2:  # High volatility dataset
        best_params = {
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 2,
            'n_estimators': 250,
            'subsample': 0.85,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }
    else:  # More stable dataset
        best_params = {
            'colsample_bytree': 0.9,
            'gamma': 0.2,
            'learning_rate': 0.03,
            'max_depth': 4,
            'min_child_weight': 3,
            'n_estimators': 200,
            'subsample': 0.95,
            'random_state': 42
        }
    
    final_model = XGBRegressor(**best_params)
    final_model.fit(train_scaled[important_features], train_scaled['cpi'])
    
    # Evaluate model on test set
    predictions = final_model.predict(test_scaled[important_features])
    rmse = np.sqrt(mean_squared_error(test_df['cpi'], predictions))
    
    # Store model data
    MODEL_DATA[dataset_id]["model"] = final_model
    MODEL_DATA[dataset_id]["scaler"] = scaler
    MODEL_DATA[dataset_id]["important_features"] = important_features
    MODEL_DATA[dataset_id]["last_trained"] = datetime.now()
    MODEL_DATA[dataset_id]["rmse"] = rmse
    
    return final_model, scaler, important_features, rmse

def generate_forecast(data, dataset_id="cards", days=14):
    """Generate forecast for the specified number of days for a specific dataset"""
    if MODEL_DATA[dataset_id]["model"] is None:
        train_model(data, dataset_id)
    
    final_model = MODEL_DATA[dataset_id]["model"]
    scaler = MODEL_DATA[dataset_id]["scaler"]
    important_features = MODEL_DATA[dataset_id]["important_features"]
    rmse = MODEL_DATA[dataset_id]["rmse"]
    
    # Calculate historical volatility to guide forecast volatility
    historical_volatility = data['cpi'].std() / data['cpi'].mean()
    
    # Determine volatility factor based on historical patterns
    volatility_factor = max(0.8, min(2.5, historical_volatility * 3))  # Cap maximum volatility
    
    # Create future dataframe
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    future_df = pd.DataFrame(index=future_dates, columns=['cpi'])
    future_df['cpi'] = np.nan
    
    # Get historical volatility pattern by day of week
    data['dayofweek'] = data.index.dayofweek
    day_volatility = data.groupby('dayofweek')['cpi'].std() / data.groupby('dayofweek')['cpi'].mean()
    
    # Calculate historical day-to-day changes
    historical_changes = data['cpi'].pct_change().dropna()
    historical_std = historical_changes.std()
    
    # Calculate historical min, max and mean for bounds checking
    hist_min = data['cpi'].min()
    hist_max = data['cpi'].max()
    hist_mean = data['cpi'].mean()
    hist_std = data['cpi'].std()
    
    # Calculate reasonable bounds for predictions
    # Allow the forecast to go 2 standard deviations outside the historical range
    lower_bound = max(0, hist_min - 0.5 * hist_std)
    upper_bound = hist_max + 2 * hist_std
    
    # Recursively generate forecasts
    forecast_df = data.copy()
    
    for i, future_date in enumerate(future_dates):
        # Create temporary dataframe with all historical data plus future dates up to current prediction date
        temp_df = pd.concat([forecast_df, future_df.loc[:future_date]])
        
        # Create features
        temp_features = create_features(temp_df)
        
        # Get the last row (which corresponds to the current future date)
        last_row = temp_features.iloc[-1:].copy()
        
        # If there are any NaN values in the features, fill with last valid value
        last_row = last_row.fillna(temp_features.iloc[-2])
        
        # Scale the features
        last_row_features = last_row[important_features].copy()
        last_row_scaled = scaler.transform(last_row_features)
        last_row_scaled_df = pd.DataFrame(last_row_scaled, columns=important_features, index=last_row.index)
        
        # Base prediction
        base_prediction = final_model.predict(last_row_scaled_df[important_features])[0]
        
        # Add day-of-week specific volatility
        day_of_week = future_date.dayofweek
        day_vol_factor = day_volatility.get(day_of_week, 1.0)
        
        # Calculate a reasonable volatility factor - more controlled than before
        # Scale by the historical patterns but cap the maximum volatility
        effective_volatility = min(0.3, historical_std * day_vol_factor * volatility_factor)
        
        # If we're on second day or later in the forecast, incorporate volatility
        if i > 0:
            # Add some day-to-day variation based on historical patterns
            # More volatile for days historically showing high volatility
            noise = np.random.normal(0, effective_volatility)
            prediction = base_prediction * (1 + noise)
            
            # Add some trend continuation (momentum) from the previous day
            previous_value = future_df.iloc[i-1]['cpi']
            momentum_factor = 0.3  # How much momentum affects the prediction
            prediction = (1 - momentum_factor) * prediction + momentum_factor * previous_value
        else:
            # For the first day, start close to the last known value
            last_known = forecast_df['cpi'].iloc[-1]
            prediction = 0.7 * base_prediction + 0.3 * last_known
            
        # Ensure prediction stays within reasonable bounds
        prediction = max(lower_bound, min(upper_bound, prediction))
        
        # Update future_df with prediction
        future_df.loc[future_date, 'cpi'] = prediction
        
        # Also update forecast_df to use for the next step's features
        forecast_df = pd.concat([forecast_df, pd.DataFrame({'cpi': [prediction]}, index=[future_date])])
    
    # Store forecast
    MODEL_DATA[dataset_id]["forecast"] = future_df
    
    # Calculate adjusted RMSE for confidence intervals
    rmse_adjustment = 1.2 if historical_volatility > 0.2 else 1.0
    adjusted_rmse = rmse * rmse_adjustment
    
    return future_df, adjusted_rmse

def create_visualization(data, forecast_df, rmse, dataset_id="cards"):
    """Create visualizations and return as base64 encoded strings and data"""
    charts = {}
    
    # 1. Create forecast chart
    fig1 = plt.figure(figsize=(10, 6))
    
    # Get the data for historical and forecast
    historical_dates = data.index[-30:].strftime('%Y-%m-%d').tolist()
    historical_values = data['cpi'][-30:].tolist()
    forecast_dates = forecast_df.index.strftime('%Y-%m-%d').tolist()
    forecast_values = forecast_df['cpi'].tolist()
    
    # Ensure lower bounds are never negative
    lower_bounds = [max(0, val - 1.96 * rmse) for val in forecast_df['cpi']]
    upper_bounds = [val + 1.96 * rmse for val in forecast_df['cpi']]
    
    # Plot data
    plt.plot(data.index[-30:], data['cpi'][-30:], label='Historical CPI', linewidth=2, color='#3366CC')
    plt.plot(forecast_df.index, forecast_df['cpi'], label='Forecasted CPI', 
             color='#DC3912', linestyle='--', marker='o', linewidth=2)
    
    # Add confidence interval with protection against negative values
    plt.fill_between(forecast_df.index, 
                     np.maximum(0, forecast_df['cpi'] - 1.96 * rmse), 
                     forecast_df['cpi'] + 1.96 * rmse, 
                     color='#DC3912', alpha=0.15, label='95% Confidence Range')
    
    display_name = DATASETS[dataset_id]["display_name"]
    plt.title(f'14-Day CPI Forecast for {display_name}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('CPI ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    forecast_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig1)
    
    # Store chart data for JSON
    forecast_chart_data = {
        "historical": {
            "dates": historical_dates,
            "values": historical_values
        },
        "forecast": {
            "dates": forecast_dates,
            "values": forecast_values,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds
        }
    }
    
    # 2. Create weekly pattern chart
    fig2 = plt.figure(figsize=(10, 6))
    
    weekday_forecast = pd.DataFrame({
        'date': forecast_df.index,
        'forecasted_cpi': forecast_df['cpi'].values,
        'day_of_week': forecast_df.index.day_name()
    })
    
    weekday_means = weekday_forecast.groupby('day_of_week')['forecasted_cpi'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    # Get data for JSON
    days = weekday_means.index.tolist()
    values = weekday_means.values.tolist()
    weekly_avg = weekday_means.mean()
    
    # Create the plot
    bars = plt.bar(weekday_means.index, weekday_means, color='#3366CC')
    plt.axhline(y=weekday_means.mean(), color='#DC3912', linestyle='--', label='Weekly Average')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'${height:.2f}', ha='center', va='bottom')
    
    plt.title(f'Expected CPI by Day of Week for {display_name}', fontsize=14)
    plt.ylabel('CPI ($)', fontsize=12)
    plt.ylim(top=weekday_means.max()*1.15)
    plt.legend()
    plt.tight_layout()
    
    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    weekly_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig2)
    
    # Store chart data for JSON
    weekly_chart_data = {
        "days": days,
        "values": values,
        "weekly_average": weekly_avg
    }
    
    # Store both charts
    charts["forecast_chart"] = ChartData(
        image_base64=forecast_image,
        chart_data=forecast_chart_data
    )
    
    charts["weekly_pattern"] = ChartData(
        image_base64=weekly_image,
        chart_data=weekly_chart_data
    )
    
    return charts

def generate_marketing_insights(data, forecast_df, rmse, dataset_id="cards"):
    """Generate marketing-friendly insights from the forecast"""
    # Create summary statistics
    avg_cpi = forecast_df['cpi'].mean()
    best_day_idx = forecast_df['cpi'].idxmin()  # Lower CPI is better
    worst_day_idx = forecast_df['cpi'].idxmax()  # Higher CPI is worse
    recent_avg = data['cpi'][-7:].mean()
    forecast_avg = forecast_df['cpi'].mean()
    percent_change = ((forecast_avg - recent_avg) / recent_avg) * 100
    
    # Weekly pattern analysis
    weekday_forecast = pd.DataFrame({
        'date': forecast_df.index,
        'forecasted_cpi': forecast_df['cpi'].values,
        'day_of_week': forecast_df.index.day_name()
    })
    
    weekday_means = weekday_forecast.groupby('day_of_week')['forecasted_cpi'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    # Historical day of week patterns
    data['day_of_week'] = data.index.day_name()
    historical_dow = data.groupby('day_of_week')['cpi'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    # Create daily forecast data
    daily_forecast = []
    for date, cpi in forecast_df['cpi'].items():
        lower_bound = max(0, cpi - 1.96 * rmse)
        upper_bound = cpi + 1.96 * rmse
        
        daily_forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "day_of_week": date.day_name(),
            "forecasted_cpi": round(cpi, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "performance": "Above Average" if cpi > avg_cpi else "Below Average"
        })
    
    # Create day of week insights
    dow_insights = []
    for day, value in weekday_means.items():
        historical = historical_dow[day]
        change = ((value - historical) / historical) * 100
        dow_insights.append({
            "day": day,
            "forecasted_cpi": round(value, 2),
            "historical_cpi": round(historical, 2),
            "percent_change": round(change, 1),
            "performance": "Above Average Cost" if value > weekday_means.mean() else "Below Average Cost"
        })
    
    display_name = DATASETS[dataset_id]["display_name"]
    
    # Generate recommendations
    recommendations = []
    if percent_change > 0:
        recommendations.append(f"{display_name} CPI is forecasted to increase by {percent_change:.1f}% - consider optimizing campaigns or targeting to reduce costs")
    else:
        recommendations.append(f"{display_name} CPI is forecasted to decrease by {abs(percent_change):.1f}% - opportunity to scale campaigns for better ROI")
    
    recommendations.append(f"Most cost-effective day for {display_name} ads: {weekday_means.idxmin()} (${weekday_means.min():.2f})")
    recommendations.append(f"Consider reducing {display_name} ad spend on {weekday_means.idxmax()} (${weekday_means.max():.2f})")
    
    # Create low-risk recommendations
    forecasted_cpi_df = pd.DataFrame(daily_forecast)
    if not forecasted_cpi_df.empty:
        low_risk_days = forecasted_cpi_df[forecasted_cpi_df['upper_bound'] < forecasted_cpi_df['forecasted_cpi'].mean()]
        if not low_risk_days.empty:
            best_low_risk_day = low_risk_days.iloc[low_risk_days['forecasted_cpi'].argmin()]
            recommendations.append(f"Safest day for guaranteed lower {display_name} CPI: {best_low_risk_day['day_of_week']} ({best_low_risk_day['date']})")
    
    # Create summary object
    summary = {
        "dataset": display_name,
        "average_forecast_cpi": round(forecast_avg, 2),
        "recent_average_cpi": round(recent_avg, 2),
        "percent_change": round(percent_change, 1),
        "lowest_cpi_day": {
            "date": best_day_idx.strftime("%Y-%m-%d"),
            "day": best_day_idx.day_name(),
            "cpi": round(forecast_df.loc[best_day_idx, 'cpi'], 2)
        },
        "highest_cpi_day": {
            "date": worst_day_idx.strftime("%Y-%m-%d"),
            "day": worst_day_idx.day_name(),
            "cpi": round(forecast_df.loc[worst_day_idx, 'cpi'], 2)
        },
        "forecast_uncertainty": round(rmse * 1.96, 2),
        "forecast_period": {
            "start_date": forecast_df.index.min().strftime("%Y-%m-%d"),
            "end_date": forecast_df.index.max().strftime("%Y-%m-%d")
        }
    }
    
    return summary, daily_forecast, dow_insights, recommendations

@app.get("/", tags=["Info"])
async def root():
    """API root with basic information"""
    available_endpoints = {f"/forecast/{dataset_id}": f"Get CPI forecast for {DATASETS[dataset_id]['display_name']}" 
                          for dataset_id in DATASETS}
    
    return {
        "name": "Marketing CPI Forecast API",
        "version": "1.0.0",
        "description": "API for forecasting CPI metrics with business-friendly insights",
        "available_datasets": list(DATASETS.keys()),
        "endpoints": available_endpoints
    }

@app.get("/forecast/{dataset_id}", response_model=ForecastResponse, tags=["Forecast"])
async def get_forecast(background_tasks: BackgroundTasks, dataset_id: str, days: int = 14):
    """Get CPI forecast with marketing insights for a specific dataset"""
    # Validate dataset_id
    if dataset_id not in DATASETS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Dataset '{dataset_id}' not found. Available datasets: {list(DATASETS.keys())}"}
        )
    
    # Load and preprocess data
    data = preprocess_data(dataset_id)
    
    # Check if model needs training
    if MODEL_DATA[dataset_id]["model"] is None or MODEL_DATA[dataset_id]["forecast"] is None:
        final_model, scaler, important_features, rmse = train_model(data, dataset_id)
        forecast_df, rmse = generate_forecast(data, dataset_id, days=days)
    else:
        # Use existing model
        forecast_df, rmse = generate_forecast(data, dataset_id, days=days)
    
    # Create visualizations and get chart data
    charts = create_visualization(data, forecast_df, rmse, dataset_id)
    
    # Generate insights
    summary, daily_forecast, dow_insights, recommendations = generate_marketing_insights(data, forecast_df, rmse, dataset_id)
    
    # Create response object using Pydantic model for validation
    response_obj = ForecastResponse(
        summary=summary,
        daily_forecast=daily_forecast,
        day_of_week_insights=dow_insights,
        recommendations=recommendations,
        charts=charts
    )
    
    # Convert the Pydantic model to a dictionary before returning as JSONResponse
    response_dict = response_obj.dict()
    
    # Return response with JSONResponse to have more control over headers
    return JSONResponse(
        content=response_dict,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.options("/forecast/{dataset_id}", include_in_schema=False)
async def options_forecast(dataset_id: str):
    """Handle preflight requests for the forecast endpoint"""
    return JSONResponse(
        content={"message": "Accepted"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",  # 24 hours in seconds
        }
    )

if __name__ == "__main__":
    uvicorn.run("cpi_forecast_api:app", host="0.0.0.0", port=8000, reload=True)