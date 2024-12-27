from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

# Load models
sales_model = joblib.load('xgboost_sales_model.joblib')
understock_model = joblib.load('xgboost_understock_model.joblib')

# Extract feature names from models
sales_model_features = sales_model.get_booster().feature_names
understock_model_features = understock_model.get_booster().feature_names


def preprocess_and_predict_in_batches(df, batch_size=1000):
    def process_chunk(chunk):
        # Check required columns
        required_columns = ['date', 'store', 'item', 'sales', 'stock_on_hand', 'supplier_id', 'supplier_lead_time']
        missing_columns = [col for col in required_columns if col not in chunk.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Parse date column
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')

        # Add time-based features
        chunk['day'] = chunk['date'].dt.day
        chunk['month'] = chunk['date'].dt.month
        chunk['year'] = chunk['date'].dt.year
        chunk['day_of_week'] = chunk['date'].dt.dayofweek
        chunk['is_weekend'] = chunk['day_of_week'].isin([5, 6]).astype(int)

        # Calculate rolling averages
        for window in [5, 10, 15]:
            col_name = f'{window}_day_moving_avg'
            if 'sales' in chunk.columns:
                chunk[col_name] = (
                    chunk.groupby(['store', 'item'])['sales']
                    .rolling(window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
                )

        # Lag features
        for lag in [1, 7, 30]:
            lag_column = f'sales_lag_{lag}'
            if 'sales' in chunk.columns:
                chunk[lag_column] = chunk.groupby(['store', 'item'])['sales'].shift(lag)

        # Derived features
        if 'sales' in chunk.columns and 'stock_on_hand' in chunk.columns:
            chunk['stock_turnover_ratio'] = chunk['sales'] / (chunk['stock_on_hand'] + 1e-5)
            chunk['days_until_stockout'] = chunk['stock_on_hand'] / (chunk['5_day_moving_avg'] + 1e-5)

        chunk['days_since_last_restock'] = (
        chunk['date'] - chunk.groupby(['store', 'item'])['date'].shift()
            ).dt.days.fillna(0).astype('float32')
        chunk['stock_to_sales_ratio'] = chunk['stock_on_hand'] / (chunk['sales'] + 1e-5)
        if 'supplier_lead_time' in chunk.columns:
            chunk['supplier_lead_time_variance'] = chunk.groupby('supplier_id')['supplier_lead_time'].transform('std').fillna(0)

        # Cumulative sales
        if 'sales' in chunk.columns:
            chunk['cumulative_sales'] = chunk.groupby(['store', 'item', 'year', 'month'])['sales'].cumsum()

        # Restocking features
        if 'restocked_inventory' in chunk.columns:
            chunk['reorder_frequency'] = (
                chunk.groupby(['store', 'item'])['restocked_inventory']
                .rolling(30, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
            )

        # Stock shortage
        if 'stock_on_hand' in chunk.columns and 'stock_reorder_level' in chunk.columns:
            chunk['stock_shortage'] = (
                (chunk['stock_on_hand'] < chunk['stock_reorder_level']) | (chunk['stock_on_hand'] <= 0)
            ).astype(int)

        # Drop rows with NaN values
        chunk.fillna(0, inplace=True)

        

        # Retain only required columns
        required_features = list(set(sales_model_features + understock_model_features))
        chunk = chunk.loc[:, chunk.columns.intersection(required_features)]
        return chunk, chunk.index

    processed_chunks = []
    valid_indices = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].copy()
        if batch.empty:
            break
        try:
            processed_chunk, indices = process_chunk(batch)
            processed_chunks.append(processed_chunk)
            valid_indices.extend(indices)
        except ValueError as e:
            print(f"Preprocessing error in batch {i}: {e}")
            continue

    # Combine processed chunks
    processed_df = pd.concat(processed_chunks, axis=0)

    # Split features for models
    sales_features = processed_df[sales_model_features]
    understock_features = processed_df[understock_model_features]

    return sales_features, understock_features, pd.Index(valid_indices)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        global uploaded_df
        uploaded_df = pd.read_csv(file)
        print("Uploaded Columns:", uploaded_df.columns)  # Debugging step
        return uploaded_df.head(20).to_html(classes="table table-bordered", index=False)
    except Exception as e:
        return f"<p>Error processing the file: {e}</p>"


@app.route('/predict_understock', methods=['POST'])
def predict_understock():
    try:
        # Preprocess data
        sales_features, understock_features, valid_indices = preprocess_and_predict_in_batches(uploaded_df, batch_size=1000)

        # Predict sales and understock risk
        sales_predictions = sales_model.predict(sales_features)
        understock_predictions = understock_model.predict(understock_features)

        # Assign predictions to the original DataFrame
        uploaded_df['Predicted Sales'] = np.nan
        uploaded_df['Understock Risk'] = np.nan

        uploaded_df.loc[valid_indices, 'Predicted Sales'] = sales_predictions
        uploaded_df.loc[valid_indices, 'Understock Risk'] = understock_predictions

        return uploaded_df[['store', 'item', 'Predicted Sales', 'Understock Risk']].to_html(classes="table table-bordered", index=False)
    except Exception as e:
        return f"<p>Error in prediction: {e}</p>"


@app.route('/plot/residuals_over_time', methods=['GET'])
def plot_residuals_over_time():
    try:
        if 'Predicted Sales' not in uploaded_df.columns:
            return "<p>Error: Predictions have not been generated yet. Please run predictions first by clicking 'Generate Predictions'.</p>"

        uploaded_df['Residual'] = uploaded_df['sales'] - uploaded_df['Predicted Sales']
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=uploaded_df, x='date', y='Residual', hue='item', legend=False)
        plt.title("Residuals Over Time")
        plt.xlabel("Date")
        plt.ylabel("Residual (Actual - Predicted)")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return f'<img src="data:image/png;base64,{image_base64}"/>'
    except Exception as e:
        return f"<p>Error while generating residuals plot: {e}</p>"
    
@app.route('/plot/sales_vs_risk', methods=['GET'])
def plot_sales_vs_risk():
    try:
        if 'Predicted Sales' not in uploaded_df.columns or 'Understock Risk' not in uploaded_df.columns:
            return "<p>Error: Predictions have not been generated yet. Please run predictions first by clicking 'Generate Predictions'.</p>"

        plot_data = uploaded_df.dropna(subset=['Predicted Sales', 'Understock Risk'])
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=plot_data, x='Predicted Sales', y='Understock Risk', hue='store', palette='viridis', legend=False)
        plt.title("Predicted Sales vs Understock Risk")
        plt.xlabel("Predicted Sales")
        plt.ylabel("Understock Risk")

        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return f'<img src="data:image/png;base64,{image_base64}"/>'
    except Exception as e:
        return f"<p>Error while generating scatter plot: {e}</p>"
    
@app.route('/plot/heatmap', methods=['GET'])
def plot_heatmap():
    try:
        if uploaded_df is None:
            return "<p>Error: No data uploaded yet. Please upload a CSV file first.</p>"

        # Calculate correlation matrix
        corr_matrix = uploaded_df.select_dtypes(include=[np.number]).corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap of Numeric Features")

        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return f'<img src="data:image/png;base64,{image_base64}"/>'
    except Exception as e:
        return f"<p>Error while generating heatmap: {e}</p>"
    
@app.route('/plot/percentage_error', methods=['GET'])
def plot_percentage_error():
    try:
        if 'Predicted Sales' not in uploaded_df.columns:
            return "<p>Error: Predictions have not been generated yet. Please run predictions first by clicking 'Generate Predictions'.</p>"

        uploaded_df['Percentage Error'] = np.abs((uploaded_df['sales'] - uploaded_df['Predicted Sales']) / uploaded_df['sales']) * 100
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=uploaded_df, x='date', y='Percentage Error', hue='item', legend=False)
        plt.title("Percentage Error Over Time")
        plt.xlabel("Date")
        plt.ylabel("Percentage Error (%)")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return f'<img src="data:image/png;base64,{image_base64}"/>'
    except Exception as e:
        return f"<p>Error while generating percentage error plot: {e}</p>"
    
@app.route('/plot/predicted_vs_actual', methods=['GET'])
def plot_predicted_vs_actual():
    try:
        if 'Predicted Sales' not in uploaded_df.columns:
            return "<p>Error: Predictions have not been generated yet. Please run predictions first by clicking 'Generate Predictions'.</p>"

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=uploaded_df, x='Predicted Sales', y='sales', hue='item', palette='viridis', legend=False)
        plt.plot([0, uploaded_df['sales'].max()], [0, uploaded_df['sales'].max()], color='red', linestyle='--')  # Line of perfect prediction
        plt.title("Predicted Sales vs Actual Sales")
        plt.xlabel("Predicted Sales")
        plt.ylabel("Actual Sales")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return f'<img src="data:image/png;base64,{image_base64}"/>'
    except Exception as e:
        return f"<p>Error while generating predicted vs actual plot: {e}</p>"



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT env variable, default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)

# https://demand-forecasting-api-201587345242.asia-northeast2.run.app/  ===-deployment url