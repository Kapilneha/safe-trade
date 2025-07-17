# Stock Recommendation Web App

This web application uses an LSTM model (TensorFlow/Keras) to predict future stock prices and generate Buy/Sell/Hold recommendations. It features a Flask backend and a responsive frontend with Bootstrap and Chart.js.

## Features
- Predict next day's closing price for any stock symbol
- Get Buy/Sell/Hold recommendation based on prediction
- Visualize historical and predicted prices on a chart
- Fallback to synthetic data when API calls fail

## Backend
- Python, Flask
- TensorFlow/Keras LSTM model
- yfinance for stock data
- Model and scaler are saved for fast predictions

## Frontend
- HTML, CSS, Bootstrap
- JavaScript (AJAX for API calls)
- Chart.js for visualizations

## Local Setup
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the Flask app:
   ```sh
   python app.py
   ```
3. Access the app at http://localhost:5000

## API Endpoint
- `/predict?symbol=AAPL`
  - Returns: current price, predicted price, recommendation, and recent history

## Deployment to GitHub and Vercel

### GitHub Deployment
1. Create a new GitHub repository
2. Initialize Git in your local project folder:
   ```sh
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

### Vercel Deployment
1. Sign up for a Vercel account and connect it to your GitHub
2. Import your GitHub repository in Vercel
3. Configure the project:
   - Framework Preset: Other
   - Build Command: None
   - Output Directory: None
   - Install Command: `pip install -r requirements.txt`
4. Deploy!

## Project Structure
- `app.py`: Flask application with LSTM model and API endpoints
- `index.html`: Frontend UI
- `static/js/main.js`: JavaScript for AJAX and chart rendering
- `requirements.txt`: Python dependencies
- `vercel.json`: Vercel configuration
- `api/index.py`: Vercel serverless function entry point

## Stock Symbols to Try
- Tech: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- Financial: JPM, BAC, V, MA
- Retail: WMT, TGT, COST
- Healthcare: JNJ, PFE, MRNA
- Entertainment: DIS, NFLX

