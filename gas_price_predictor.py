# Natural Gas Price Predictor - Using CSV File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta

print("Starting Natural Gas Price Analysis...")
print("Loading data from Nat_Gas.csv...")

# Step 1: Load data from CSV file
try:
    df = pd.read_csv('Nat_Gas.csv')
    print("CSV file loaded successfully!")

    # Convert scientific notation to regular numbers
    df['Prices'] = pd.to_numeric(df['Prices'], errors='coerce')

    # Convert dates to proper format
    df['Dates'] = pd.to_datetime(df['Dates'])

    # Remove any rows with missing values
    df = df.dropna()

    print(f"Data overview:")
    print(f"   - Period: {df['Dates'].min().strftime('%b %Y')} to {df['Dates'].max().strftime('%b %Y')}")
    print(f"   - Number of months: {len(df)}")
    print(f"   - Price range: ${df['Prices'].min():.2f} - ${df['Prices'].max():.2f}")
    print(f"   - Average price: ${df['Prices'].mean():.2f}")

except FileNotFoundError:
    print("ERROR: Nat_Gas.csv file not found!")
    print("Make sure the CSV file is in the same folder as this Python file")
    exit()
except Exception as e:
    print(f"ERROR loading file: {e}")
    exit()

# Step 2: Display first few rows of data
print("\nFirst 5 rows of data:")
print(df.head())

# Step 3: Create historical price chart
print("\nCreating historical price chart...")
plt.figure(figsize=(12, 6))
plt.plot(df['Dates'], df['Prices'], marker='o', linewidth=2, markersize=4, color='blue', label='Monthly Prices')
plt.title('Natural Gas Prices (Oct 2020 - Sep 2024)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add average price line
avg_price = df['Prices'].mean()
plt.axhline(y=avg_price, color='red', linestyle='--', alpha=0.7, label=f'Average (${avg_price:.2f})')
plt.legend()

plt.tight_layout()
plt.show()

# Step 4: Analyze seasonal patterns
print("\nAnalyzing seasonal patterns...")
df['Month'] = df['Dates'].dt.month
df['Year'] = df['Dates'].dt.year

monthly_avg = df.groupby('Month')['Prices'].mean()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print("\nAverage Prices by Month:")
for i, (month, avg_price) in enumerate(monthly_avg.items()):
    print(f"   {month_names[i]}: ${avg_price:.2f}")

# Find highest and lowest months
highest_month = monthly_avg.idxmax()
lowest_month = monthly_avg.idxmin()

print(f"\n Seasonal Insights:")
print(f"   - Highest prices: {month_names[highest_month - 1]} (${monthly_avg[highest_month]:.2f})")
print(f"   - Lowest prices: {month_names[lowest_month - 1]} (${monthly_avg[lowest_month]:.2f})")
print(f"   - Seasonal spread: ${monthly_avg[highest_month] - monthly_avg[lowest_month]:.2f}")

# Plot seasonal pattern
plt.figure(figsize=(10, 5))
bars = plt.bar(month_names, monthly_avg, color='lightblue', edgecolor='darkblue')
plt.title('Natural Gas Prices: Seasonal Pattern', fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average Price ($)')
plt.grid(True, alpha=0.3)

# Highlight highest and lowest
bars[highest_month - 1].set_color('red')
bars[lowest_month - 1].set_color('green')

plt.tight_layout()
plt.show()

# Step 5: Prepare data for machine learning
print("\nPreparing machine learning model...")
df['Days'] = (df['Dates'] - df['Dates'].min()).dt.days

X = df['Days'].values.reshape(-1, 1)  # Input features (days)
y = df['Prices'].values  # Target variable (prices)

print(f"Training data prepared:")
print(f"   - Input shape: {X.shape}")
print(f"   - Date range: {df['Days'].min()} to {df['Days'].max()} days")

# Step 6: Train the prediction model
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Check model accuracy
y_pred = model.predict(X_poly)
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Model trained successfully!")
print(f"   - Mean Absolute Error: ${mae:.2f}")
print(f"   - R-squared: {r2:.4f}")

if r2 > 0.8:
    print("   Excellent model fit!")
elif r2 > 0.6:
    print("   Good model fit!")
else:
    print("   Model might need improvement")


# Step 7: Create prediction function
def predict_gas_price(input_date):
    """
    Predict natural gas price for any date
    """
    try:
        if isinstance(input_date, str):
            input_date = pd.to_datetime(input_date)

        start_date = df['Dates'].min()
        days_from_start = (input_date - start_date).days

        days_array = np.array([[days_from_start]])
        days_poly = poly.transform(days_array)

        predicted_price = model.predict(days_poly)[0]
        return max(0, predicted_price)  # Ensure no negative prices

    except Exception as e:
        print(f"Prediction error: {e}")
        return None


# Step 8: Test the prediction function
print("\nTesting predictions with sample dates...")
test_dates = [
    '2023-12-25',  # Christmas
    '2024-06-15',  # Summer
    '2024-12-31',  # Year-end
    '2025-03-20',  # Spring next year
    '2025-07-04'  # Independence Day next year
]

print("\nSample Predictions:")
for date in test_dates:
    price = predict_gas_price(date)
    if price is not None:
        print(f"   {date}: ${price:.2f}")

# Step 9: Generate future predictions (1 year ahead)
print("\nGenerating 1-year future forecast...")
last_date = df['Dates'].max()
future_dates = [last_date + timedelta(days=30 * i) for i in range(13)]  # Next 12 months
future_prices = [predict_gas_price(date) for date in future_dates]

print(f"Future price trend:")
print(f"   - Current ({last_date.strftime('%b %Y')}): ${df['Prices'].iloc[-1]:.2f}")
print(f"   - 6 months from now: ${future_prices[6]:.2f}")
print(f"   - 1 year from now: ${future_prices[12]:.2f}")

# Step 10: Create comprehensive visualization
plt.figure(figsize=(14, 8))

# Plot historical data
plt.plot(df['Dates'], df['Prices'], 'bo-', label='Historical Prices',
         linewidth=2, markersize=6, alpha=0.8)

# Plot model fit
plt.plot(df['Dates'], y_pred, 'r-', label='Model Fit', linewidth=2, alpha=0.7)

# Plot future predictions
plt.plot(future_dates, future_prices, 'g--', label='Future Predictions',
         linewidth=2, marker='s', markersize=4)

plt.title('Natural Gas Price Analysis\nHistorical Data + 1-Year Future Forecast',
          fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add prediction region background
prediction_start = future_dates[0]
prediction_end = future_dates[-1]
plt.axvspan(prediction_start, prediction_end, alpha=0.1, color='green',
            label='Prediction Region')

plt.tight_layout()
plt.show()

# Step 11: Interactive prediction tool
print("\n" + "=" * 60)
print("NATURAL GAS PRICE PREDICTOR - INTERACTIVE TOOL")
print("=" * 60)


def interactive_predictor():
    while True:
        print("\n" + "-" * 40)
        print("What would you like to do?")
        print("1.Predict price for a specific date")
        print("2.See seasonal analysis")
        print("3.View price trends")
        print("4.Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            print("\nEnter a date to predict natural gas price")
            print("Format: YYYY-MM-DD (e.g., 2024-12-25)")
            date_input = input("Date: ").strip()

            price = predict_gas_price(date_input)
            if price is not None:
                avg_price = df['Prices'].mean()
                current_price = df['Prices'].iloc[-1]

                print(f"\nPrediction Results for {date_input}:")
                print(f"Predicted Price: ${price:.2f}")
                print(f"Historical Average: ${avg_price:.2f}")
                print(f"Current Price: ${current_price:.2f}")

                # Give insights
                if price > avg_price * 1.15:
                    print("Significantly above average - peak season")
                elif price > avg_price * 1.05:
                    print("Above average - seasonal high")
                elif price < avg_price * 0.85:
                    print("Significantly below average - good buying opportunity")
                elif price < avg_price * 0.95:
                    print("Below average - off-peak season")
                else:
                    print("Around historical average")

        elif choice == '2':
            print("\nSEASONAL ANALYSIS")
            print("Average prices by month:")
            for i, (month, avg_price) in enumerate(monthly_avg.items()):
                trend = "HIGH" if month == highest_month else "LOW" if month == lowest_month else ""
                print(f"   {month_names[i]}: ${avg_price:.2f} {trend}")

            print(f"\nBest time to buy: {month_names[lowest_month - 1]}")
            print(f"Typically most expensive: {month_names[highest_month - 1]}")

        elif choice == '3':
            print("\nPRICE TRENDS")
            print(f"Historical Trends:")
            print(
                f"   - Overall trend: {'Increasing' if df['Prices'].iloc[-1] > df['Prices'].iloc[0] else 'Decreasing'}")
            print(f"   - Volatility: ${df['Prices'].std():.2f} standard deviation")
            print(f"   - Current price: ${df['Prices'].iloc[-1]:.2f}")

            print(f"\nFuture Outlook (1 year):")
            trend = "Increasing" if future_prices[-1] > future_prices[0] else "Decreasing"
            print(f"   - Predicted trend: {trend}")
            print(f"   - Forecast range: ${min(future_prices):.2f} - ${max(future_prices):.2f}")

        elif choice == '4':
            print("\nThank you for using the Natural Gas Price Predictor!")
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


# Start the interactive tool
interactive_predictor()