import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load('customer_priority_model_test.joblib')  # Replace with your actual model filename

def load_data(file_name):
    """Load data from a CSV file."""
    return pd.read_csv(file_name)

def apply_model_to_label_customer_value(df):
    """Predict Customer_Value for each row using the model."""
    # Ensure only the relevant columns used for prediction are passed to the model
    relevant_columns = ['Purchased (in thousands)', 'Price (single piece)']
    df['Customer_Value'] = model.predict(df[relevant_columns])  # Adjust columns as needed
    return df

def calculate_priority_scores(df):
    """Calculate priority scores for each customer."""
    # Group by Customer and calculate percentage of each labeled value
    customer_value_counts = df.groupby('Customer')['Customer_Value'].value_counts(normalize=True).unstack().fillna(0) * 100
    
    # Assign weights for high, normal, low values and calculate priority score
    weights = {2: 3, 1: 2, 0: 1}
    customer_value_counts['Priority Score'] = (
        customer_value_counts.get(2, 0) * weights[2] +  # use 0 if column 2 doesn't exist
        customer_value_counts.get(1, 0) * weights[1] +  # use 0 if column 1 doesn't exist
        customer_value_counts.get(0, 0) * weights[0]    # use 0 if column 0 doesn't exist
    )
    
    return customer_value_counts[['Priority Score']].sort_values(by='Priority Score', ascending=False)

def plot_priority_scores(customer_value_counts_df):
    """Plot the priority scores for each customer."""
    plt.figure(figsize=(12, 8))
    plt.bar(customer_value_counts_df.index, customer_value_counts_df['Priority Score'], color='navy')
    
    # Automatically set the y-axis
    min_score = customer_value_counts_df['Priority Score'].min()
    max_score = customer_value_counts_df['Priority Score'].max()
    plt.ylim(min_score - 5, max_score + 5)
    
    plt.xlabel("Customer")
    plt.ylabel("Priority Score")
    plt.title("Customer Priority")
    plt.xticks(rotation=90)
    plt.show()

# Run the functions
df = load_data("european_sales_no_CV.csv")  # Replace with the specific file name
df = apply_model_to_label_customer_value(df)  # Apply model to fill Customer_Value
customer_value_counts_df = calculate_priority_scores(df)
plot_priority_scores(customer_value_counts_df)