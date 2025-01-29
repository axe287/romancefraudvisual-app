import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import chardet
import logging
import numpy as np
import statsmodels.formula.api as smf  # Added for regression models

# ---------------------------
# Configuration and Setup
# ---------------------------

st.set_page_config(page_title="Carla vs. Charles - Focused Dashboard", layout="wide")

# **Define current_dir here**
current_dir = Path(__file__).parent  # This line defines current_dir

#paths to your CSV files
DATA_DIRS = [
    current_dir / '#new sim_gemini_gemini_final' / 'simulation_results100_updated_verified.csv',
    current_dir / '#new sim_gemini_gpt3.5_final' / 'simulation_results_100results_updated_verified.csv',
    current_dir / '#new sim_gemini_gpt4omini_final' / 'simulation_results_100results_updated_verified.csv',
    current_dir / '#new sim_gpt3.5_gemini_final' / 'simulation_results100results_updated_verified.csv',
    current_dir / '#new sim_gpt3.5_gpt3.5_final' / 'simulation_results100results_updated_verified.csv',
    current_dir / '#new sim_gpt3.5_gpt4omini_final' / 'simulation_results_100results_updated_verified.csv',
    current_dir / '#new sim_gpt4omini_gemini_final_test' / 'simulation_results_100results_updated_verified.csv'
]

# Define the mapping from folder names to human-readable titles
dataset_title_map = {
    "#new sim_gemini_gemini_final": "Gemini (Carla) vs Gemini (Charles)",
    "#new sim_gemini_gpt3.5_final": "Gemini (Carla) vs GPT3.5 (Charles)",
    "#new sim_gemini_gpt4omini_final": "Gemini (Carla) vs GPT4-o-mini (Charles)",
    "#new sim_gpt3.5_gemini_final": "GPT3.5 (Carla) vs Gemini (Charles)",
    "#new sim_gpt3.5_gpt3.5_final": "GPT3.5 (Carla) vs GPT3.5 (Charles)",
    "#new sim_gpt3.5_gpt4omini_final": "GPT3.5 (Carla) vs GPT4-o-mini (Charles)",
    "#new sim_gpt4omini_gemini_final_test": "GPT4-o-mini (Carla) vs Gemini (Charles)"
}

# Configure logging
logging.basicConfig(
    filename='dashboard_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# ---------------------------
# Helper Functions
# ---------------------------

def detect_file_encoding(file_path, num_bytes=100000):
    """
    Detect the encoding of a file using chardet.

    Parameters:
        file_path (Path): Path to the file.
        num_bytes (int): Number of bytes to read for detection.

    Returns:
        str: Detected encoding.
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(num_bytes)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        logging.info(f"Detected encoding for {file_path.name}: {encoding} with confidence {confidence}")
        return encoding
    except Exception as e:
        logging.error(f"Error detecting encoding for {file_path.name}: {e}")
        return None

def load_csv_with_fallback(file_path):
    """
    Load a CSV file with multiple encoding fallbacks.

    Parameters:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if failed.
    """
    # Step 1: Detect encoding
    encoding = detect_file_encoding(file_path)

    if encoding:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            df['Dataset'] = f"{file_path.parent.name}/{file_path.stem}"
            logging.info(f"Successfully loaded {file_path.name} with encoding {encoding}")
            return df
        except UnicodeDecodeError as e:
            logging.warning(f"UnicodeDecodeError for {file_path.name} with encoding {encoding}: {e}")
        except Exception as e:
            logging.error(f"Error loading {file_path.name} with encoding {encoding}: {e}")

    # Step 2: Fallback to 'cp1252'
    try:
        df = pd.read_csv(file_path, encoding='cp1252')
        df['Dataset'] = f"{file_path.parent.name}/{file_path.stem}"
        logging.info(f"Successfully loaded {file_path.name} with fallback encoding 'cp1252'")
        return df
    except UnicodeDecodeError as e:
        logging.warning(f"UnicodeDecodeError for {file_path.name} with encoding 'cp1252': {e}")
    except Exception as e:
        logging.error(f"Error loading {file_path.name} with encoding 'cp1252': {e}")

    # Step 3: Fallback to 'latin1'
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        df['Dataset'] = f"{file_path.parent.name}/{file_path.stem}"
        logging.info(f"Successfully loaded {file_path.name} with fallback encoding 'latin1'")
        return df
    except UnicodeDecodeError as e:
        logging.warning(f"UnicodeDecodeError for {file_path.name} with encoding 'latin1': {e}")
    except Exception as e:
        logging.error(f"Error loading {file_path.name} with encoding 'latin1': {e}")

    # Step 4: Final attempt with 'utf-8' and ignoring errors
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        df['Dataset'] = f"{file_path.parent.name}/{file_path.stem}"
        logging.info(f"Successfully loaded {file_path.name} with encoding 'utf-8' and ignoring errors")
        return df
    except Exception as e:
        logging.error(f"Failed to load {file_path.name} after all encoding attempts: {e}")
        st.error(f"Failed to load {file_path.name} after all encoding attempts: {e}")
        return None

@st.cache_data
def load_data(data_paths):
    """
    Load and combine multiple CSV files from given paths.

    Parameters:
        data_paths (list of Path): List of file paths.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    df_list = []
    for path in data_paths:
        if not path.exists():
            st.warning(f"Path {path} does not exist. Skipping.")
            logging.warning(f"Path {path} does not exist. Skipping.")
            continue
        if path.is_file() and path.suffix.lower() == '.csv':
            df = load_csv_with_fallback(path)
            if df is not None:
                df_list.append(df)
        else:
            st.warning(f"Path {path} is not a CSV file. Skipping.")
            logging.warning(f"Path {path} is not a CSV file. Skipping.")
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # Map the 'Dataset' to 'Dataset_Title' using the mapping dictionary
        combined_df['Dataset_Title'] = combined_df['Dataset'].apply(
            lambda x: dataset_title_map.get(x.split('/')[0], x.split('/')[0])
        )
        logging.info("Successfully mapped 'Dataset' to 'Dataset_Title'")
        
        return combined_df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no files found

def extract_models(dataset_title):
    """
    Extract Carla and Charles models from the Dataset_Title.

    Parameters:
        dataset_title (str): The title of the dataset.

    Returns:
        tuple: (Carla_model, Charles_model)
    """
    try:
        # Expected format: "Model (Carla) vs Model (Charles)"
        parts = dataset_title.split(' vs ')
        carla_part = parts[0].strip()
        charles_part = parts[1].strip()
        
        # Extract model names with labels
        carla_model = carla_part.split('(')[0].strip() + " (Carla)"
        charles_model = charles_part.split('(')[0].strip() + " (Charles)"
        
        return carla_model, charles_model
    except Exception as e:
        logging.error(f"Error extracting models from '{dataset_title}': {e}")
        return "Unknown (Carla)", "Unknown (Charles)"

def format_matrix_table(matrix_pivot):
    """
    Format the pivot table by rounding, replacing NaNs with 'N/A', and appending '%'.

    Parameters:
        matrix_pivot (pd.DataFrame): Pivot table with numerical values.

    Returns:
        pd.DataFrame: Formatted pivot table as strings with '%' or 'N/A'.
    """
    # Round the values
    matrix_pivot = matrix_pivot.round(0)

    # Function to format each cell
    def format_cell(x):
        if pd.notnull(x):
            return f"{int(x)}%"
        else:
            return "N/A"

    # Apply the formatting
    return matrix_pivot.applymap(format_cell)

# ---------------------------
# Load the Data
# ---------------------------

data = load_data(DATA_DIRS)

# Check if data is loaded
if data.empty:
    st.warning("No CSV files were successfully loaded. Please check the logs for details.")
    st.stop()

# ---------------------------
# Data Validation
# ---------------------------

# Define required columns
required_columns = [
    'Iteration', 'Result_Verified', 'Num_Turns', 'Avg_Carla_Response_Length',
    'Avg_Charles_Response_Length', 'Avg_Sentiment_Charles',
    'Predominant_Strategy_Carla', 'Question_Count_Carla',
    'Question_Count_Charles', 'Financial_Requests_Made',
    'Response_to_Requests', 'Emoji_Count_Carla', 'Conversation - Full',
    'Carla_Request_Message_Number'  # Added Carla_Request_Message_Number
]

# Check for missing columns
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"The following required columns are missing from the data: {', '.join(missing_columns)}")
    st.stop()

# ---------------------------
# Data Preprocessing
# ---------------------------

# Convert numerical columns to numeric types
numeric_columns = [
    'Result_Verified', 'Num_Turns', 'Avg_Carla_Response_Length',
    'Avg_Charles_Response_Length', 'Avg_Sentiment_Charles',
    'Question_Count_Carla', 'Question_Count_Charles',
    'Financial_Requests_Made', 'Emoji_Count_Carla',
    'Iteration', 'Carla_Request_Message_Number'  # Added 'Carla_Request_Message_Number'
]

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, set errors to NaN

# Handle missing values by dropping rows with critical missing data
data.dropna(subset=['Dataset', 'Result_Verified', 'Response_to_Requests', 'Iteration'], inplace=True)  # Updated 'Result' to 'Result_Verified'

# ---------------------------
# Define Success Label and Sentiment Threshold
# ---------------------------

# Define what constitutes a successful result
success_label = 1  # Assuming 'Result_Verified' uses 1 for success

# Define sentiment threshold
sentiment_threshold = 0.5

# ---------------------------
# Sidebar Filters
# ---------------------------

st.sidebar.header("Filters")

# Select Datasets (Use 'Dataset_Title' for display)
datasets = data['Dataset_Title'].unique().tolist()
selected_datasets = st.sidebar.multiselect("Select Datasets", options=datasets, default=datasets)

# Filter data based on selection
filtered_data = data[data['Dataset_Title'].isin(selected_datasets)]

# ---------------------------
# Dashboard Title and Description
# ---------------------------

st.title("Carla vs. Charles - Focused Dashboard")

st.markdown("""
This dashboard provides insights into the performance and interactions between Carla and Charles across various CSV datasets. Below are the key visualizations:
1. **CSV with the Most Successful Results**
2. **CSV with Charles Being Mostly Supportive**
3. **CSV with Charles Being Mostly Skeptical**
4. **Impact of Charles' Skepticism on Success Rate**
5. **Correlation Between Carla's Emoji Usage and Success in Gemini-Assigned Datasets**
6. **Distribution of Charles' Sentiment Scores**
7. **Trend of Successful Results Over Iterations**
8. **Heatmap of Correlations Between Numerical Metrics**
9. **Engagement Metrics Over Datasets**
10. **Scatter Plot of Question Counts vs. Success Rate**
11. **Carla's Average Request Message Number**
12. **Percentage of Verified Success per Dataset**
13. **Finances Being Mentioned per Dataset**
14. **Percentage of Conversations with Emojis in Carla's Messages per Dataset**
15. **Percentage of Conversations by Strategy Type per Dataset**
16. **Percentage of Conversations with High Sentiment Scores per Dataset**
17. **Percentage of Conversations with Financial Requests Made per Dataset**
18. **3D Linear Regression Model: Sentiment and Financial Requests vs. Success Rate**
19. **3D Linear Regression Model: Carla's Average Request Message Number and Sentiment Score vs. Success Rate**
""")

# ---------------------------
# Visualization 1: CSV with the Most Successful Results
# ---------------------------

st.subheader("1. CSV with the Most Successful Results")

# Calculate the number of successful results per Dataset using 'Result_Verified'
success_counts = filtered_data.groupby('Dataset_Title')['Result_Verified'].apply(lambda x: (x == success_label).sum()).reset_index()
success_counts.columns = ['Dataset_Title', 'Successful_Results']

# Sort the datasets by the number of successful results
success_counts_sorted = success_counts.sort_values(by='Successful_Results', ascending=False)

# Create a bar chart
fig_success = px.bar(
    success_counts_sorted,
    x='Dataset_Title',
    y='Successful_Results',
    title='Number of Successful Results per Dataset',
    labels={'Successful_Results': 'Number of Successes', 'Dataset_Title': 'Dataset'},
    color='Successful_Results',
    color_continuous_scale='Viridis'
)

st.plotly_chart(fig_success, use_container_width=True, key='fig_success')

# Display the table of Successful Results
st.markdown("**Successful Results per Dataset:**")
st.table(success_counts_sorted[['Dataset_Title', 'Successful_Results']].style.format({'Successful_Results': '{:.0f}'}))

# Add Percentage Table
try:
    # Calculate total results per dataset
    total_results = filtered_data.groupby('Dataset_Title')['Result_Verified'].count().reset_index()
    total_results.columns = ['Dataset_Title', 'Total_Results']

    # Merge with success counts
    success_percentage = pd.merge(success_counts_sorted, total_results, on='Dataset_Title')

    # Calculate Success Rate Percentage
    success_percentage['Success_Rate_Percentage'] = (success_percentage['Successful_Results'] / success_percentage['Total_Results']) * 100

    # Sort by Success Rate Percentage
    success_percentage_sorted = success_percentage.sort_values(by='Success_Rate_Percentage', ascending=False)

    # Display the percentage table
    st.markdown("**Success Rate Percentage per Dataset:**")
    st.table(success_percentage_sorted[['Dataset_Title', 'Success_Rate_Percentage']].style.format({'Success_Rate_Percentage': '{:.0f}%'}))
except Exception as e:
    logging.error(f"Error calculating Success Rate Percentage for Visualization 1: {e}")
    st.error(f"An error occurred while calculating Success Rate Percentage: {e}")

# ---------------------------
# Matrix Table for Visualization 1
# ---------------------------

st.markdown("### Success Rates Matrix")

try:
    # Extract Carla and Charles models
    filtered_data[['Carla_Model', 'Charles_Model']] = filtered_data['Dataset_Title'].apply(
        lambda x: pd.Series(extract_models(x))
    )

    # Calculate success rates for each pairing
    matrix_data = filtered_data.groupby(['Carla_Model', 'Charles_Model'])['Result_Verified'].apply(lambda x: (x == success_label).mean() * 100).reset_index()

    matrix_pivot = matrix_data.pivot(index='Carla_Model', columns='Charles_Model', values='Result_Verified')

    # Format the matrix table
    formatted_matrix = format_matrix_table(matrix_pivot)

    # Display the matrix table
    st.markdown("**Success Rates Matrix (Percentage):**")
    st.table(formatted_matrix)
except Exception as e:
    logging.error(f"Error creating Success Rates Matrix for Visualization 1: {e}")
    st.error(f"An error occurred while creating Success Rates Matrix: {e}")

# ---------------------------
# Visualization 2: CSVs with Mostly Supportive Responses
# ---------------------------

st.subheader("2. CSV with Charles Being Mostly Supportive")

# Define the supportive response label
supportive_label = 'supportive'  # Adjust based on your data's actual label

# Standardize the 'Response_to_Requests' column
filtered_data['Response_to_Requests'] = filtered_data['Response_to_Requests'].str.lower().str.strip()

# Calculate the number of supportive responses per Dataset
supportive_counts = filtered_data.groupby('Dataset_Title')['Response_to_Requests'].apply(lambda x: (x == supportive_label).sum()).reset_index()
supportive_counts.columns = ['Dataset_Title', 'Supportive_Responses']

# Sort the datasets by the number of supportive responses
supportive_counts_sorted = supportive_counts.sort_values(by='Supportive_Responses', ascending=False)

# Create a bar chart
fig_supportive = px.bar(
    supportive_counts_sorted,
    x='Dataset_Title',
    y='Supportive_Responses',
    title='Number of Supportive Responses per Dataset',
    labels={'Supportive_Responses': 'Number of Supportive Responses', 'Dataset_Title': 'Dataset'},
    color='Supportive_Responses',
    color_continuous_scale='Blues'
)

st.plotly_chart(fig_supportive, use_container_width=True, key='fig_supportive')

# Display the table of Supportive Responses
st.markdown("**Supportive Responses per Dataset:**")
st.table(supportive_counts_sorted[['Dataset_Title', 'Supportive_Responses']].style.format({'Supportive_Responses': '{:.0f}'}))

# Add Percentage Table
try:
    # Calculate total responses per dataset
    total_responses = filtered_data.groupby('Dataset_Title')['Response_to_Requests'].count().reset_index()
    total_responses.columns = ['Dataset_Title', 'Total_Responses']

    # Merge with supportive counts
    supportive_percentage = pd.merge(supportive_counts_sorted, total_responses, on='Dataset_Title')

    # Calculate Supportive Responses Percentage
    supportive_percentage['Supportive_Responses_Percentage'] = (supportive_percentage['Supportive_Responses'] / supportive_percentage['Total_Responses']) * 100

    # Sort by Supportive Responses Percentage
    supportive_percentage_sorted = supportive_percentage.sort_values(by='Supportive_Responses_Percentage', ascending=False)

    # Display the percentage table
    st.markdown("**Supportive Responses Percentage per Dataset:**")
    st.table(supportive_percentage_sorted[['Dataset_Title', 'Supportive_Responses_Percentage']].style.format({'Supportive_Responses_Percentage': '{:.0f}%'}))
except Exception as e:
    logging.error(f"Error calculating Supportive Responses Percentage for Visualization 2: {e}")
    st.error(f"An error occurred while calculating Supportive Responses Percentage: {e}")

# ---------------------------
# Matrix Table for Visualization 2
# ---------------------------

st.markdown("### Supportive Responses Rates Matrix")

try:
    # Calculate supportive response rates for each pairing
    matrix_data_supportive = filtered_data.groupby(['Carla_Model', 'Charles_Model'])['Response_to_Requests'].apply(
        lambda x: (x == supportive_label).mean() * 100
    ).reset_index()
    matrix_pivot_supportive = matrix_data_supportive.pivot(index='Carla_Model', columns='Charles_Model', values='Response_to_Requests')

    # Format the matrix table
    formatted_matrix_supportive = format_matrix_table(matrix_pivot_supportive)

    # Display the matrix table
    st.markdown("**Supportive Responses Rates Matrix (Percentage):**")
    st.table(formatted_matrix_supportive)
except Exception as e:
    logging.error(f"Error creating Supportive Responses Rates Matrix for Visualization 2: {e}")
    st.error(f"An error occurred while creating Supportive Responses Rates Matrix: {e}")

# ---------------------------
# Visualization 3: CSVs with Mostly Skeptical Responses
# ---------------------------

st.subheader("3. CSV with Charles Being Mostly Skeptical")

# Define the skeptical response label
skeptical_label = 'skeptical'  

# Calculate the number of skeptical responses per Dataset
skeptical_counts = filtered_data.groupby('Dataset_Title')['Response_to_Requests'].apply(lambda x: (x == skeptical_label).sum()).reset_index()
skeptical_counts.columns = ['Dataset_Title', 'Skeptical_Responses']

# Sort the datasets by the number of skeptical responses
skeptical_counts_sorted = skeptical_counts.sort_values(by='Skeptical_Responses', ascending=False)

# Create a bar chart
fig_skeptical = px.bar(
    skeptical_counts_sorted,
    x='Dataset_Title',
    y='Skeptical_Responses',
    title='Number of Skeptical Responses per Dataset',
    labels={'Skeptical_Responses': 'Number of Skeptical Responses', 'Dataset_Title': 'Dataset'},
    color='Skeptical_Responses',
    color_continuous_scale='Reds'
)

st.plotly_chart(fig_skeptical, use_container_width=True, key='fig_skeptical')

# Display the table of Skeptical Responses
st.markdown("**Skeptical Responses per Dataset:**")
st.table(skeptical_counts_sorted[['Dataset_Title', 'Skeptical_Responses']].style.format({'Skeptical_Responses': '{:.0f}'}))

# Add Percentage Table
try:
    # Calculate total responses per dataset
    total_responses = filtered_data.groupby('Dataset_Title')['Response_to_Requests'].count().reset_index()
    total_responses.columns = ['Dataset_Title', 'Total_Responses']

    # Merge with skeptical counts
    skeptical_percentage = pd.merge(skeptical_counts_sorted, total_responses, on='Dataset_Title')

    # Calculate Skeptical Responses Percentage
    skeptical_percentage['Skeptical_Responses_Percentage'] = (skeptical_percentage['Skeptical_Responses'] / skeptical_percentage['Total_Responses']) * 100

    # Sort by Skeptical Responses Percentage
    skeptical_percentage_sorted = skeptical_percentage.sort_values(by='Skeptical_Responses_Percentage', ascending=False)

    # Display the percentage table
    st.markdown("**Skeptical Responses Percentage per Dataset:**")
    st.table(skeptical_percentage_sorted[['Dataset_Title', 'Skeptical_Responses_Percentage']].style.format({'Skeptical_Responses_Percentage': '{:.0f}%'}))
except Exception as e:
    logging.error(f"Error calculating Skeptical Responses Percentage for Visualization 3: {e}")
    st.error(f"An error occurred while calculating Skeptical Responses Percentage: {e}")

# ---------------------------
# Matrix Table for Visualization 3
# ---------------------------

st.markdown("### Skeptical Responses Rates Matrix")

try:
    # Calculate skeptical response rates for each pairing
    matrix_data_skeptical = filtered_data.groupby(['Carla_Model', 'Charles_Model'])['Response_to_Requests'].apply(
        lambda x: (x == skeptical_label).mean() * 100
    ).reset_index()
    matrix_pivot_skeptical = matrix_data_skeptical.pivot(index='Carla_Model', columns='Charles_Model', values='Response_to_Requests')

    # Format the matrix table
    formatted_matrix_skeptical = format_matrix_table(matrix_pivot_skeptical)

    # Display the matrix table
    st.markdown("**Skeptical Responses Rates Matrix (Percentage):**")
    st.table(formatted_matrix_skeptical)
except Exception as e:
    logging.error(f"Error creating Skeptical Responses Rates Matrix for Visualization 3: {e}")
    st.error(f"An error occurred while creating Skeptical Responses Rates Matrix: {e}")

# ---------------------------
# Visualization 4: Impact of Skepticism on Success Rate
# ---------------------------

st.subheader("4. Impact of Charles Being More Skeptical on Success Rate")

try:
    # Calculate total responses and skeptical responses per Dataset
    total_responses = filtered_data.groupby('Dataset_Title')['Response_to_Requests'].count().reset_index()
    total_responses.columns = ['Dataset_Title', 'Total_Responses']

    # Merge total responses with skeptical responses
    skeptical_ratio = pd.merge(total_responses, skeptical_counts, on='Dataset_Title')
    skeptical_ratio['Skeptical_Ratio'] = skeptical_ratio['Skeptical_Responses'] / skeptical_ratio['Total_Responses']

    # Calculate success rate per Dataset using 'Result_Verified'
    success_rate = success_counts.copy()
    # Get total results per Dataset
    total_results = filtered_data.groupby('Dataset_Title')['Result_Verified'].count().reset_index()
    total_results.columns = ['Dataset_Title', 'Total_Results']
    success_rate = pd.merge(success_rate, total_results, on='Dataset_Title')
    success_rate['Success_Rate'] = success_rate['Successful_Results'] / success_rate['Total_Results']

    # Merge skeptical ratio with success rate
    analysis_df = pd.merge(skeptical_ratio, success_rate[['Dataset_Title', 'Success_Rate']], on='Dataset_Title')

    # Plotting the Impact of Skepticism on Success Rate
    fig_skepticism = px.scatter(
        analysis_df,
        x='Skeptical_Ratio',
        y='Success_Rate',
        text='Dataset_Title',
        title="Impact of Skepticism Ratio on Success Rate",
        labels={
            'Skeptical_Ratio': 'Skeptical Responses Ratio',
            'Success_Rate': 'Success Rate'
        },
        size='Success_Rate',
        hover_data=['Dataset_Title'],
        trendline='ols'
    )

    fig_skepticism.update_traces(textposition='top center')

    st.plotly_chart(fig_skepticism, use_container_width=True, key='fig_skepticism')

    # Add a note beneath the visualization
    st.markdown("""
        **Note:** The size of the circles are proportionate to the **Success Rate** for each dataset.
        """)


    # Calculate and display Pearson correlation coefficient
    correlation_skepticism = analysis_df['Skeptical_Ratio'].corr(analysis_df['Success_Rate'])
    st.markdown(f"**Pearson Correlation Coefficient:** {correlation_skepticism:.2f}")

    # Interpretation
    if correlation_skepticism > 0:
        interpretation_skepticism = "a positive correlation"
    elif correlation_skepticism < 0:
        interpretation_skepticism = "a negative correlation"
    else:
        interpretation_skepticism = "no correlation"

    st.markdown(f"This indicates **{interpretation_skepticism}** between the skepticism ratio and the success rate.")
except Exception as e:
    logging.error(f"Error in Visualization 4: {e}")
    st.error(f"An error occurred while generating Visualization 4: {e}")

# ---------------------------
# Aggregate Additional Metrics for Correlation Matrix
# ---------------------------

# Define additional numerical metrics to include
additional_metrics = [
    'Num_Turns', 'Avg_Carla_Response_Length', 'Avg_Charles_Response_Length',
    'Avg_Sentiment_Charles', 'Financial_Requests_Made', 'Emoji_Count_Carla',
    'Question_Count_Carla', 'Question_Count_Charles', 'Carla_Request_Message_Number'
]

# Compute per-dataset aggregates
metrics_aggregated = filtered_data.groupby('Dataset_Title').agg(
    Num_Turns_mean=('Num_Turns', 'mean'),
    Avg_Carla_Response_Length_mean=('Avg_Carla_Response_Length', 'mean'),
    Avg_Charles_Response_Length_mean=('Avg_Charles_Response_Length', 'mean'),
    Avg_Sentiment_Charles_mean=('Avg_Sentiment_Charles', 'mean'),
    Financial_Requests_Made_mean=('Financial_Requests_Made', 'mean'),
    Emoji_Count_Carla_sum=('Emoji_Count_Carla', 'sum'),
    Question_Count_Carla_sum=('Question_Count_Carla', 'sum'),
    Question_Count_Charles_sum=('Question_Count_Charles', 'sum'),
    Average_Carla_Request_Message_Number_mean=('Carla_Request_Message_Number', 'mean')  # Added this line
).reset_index()

# Merge aggregated metrics into analysis_df
analysis_df = pd.merge(analysis_df, metrics_aggregated, on='Dataset_Title', how='left')

# **Temporary Check:** Display columns of analysis_df
# st.markdown("**Debugging: Columns in `analysis_df` after Aggregation:**")
# st.write(analysis_df.columns.tolist())

# **Temporary Check:** Display sample data
# st.markdown("**Debugging: Sample Data from `analysis_df`:**")
# st.write(analysis_df[['Dataset_Title', 'Average_Carla_Request_Message_Number_mean']].head())



# ---------------------------
# Visualization 5: Correlation Between Carla's Emoji Usage and Success in Gemini Datasets
# ---------------------------

st.subheader("5. Correlation Between Carla's Emoji Usage and Success in Gemini-Assigned Datasets")

# Identify Gemini-Assigned Datasets where Carla is Gemini (Case-Insensitive)
gemini_datasets = filtered_data[
    filtered_data['Dataset_Title'].str.lower().str.startswith('gemini (carla)', na=False)
]['Dataset_Title'].unique().tolist()

# Verify the filtered datasets
st.markdown("**Filtered Gemini-Assigned Datasets (Carla as Gemini):**")
st.write(gemini_datasets)

# Filter data for Gemini Datasets
gemini_data = filtered_data[filtered_data['Dataset_Title'].isin(gemini_datasets)]

if not gemini_data.empty:
    # Calculate average emoji count and success rate per Dataset
    gemini_analysis = gemini_data.groupby('Dataset_Title').agg(
        Avg_Emoji_Count=('Emoji_Count_Carla', 'mean'),
        Successful_Results=('Result_Verified', lambda x: (x == success_label).sum()),
        Total_Results=('Result_Verified', 'count')
    ).reset_index()
    gemini_analysis['Success_Rate'] = gemini_analysis['Successful_Results'] / gemini_analysis['Total_Results']

    # Create a scatter plot to visualize the correlation
    fig_gemini_emoji = px.scatter(
        gemini_analysis,
        x='Avg_Emoji_Count',
        y='Success_Rate',
        text='Dataset_Title',
        title="Carla's Average Emoji Count vs. Success Rate (Gemini Datasets)",
        labels={
            'Avg_Emoji_Count': "Average Emoji Count",
            'Success_Rate': "Success Rate"
        },
        size='Success_Rate',
        hover_data=['Dataset_Title'],
        trendline='ols'  # Adds a linear trendline using statsmodels
    )

    fig_gemini_emoji.update_traces(textposition='top center')

    st.plotly_chart(fig_gemini_emoji, use_container_width=True, key='fig_gemini_emoji')

    # Add a note beneath the visualization
    st.markdown("""
        **Note:** The size of the circles is proportionate to the **Success Rate** for each dataset.
        """)

    # Calculate and display Pearson correlation coefficient
    correlation_gemini = gemini_analysis['Avg_Emoji_Count'].corr(gemini_analysis['Success_Rate'])
    st.markdown(f"**Pearson Correlation Coefficient:** {correlation_gemini:.2f}")

    # Interpretation
    if correlation_gemini > 0:
        interpretation_gemini = "a positive correlation"
    elif correlation_gemini < 0:
        interpretation_gemini = "a negative correlation"
    else:
        interpretation_gemini = "no correlation"

    st.markdown(f"This indicates **{interpretation_gemini}** between Carla's emoji usage and the success rate in Gemini-assigned datasets.")
else:
    st.warning("No Gemini-assigned datasets found in the selected data.")



# ---------------------------
# Visualization 6: Distribution of Charles' Sentiment Scores
# ---------------------------

st.subheader("6. Distribution of Charles' Sentiment Scores")

# 6.3 Faceted Histograms for Each Dataset
st.markdown("### Faceted Histograms for Each Dataset")

if 'Avg_Sentiment_Charles' in filtered_data.columns and 'Dataset_Title' in filtered_data.columns:
    fig_facet_hist = px.histogram(
        filtered_data,
        x="Avg_Sentiment_Charles",
        facet_col="Dataset_Title",
        color="Dataset_Title",
        nbins=20,
        title="Faceted Histogram of Charles' Sentiment Scores",
        labels={"Avg_Sentiment_Charles": "Average Sentiment"},
        hover_data=['Dataset_Title']
    )

    # Increase figure size
    fig_facet_hist.update_layout(
        width=1500,  
        height=800,  
        showlegend=False,
        bargap=0.1,
        title_x=0.5,
        margin=dict(l=40, r=40, t=100, b=40),
        plot_bgcolor='white'
    )

    # Rotate facet titles to prevent overlap
    for annotation in fig_facet_hist.layout.annotations:
        annotation.update(textangle=-45)

    st.plotly_chart(fig_facet_hist, use_container_width=True, key='fig_facet_hist')

    # Add Average Sentiment Percentage Table
    try:
        # Calculate average sentiment per dataset
        avg_sentiment = filtered_data.groupby('Dataset_Title')['Avg_Sentiment_Charles'].mean().reset_index()
        avg_sentiment.columns = ['Dataset_Title', 'Average_Sentiment_Score']

        # Sort by average sentiment descending
        avg_sentiment_sorted = avg_sentiment.sort_values(by='Average_Sentiment_Score', ascending=False)

        # Display the average sentiment table
        st.markdown("**Average Sentiment Score per Dataset:**")
        st.table(avg_sentiment_sorted[['Dataset_Title', 'Average_Sentiment_Score']].style.format({'Average_Sentiment_Score': '{:.2f}'}))
    except Exception as e:
        logging.error(f"Error calculating Average Sentiment Score for Visualization 6: {e}")
        st.error(f"An error occurred while calculating Average Sentiment Score: {e}")

    # Export the faceted histogram as PNG
    try:
        fig_facet_hist.write_image("fig_facet_hist.png")
    except Exception as e:
        logging.error(f"Error exporting faceted histogram: {e}")
        st.warning("Could not export the faceted histogram as PNG.")

    # Provide a download button for the faceted histogram
    try:
        with open("fig_facet_hist.png", "rb") as image_file:
            st.download_button(
                label="Download Faceted Histogram as PNG",
                data=image_file,
                file_name="fig_facet_hist.png",
                mime="image/png"
            )
    except FileNotFoundError:
        st.warning("Faceted histogram image not found for download.")

    # --- Modification Start ---
    # Adding Linear Regression Model under Visualization 6

    st.markdown("### Linear Regression Model: Sentiment Score vs. Success Rate")

    # Prepare data for the linear model
    model_data = analysis_df[['Dataset_Title', 'Avg_Sentiment_Charles_mean', 'Success_Rate']].dropna()

    if not model_data.empty:
        try:
            # Define the formula for the linear model
            formula = 'Success_Rate ~ Avg_Sentiment_Charles_mean'

            # Fit the linear regression model using statsmodels
            model = smf.ols(formula=formula, data=model_data).fit()

            # Display the model summary
            st.markdown("**Linear Regression Model Summary:**")
            st.text(model.summary())

            # Create the regression line plot
            fig_linear = px.scatter(
                model_data,
                x='Avg_Sentiment_Charles_mean',
                y='Success_Rate',
                trendline='ols',
                title="Linear Regression: Sentiment Score vs. Success Rate",
                labels={'Avg_Sentiment_Charles_mean': 'Average Sentiment Score', 'Success_Rate': 'Success Rate'}
            )

            st.plotly_chart(fig_linear, use_container_width=True, key='fig_linear_regression')
        except Exception as e:
            logging.error(f"Error building linear regression model for Visualization 6: {e}")
            st.error(f"An error occurred while building the linear regression model: {e}")
    else:
        st.warning("Not enough data to build a linear regression model.")

    # --- Modification End ---

    # ---------------------------
    # Matrix Table for Visualization 6
    # ---------------------------

    st.markdown("### Sentiment Scores Success Rates Matrix")

    try:
        # Calculate success rates based on sentiment scores for each pairing
        matrix_data_sentiment = filtered_data.groupby(['Carla_Model', 'Charles_Model']).apply(
            lambda df: (df['Avg_Sentiment_Charles'] > sentiment_threshold).mean() * 100
        ).reset_index(name='High_Sentiment_Success_Rate')

        matrix_pivot_sentiment = matrix_data_sentiment.pivot(index='Carla_Model', columns='Charles_Model', values='High_Sentiment_Success_Rate')

        # Format the matrix table
        formatted_matrix_sentiment = format_matrix_table(matrix_pivot_sentiment)

        # Display the matrix table
        st.markdown("**High Sentiment Success Rates Matrix (Percentage):**")
        st.table(formatted_matrix_sentiment)
    except Exception as e:
        logging.error(f"Error creating Sentiment Scores Success Rates Matrix for Visualization 6: {e}")
        st.error(f"An error occurred while creating Sentiment Scores Success Rates Matrix: {e}")

else:
    st.warning("'Avg_Sentiment_Charles' or 'Dataset_Title' column is missing from the data.")

# ---------------------------
# Visualization 7: Trend of Successful Results Over Iterations
# ---------------------------

st.subheader("7. Trend of Successful Results Over Iterations")

# Ensure 'Iteration' is numeric (already converted in preprocessing)
if 'Iteration' in filtered_data.columns:
    try:
        # Calculate cumulative successes over iterations using 'Result_Verified'
        trend_data = filtered_data.sort_values('Iteration').groupby('Iteration').agg(
            Successful_Results=('Result_Verified', lambda x: (x == success_label).sum())
        ).reset_index()

        # Apply rolling average for smoothing
        trend_data['Rolling_Success'] = trend_data['Successful_Results'].rolling(window=5).mean()

        # Create the main line plot without the 'name' parameter
        fig_trend = px.line(
            trend_data,
            x='Iteration',
            y='Successful_Results',
            title='Trend of Successful Results Over Iterations',
            labels={'Successful_Results': 'Number of Successes', 'Iteration': 'Iteration'},
            markers=True
        )

        # Add the Rolling Average line using Plotly Graph Objects
        fig_trend.add_trace(go.Scatter(
            x=trend_data['Iteration'],
            y=trend_data['Rolling_Success'],
            mode='lines',
            name='Rolling Average (5)',
            line=dict(color='red', dash='dash')
        ))

        # Update layout for better visualization
        fig_trend.update_layout(
            legend=dict(title='Legend'),
            hovermode='x unified'
        )

        st.plotly_chart(fig_trend, use_container_width=True, key='fig_trend')

        # ---------------------------
        # Removed Matrix Tables from Visualization 7
        # ---------------------------

        # ---------------------------
        # Added: Iteration 63 Analysis
        # ---------------------------
        
        st.markdown("---")  # Separator for clarity
        st.subheader("Iteration 63 Analysis")

        # Filter data for Iteration 63
        iteration_63_data = filtered_data[filtered_data['Iteration'] == 63]

        if not iteration_63_data.empty:
            st.markdown("**Overview of Iteration 63:**")
            st.write(f"**Total Conversations:** {iteration_63_data.shape[0]}")

            # Display key metrics
            metrics_63 = {
                'Successful Results': (iteration_63_data['Result_Verified'] == success_label).sum(),
                'Average Sentiment Score': iteration_63_data['Avg_Sentiment_Charles'].mean(),
                'Average Carla Response Length': iteration_63_data['Avg_Carla_Response_Length'].mean(),
                'Average Charles Response Length': iteration_63_data['Avg_Charles_Response_Length'].mean(),
                'Total Emojis by Carla': iteration_63_data['Emoji_Count_Carla'].sum(),
                'Total Financial Requests Made by Carla': iteration_63_data['Carla_Request_Message_Number'].sum(),
                'Total Questions by Carla': iteration_63_data['Question_Count_Carla'].sum(),
                'Total Questions by Charles': iteration_63_data['Question_Count_Charles'].sum()
            }

            metrics_df = pd.DataFrame(list(metrics_63.items()), columns=['Metric', 'Value'])
            st.table(metrics_df.style.format({'Value': '{:.2f}'}))

            # Visualizations for Iteration 63

            # 1. Success Rate in Iteration 63
            success_rate_63 = (iteration_63_data['Result_Verified'] == success_label).mean() * 100
            st.markdown(f"**Success Rate:** {success_rate_63:.2f}%")

            # 2. Sentiment Score Distribution
            fig_sentiment_63 = px.histogram(
                iteration_63_data,
                x='Avg_Sentiment_Charles',
                nbins=20,
                title="Distribution of Charles' Sentiment Scores in Iteration 63",
                labels={'Avg_Sentiment_Charles': 'Average Sentiment Score'},
                color_discrete_sequence=['teal']
            )
            st.plotly_chart(fig_sentiment_63, use_container_width=True, key='fig_sentiment_63')

            # 3. Response Lengths
            fig_response_length_63 = go.Figure()
            fig_response_length_63.add_trace(go.Box(
                y=iteration_63_data['Avg_Carla_Response_Length'],
                name='Carla Response Length',
                marker_color='orange'
            ))
            fig_response_length_63.add_trace(go.Box(
                y=iteration_63_data['Avg_Charles_Response_Length'],
                name='Charles Response Length',
                marker_color='blue'
            ))
            fig_response_length_63.update_layout(
                title="Distribution of Response Lengths in Iteration 63",
                yaxis_title="Response Length",
                boxmode='group'
            )
            st.plotly_chart(fig_response_length_63, use_container_width=True, key='fig_response_length_63')

            # 4. Emoji Usage by Carla
            #fig_emoji_63 = px.histogram(
           #     iteration_63_data,
           #      x='Emoji_Count_Carla',
           #     nbins=10,
           #     title="Emoji Usage by Carla in Iteration 63",
           #     labels={'Emoji_Count_Carla': 'Number of Emojis'},
           #     color_discrete_sequence=['purple']
           # )
           # st.plotly_chart(fig_emoji_63, use_container_width=True, key='fig_emoji_63')

            # 5. Financial Requests by Carla
           # fig_financial_63 = px.histogram(
           #     iteration_63_data,
           #     x='Carla_Request_Message_Number',
           #     nbins=10,
           #     title="Financial Requests Made by Carla in Iteration 63",
           #     labels={'Carla_Request_Message_Number': 'Number of Financial Requests'},
           #     color_discrete_sequence=['green']
           # )
           # st.plotly_chart(fig_financial_63, use_container_width=True, key='fig_financial_63')

            # 6. Questions Asked
            fig_questions_63 = go.Figure()
            fig_questions_63.add_trace(go.Bar(
                x=['Carla', 'Charles'],
                y=[
                    iteration_63_data['Question_Count_Carla'].sum(),
                    iteration_63_data['Question_Count_Charles'].sum()
                ],
                marker_color=['salmon', 'skyblue']
            ))
            fig_questions_63.update_layout(
                title="Total Questions Asked in Iteration 63",
                xaxis_title="Participant",
                yaxis_title="Number of Questions"
            )
            st.plotly_chart(fig_questions_63, use_container_width=True, key='fig_questions_63')

            # 7. Comparison with Overall Metrics
            st.markdown("**Comparison with Overall Metrics:**")

            # Calculate overall metrics excluding Iteration 63 for comparison
            overall_data = filtered_data[filtered_data['Iteration'] != 63]

            overall_metrics = {
                'Overall Success Rate': (overall_data['Result_Verified'] == success_label).mean() * 100,
                'Overall Average Sentiment Score': overall_data['Avg_Sentiment_Charles'].mean(),
                'Overall Average Carla Response Length': overall_data['Avg_Carla_Response_Length'].mean(),
                'Overall Average Charles Response Length': overall_data['Avg_Charles_Response_Length'].mean(),
                'Overall Total Emojis by Carla': overall_data['Emoji_Count_Carla'].sum(),
                'Overall Total Financial Requests Made by Carla': overall_data['Carla_Request_Message_Number'].sum(),
                'Overall Total Questions by Carla': overall_data['Question_Count_Carla'].sum(),
                'Overall Total Questions by Charles': overall_data['Question_Count_Charles'].sum()
            }

            overall_metrics_df = pd.DataFrame(list(overall_metrics.items()), columns=['Metric', 'Overall_Value'])
            st.table(overall_metrics_df.style.format({'Overall_Value': '{:.2f}'}))

        else:
            st.warning("No data available for Iteration 63.")
        
    except Exception as e:
        logging.error(f"Error in Visualization 7: {e}")
        st.error(f"An error occurred while generating Visualization 7: {e}")
else:
    st.warning("'Iteration' column is missing from the data.")


# ---------------------------
# Visualization 8: Heatmap of Correlations Between Numerical Metrics
# ---------------------------

st.subheader("8. Heatmap of Correlations Between Numerical Metrics")

# Select numerical columns for correlation matrix
all_numerical_metrics = [
    'Num_Turns_mean', 'Avg_Carla_Response_Length_mean',
    'Avg_Charles_Response_Length_mean', 'Avg_Sentiment_Charles_mean',
    'Financial_Requests_Made_mean', 'Emoji_Count_Carla_sum',
    'Question_Count_Carla_sum', 'Question_Count_Charles_sum',
    'Success_Rate'
]

# Check if all numerical metrics exist in analysis_df
missing_corr_metrics = [col for col in all_numerical_metrics if col not in analysis_df.columns]
if not missing_corr_metrics:
    # Calculate correlation matrix using analysis_df
    corr_matrix_all = analysis_df[all_numerical_metrics].corr()

    fig_corr_heatmap = px.imshow(
        corr_matrix_all,
        text_auto=True,
        aspect="auto",
        title='Correlation Matrix of Numerical Metrics',
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )

    # Assign a unique key to avoid duplicate ID errors
    st.plotly_chart(fig_corr_heatmap, use_container_width=True, key='corr_heatmap_v8')

    # ---------------------------
    # Removed Matrix Table for Visualization 8
    # ---------------------------

    # 
else:
    st.warning(f"The following numerical metrics are missing from the data and won't be included in the correlation matrix: {', '.join(missing_corr_metrics)}")

# ---------------------------
# Visualization 9: Engagement Metrics Over Datasets
# ---------------------------

st.subheader("9. Engagement Metrics Over Datasets")

# Define engagement metrics with correct aggregated column names
engagement_metrics = {
    'Num_Turns_mean': 'Average Number of Turns',
    'Avg_Carla_Response_Length_mean': 'Average Carla Response Length',
    'Avg_Charles_Response_Length_mean': 'Average Charles Response Length',
    'Avg_Sentiment_Charles_mean': 'Average Sentiment Score',
    'Financial_Requests_Made_mean': 'Average Financial Requests Made',
    'Emoji_Count_Carla_sum': 'Total Emojis by Carla',
    'Question_Count_Carla_sum': 'Total Questions by Carla',
    'Question_Count_Charles_sum': 'Total Questions by Charles'
}

# Check if all engagement metrics exist in analysis_df
missing_engagement = [col for col in engagement_metrics.keys() if col not in analysis_df.columns]
if not missing_engagement:
    try:
        # Extract relevant columns for engagement metrics
        engagement_data = analysis_df[['Dataset_Title'] + list(engagement_metrics.keys())].copy()
        
        # Rename columns for clarity
        engagement_data.rename(columns=engagement_metrics, inplace=True)
        
        # Select the visualization type
        viz_options = {
            "Grouped Bar Chart": "grouped_bar",
            "Parallel Coordinates Plot": "parallel_coordinates",
            "Heatmap of Engagement Metrics": "heatmap"
        }
        
        selected_viz = st.selectbox(
            "Select Visualization Type for Engagement Metrics",
            options=list(viz_options.keys()),
            index=0
        )
        
        # Implement the selected visualization
        if viz_options[selected_viz] == "grouped_bar":
            # 1. Grouped Bar Chart
            
            # Reshape data for a grouped bar chart
            engagement_melted = engagement_data.melt(
                id_vars='Dataset_Title',
                value_vars=list(engagement_metrics.values()),
                var_name='Metric',
                value_name='Average Value'
            )
            
            # Create Grouped Bar Chart
            fig_grouped_bar = px.bar(
                engagement_melted,
                x='Metric',
                y='Average Value',
                color='Dataset_Title',
                barmode='group',
                title="Grouped Bar Chart of Engagement Metrics Over Datasets",
                labels={'Average Value': 'Average Value', 'Metric': 'Engagement Metric'},
                height=600
            )
            
            fig_grouped_bar.update_layout(
                xaxis_title="Engagement Metric",
                yaxis_title="Average Value",
                legend_title="Dataset",
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig_grouped_bar, use_container_width=True, key='fig_grouped_bar_v9')
        
        elif viz_options[selected_viz] == "parallel_coordinates":
            # 2. Parallel Coordinates Plot
            
            # Create Parallel Coordinates Plot
            fig_parallel = px.parallel_coordinates(
                engagement_data,
                color="Average Sentiment Score",
                dimensions=list(engagement_metrics.values()),
                color_continuous_scale=px.colors.diverging.Tealrose,
                color_continuous_midpoint=np.mean(engagement_data["Average Sentiment Score"]),
                title="Parallel Coordinates of Engagement Metrics Over Datasets",
                labels={metric: metric for metric in engagement_metrics.values()},
                height=600
            )
            
            st.plotly_chart(fig_parallel, use_container_width=True, key='fig_parallel_coordinates_v9')
        
        elif viz_options[selected_viz] == "heatmap":
            # 3. Heatmap of Engagement Metrics
            
            # Calculate correlation matrix
            corr_matrix_engagement = engagement_data[list(engagement_metrics.values())].corr()
            
            # Create Heatmap
            fig_heatmap_engagement = px.imshow(
                corr_matrix_engagement,
                text_auto=True,
                aspect="auto",
                title='Correlation Matrix of Engagement Metrics',
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            
            fig_heatmap_engagement.update_layout(
                xaxis_title="Engagement Metric",
                yaxis_title="Engagement Metric",
                margin=dict(l=100, r=100, t=100, b=100)
            )
            
            st.plotly_chart(fig_heatmap_engagement, use_container_width=True, key='fig_heatmap_engagement_v9')

        # ---------------------------
        # Matrix Table for Visualization 9
        # ---------------------------

        st.markdown("### Engagement Metrics Success Rates Matrix")

        try:
            # Assuming 'Engagement Metrics' relate to success rates, create a matrix based on 'Success_Rate'
            # Extract Carla and Charles models
            engagement_data[['Carla_Model', 'Charles_Model']] = engagement_data['Dataset_Title'].apply(
                lambda x: pd.Series(extract_models(x))
            )

            # Merge with success rates
            engagement_data = pd.merge(engagement_data, success_percentage_sorted[['Dataset_Title', 'Success_Rate_Percentage']], on='Dataset_Title', how='left')

            # Create pivot table for Success Rate Percentage
            matrix_data_engagement = engagement_data.groupby(['Carla_Model', 'Charles_Model'])['Success_Rate_Percentage'].mean().reset_index()
            matrix_pivot_engagement = matrix_data_engagement.pivot(index='Carla_Model', columns='Charles_Model', values='Success_Rate_Percentage')

            # Format the matrix table
            formatted_matrix_engagement = format_matrix_table(matrix_pivot_engagement)

            # Display the matrix table
            st.markdown("**Success Rates Matrix Based on Engagement Metrics (Percentage):**")
            st.table(formatted_matrix_engagement)
        except Exception as e:
            logging.error(f"Error creating Engagement Metrics Success Rates Matrix for Visualization 9: {e}")
            st.error(f"An error occurred while creating Engagement Metrics Success Rates Matrix: {e}")

    except Exception as e:
        logging.error(f"Error in Visualization 9: {e}")
        st.error(f"An error occurred while generating Visualization 9: {e}")
else:
    st.warning(f"The following engagement metrics are missing from the data: {', '.join(missing_engagement)}")

# ---------------------------
# Visualization 10: Scatter Plot of Question Counts vs. Success Rate
# ---------------------------

st.subheader("10. Scatter Plot of Question Counts vs. Success Rate")

# Choose which question count to plot: Carla or Charles
question_types = ['Question_Count_Carla_sum', 'Question_Count_Charles_sum']
selected_question = st.selectbox(
    "Select Question Type", 
    options=question_types, 
    format_func=lambda x: 'Questions by Carla' if x == 'Question_Count_Carla_sum' else 'Questions by Charles'
)

if selected_question in analysis_df.columns:
    try:
        # Calculate total questions and success rate per Dataset
        question_success = analysis_df.groupby('Dataset_Title').agg(
            Total_Questions=(selected_question, 'sum'),
            Success_Rate=('Success_Rate', 'mean')  # Already present
        ).reset_index()
        question_success.columns = ['Dataset_Title', 'Total_Questions', 'Success_Rate']

        # Create Scatter Plot
        fig_question_scatter = px.scatter(
            question_success,
            x='Total_Questions',
            y='Success_Rate',
            text='Dataset_Title',
            title=f"Total Questions ({'Carla' if selected_question == 'Question_Count_Carla_sum' else 'Charles'}) vs. Success Rate",
            labels={
                'Total_Questions': f"Total Questions ({'Carla' if selected_question == 'Question_Count_Carla_sum' else 'Charles'})",
                'Success_Rate': 'Success Rate'
            },
            size='Success_Rate',
            hover_data=['Dataset_Title'],
            trendline='ols'
        )

        fig_question_scatter.update_traces(textposition='top center')

        st.plotly_chart(fig_question_scatter, use_container_width=True, key='fig_question_scatter')

        # Add a note beneath the visualization
        st.markdown("""
        **Note:** The size of the circles are proportionate to the **Success Rate** for each dataset.
        """)

        # Display the table of Total Questions
        st.markdown("**Total Questions per Dataset:**")
        st.table(question_success[['Dataset_Title', 'Total_Questions']].style.format({'Total_Questions': '{:.0f}'}))

        # Add Percentage Table
        try:
            # Calculate Success Rate Percentage
            question_success['Success_Rate_Percentage'] = question_success['Success_Rate'] * 100

            # Sort by Success Rate Percentage
            question_success_sorted = question_success.sort_values(by='Success_Rate_Percentage', ascending=False)

            # Display the percentage table
            st.markdown("**Success Rate Percentage per Dataset:**")
            st.table(question_success_sorted[['Dataset_Title', 'Success_Rate_Percentage']].style.format({'Success_Rate_Percentage': '{:.0f}%'}))
        except Exception as e:
            logging.error(f"Error calculating Success Rate Percentage for Visualization 10: {e}")
            st.error(f"An error occurred while calculating Success Rate Percentage: {e}")

        # Calculate and display Pearson correlation coefficient
        correlation_q = question_success['Total_Questions'].corr(question_success['Success_Rate'])
        st.markdown(f"**Pearson Correlation Coefficient:** {correlation_q:.2f}")

        # Interpretation
        if correlation_q > 0:
            interpretation_q = "a positive correlation"
        elif correlation_q < 0:
            interpretation_q = "a negative correlation"
        else:
            interpretation_q = "no correlation"

        st.markdown(f"This indicates **{interpretation_q}** between the number of questions {'Carla' if selected_question == 'Question_Count_Carla_sum' else 'Charles'} asks and the success rate.")
    except Exception as e:
        logging.error(f"Error in Visualization 10: {e}")
        st.error(f"An error occurred while generating Visualization 10: {e}")
else:
    st.warning(f"Selected question count column '{selected_question}' is missing from the data.")

# ---------------------------
# Visualization 11: Carla's Average Request Message Number with Linear Regression
# ---------------------------

st.subheader("11. Carla's Average Request Message Number")

# Check if 'Carla_Request_Message_Number' exists in the data
if 'Carla_Request_Message_Number' in filtered_data.columns:
    try:
        # Filter out zero values
        carla_requests_non_zero = filtered_data[filtered_data['Carla_Request_Message_Number'] > 0]

        # Calculate average Carla_Request_Message_Number per Dataset
        carla_avg = carla_requests_non_zero.groupby('Dataset_Title')['Carla_Request_Message_Number'].mean().reset_index()
        carla_avg.columns = ['Dataset_Title', 'Average_Carla_Request_Message_Number']

        # Merge with Success Rate Percentage
        carla_avg = pd.merge(carla_avg, success_percentage_sorted[['Dataset_Title', 'Success_Rate_Percentage']], on='Dataset_Title', how='left')

        # Create two columns for side-by-side visualization
        col1, col2 = st.columns(2)

        with col1:
            # Create a bar chart
            fig_carla_avg = px.bar(
                carla_avg,
                x='Dataset_Title',
                y='Average_Carla_Request_Message_Number',
                title="Carla's Average Request Message Number per Dataset (Zeros Ignored)",
                labels={
                    'Average_Carla_Request_Message_Number': "Average Request Message Number",
                    'Dataset_Title': 'Dataset'
                },
                color='Average_Carla_Request_Message_Number',
                color_continuous_scale='Oranges'
            )

            st.plotly_chart(fig_carla_avg, use_container_width=True, key='fig_carla_avg')

            # Display the overall average across all datasets
            overall_avg = carla_requests_non_zero['Carla_Request_Message_Number'].mean()
            st.markdown(f"**Overall Average Carla_Request_Message_Number (Zeros Ignored):** {overall_avg:.2f}")

        with col2:
            # Create a scatter plot with regression line
            fig_linear_carla = px.scatter(
                carla_avg,
                x='Average_Carla_Request_Message_Number',
                y='Success_Rate_Percentage',
                text='Dataset_Title',
                title="Carla's Request Timing vs. Success Rate",
                labels={
                    'Average_Carla_Request_Message_Number': "Average Request Message Number",
                    'Success_Rate_Percentage': "Success Rate (%)"
                },
                size='Success_Rate_Percentage',
                hover_data=['Dataset_Title'],
                trendline='ols',  # Adds a linear regression line
                color='Success_Rate_Percentage',
                color_continuous_scale='Viridis'
            )

            fig_linear_carla.update_traces(textposition='top center')

            st.plotly_chart(fig_linear_carla, use_container_width=True, key='fig_linear_carla_regression')

            # Add a note beneath the scatter plot
            st.markdown("""
                **Note:** The size and color of the circles are proportionate to the **Success Rate (%)** for each dataset.
                """)

        # Display the overall average if needed
        # st.markdown(f"**Overall Average Carla_Request_Message_Number (Zeros Ignored):** {overall_avg:.2f}")

        # --- Adding Linear Regression Model Summary ---
        st.markdown("### Linear Regression Model Summary")

        # Prepare data for the linear model
        model_data_carla = carla_avg[['Average_Carla_Request_Message_Number', 'Success_Rate_Percentage']].dropna()

        if not model_data_carla.empty:
            try:
                # Define the formula for the linear model
                formula_carla = 'Success_Rate_Percentage ~ Average_Carla_Request_Message_Number'

                # Fit the linear regression model using statsmodels
                model_carla = smf.ols(formula=formula_carla, data=model_data_carla).fit()

                # Display the model summary
                st.text(model_carla.summary())

                # Display Pearson correlation coefficient
                correlation_carla = model_data_carla['Average_Carla_Request_Message_Number'].corr(model_data_carla['Success_Rate_Percentage'])
                st.markdown(f"**Pearson Correlation Coefficient:** {correlation_carla:.2f}")

                # Interpretation
                if correlation_carla > 0:
                    interpretation_carla = "a positive correlation"
                elif correlation_carla < 0:
                    interpretation_carla = "a negative correlation"
                else:
                    interpretation_carla = "no correlation"

                st.markdown(f"This indicates **{interpretation_carla}** between the average point at which Carla makes financial requests and the success rate.")

            except Exception as e:
                logging.error(f"Error building linear regression model for Visualization 11: {e}")
                st.error(f"An error occurred while building the linear regression model: {e}")
        else:
            st.warning("Not enough data to build a linear regression model.")

        # ---------------------------
        # Matrix Table for Visualization 11
        # ---------------------------

        st.markdown("### Carla's Request Success Rates Matrix")

        try:
            # Calculate success rates for each pairing based on request messages
            matrix_data_carla_requests = filtered_data.groupby(['Carla_Model', 'Charles_Model']).apply(
                lambda df: (df['Carla_Request_Message_Number'] > 0).mean() * 100
            ).reset_index(name='Request_Success_Rate')

            matrix_pivot_carla_requests = matrix_data_carla_requests.pivot(index='Carla_Model', columns='Charles_Model', values='Request_Success_Rate')

            # Format the matrix table
            formatted_matrix_carla_requests = format_matrix_table(matrix_pivot_carla_requests)

            # Display the matrix table
            st.markdown("**Carla's Request Success Rates Matrix (Percentage):**")
            st.table(formatted_matrix_carla_requests)
        except Exception as e:
            logging.error(f"Error creating Carla's Request Success Rates Matrix for Visualization 11: {e}")
            st.error(f"An error occurred while creating Carla's Request Success Rates Matrix: {e}")

    except Exception as e:
        logging.error(f"Error in Visualization 11: {e}")
        st.error(f"An error occurred while generating Visualization 11: {e}")
else:
    st.warning("'Carla_Request_Message_Number' column is missing from the data.")


# ---------------------------
# Visualization 12: Percentage of Verified Success per Dataset
# ---------------------------

st.subheader("12. Percentage of Verified Success per Dataset")

if 'Success_Rate' in analysis_df.columns:
    try:
        # Create a dataframe with Success Rate as percentage
        success_percentage = analysis_df[['Dataset_Title', 'Success_Rate']].copy()
        success_percentage['Success_Rate_Percentage'] = success_percentage['Success_Rate'] * 100
        success_percentage = success_percentage[['Dataset_Title', 'Success_Rate_Percentage']].sort_values(by='Success_Rate_Percentage', ascending=False)

        # Display as a table
        st.markdown("**Success Rate Percentage per Dataset:**")
        st.table(success_percentage.style.format({'Success_Rate_Percentage': '{:.0f}%'}))
        
        # Optional: Bar Chart Visualization
        fig_success_percentage = px.bar(
            success_percentage,
            x='Dataset_Title',
            y='Success_Rate_Percentage',
            title='Percentage of Verified Success per Dataset',
            labels={'Success_Rate_Percentage': 'Success Rate (%)', 'Dataset_Title': 'Dataset'},
            color='Success_Rate_Percentage',
            color_continuous_scale='Greens'
        )
        
        st.plotly_chart(fig_success_percentage, use_container_width=True, key='fig_success_percentage')

        # ---------------------------
        # Removed Matrix Table for Visualization 12
        # ---------------------------

    except Exception as e:
        logging.error(f"Error in Visualization 12: {e}")
        st.error(f"An error occurred while generating Visualization 12: {e}")
else:
    st.warning("'Success_Rate' column is missing from the analysis data.")

# ---------------------------
# Visualization 13: Finances Being Mentioned per Dataset
# ---------------------------

st.subheader("13. Finances Being Mentioned per Dataset")

if 'Financial_Requests_Made' in filtered_data.columns and 'Dataset_Title' in filtered_data.columns:
    try:
        # Define conversations where finances are mentioned
        filtered_data['Finances_Mentioned'] = filtered_data['Financial_Requests_Made'] > 0

        # Calculate number of conversations with finances mentioned per Dataset
        finances_mentioned_counts = filtered_data.groupby('Dataset_Title')['Finances_Mentioned'].sum().reset_index()
        finances_mentioned_counts.columns = ['Dataset_Title', 'Conversations_with_Finances_Mentioned']

        # Calculate total conversations per Dataset
        total_conversations = filtered_data.groupby('Dataset_Title')['Finances_Mentioned'].count().reset_index()
        total_conversations.columns = ['Dataset_Title', 'Total_Conversations']

        # Merge the two dataframes
        finances_percentage = pd.merge(finances_mentioned_counts, total_conversations, on='Dataset_Title')

        # Calculate the percentage
        finances_percentage['Finances_Mentioned_Percentage'] = (finances_percentage['Conversations_with_Finances_Mentioned'] / finances_percentage['Total_Conversations']) * 100

        # Sort by percentage descending
        finances_percentage_sorted = finances_percentage.sort_values(by='Finances_Mentioned_Percentage', ascending=False)

        # Display as a table
        st.markdown("**Finances Being Mentioned Percentage per Dataset:**")
        st.table(finances_percentage_sorted[['Dataset_Title', 'Finances_Mentioned_Percentage']].style.format({'Finances_Mentioned_Percentage': '{:.0f}%'}))

        # Optional: Bar Chart Visualization
        fig_finances_percentage_v15 = px.bar(
            finances_percentage_sorted,
            x='Dataset_Title',
            y='Finances_Mentioned_Percentage',
            title='Percentage of Conversations with Finances Mentioned per Dataset',
            labels={'Finances_Mentioned_Percentage': 'Finances Mentioned (%)', 'Dataset_Title': 'Dataset'},
            color='Finances_Mentioned_Percentage',
            color_continuous_scale='Purples'
        )

        st.plotly_chart(fig_finances_percentage_v15, use_container_width=True, key='fig_finances_percentage_v15')

        # ---------------------------
        # Matrix Table for Visualization 13
        # ---------------------------

        st.markdown("### Finances Mentioned Rates Matrix")

        try:
            # Calculate finances mentioned rates for each pairing
            matrix_data_finances = filtered_data.groupby(['Carla_Model', 'Charles_Model'])['Finances_Mentioned'].mean().reset_index()
            matrix_data_finances['Finances_Mentioned_Percentage'] = matrix_data_finances['Finances_Mentioned'] * 100

            matrix_pivot_finances = matrix_data_finances.pivot(index='Carla_Model', columns='Charles_Model', values='Finances_Mentioned_Percentage')

            # Format the matrix table
            formatted_matrix_finances = format_matrix_table(matrix_pivot_finances)

            # Display the matrix table
            st.markdown("**Finances Mentioned Rates Matrix (Percentage):**")
            st.table(formatted_matrix_finances)
        except Exception as e:
            logging.error(f"Error creating Finances Mentioned Rates Matrix for Visualization 13: {e}")
            st.error(f"An error occurred while creating Finances Mentioned Rates Matrix: {e}")

    except Exception as e:
        logging.error(f"Error in Visualization 13: {e}")
        st.error(f"An error occurred while generating Visualization 13: {e}")
else:
    st.warning("Required columns for Finances Being Mentioned Percentage are missing from the data.")

# ---------------------------
# Visualization 14: Percentage of Conversations with Emojis in Carla's Messages per Dataset
# ---------------------------

st.subheader("14. Percentage of Conversations with Emojis in Carla's Messages per Dataset")

if 'Emoji_Count_Carla' in filtered_data.columns and 'Dataset_Title' in filtered_data.columns:
    try:
        # Define conversations with at least one emoji
        filtered_data['Contains_Emoji'] = filtered_data['Emoji_Count_Carla'] > 0

        # Calculate number of conversations with emojis per Dataset
        emoji_counts = filtered_data.groupby('Dataset_Title')['Contains_Emoji'].sum().reset_index()
        emoji_counts.columns = ['Dataset_Title', 'Conversations_with_Emojis']

        # Calculate total conversations per Dataset
        total_conversations = filtered_data.groupby('Dataset_Title')['Contains_Emoji'].count().reset_index()
        total_conversations.columns = ['Dataset_Title', 'Total_Conversations']

        # Merge and calculate percentage
        emoji_percentage = pd.merge(emoji_counts, total_conversations, on='Dataset_Title')
        emoji_percentage['Emoji_Usage_Percentage'] = (emoji_percentage['Conversations_with_Emojis'] / emoji_percentage['Total_Conversations']) * 100

        # Sort by percentage descending
        emoji_percentage_sorted = emoji_percentage.sort_values(by='Emoji_Usage_Percentage', ascending=False)

        # Display as a table
        st.markdown("**Emoji Usage Percentage per Dataset:**")
        st.table(emoji_percentage_sorted[['Dataset_Title', 'Emoji_Usage_Percentage']].style.format({'Emoji_Usage_Percentage': '{:.0f}%'}))

        # Optional: Bar Chart Visualization
        fig_emoji_percentage_v16 = px.bar(
            emoji_percentage_sorted,
            x='Dataset_Title',
            y='Emoji_Usage_Percentage',
            title="Percentage of Conversations with Emojis in Carla's Messages per Dataset",
            labels={'Emoji_Usage_Percentage': 'Emoji Usage (%)', 'Dataset_Title': 'Dataset'},
            color='Emoji_Usage_Percentage',
            color_continuous_scale='Blues'
        )

        st.plotly_chart(fig_emoji_percentage_v16, use_container_width=True, key='fig_emoji_usage_percentage_v16')

        # ---------------------------
        # Matrix Table for Visualization 14
        # ---------------------------

        st.markdown("### Emojis Usage Rates Matrix")

        try:
            # Calculate emojis usage rates for each pairing
            matrix_data_emojis = filtered_data.groupby(['Carla_Model', 'Charles_Model'])['Contains_Emoji'].mean().reset_index()
            matrix_data_emojis['Emoji_Usage_Percentage'] = matrix_data_emojis['Contains_Emoji'] * 100

            matrix_pivot_emojis = matrix_data_emojis.pivot(index='Carla_Model', columns='Charles_Model', values='Emoji_Usage_Percentage')

            # Format the matrix table
            formatted_matrix_emojis = format_matrix_table(matrix_pivot_emojis)

            # Display the matrix table
            st.markdown("**Emojis Usage Rates Matrix (Percentage):**")
            st.table(formatted_matrix_emojis)
        except Exception as e:
            logging.error(f"Error creating Emojis Usage Rates Matrix for Visualization 14: {e}")
            st.error(f"An error occurred while creating Emojis Usage Rates Matrix: {e}")

    except Exception as e:
        logging.error(f"Error in Visualization 14: {e}")
        st.error(f"An error occurred while generating Visualization 14: {e}")
else:
    st.warning("Required columns for Emoji Usage Percentage are missing from the data.")

# ---------------------------
# Visualization 15: Percentage of Conversations by Strategy Type per Dataset
# ---------------------------

st.subheader("15. Percentage of Conversations by Strategy Type per Dataset")

if 'Predominant_Strategy_Carla' in filtered_data.columns and 'Dataset_Title' in filtered_data.columns:
    try:
        # Calculate strategy counts per Dataset
        strategy_counts = filtered_data.groupby(['Dataset_Title', 'Predominant_Strategy_Carla']).size().reset_index(name='Count')

        # Calculate total conversations per Dataset
        total_conversations = filtered_data.groupby('Dataset_Title').size().reset_index(name='Total_Conversations')

        # Merge and calculate percentage
        strategy_percentage = pd.merge(strategy_counts, total_conversations, on='Dataset_Title')
        strategy_percentage['Strategy_Usage_Percentage'] = (strategy_percentage['Count'] / strategy_percentage['Total_Conversations']) * 100

        # Sort values for better visualization
        strategy_percentage_sorted = strategy_percentage.sort_values(by=['Dataset_Title', 'Strategy_Usage_Percentage'], ascending=[True, False])

        # Display as a table
        st.markdown("**Strategy Usage Percentage per Dataset:**")
        st.table(strategy_percentage_sorted[['Dataset_Title', 'Predominant_Strategy_Carla', 'Strategy_Usage_Percentage']].style.format({'Strategy_Usage_Percentage': '{:.0f}%'}))

        # Optional: Stacked Bar Chart Visualization
        fig_strategy_percentage_v17 = px.bar(
            strategy_percentage_sorted,
            x='Dataset_Title',
            y='Strategy_Usage_Percentage',
            color='Predominant_Strategy_Carla',
            title='Percentage Distribution of Carla\'s Strategies per Dataset',
            labels={'Strategy_Usage_Percentage': 'Strategy Usage (%)', 'Dataset_Title': 'Dataset', 'Predominant_Strategy_Carla': 'Strategy'},
            color_discrete_sequence=px.colors.qualitative.Pastel,
            barmode='stack'
        )

        st.plotly_chart(fig_strategy_percentage_v17, use_container_width=True, key='fig_strategy_usage_percentage_v17')

        # ---------------------------
        # Removed Matrix Table for Visualization 15
        # ---------------------------
        # 

    except Exception as e:
        logging.error(f"Error in Visualization 15: {e}")
        st.error(f"An error occurred while generating Visualization 15: {e}")
else:
    st.warning("Required columns for Strategy Usage Percentage are missing from the data.")


# ---------------------------
# Visualization 16: Percentage of Conversations with High Sentiment Scores per Dataset
# ---------------------------

st.subheader("16. Percentage of Conversations with High Sentiment Scores per Dataset")

if 'Avg_Sentiment_Charles' in filtered_data.columns and 'Dataset_Title' in filtered_data.columns:
    try:
        # Determine if a conversation has high sentiment
        filtered_data['High_Sentiment'] = filtered_data['Avg_Sentiment_Charles'] > sentiment_threshold

        # Calculate number of high sentiment conversations per Dataset
        high_sentiment_counts = filtered_data.groupby('Dataset_Title')['High_Sentiment'].sum().reset_index()
        high_sentiment_counts.columns = ['Dataset_Title', 'High_Sentiment_Conversations']

        # Calculate total conversations per Dataset
        total_conversations = filtered_data.groupby('Dataset_Title')['High_Sentiment'].count().reset_index()
        total_conversations.columns = ['Dataset_Title', 'Total_Conversations']

        # Merge and calculate percentage
        high_sentiment_percentage = pd.merge(high_sentiment_counts, total_conversations, on='Dataset_Title')
        high_sentiment_percentage['High_Sentiment_Percentage'] = (high_sentiment_percentage['High_Sentiment_Conversations'] / high_sentiment_percentage['Total_Conversations']) * 100

        # Sort by percentage descending
        high_sentiment_percentage_sorted = high_sentiment_percentage.sort_values(by='High_Sentiment_Percentage', ascending=False)

        # Display as a table
        st.markdown(f"**High Sentiment Percentage (Sentiment > {sentiment_threshold}) per Dataset:**")
        st.table(high_sentiment_percentage_sorted[['Dataset_Title', 'High_Sentiment_Percentage']].style.format({'High_Sentiment_Percentage': '{:.0f}%'}))

        # Optional: Bar Chart Visualization
        fig_high_sentiment_percentage_v18 = px.bar(
            high_sentiment_percentage_sorted,
            x='Dataset_Title',
            y='High_Sentiment_Percentage',
            title=f'Percentage of Conversations with Sentiment > {sentiment_threshold} per Dataset',
            labels={'High_Sentiment_Percentage': f'High Sentiment Conversations (%)', 'Dataset_Title': 'Dataset'},
            color='High_Sentiment_Percentage',
            color_continuous_scale='Oranges'
        )

        st.plotly_chart(fig_high_sentiment_percentage_v18, use_container_width=True, key='fig_high_sentiment_percentage_v18')

        # ---------------------------
        # Matrix Table for Visualization 16
        # ---------------------------

        st.markdown("### High Sentiment Success Rates Matrix")

        try:
            # Calculate high sentiment success rates for each pairing
            matrix_data_high_sentiment = filtered_data.groupby(['Carla_Model', 'Charles_Model'])['High_Sentiment'].mean().reset_index()
            matrix_data_high_sentiment['High_Sentiment_Percentage'] = matrix_data_high_sentiment['High_Sentiment'] * 100

            matrix_pivot_high_sentiment = matrix_data_high_sentiment.pivot(index='Carla_Model', columns='Charles_Model', values='High_Sentiment_Percentage')

            # Format the matrix table
            formatted_matrix_high_sentiment = format_matrix_table(matrix_pivot_high_sentiment)

            # Display the matrix table
            st.markdown("**High Sentiment Success Rates Matrix (Percentage):**")
            st.table(formatted_matrix_high_sentiment)
        except Exception as e:
            logging.error(f"Error creating High Sentiment Success Rates Matrix for Visualization 16: {e}")
            st.error(f"An error occurred while creating High Sentiment Success Rates Matrix: {e}")

    except Exception as e:
        logging.error(f"Error in Visualization 16: {e}")
        st.error(f"An error occurred while generating Visualization 16: {e}")
else:
    st.warning("'Avg_Sentiment_Charles' or 'Dataset_Title' column is missing from the data.")

# ---------------------------
# Visualization 17: Percentage of Conversations with Financial Requests Made per Dataset
# ---------------------------

st.subheader("17. Percentage of Conversations with Financial Requests Made per Dataset")

if 'Carla_Request_Message_Number' in filtered_data.columns and 'Dataset_Title' in filtered_data.columns:
    try:
        # Define conversations where financial requests were made
        filtered_data['Financial_Request_Made'] = filtered_data['Carla_Request_Message_Number'] > 0

        # Calculate number of conversations with financial requests per Dataset
        financial_requests_counts = filtered_data.groupby('Dataset_Title')['Financial_Request_Made'].sum().reset_index()
        financial_requests_counts.columns = ['Dataset_Title', 'Conversations_with_Financial_Requests']

        # Calculate total conversations per Dataset
        total_conversations = filtered_data.groupby('Dataset_Title')['Financial_Request_Made'].count().reset_index()
        total_conversations.columns = ['Dataset_Title', 'Total_Conversations']

        # Merge the two dataframes
        financial_requests_percentage = pd.merge(financial_requests_counts, total_conversations, on='Dataset_Title')

        # Calculate the percentage
        financial_requests_percentage['Financial_Requests_Percentage'] = (financial_requests_percentage['Conversations_with_Financial_Requests'] / financial_requests_percentage['Total_Conversations']) * 100

        # Sort by percentage descending
        financial_requests_percentage_sorted = financial_requests_percentage.sort_values(by='Financial_Requests_Percentage', ascending=False)

        # Display as a table
        st.markdown("**Financial Requests Made Percentage per Dataset:**")
        st.table(financial_requests_percentage_sorted[['Dataset_Title', 'Financial_Requests_Percentage']].style.format({'Financial_Requests_Percentage': '{:.0f}%'}))

        # Optional: Bar Chart Visualization
        fig_financial_requests_percentage_v19 = px.bar(
            financial_requests_percentage_sorted,
            x='Dataset_Title',
            y='Financial_Requests_Percentage',
            title='Percentage of Conversations with Financial Requests Made per Dataset',
            labels={'Financial_Requests_Percentage': 'Financial Requests Made (%)', 'Dataset_Title': 'Dataset'},
            color='Financial_Requests_Percentage',
            color_continuous_scale='Greens'
        )

        st.plotly_chart(fig_financial_requests_percentage_v19, use_container_width=True, key='fig_financial_requests_percentage_v19')

        # ---------------------------
        # Matrix Table for Visualization 17
        # ---------------------------

        st.markdown("### Financial Requests Made Rates Matrix")

        try:
            # Calculate financial requests made rates for each pairing
            matrix_data_financial_requests = filtered_data.groupby(['Carla_Model', 'Charles_Model'])['Financial_Request_Made'].mean().reset_index()
            matrix_data_financial_requests['Financial_Requests_Percentage'] = matrix_data_financial_requests['Financial_Request_Made'] * 100

            matrix_pivot_financial_requests = matrix_data_financial_requests.pivot(index='Carla_Model', columns='Charles_Model', values='Financial_Requests_Percentage')

            # Format the matrix table
            formatted_matrix_financial_requests = format_matrix_table(matrix_pivot_financial_requests)

            # Display the matrix table
            st.markdown("**Financial Requests Made Rates Matrix (Percentage):**")
            st.table(formatted_matrix_financial_requests)
        except Exception as e:
            logging.error(f"Error creating Financial Requests Made Rates Matrix for Visualization 17: {e}")
            st.error(f"An error occurred while creating Financial Requests Made Rates Matrix: {e}")

    except Exception as e:
        logging.error(f"Error in Visualization 17: {e}")
        st.error(f"An error occurred while generating Visualization 17: {e}")
else:
    st.warning("Required columns for Financial Requests Made Percentage are missing from the data.")

# ---------------------------
# Visualization 18: 3D Linear Regression Model: Sentiment and Financial Requests vs. Success Rate
# ---------------------------

st.subheader("18. 3D Linear Regression Model: Sentiment and Financial Requests vs. Success Rate")

try:
    # Check if required metrics exist in analysis_df
    required_metrics_v18 = ['Avg_Sentiment_Charles_mean', 'Financial_Requests_Made_mean', 'Success_Rate']
    missing_metrics_v18 = [metric for metric in required_metrics_v18 if metric not in analysis_df.columns]
    
    if not missing_metrics_v18:
        # Prepare data for 3D regression
        model_data_v18 = analysis_df[['Dataset_Title'] + required_metrics_v18].dropna()
        
        if not model_data_v18.empty:
            # Create 3D scatter plot
            fig_3d = px.scatter_3d(
                model_data_v18,
                x='Avg_Sentiment_Charles_mean',
                y='Financial_Requests_Made_mean',
                z='Success_Rate',
                color='Success_Rate',
                size='Success_Rate',
                hover_data=['Dataset_Title'],
                title="3D Linear Regression: Sentiment and Financial Requests vs. Success Rate",
                labels={
                    'Avg_Sentiment_Charles_mean': "Average Sentiment Score",
                    'Financial_Requests_Made_mean': "Average Financial Requests Made",
                    'Success_Rate': "Success Rate"
                }
            )
            
            # Fit linear regression model
            formula_v18 = 'Success_Rate ~ Avg_Sentiment_Charles_mean + Financial_Requests_Made_mean'
            model_v18 = smf.ols(formula=formula_v18, data=model_data_v18).fit()
            
            # Create grid for regression plane
            x_range = np.linspace(model_data_v18['Avg_Sentiment_Charles_mean'].min(), model_data_v18['Avg_Sentiment_Charles_mean'].max(), 10)
            y_range = np.linspace(model_data_v18['Financial_Requests_Made_mean'].min(), model_data_v18['Financial_Requests_Made_mean'].max(), 10)
            x_grid, y_grid = np.meshgrid(x_range, y_range)
            z_grid = model_v18.predict(pd.DataFrame({
                'Avg_Sentiment_Charles_mean': x_grid.ravel(),
                'Financial_Requests_Made_mean': y_grid.ravel()
            })).values.reshape(x_grid.shape)
            
            # Add regression plane to the 3D scatter plot
            fig_3d.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale='Viridis',
                opacity=0.5,
                showscale=False,
                name='Regression Plane'
            ))
            
            # Update layout for better visualization
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="Average Sentiment Score",
                    yaxis_title="Average Financial Requests Made",
                    zaxis_title="Success Rate"
                ),
                legend=dict(title='Success Rate'),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            # Display the 3D scatter plot with regression plane
            st.plotly_chart(fig_3d, use_container_width=True, key='fig_3d_regression_v18')
            
            # Display model summary
            st.markdown("### Linear Regression Model Summary")
            st.text(model_v18.summary())
            
            # Display Pearson correlation coefficients
            correlation_carla = model_data_v18['Avg_Sentiment_Charles_mean'].corr(model_data_v18['Success_Rate'])
            correlation_financial = model_data_v18['Financial_Requests_Made_mean'].corr(model_data_v18['Success_Rate'])
            st.markdown(f"**Pearson Correlation Coefficient (Sentiment vs. Success Rate):** {correlation_carla:.2f}")
            st.markdown(f"**Pearson Correlation Coefficient (Financial Requests vs. Success Rate):** {correlation_financial:.2f}")
            
            # Interpretation of correlations
            if correlation_carla > 0:
                interpretation_carla = "a positive correlation"
            elif correlation_carla < 0:
                interpretation_carla = "a negative correlation"
            else:
                interpretation_carla = "no correlation"
                
            if correlation_financial > 0:
                interpretation_financial = "a positive correlation"
            elif correlation_financial < 0:
                interpretation_financial = "a negative correlation"
            else:
                interpretation_financial = "no correlation"
                
            st.markdown(f"This indicates **{interpretation_carla}** between Charles' sentiment scores and the success rate.")
            st.markdown(f"This indicates **{interpretation_financial}** between the number of financial requests made by Carla and the success rate.")
        
        else:
            st.warning("No data available for Visualization 18 after dropping missing values.")
    else:
        st.warning(f"The following required metrics for Visualization 18 are missing from 'analysis_df': {', '.join(missing_metrics_v18)}")
    
except Exception as e:
    logging.error(f"Error in Visualization 18: {e}")
    st.error(f"An error occurred while generating Visualization 18: {e}")

# ---------------------------
# Visualization 19: 3D Linear Regression Model: Carla's Average Request Message Number and Sentiment Score vs. Success Rate
# ---------------------------

def visualization_19(analysis_df):
    st.subheader("19. 3D Linear Regression Model: Carla's Average Request Message Number and Sentiment Score vs. Success Rate")
    
    try:
        # Check if required metrics exist in analysis_df
        required_metrics_v19 = ['Average_Carla_Request_Message_Number_mean', 'Avg_Sentiment_Charles_mean', 'Success_Rate']
        missing_metrics_v19 = [metric for metric in required_metrics_v19 if metric not in analysis_df.columns]
        
        if not missing_metrics_v19:
            # Prepare data for 3D regression
            model_data_v19 = analysis_df[['Dataset_Title'] + required_metrics_v19].dropna()
            
            if not model_data_v19.empty:
                # Create 3D scatter plot
                fig_3d_v19 = px.scatter_3d(
                    model_data_v19,
                    x='Average_Carla_Request_Message_Number_mean',
                    y='Avg_Sentiment_Charles_mean',
                    z='Success_Rate',
                    color='Success_Rate',
                    size='Success_Rate',
                    hover_data=['Dataset_Title'],
                    title="3D Linear Regression: Carla's Avg Request Msg Num & Sentiment Score vs. Success Rate",
                    labels={
                        'Average_Carla_Request_Message_Number_mean': "Average Request Message Number",
                        'Avg_Sentiment_Charles_mean': "Average Sentiment Score",
                        'Success_Rate': "Success Rate"
                    }
                )
                
                # Fit linear regression model
                formula_v19 = 'Success_Rate ~ Average_Carla_Request_Message_Number_mean + Avg_Sentiment_Charles_mean'
                model_v19 = smf.ols(formula=formula_v19, data=model_data_v19).fit()
                
                # Create grid for regression plane
                x_range_v19 = np.linspace(model_data_v19['Average_Carla_Request_Message_Number_mean'].min(), model_data_v19['Average_Carla_Request_Message_Number_mean'].max(), 10)
                y_range_v19 = np.linspace(model_data_v19['Avg_Sentiment_Charles_mean'].min(), model_data_v19['Avg_Sentiment_Charles_mean'].max(), 10)
                x_grid_v19, y_grid_v19 = np.meshgrid(x_range_v19, y_range_v19)
                z_grid_v19 = model_v19.predict(pd.DataFrame({
                    'Average_Carla_Request_Message_Number_mean': x_grid_v19.ravel(),
                    'Avg_Sentiment_Charles_mean': y_grid_v19.ravel()
                })).values.reshape(x_grid_v19.shape)
                
                # Add regression plane to the 3D scatter plot
                fig_3d_v19.add_trace(go.Surface(
                    x=x_grid_v19,
                    y=y_grid_v19,
                    z=z_grid_v19,
                    colorscale='Viridis',
                    opacity=0.5,
                    showscale=False,
                    name='Regression Plane'
                ))
                
                # Update layout for better visualization
                fig_3d_v19.update_layout(
                    scene=dict(
                        xaxis_title="Average Request Message Number",
                        yaxis_title="Average Sentiment Score",
                        zaxis_title="Success Rate"
                    ),
                    legend=dict(title='Success Rate'),
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                # Display the 3D scatter plot with regression plane
                st.plotly_chart(fig_3d_v19, use_container_width=True, key='fig_3d_regression_v19')
                
                # Display model summary
                st.markdown("### Linear Regression Model Summary")
                st.text(model_v19.summary())
                
                # Display Pearson correlation coefficients
                correlation_carla_v19 = model_data_v19['Average_Carla_Request_Message_Number_mean'].corr(model_data_v19['Success_Rate'])
                correlation_sentiment_v19 = model_data_v19['Avg_Sentiment_Charles_mean'].corr(model_data_v19['Success_Rate'])
                st.markdown(f"**Pearson Correlation Coefficient (Carla's Request Msg Num vs. Success Rate):** {correlation_carla_v19:.2f}")
                st.markdown(f"**Pearson Correlation Coefficient (Sentiment Score vs. Success Rate):** {correlation_sentiment_v19:.2f}")
                
                # Interpretation of correlations
                if correlation_carla_v19 > 0:
                    interpretation_carla_v19 = "a positive correlation"
                elif correlation_carla_v19 < 0:
                    interpretation_carla_v19 = "a negative correlation"
                else:
                    interpretation_carla_v19 = "no correlation"
                    
                if correlation_sentiment_v19 > 0:
                    interpretation_sentiment_v19 = "a positive correlation"
                elif correlation_sentiment_v19 < 0:
                    interpretation_sentiment_v19 = "a negative correlation"
                else:
                    interpretation_sentiment_v19 = "no correlation"
                    
                st.markdown(f"This indicates **{interpretation_carla_v19}** between Carla's average request message number and the success rate.")
                st.markdown(f"This indicates **{interpretation_sentiment_v19}** between Charles' sentiment scores and the success rate.")
            
            else:
                st.warning("No data available for Visualization 19 after dropping missing values.")
        else:
            st.warning(f"The following required metrics for Visualization 19 are missing from 'analysis_df': {', '.join(missing_metrics_v19)}")
        
    except Exception as e:
        logging.error(f"Error in Visualization 19: {e}")
        st.error(f"An error occurred while generating Visualization 19: {e}")

# **Call Visualization 19**
visualization_19(analysis_df)



# ---------------------------
# Optional: Dataset Overview and Download
# ---------------------------

st.header("Dataset Overview")

st.dataframe(filtered_data.head())

# Provide download link for the filtered data
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv_filtered = convert_df(filtered_data)

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv_filtered,
    file_name='filtered_data.csv',
    mime='text/csv',
)

# ---------------------------
# Footer
# ---------------------------

st.markdown("---")
st.markdown("© 2024 Visualization. Aaron Halberstadt-Twum @ UCL")
