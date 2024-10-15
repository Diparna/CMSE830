import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/andandandand/CSV-datasets/master/plane_crashes_data.csv"
    df = pd.read_csv(url)
    
    # Display raw data types
    st.write("Data types before conversion:")
    st.write(df.dtypes)
    
    # Convert year to numeric, invalid values become NaN
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Drop rows with NaN years
    df = df.dropna(subset=['year'])
    
    # Convert year to integer
    df['year'] = df['year'].astype(int)
    
    # Display data types after conversion
    st.write("Data types after conversion:")
    st.write(df.dtypes)
    
    # Create date column
    df['date'] = pd.to_datetime(df['month'].astype(str) + ' ' + df['year'].astype(str), format='%B %Y', errors='coerce')
    
    # Convert other numeric columns
    numeric_columns = ['aboard', 'fatalities']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Load data
df = load_data()

st.title("Plane Crash Data Analysis")

# Display sample data
st.write("Sample data:")
st.write(df.head())

# Sidebar
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", 
                               min_value=int(df['year'].min()), 
                               max_value=int(df['year'].max()), 
                               value=(int(df['year'].min()), int(df['year'].max())))

# Filter data based on year range
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# Overview
st.header("Dataset Overview")
st.write(f"Total number of crashes: {len(filtered_df)}")

# Handle date range display
valid_dates = filtered_df['date'].dropna()
if len(valid_dates) > 0:
    min_date = valid_dates.min()
    max_date = valid_dates.max()
    st.write(f"Date range: {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}")
else:
    st.write("No valid dates available in the selected range.")

# Crashes over time
st.header("Crashes Over Time")
crashes_by_year = filtered_df.groupby('year').size().reset_index(name='Count')
fig_timeline = px.line(crashes_by_year, x='year', y='Count', title='Number of Crashes per Year')
st.plotly_chart(fig_timeline)

# Top 10 Operators
st.header("Top 10 Operators with Most Crashes")
top_operators = filtered_df['operator'].value_counts().nlargest(10)
fig_operators = px.bar(top_operators, x=top_operators.index, y=top_operators.values, title='Top 10 Operators with Most Crashes')
st.plotly_chart(fig_operators)

# Fatalities Analysis
st.header("Fatalities Analysis")
total_fatalities = filtered_df['fatalities'].sum()
total_aboard = filtered_df['aboard'].sum()
if total_aboard > 0:
    survival_rate = (1 - (total_fatalities / total_aboard)) * 100
else:
    survival_rate = 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Fatalities", f"{total_fatalities:,}")
col2.metric("Total People Aboard", f"{total_aboard:,}")
col3.metric("Overall Survival Rate", f"{survival_rate:.2f}%")

# Aircraft Types
st.header("Most Common Aircraft Types")
top_aircraft = filtered_df['type'].value_counts().nlargest(10)
fig_aircraft = px.pie(values=top_aircraft.values, names=top_aircraft.index, title='Top 10 Aircraft Types Involved in Crashes')
st.plotly_chart(fig_aircraft)

# Correlation between Aboard and Fatalities
st.header("Correlation between People Aboard and Fatalities")
fig_correlation = px.scatter(filtered_df, x="aboard", y="fatalities", trendline="ols", 
                             title="Correlation between People Aboard and Fatalities")
st.plotly_chart(fig_correlation)



# Mapping of month numbers to month names
month_map = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Create a 'month_name' column in the DataFrame using the month_map
filtered_df['month_name'] = filtered_df['month'].map(month_map)
crashes_by_month = filtered_df.groupby('month').size().reset_index(name='Count') # Group crashes by 'month' (number) and count
crashes_by_month = crashes_by_month.sort_values('month') # Sort by month number to ensure the correct order
# Create the bar chart using month number for sorting, but show month names on the x-axis
fig_months = px.bar(crashes_by_month, 
                    x=crashes_by_month['month'].map(month_map),  # Display month names on x-axis
                    y='Count', 
                    title='Number of Crashes by Month',
                    labels={'x': 'Month', 'Count': 'Number of Crashes'})  # Labels for axes
# Display the chart
st.plotly_chart(fig_months)



# Crashes by Hour
st.header("Crashes by Hour")
filtered_df['hour'] = pd.to_numeric(filtered_df['hour'], errors='coerce')
crashes_by_hour = filtered_df['hour'].value_counts().sort_index()
fig_hours = px.bar(x=crashes_by_hour.index, y=crashes_by_hour.values, title='Number of Crashes by Hour', labels={'x': 'Hour', 'y': 'Number of Crashes'})
fig_hours.update_xaxes(type='category')
st.plotly_chart(fig_hours)

# Word Cloud of Locations
st.header("Common Crash Locations")
text = ' '.join(filtered_df['location'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Display raw data
if st.checkbox("Show Raw Data"):
    num_rows = st.slider("Number of rows to view", min_value=5, max_value=len(filtered_df), value=10)
    st.subheader(f"Showing {num_rows} rows")
    st.write(filtered_df.head(num_rows))


fig_timeline.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Crashes",
    hovermode="x unified"
)

if st.sidebar.button("Reset Filters"):
    year_range = (int(df['year'].min()), int(df['year'].max()))