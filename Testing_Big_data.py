import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import calendar
import time


# Load the data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Diparna/CMSE830/refs/heads/main/appended_plane_crash_data.csv"
    df = pd.read_csv(url)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    #df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    #df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    #df['date'] = pd.to_datetime(df['month'].astype(str) + ' ' + df['year'].astype(str), format='%B %Y', errors='coerce')
    
    numeric_columns = ['aboard', 'fatalities']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Load data
df = load_data()

st.title("Plane Crash Data Analysis")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Missing Data Analysis", "Initial Data Analysis", "Exploratory Data Analysis", "Temporal Analysis", "Operator Analysis", "Aircraft Analysis", "Aircraft Survival Analysis" ,"Location Analysis", "Animation Analysis", "Summary"])


# Year range filter
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", 
                               min_value=int(df['year'].min()), 
                               max_value=int(df['year'].max()), 
                               value=(int(df['year'].min()), int(df['year'].max())))

st.sidebar.markdown("<br>" * 7, unsafe_allow_html=True)
st.sidebar.subheader("GitHub Repository")
st.sidebar.markdown("[View the source code on GitHub](https://github.com/Diparna/CMSE830)", unsafe_allow_html=True)

def clean_locations(df):
    # Replace the ellipsis and any other special characters
    if 'location' in df.columns:
        # Remove ellipsis and other special characters
        df['location'] = df['location'].str.replace('…', '')
        df['location'] = df['location'].str.replace(',', '')
        # Strip any leading/trailing whitespace
        df['location'] = df['location'].str.strip()
        # Replace empty strings with NaN
        df['location'] = df['location'].replace('', pd.NA)
    
    return df

df = clean_locations(df)


# Filter data based on year range
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

if page == "Overview":
    st.header("Dataset Overview")
    st.markdown("This is a page showing the overview from the dataset.")
    st.write(f"Total number of crashes: {len(filtered_df)}")

    valid_dates = filtered_df['date'].dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        st.write(f"Date range: {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}")
    else:
        st.write("No valid dates available in the selected range.")

    # Fatalities Analysis
    st.subheader("Fatalities Analysis")
    st.markdown("Below we have summary of metrics from the dataset.")
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

    # Display sample data
    st.subheader("Sample Data")
    st.write(filtered_df.head())

elif page == "Missing Data Analysis":
    st.title("Missing Data Analysis")
    
    # Display missing data information
    st.header("Missing Data Overview")
    st.markdown("This is a page for checking the missingness of the dataset.")
    missing_data = filtered_df.isnull().sum().sort_values(ascending=False)
    missing_percent = 100 * filtered_df.isnull().sum() / len(filtered_df)
    missing_table = pd.concat([missing_data, missing_percent], axis=1, keys=['Missing Values', 'Percentage'])
    st.write(missing_table)
    
    # Visualize missing data
    st.header("Missing Data Visualization")
    st.markdown("This is a plot for showing the missingness of the dataset.")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(filtered_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    st.pyplot(fig)
    
    # Imputation options
    st.header("Data Imputation")
    st.markdown("This is a for trying various imputation methods. I have used only simple imputer(I felt like there's no need to go for advanced imputation for my project.).The error pops up when there's nothing selected in the columns section. All the imputation do not work for all the columns. The Most Frequent work on both numeric and text based data.")
    imputation_method = st.selectbox("Select imputation method", ["Mean", "Median", "Most Frequent", "Constant"])
    columns_to_impute = st.multiselect("Select columns to impute", filtered_df.columns)
    try:
        if len(columns_to_impute) == 0:
            st.error("Please select at least one column to impute!")
        else:
            if st.button("Perform Imputation"):
                if imputation_method == "Mean":
                    imputer = SimpleImputer(strategy='mean')
                elif imputation_method == "Median":
                    imputer = SimpleImputer(strategy='median')
                elif imputation_method == "Most Frequent":
                    imputer = SimpleImputer(strategy='most_frequent')
                else:  # Constant
                    constant_value = st.text_input("Enter constant value for imputation", "0")
                    imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
        
            filtered_df_imputed = filtered_df.copy()
            filtered_df_imputed[columns_to_impute] = imputer.fit_transform(filtered_df[columns_to_impute])
        
            st.write("Data after imputation:")
            st.write(filtered_df_imputed[columns_to_impute].head())
    
            # Display the existing data

            st.header("Missing Data Overview Before Imputation")
            missing_data = filtered_df.isnull().sum().sort_values(ascending=False)
            missing_percent = 100 * filtered_df.isnull().sum() / len(filtered_df)
            missing_table = pd.concat([missing_data, missing_percent], axis=1, keys=['Missing Values', 'Percentage'])
            st.write(missing_table)

            # Update missing data information after imputation
            st.header("Missing Data Overview After Imputation")
            missing_data_after = filtered_df_imputed.isnull().sum().sort_values(ascending=False)
            missing_percent_after = 100 * filtered_df_imputed.isnull().sum() / len(filtered_df_imputed)
            missing_table_after = pd.concat([missing_data_after, missing_percent_after], axis=1, keys=['Missing Values', 'Percentage'])
            st.write(missing_table_after)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Common errors:\n"
            "- Mean and Median imputation only work with numeric data\n"
            "- Make sure the selected columns contain valid data for the chosen imputation method\n"
            "- You have not clicked on Perform Imputation!")

elif page == "Initial Data Analysis":
    st.title("Initial Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Data Quality Check", "Basic Statistical Summary", "Data Distribution"])
    
    with tab1:
        st.header("Dataset Overview")
        st.markdown(" This tab includes the generic info about the Dataset. ")
        st.write(f"Number of rows: {filtered_df.shape[0]}")
        st.write(f"Number of columns: {filtered_df.shape[1]}")
        st.write("Column names and data types:")
        st.write(filtered_df.dtypes)
    
    with tab2:
        st.header("Data Quality Check")
        missing_values = filtered_df.isnull().sum()
        st.write("Missing values per column:")
        st.write(missing_values[missing_values > 0])
        st.write(f"Number of duplicate rows: {filtered_df.duplicated().sum()}")
    
    with tab3:
        st.header("Basic Statistical Summary")
        st.write(filtered_df.describe())
    
    with tab4:
        st.header("Data Distribution")
        st.markdown("As the name suggests, it shows the count of the selected columns. I feel that the graph For aboard and fatalities is difficult to read. So, let's say, it is at 200 and the count is at 10. So that means there are 10 incidents where the number of people abord the flight and the number of people who died is 200.")
        # Select only numeric columns for bar chart
        numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns

        # Dropdown to select a column for the bar chart
        selected_col = st.selectbox("Select a numeric column for bar chart", numeric_cols)

        # If the selected column is the 'month', map the month numbers to names and sort by month number
        if selected_col == 'month':
            filtered_df['month_name'] = filtered_df['month'].apply(lambda x: calendar.month_name[int(x)] if pd.notnull(x) else x)
    
            # Group by month numbers to keep sorting correct, then map to month names
            data_grouped = filtered_df.groupby('month')['month_name'].count().reset_index()
            data_grouped.columns = ['Month Number', 'Count']  # Rename columns
            data_grouped['Month Name'] = data_grouped['Month Number'].apply(lambda x: calendar.month_name[int(x)])
    
            # Sort by month number to ensure correct order
            data_grouped = data_grouped.sort_values('Month Number')
    
            # Create the bar chart with month names in the correct order
            fig = px.bar(data_grouped, x='Month Name', y='Count', title='Data Distribution by Month',
                      labels={'Month Name': 'Month', 'Count': 'Count of Occurrences'})
    
        else:
            # Group the data by the selected column and count the occurrences
            data_grouped = filtered_df[selected_col].value_counts().reset_index()
            data_grouped.columns = [selected_col, 'Count']  # Rename the columns for clarity
            # Create the bar chart for the selected column
            fig = px.bar(data_grouped, x=selected_col, y='Count', title='Data Distribution',
                      labels={selected_col: selected_col, 'Count': 'Count of Occurrences'})
        
        
        # Display the bar graph in the Streamlit app
        st.plotly_chart(fig)


elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("This page shows various things that were done for Exploratory Data Analysis(EDA) of the dataset. I did an analysis on most of the categories I could think of such as Operators, Aircrafts, Common Locations and Number of people who died in the crash.")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Fatality Analysis", "Operator and Aircraft Analysis", "Geographical Analysis", "Correlation Analysis", "Word Map"])
      
    with tab1:
        st.header("Fatality Analysis")
        fig = px.scatter(filtered_df, x="aboard", y="fatalities", title="Fatalities vs People Aboard")
        st.plotly_chart(fig)
    
    with tab2:
        st.header("Operator and Aircraft Analysis")
        top_operators = filtered_df['operator'].value_counts().nlargest(10)
        fig = px.bar(top_operators, x=top_operators.index, y=top_operators.values, title='Top 10 Operators with Most Crashes')
        st.plotly_chart(fig)
    
    with tab3:
        st.header("Geographical Analysis")
        top_locations = filtered_df['location'].value_counts().nlargest(10)
        fig = px.bar(top_locations, x=top_locations.index, y=top_locations.values, title='Top 10 Locations with Most Crashes', labels={'x': 'Location', 'y': 'Number of Crashes'})
        st.plotly_chart(fig)
    
    with tab4:
        st.header("Correlation Analysis")
        st.markdown("For the correlation matrix, I tried including the other columns such as operators, countries and location but it said that it cannot convert these columns into float. So, I was not sure how to show correlation between them.")
        #corr_matrix = filtered_df[['aboard', 'fatalities', '']].corr()
        #fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        #st.plotly_chart(fig)
        corr_matrix_1 = filtered_df[['aboard', 'fatalities', 'year']].corr()
        fig_1 = px.imshow(corr_matrix_1, text_auto=True, aspect="auto")
        st.plotly_chart(fig_1)
    
    with tab5:
        st.header("Word Map")
        st.markdown("This is a rough word map made from location data to try and make the common countries easier to notice.")
        text = ' '.join(filtered_df['country'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

elif page == "Temporal Analysis":
    st.header("Temporal Analysis")
    st.markdown("We discuss various statistics based on various timely statistics such as yearly data or monthly data.")
    # Crashes over time
    st.subheader("Crashes Over Time")
    crashes_by_year = filtered_df.groupby('year').size().reset_index(name='Count')
    fig_timeline = px.line(crashes_by_year, x='year', y='Count', title='Number of Crashes per Year')
    st.plotly_chart(fig_timeline)

    # Crashes by Month
    st.subheader("Crashes by Month")
    if pd.api.types.is_numeric_dtype(filtered_df['month']):
        filtered_df['month'] = pd.to_datetime(filtered_df['month'], format='%m').dt.strftime('%B')
    
    # Define the correct month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    crashes_by_month = (filtered_df['month'].value_counts().reindex(month_order).fillna(0))
    fig_months = px.bar(x=crashes_by_month.index, y=crashes_by_month.values,title='Number of Crashes by Month',labels={'x': 'Month', 'y': 'Number of Crashes'})
    fig_months.update_layout( xaxis_tickangle=-45, showlegend=False, height=500)
    st.plotly_chart(fig_months)

    # Crashes by Hour
    st.subheader("Crashes by Hour")
    st.markdown("For the hour, we can see lots of count in 0 because the hour was missing in the data so it was put in as 0 instead.")
    filtered_df['hour'] = pd.to_numeric(filtered_df['hour'], errors='coerce')
    crashes_by_hour = filtered_df['hour'].value_counts().sort_index()
    fig_hours = px.bar( x=crashes_by_hour.index, y=crashes_by_hour.values, title='Number of Crashes by Hour', labels={'x': 'Hour of the Day', 'y': 'Count of Crashes'})
    fig_hours.update_xaxes(type='category')
    st.plotly_chart(fig_hours)


elif page == "Operator Analysis":
    st.header("Operator Analysis")

    # Top 10 Operators
    st.subheader("Top 10 Operators with Most Crashes")
    st.markdown("I know this is a repeat. This was supposed to be a placeholder for something else I was trying. This is included in the Exploratory Data Analysis as well.")
    top_operators = filtered_df['operator'].value_counts().nlargest(10)
    fig_operators = px.bar(top_operators, x=top_operators.index, y=top_operators.values, title='Top 10 Operators with Most Crashes', labels={'x': 'Operators', 'y': 'Count of Crashes'})
    st.plotly_chart(fig_operators)

elif page == "Aircraft Analysis":
    st.header("Aircraft Analysis")

    # Aircraft Types
    st.subheader("Most Common Aircraft Types")
    st.markdown(" As the name suggests, this shows the number of crashes for each aircraft and it was surprising to see the dominating aircraft in type of aircrafts.")
    top_aircraft = filtered_df['type'].value_counts().nlargest(10)
    fig_aircraft = px.pie(values=top_aircraft.values, names=top_aircraft.index, title='Top 10 Aircraft Types Involved in Crashes')
    st.plotly_chart(fig_aircraft)

    # Correlation between Aboard and Fatalities
    st.subheader("Correlation between People Aboard and Fatalities")
    fig_correlation = px.scatter(filtered_df, x="aboard", y="fatalities", trendline="ols", 
                                 title="Correlation between People Aboard and Fatalities")
    st.plotly_chart(fig_correlation)

elif page == "Location Analysis":
    st.header("Location Analysis")

    # Word Cloud of Locations
    st.subheader("Common Crash Locations")
    
    # Prepare location data
    locations = filtered_df['location'].dropna().tolist()
    location_freq = {}
    for loc in locations:
        if loc in location_freq:
            location_freq[loc] += 1
        else:
            location_freq[loc] = 1

     # Prepare country data
    countries = filtered_df['country'].dropna().tolist()
    country_freq = {}
    for loc in countries:
        if loc in country_freq:
            country_freq[loc] += 1
        else:
            country_freq[loc] = 1
    
    # Generate word cloud for locations
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(location_freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Display top locations
    st.subheader("Top 10 Crash Locations")
    top_locations = pd.Series(location_freq).nlargest(10)
    fig_locations = px.bar(x=top_locations.index, y=top_locations.values, 
                           title='Top 10 Locations with Most Crashes', labels={'x': 'Location', 'y': 'Count of Crashes'})
    st.plotly_chart(fig_locations)


    # Generate word cloud for locations
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(country_freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Display top countries
    st.subheader("Top 10 Crash Countries")
    top_countries = pd.Series(country_freq).nlargest(10)
    fig_countries = px.bar(x=top_countries.index, y=top_countries.values, 
                           title='Top 10 Locations with Most Crashes', labels={'x': 'Location', 'y': 'Count of Crashes'})
    st.plotly_chart(fig_countries)

elif page == "Animation Analysis":
     st.title("Animated Analysis of Plane Crashes")
     st.markdown("This is a fun animated chart to see the number of crashes over the years! Just because Animations are fun! This shows a timelapse of crashes over the years.")
     animation_tab1, animation_tab2, animation_tab3 = st.tabs([
        "Cumulative Crashes Over Time", 
        "Yearly Accident Trends", 
        "Operator Evolution"
    ])
     with animation_tab1:
        st.header("Cumulative Crashes Over Time")
        
        # Create cumulative data
        yearly_crashes = filtered_df.groupby('year').size().reset_index(name='crashes')
        yearly_crashes['cumulative_crashes'] = yearly_crashes['crashes'].cumsum()
        
        # Create the animated plot
        fig = go.Figure()
        
        # Add the base line
        fig.add_trace(
            go.Scatter(
                x=yearly_crashes['year'],
                y=yearly_crashes['cumulative_crashes'],
                mode='lines',
                name='Cumulative Crashes',
                line=dict(color='red')
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Cumulative Number of Plane Crashes Over Time',
            xaxis_title='Year',
            yaxis_title='Total Number of Crashes',
            showlegend=True
        )
        
        # Add animation
        if st.button('Play Animation', key='cumulative'):
            placeholder = st.empty()
            for i in range(len(yearly_crashes)):
                fig.update_traces(
                    x=yearly_crashes['year'][:i+1],
                    y=yearly_crashes['cumulative_crashes'][:i+1]
                )
                with placeholder:
                    st.plotly_chart(fig)
                time.sleep(0.5)
    
     with animation_tab2:
        st.header("Yearly Accident Trends")
        
        # Create yearly data with additional metrics
        yearly_data = filtered_df.groupby('year').agg({
            'fatalities': 'sum',
            'aboard': 'sum'
        }).reset_index()
        yearly_data['survival_rate'] = (1 - (yearly_data['fatalities'] / yearly_data['aboard'])) * 100
        
        # Create animated bar chart
        fig = go.Figure()
        
        # Add initial empty bar chart
        fig.add_trace(
            go.Bar(
                x=yearly_data['year'],
                y=yearly_data['fatalities'],
                name='Fatalities',
                marker_color='red'
            )
        )
        
        fig.update_layout(
            title='Yearly Fatalities in Plane Crashes',
            xaxis_title='Year',
            yaxis_title='Number of Fatalities',
            showlegend=True
        )
        
        if st.button('Play Animation', key='yearly'):
            placeholder = st.empty()
            for i in range(len(yearly_data)):
                fig.update_traces(
                    x=yearly_data['year'][:i+1],
                    y=yearly_data['fatalities'][:i+1]
                )
                with placeholder:
                    st.plotly_chart(fig)
                time.sleep(0.5)
    
     with animation_tab3:
        st.header("Evolution of Operator Accidents")
        
        # Get top 10 operators
        top_operators = filtered_df['operator'].value_counts().nlargest(10).index
        
        # Create data for top operators by year
        operator_yearly = filtered_df[filtered_df['operator'].isin(top_operators)].groupby(
            ['year', 'operator']).size().reset_index(name='crashes')
        
        # Create animated line chart for operators
        fig = go.Figure()
        
        # Add a line for each operator
        for operator in top_operators:
            operator_data = operator_yearly[operator_yearly['operator'] == operator]
            fig.add_trace(
                go.Scatter(
                    x=operator_data['year'],
                    y=operator_data['crashes'],
                    name=operator,
                    mode='lines+markers'
                )
            )
        
        fig.update_layout(
            title='Evolution of Accidents by Top Operators',
            xaxis_title='Year',
            yaxis_title='Number of Crashes',
            showlegend=True
        )
        
        if st.button('Play Animation', key='operator'):
            placeholder = st.empty()
            years = sorted(operator_yearly['year'].unique())
            for year in years:
                fig.update_layout(
                    xaxis_range=[years[0], year]
                )
                with placeholder:
                    st.plotly_chart(fig)
                time.sleep(0.5)


elif page == "Aircraft Survival Analysis":
    st.title("Aircraft Type Survival Analysis")
    st.markdown("This page shows the survival rate of the passengers in case of an accident. ")
    # Calculate survival statistics for each aircraft type
    def get_aircraft_survival_stats(df):
        stats = df.groupby('type').agg({
            'aboard': 'sum',
            'fatalities': 'sum',
            'year': 'count'  # Count of incidents
        }).reset_index()
        
        stats['survivors'] = stats['aboard'] - stats['fatalities']
        stats['survival_rate'] = (stats['survivors'] / stats['aboard'] * 100).round(2)
        stats['incidents'] = stats['year']
        return stats
    
    aircraft_stats = get_aircraft_survival_stats(filtered_df)
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Aircraft Type")
        
        # Add a "Show All" option at the top of the dropdown
        all_aircraft_types = ['Show All'] + sorted(aircraft_stats['type'].unique().tolist())
        
        selected_aircraft = st.selectbox(
            "Choose an aircraft type:",
            all_aircraft_types
        )
        
        # Filter minimum incidents
        min_incidents = st.slider(
            "Minimum number of incidents",
            1, 50, 5,
            help="Filter aircraft types with at least this many incidents"
        )
        
        # Filter the data based on minimum incidents
        filtered_stats = aircraft_stats[aircraft_stats['incidents'] >= min_incidents]
        
    with col2:
        if selected_aircraft == "Show All":
            st.subheader(f"Survival Rates for All Aircraft Types (with ≥{min_incidents} incidents)")
        else:
            st.subheader(f"Survival Rate for {selected_aircraft}")
    
    # Create visualization based on selection
    if selected_aircraft == "Show All":
        # Sort by survival rate for better visualization
        filtered_stats = filtered_stats.sort_values('survival_rate', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=filtered_stats['type'],
            x=filtered_stats['survival_rate'],
            orientation='h',
            marker_color='lightblue',
            customdata=np.stack((
                filtered_stats['incidents'],
                filtered_stats['aboard'],
                filtered_stats['survivors'],
                filtered_stats['fatalities']
            ), axis=-1),
            hovertemplate=
            "<b>%{y}</b><br>" +
            "Survival Rate: %{x:.1f}%<br>" +
            "Incidents: %{customdata[0]}<br>" +
            "Total Aboard: %{customdata[1]}<br>" +
            "Survivors: %{customdata[2]}<br>" +
            "Fatalities: %{customdata[3]}<br>" +
            "<extra></extra>"
        ))
        
        fig.update_layout(
            title_text="Aircraft Types by Survival Rate",
            xaxis_title="Survival Rate (%)",
            yaxis_title="Aircraft Type",
            height=max(400, len(filtered_stats) * 25),  # Dynamic height based on number of aircraft types
            showlegend=False
        )
        
    else:
        # Get stats for selected aircraft
        selected_stats = aircraft_stats[aircraft_stats['type'] == selected_aircraft].iloc[0]
        
        # Create gauge chart for single aircraft
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = selected_stats['survival_rate'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            delta = {'reference': aircraft_stats['survival_rate'].mean()},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': aircraft_stats['survival_rate'].mean()
                }
            }
        ))
        
        fig.update_layout(
            title_text=f"Survival Rate for {selected_aircraft}",
            height=400
        )
    
    # Display the plot
    st.plotly_chart(fig)
    
    # Display detailed statistics
    st.subheader("Detailed Statistics")
    
    if selected_aircraft == "Show All":
        # Show sortable table for all aircraft
        st.dataframe(
            filtered_stats[['type', 'incidents', 'aboard', 'survivors', 'fatalities', 'survival_rate']]
            .sort_values('survival_rate', ascending=False)
            .style.format({
                'survival_rate': '{:.2f}%',
                'incidents': '{:,}',
                'aboard': '{:,}',
                'survivors': '{:,}',
                'fatalities': '{:,}'
            })
        )
    else:
        # Show detailed stats for selected aircraft
        selected_stats = aircraft_stats[aircraft_stats['type'] == selected_aircraft].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Total Incidents",
            f"{selected_stats['incidents']:,}"
        )
        
        col2.metric(
            "Total Aboard",
            f"{selected_stats['aboard']:,}"
        )
        
        col3.metric(
            "Survivors",
            f"{selected_stats['survivors']:,}"
        )
        
        col4.metric(
            "Fatalities",
            f"{selected_stats['fatalities']:,}"
        )
        
        # Compare with average
        st.subheader("Comparison with Average")
        avg_survival_rate = aircraft_stats['survival_rate'].mean()
        
        comparison_text = (
            f"The survival rate for {selected_aircraft} is "
            f"{selected_stats['survival_rate']:.2f}%, compared to the "
            f"average survival rate of {avg_survival_rate:.2f}% across all aircraft types. "
            f"This is {abs(selected_stats['survival_rate'] - avg_survival_rate):.2f}% "
            f"{'higher' if selected_stats['survival_rate'] > avg_survival_rate else 'lower'} "
            f"than the average."
        )
        
        st.write(comparison_text)
elif page == "Summary":
    st.title("Findings and Future Outlook")
    st.markdown("I was searching the internet for a complete dataset. My original idea was to investigate the Bermuda Triangle using flight statistics and Weather data to see if there's any correlation with the incidents that occured in Bermuda Triangle with the weather patterns of that period. ")
    st.markdown(" But, I have been unable to locate flight data for the Bermuda Triangle. I have scraped data from a few websites for crashes all over the world and the shipwrecks. Till now for the Bermuda region, I have found very few recorded shipwrecks and aircraft crashes. I have found a website for the weather data luckily. But, I am curerntly working on scraping the data and then combining it on my local computer. Then, the next step would be filter it down to the locations of San Juan, Bermuda and Florida.")
    st.markdown(" So, I merged two different datasets for crash data for the airplanes. Then, did an analysis on the survival rate, common crash locations and analysis based on different aircrafts, how many crashes occured around the world throughout the years. Since I was originally focusing on data from 1950-1980, I gathered crash data before 2000.")
    st.markdown(" I think after collecting the weather data and doing some analysis, I will need to decide whether I should continue to focus on the Bermuda Triangle. I hoped for better data available. It is hard to find historical data. Lots of data are available in paper format. It will take wayyy too much time if I were to convert the data into a digital format. ")
    st.markdown(" I tried to salvage as much as I could from the data that I had. I tried merging the marine shipwreck logs but the columns are completely different. I ran a separate analysis on that dataset. I am thinking of including that in my final project and try to collect more data related to shipwrecks and aircraft crashes. ")
    st.markdown(" The dates are missing for half the dataset as they have the month but I could not find the exact date. I was thinking of imputing the data(the missing dates) and then do the analysis but it did not feel correct to me as it won't correctly show the timeline of the crashes.")
    st.markdown(" My next steps in terms of project is going to look at the weather data and then check for Bermuda region. If there's any sensible data, I will work with that. Other than that, I am a bit confused on how to proceed further. I am considering if I should continue on the same path or change the project entirely.")

# Display raw data option (available on all pages)
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(filtered_df)
