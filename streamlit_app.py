import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the app
st.title('Interactive Line Chart Example')

# Generate a random dataset
@st.cache  # Use caching to speed up load times when the data doesn't change
def generate_data(num_points):
    """Generates a simple dataset with the specified number of points."""
    time = pd.date_range('2023-01-01', periods=num_points, freq='D')
    values = np.random.randn(num_points).cumsum()
    data = pd.DataFrame({'Time': time, 'Value': values})
    return data

# Slider for the number of data points to generate
num_points = st.slider('Select the number of data points', min_value=10, max_value=1000, value=300)

# Generate and display the data
data = generate_data(num_points)
st.write(f"Data with {num_points} points")

# Plotting the data using a line chart
st.line_chart(data.set_index('Time'))

# Display the raw data as a table
if st.checkbox('Show raw data'):
    st.write(data)
