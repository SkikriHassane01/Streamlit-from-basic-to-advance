#____________________________________
# 1.Importing the necessary libraries
import streamlit as st  
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import plotly.express as px
import time 
#____________________________________


#____________________________________
# 2. core components

def main():
    
    # page config
    st.set_page_config(
        page_title= "Streamlit Data Science Dashboard",
        page_icon= "📊",
        layout= "wide",
    )
    
    # title / header
    st.title("Data Science Dashboard")
    st.header("Interactive Data Analysis with Streamlit")

    # load the data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
      # 3. Text Elements
    st.markdown("## Data Overview")
    st.text("This is a simple text element")
    st.write("Write can handle multiple data types automatically")
    st.info("This is an informative message")
    st.warning("This is a warning message")
    st.error("This is an error message")
    st.success("This is a success message")
    
    # 4. Data Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Display")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Basic Statistics")
        st.table(df.describe().round(2))
        # 5. Input Widgets
    st.sidebar.header("Controls")
    
    # Select features for analysis
    feature_x = st.sidebar.selectbox("Select X-axis feature", df.columns[:-1])
    feature_y = st.sidebar.selectbox("Select Y-axis feature", df.columns[:-1])

   # Slider for sample size
    sample_size = st.sidebar.slider("Sample Size", 10, len(df), 50)
    
    # Checkbox for showing correlation matrix
    show_corr = st.sidebar.checkbox("Show Correlation Matrix")

    # 6. Visualization
    st.markdown("## Data Visualization")
    
    # Interactive Plotly scatter plot
    fig_scatter = px.scatter(
        df.sample(sample_size),
        x=feature_x,
        y=feature_y,
        color='target',
        title=f"{feature_x} vs {feature_y}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

   # Conditional display of correlation matrix
    if show_corr:
        st.sidebar.markdown("## Correlation Matrix")
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.sidebar.pyplot(fig_corr)

    # 7. File Upload
    st.markdown("## File Upload")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(user_df.head())

    # 8. Download Button
    st.markdown("## Download Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="iris_data.csv",
        mime="text/csv"
    )
    
  # 9. Expander
    with st.expander("See detailed explanation"):
        st.write("""
            This dashboard demonstrates various Streamlit features:
            - Data visualization
            - Interactive controls
            - File upload/download
            - Different display options
            - Conditional rendering
        """)

    # 10. Progress and Status
    st.markdown("## Progress Demonstration")
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        # Update progress bar every 0.1 seconds
        time.sleep(0.1)
        progress_bar.progress(i + 1)
        status_text.text(f'Progress: {i+1}%')
        
    status_text.text('Done!')
if __name__ == "__main__":
    main()