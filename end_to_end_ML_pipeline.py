import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes, load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io
import re

# Set page config for a professional look
st.set_page_config(page_title="Multi-Dataset ML Pipeline App", layout="wide", initial_sidebar_state="expanded")

# --- Helper Function for Formatting Description ---
def display_formatted_description(description):
    """
    Parses the scikit-learn dataset description string (reST format)
    and displays it in a clean, formatted way in Streamlit.
    """
    # Keep the initial title and intro paragraph
    try:
        title, rest = description.split('.. _', 1)
        st.markdown(title.strip())
    except ValueError:
        st.markdown(description) # Fallback for different formats
        return

    # Use regex to find all key-value pairs like ':key: value'
    characteristics = re.findall(r':(.*?):\s*(.*)', description)
    info_dict = {}

    for key, value in characteristics:
        key = key.strip()
        value = value.strip()
        if value and key not in ["Attribute Information", "Summary Statistics"]:
            info_dict[key] = value

    st.subheader("Dataset Characteristics")
    for key, value in info_dict.items():
        st.markdown(f"**{key}:** {value}")

    # Handle multi-line Attribute Information
    if ":Attribute Information:" in description:
        st.markdown("**Attribute Information:**")
        # Extract the block of text for attributes
        attr_text = description.split(':Attribute Information:')[1].split(':Summary Statistics:')[0]
        # Find all lines starting with '- '
        attributes = [line.strip() for line in attr_text.split('\n') if line.strip().startswith('- ')]
        for attr in attributes:
            st.markdown(attr)
            
    # Handle Summary Statistics by converting the text table to a DataFrame
    if ":Summary Statistics:" in description and '=============================' in description:
        st.subheader("Summary Statistics")
        try:
            # Isolate the table string based on the '===' separators
            table_str = description.split('=============================')[1]
            table_str = "=============================\n" + table_str.strip() + "\n============================="
            
            # Use pandas to read the fixed-width format table
            df = pd.read_fwf(io.StringIO(table_str))
            
            # Clean up the DataFrame
            df = df.rename(columns=lambda x: x.replace(':', '').strip())
            df = df.dropna(how='all')
            df = df[~df.iloc[:, 0].str.contains('===', na=False)]
            df = df.set_index(df.columns[0])
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.warning("Could not automatically format the summary statistics table.")
            # Fallback to show the raw text if parsing fails
            raw_stats = description.split(':Summary Statistics:')[1]
            st.text(raw_stats)


# Sidebar for navigation and dataset selection
st.sidebar.title("Navigation")
dataset = st.sidebar.selectbox("Select Dataset", ["Iris", "Digits", "Wine", "Breast Cancer", "Diabetes", "Linnerud"])
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Model Training & Evaluation", "Interactive Prediction", "Conclusion"])

# Cache the data loading for efficiency
@st.cache_data
def load_data(dataset):
    if dataset == "Iris":
        bunch = load_iris()
    elif dataset == "Digits":
        bunch = load_digits()
    elif dataset == "Wine":
        bunch = load_wine()
    elif dataset == "Breast Cancer":
        bunch = load_breast_cancer()
    elif dataset == "Diabetes":
        bunch = load_diabetes()
    elif dataset == "Linnerud":
        bunch = load_linnerud()
    
    original_target = bunch.target
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    
    # Determine if classification based on dataset
    is_classification = dataset in ["Iris", "Digits", "Wine", "Breast Cancer"]
    
    if original_target.ndim == 1:
        df['target'] = original_target
        target = df['target']
        target_cols = ['target']
        is_multi_target = False
    else:
        target_cols = bunch.target_names if hasattr(bunch, 'target_names') else bunch.target_feature_names if hasattr(bunch, 'target_feature_names') else [f'target_{i}' for i in range(original_target.shape[1])]
        df_targets = pd.DataFrame(original_target, columns=target_cols)
        df = pd.concat([df, df_targets], axis=1)
        target = original_target
        is_multi_target = True
    
    target_names_str = None
    if is_classification:
        if hasattr(bunch, 'target_names') and bunch.target_names is not None:
            target_names_str = [str(name) for name in bunch.target_names]
        else:
            target_names_str = [str(i) for i in np.unique(original_target)]
        if original_target.ndim == 1:
            df['class'] = [target_names_str[int(i)] for i in original_target]
    
    images = bunch.images if hasattr(bunch, 'images') else None
    description = bunch.DESCR if hasattr(bunch, 'DESCR') else "No description available."
    
    return df, target_names_str, bunch.feature_names, description, images, target, is_classification, is_multi_target, target_cols

# Load data based on selection
data, target_names, feature_names, description, images, target, is_classification, is_multi_target, target_cols = load_data(dataset)
features = data[feature_names]

# Cache model training
@st.cache_resource
def train_model(_features, _target, dataset, is_classification, is_multi_target, target_names):
    X_train, X_test, y_train, y_test = train_test_split(_features, _target, test_size=0.2, random_state=42)
    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if is_classification:
        metric1 = accuracy_score(y_test, y_pred)
        metric2 = None
        report = classification_report(y_test, y_pred, target_names=target_names)
        cm = confusion_matrix(y_test, y_pred)
    else:
        if is_multi_target:
            metric1 = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
            metric2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        else:
            metric1 = mean_squared_error(y_test, y_pred)
            metric2 = r2_score(y_test, y_pred)
        report = f"Mean Squared Error: {metric1:.4f}\nR² Score: {metric2:.4f}"
        cm = None
    
    return model, metric1, metric2, report, cm, X_test, y_test, y_pred

# Train model
model, metric1, metric2, report, cm, X_test, y_test, y_pred = train_model(features, target, dataset, is_classification, is_multi_target, target_names)

# --- App Pages ---

# Home Page
if page == "Home":
    st.title(f"{dataset} Dataset ML Pipeline")
    st.markdown("""
    ### Project Overview
    This application demonstrates an end-to-end Machine Learning pipeline, including data loading, EDA, preprocessing, model training, evaluation, visualization, and interactive predictions.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_classification:
            st.markdown("**Problem Definition:** Classification task to predict the class using the provided features.")
        else:
            st.markdown("**Problem Definition:** Regression task to predict the target value(s) using the provided features.")
        
        st.subheader("Dataset Description")
        display_formatted_description(description) # Using the new formatting function
        
    with col2:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        if dataset == "Iris":
            st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Example: Iris Versicolor", width=300)
        elif dataset == "Digits":
            st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_digits_classification_001.png", caption="Example: Handwritten Digits", width=300)
        elif dataset == "Wine":
            st.image("https://media.winefolly.com/red-wine-boldness-chart-by-wine-folly.png", caption="Example: Wine Classification", width=300)
        elif dataset == "Breast Cancer":
            st.image("https://psu-gatsby-files-prod.s3.amazonaws.com/s3fs-public/2022/10/bioprinted_tumor.jpg", caption="Example: Breast Cancer Cells", width=300)
        elif dataset == "Diabetes":
            st.image("https://www.shalom-education.com/wp-content/uploads/2022/11/Shutterstock_792237643-843x1024.jpg", caption="Example: Blood Glucose Meter", width=300)
        elif dataset == "Linnerud":
            st.image("https://i.pinimg.com/originals/97/92/6e/97926e20f47d548d612c70ba94c9eb22.jpg", caption="Example: Fitness Exercise", width=300)
    
    st.markdown("""
    ---
    Navigate using the sidebar to explore each step! Select different datasets to see how the app adapts to classification and regression tasks.
    """)

# EDA Page
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Overview")
        st.dataframe(data.head(10), use_container_width=True)
        st.markdown(f"**Shape:** {data.shape}")
        st.markdown("**Missing Values:** None (clean dataset)")
    
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(data.describe(), use_container_width=True)
    
    if dataset == "Digits" and images is not None:
        st.subheader("Sample Images")
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(images[i], cmap='gray')
                ax.axis('off')
                if 'class' in data:
                    ax.set_title(data['class'][i])
                st.pyplot(fig)
    
    st.subheader("Data Distribution")
    with st.expander("Pairplot (may take time for high dimensions)", expanded=True):
        hue = 'class' if 'class' in data else None
        pairplot_fig = sns.pairplot(data, hue=hue, palette='Set2' if hue else None, diag_kind='kde')
        st.pyplot(pairplot_fig)
    
    st.subheader("Feature Correlations")
    corr = data.corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=(len(data.columns) < 20), cmap='coolwarm', ax=ax_corr, fmt=".2f")
    st.pyplot(fig_corr)

# Model Training & Evaluation Page
elif page == "Model Training & Evaluation":
    st.title("Model Training & Evaluation")
    
    st.subheader("Train-Test Split")
    st.markdown("Data split: 80% training, 20% testing (random_state=42 for reproducibility).")
    
    st.subheader("Model Selection")
    if is_classification:
        st.markdown("Using Random Forest Classifier (ensemble method for high accuracy).")
    else:
        st.markdown("Using Random Forest Regressor (ensemble method for accurate predictions).")
    
    st.subheader("Evaluation Metrics")
    if is_classification:
        st.markdown(f"**Accuracy:** {metric1 * 100:.2f}%")
    else:
        st.markdown(f"**Mean Squared Error:** {metric1:.4f}")
        st.markdown(f"**R² Score:** {metric2:.4f}")
    st.text("Report:")
    if is_classification:
        # Parse classification report into dataframe for better display with lines
        report_lines = report.split('\n')
        report_data = []
        for line in report_lines[2:-5]:
            row_data = line.split()
            if len(row_data) > 0:
                report_data.append(row_data)
        report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        st.dataframe(report_df, use_container_width=True)
        
        # Display averages separately
        avg_lines = report_lines[-4:]
        st.text("\n".join(avg_lines))
    else:
        st.text(report)
    
    if is_classification:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap='Blues', ax=ax_cm)
        st.pyplot(fig_cm)
    
    if not is_classification:
        st.subheader("Predicted vs Actual")
        if not is_multi_target:
            fig_pv, ax_pv = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax_pv)
            ax_pv.set_xlabel("Actual Target")
            ax_pv.set_ylabel("Predicted Target")
            st.pyplot(fig_pv)
        else:
            cols = st.columns(min(3, y_test.shape[1]))
            for i in range(y_test.shape[1]):
                with cols[i % 3]:
                    fig_pv, ax_pv = plt.subplots(figsize=(4, 3))
                    sns.scatterplot(x=y_test[:, i], y=y_pred[:, i], ax=ax_pv)
                    ax_pv.set_title(target_cols[i])
                    ax_pv.set_xlabel("Actual")
                    ax_pv.set_ylabel("Predicted")
                    st.pyplot(fig_pv)
    
    st.subheader("PCA Visualization")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig_db, ax_db = plt.subplots(figsize=(10, 6))
    if is_classification:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['class'], palette='Set2', ax=ax_db)
        # Decision boundaries (simplified, using PCA-trained model)
        X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, target, test_size=0.2, random_state=42)
        model_pca = RandomForestClassifier(n_estimators=100, random_state=42) if is_classification else RandomForestRegressor(n_estimators=100, random_state=42)
        model_pca.fit(X_train_pca, y_train)
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
        if not is_classification:
            Z = Z[:, 0] if is_multi_target else Z
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='Set2' if is_classification else 'viridis')
    else:
        color_values = target[:, 0] if is_multi_target else target
        sc = ax_db.scatter(X_pca[:, 0], X_pca[:, 1], c=color_values, cmap='viridis')
        plt.colorbar(sc, label=target_cols[0] if is_multi_target else 'Target Value')
    plt.title("PCA 2D Projection of the Dataset")
    st.pyplot(fig_db)

# Interactive Prediction Page
elif page == "Interactive Prediction":
    st.title("Interactive Prediction")
    
    st.markdown(f"Provide input for the {dataset} dataset and click 'Predict' to get the model's prediction.")
    
    if dataset == "Digits":
        uploaded_file = st.file_uploader("Upload a grayscale digit image (e.g., 28x28 PNG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('L').resize((8, 8))
            arr = np.array(img) / 255.0 * 16.0
            st.image(img, caption="Processed 8x8 Image", width=200)
            
            if st.button("Predict"):
                input_data = arr.flatten().reshape(1, -1)
                prediction = model.predict(input_data)
                class_label = target_names[int(prediction[0])]
                st.success(f"Predicted Class: **{class_label}**")
                st.markdown(f"This means the uploaded digit image is predicted to be the number **{class_label}**.")
    else:
        input_values = []
        num_cols = 4 if len(feature_names) > 20 else 3
        cols = st.columns(num_cols)
        for i, feat in enumerate(feature_names):
            with cols[i % num_cols]:
                min_val = float(features[feat].min())
                max_val = float(features[feat].max())
                value = float(features[feat].mean())
                step = 0.1 if max_val - min_val < 10 else 1.0
                input_values.append(st.slider(feat, min_val, max_val, value, step=step))
        
        if st.button("Predict"):
            # Create a DataFrame with column names to avoid the UserWarning
            input_data = pd.DataFrame([input_values], columns=feature_names)
            prediction = model.predict(input_data)
            
            if is_classification:
                class_label = target_names[int(prediction[0])]
                st.success(f"Predicted Class: **{class_label}**")
                if dataset == "Iris":
                    st.markdown(f"The model predicts the flower is a **{class_label}** species based on the provided sepal and petal measurements.")
                elif dataset == "Wine":
                    st.markdown(f"The model predicts the wine belongs to class **{class_label}** based on its chemical properties.")
                elif dataset == "Breast Cancer":
                    st.markdown(f"The model predicts the tumor is **{class_label}** (malignant or benign) based on the cell features.")
            else:
                if not is_multi_target:
                    st.success(f"Predicted Value: **{prediction[0]:.4f}**")
                    if dataset == "Diabetes":
                        st.markdown(f"The predicted disease progression measure is **{prediction[0]:.4f}**, indicating the level of diabetes progression one year after baseline.")
                else:
                    for i, pred in enumerate(prediction[0]):
                        st.success(f"Predicted {target_cols[i]}: **{pred:.4f}**")
                    if dataset == "Linnerud":
                        st.markdown("These are the predicted physiological measurements (Weight, Waist, and Pulse) based on the exercise data.")
            
            # Input visualization
            st.subheader("Input Visualization")
            input_df = pd.DataFrame([input_values], columns=feature_names)
            fig_input, ax_input = plt.subplots(figsize=(10, 4))
            # Fixed the FutureWarning by assigning 'x' to 'hue' and disabling the legend
            sns.barplot(x=input_df.columns, y=input_df.iloc[0], palette='pastel', hue=input_df.columns, legend=False, ax=ax_input)
            plt.title("Input Features")
            plt.xticks(rotation=45)
            st.pyplot(fig_input)

# Conclusion Page
elif page == "Conclusion":
    st.title("Conclusion")
    st.markdown(f"""
    ### Summary for {dataset} Dataset
    - **Performance:** The Random Forest model performed well, with key metrics shown in the evaluation section.
    - **Visualizations:** EDA and PCA revealed patterns in the data. For classification, the confusion matrix and decision boundaries confirm robustness; for regression, predicted vs actual plots show fit.
    - **Enhancements:** Incorporate pipelines for preprocessing, hyperparameter tuning, or advanced models. The interactive prediction serves as a great demo!
    
    This app showcases a flexible end-to-end ML pipeline adapting to multiple datasets and task types (classification/regression).
    """)
    st.markdown("Built with ❤️ using Python, Scikit-learn, Matplotlib, Seaborn, Streamlit, and PIL.")