# Multi-Dataset Machine Learning Pipeline

## Overview
This project implements an end-to-end machine learning pipeline with an interactive Streamlit application. The pipeline supports multiple datasets (Iris, Digits, Wine, Breast Cancer, Diabetes, Linnerud) for classification and regression tasks. It includes data exploration, preprocessing, model training, evaluation, visualization, and real-time predictions, leveraging Python libraries such as Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and PIL.

Key features:
- **Exploratory Data Analysis (EDA)**: Visualizes dataset characteristics with summary statistics, pairplots, correlation heatmaps, and sample images (for Digits).
- **Preprocessing**: Standardizes features using `StandardScaler` and applies `PCA` for dimensionality reduction.
- **Modeling**: Trains `RandomForestClassifier` for classification tasks and `RandomForestRegressor` for regression tasks.
- **Evaluation**: Computes metrics like accuracy, confusion matrices, MSE, and R² scores.
- **Visualization**: Generates PCA 2D projections, decision boundaries, and predicted vs. actual plots.
- **Interactive Predictions**: Allows users to input feature values or upload images (for Digits) via Streamlit for real-time predictions.
- **Optimization**: Uses Streamlit caching (`@st.cache_data`, `@st.cache_resource`) for efficient performance.

## Datasets
The pipeline supports the following Scikit-learn datasets:
- **Iris**: Classification (3 classes, 4 features)
- **Digits**: Classification (10 classes, 64 features, image data)
- **Wine**: Classification (3 classes, 13 features)
- **Breast Cancer**: Classification (2 classes, 30 features)
- **Diabetes**: Regression (1 target, 10 features)
- **Linnerud**: Regression (3 targets, 3 features)

## Requirements
- Python 3.8+
- Libraries:
  - `streamlit>=1.24.0`
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `matplotlib>=3.7.0`
  - `seaborn>=0.12.0`
  - `scikit-learn>=1.2.0`
  - `pillow>=9.5.0`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-dataset-ml-pipeline.git
   cd multi-dataset-ml-pipeline
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run ex.py
   ```

## Usage
1. Launch the Streamlit app by running `streamlit run ex.py`.
2. Access the app in your browser (typically at `http://localhost:8501`).
3. Use the sidebar to:
   - Select a dataset (Iris, Digits, Wine, Breast Cancer, Diabetes, Linnerud).
   - Navigate to different pages: Home, Exploratory Data Analysis, Model Training & Evaluation, Interactive Prediction, Conclusion.
4. **Exploratory Data Analysis**:
   - View dataset overview (first 10 rows, summary statistics).
   - Explore pairplots, correlation heatmaps, and sample images (for Digits).
5. **Model Training & Evaluation**:
   - Train Random Forest models and view performance metrics (accuracy, MSE, R², confusion matrices).
   - Visualize predicted vs. actual values and PCA 2D projections.
6. **Interactive Prediction**:
   - Input feature values via sliders or upload an image (for Digits) to get real-time predictions.
   - View bar plots of input features and prediction results.
7. **Conclusion**: Review project summary and future enhancements.

## Project Structure
```
multi-dataset-ml-pipeline/
├── ex.py                # Main Streamlit app and ML pipeline code
├── requirements.txt     # List of dependencies
├── README.md            # Project documentation
```

## Key Features
- **Modular Design**: The pipeline is structured for reusability and scalability across datasets.
- **Interactive UI**: Streamlit provides a user-friendly interface with dynamic dataset selection and navigation.
- **Comprehensive Visualizations**: Includes pairplots, heatmaps, PCA plots, confusion matrices, and more.
- **Performance Optimization**: Uses caching to ensure fast data loading and model training.
- **Image Handling**: Supports image-based predictions for the Digits dataset using PIL.

## Example Visualizations
- **Dataset Overview**: First 10 rows and summary statistics in a tabular format.
- **Pairplot**: Visualizes feature relationships for datasets with up to 20 features.
- **Correlation Heatmap**: Shows feature correlations using `sns.heatmap`.
- **Sample Images**: Displays sample digit images for the Digits dataset.
- **Confusion Matrix**: Evaluates classification performance for datasets like Iris and Breast Cancer.
- **PCA Visualization**: 2D scatter plots with decision boundaries (classification) or color gradients (regression).
- **Predicted vs. Actual Plot**: Scatter plots for regression datasets like Diabetes.
- **Input Feature Bar Plot**: Visualizes user inputs in the interactive prediction module.

## Future Enhancements
- Implement hyperparameter tuning for Random Forest models using GridSearchCV.
- Add advanced preprocessing techniques (e.g., feature selection, outlier handling).
- Support real-time data integration or custom dataset uploads.
- Deploy the Streamlit app to a cloud platform for broader accessibility.
- Incorporate additional ML algorithms (e.g., SVM, XGBoost) for comparison.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built as part of a 15-day internship at BrainyBeam Info-Tech Pvt. Ltd., Ahmedabad (26th May 2025 - 13th June 2025).
- Thanks to the open-source community for libraries like Scikit-learn, Streamlit, and Seaborn.
- Inspired by tutorials from Data Professor, StatQuest, and Corey Schafer.

## Contact
For questions or feedback, please open an issue on GitHub or contact [your-email@example.com].