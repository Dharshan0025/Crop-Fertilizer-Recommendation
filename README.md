
# Crop Recommendation System

## Project Description
The Crop Recommendation System is a machine learning-based application that helps farmers and agriculturalists determine the most suitable crop to grow based on soil properties and environmental factors. Using a dataset containing information on soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH, and rainfall, the system predicts the crop that will thrive in the given conditions. This application provides valuable insights to improve crop yield and optimize farming practices.

---

## Features

1. **Data Analysis:**
   - Explore the dataset and understand its characteristics.
   - Identify missing or duplicate values.
   - Analyze the distribution of features (e.g., Nitrogen, temperature).

2. **Data Visualization:**
   - Correlation matrix to identify relationships between features.
   - Feature distribution plots for a detailed understanding of the dataset.

3. **Machine Learning Model:**
   - A Random Forest Classifier is trained to predict the most suitable crop.
   - Model performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

4. **Interactive Prediction System:**
   - Accepts user input (e.g., soil and environmental parameters).
   - Recommends the best crop based on real-time input.

5. **Feature Importance Analysis:**
   - Visualizes the importance of features in the crop prediction model.

---

## Dataset
The dataset contains the following columns:
- **N**: Nitrogen content in soil
- **P**: Phosphorus content in soil
- **K**: Potassium content in soil
- **temperature**: Temperature (°C)
- **humidity**: Humidity (%)
- **ph**: Soil pH value
- **rainfall**: Annual rainfall (mm)
- **label**: Target variable, representing the crop type (e.g., rice, wheat).

---

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.8+
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

Install dependencies using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd crop-recommendation-system
   ```

2. Place the dataset in the `dataset/` folder.

3. Run the script:
   ```bash
   python crop_recommendation.py
   ```

4. Follow the prompts for interactive crop prediction.

---

## Project Workflow

1. **Data Loading and Exploration:**
   - Load the dataset using Pandas.
   - View the first few records to verify the dataset.
   
2. **Data Preprocessing:**
   - Handle missing values and duplicates.
   - Encode categorical labels using `LabelEncoder`.
   - Scale numerical features using `StandardScaler`.

3. **Exploratory Data Analysis (EDA):**
   - Visualize the distribution of each feature.
   - Identify correlations between variables.

4. **Model Training and Evaluation:**
   - Split the dataset into training and testing sets.
   - Train a Random Forest Classifier.
   - Evaluate model performance using accuracy, classification report, and confusion matrix.

5. **Interactive System:**
   - Input real-time values for soil and environmental conditions.
   - Receive crop recommendations based on the trained model.

---

## Example Output

### Feature Correlation Heatmap
![Heatmap Example](heatmap_example.png)

### Classification Report
```
              precision    recall  f1-score   support

      rice       0.98       0.97      0.97       100
      wheat      0.99       0.98      0.98       100

   accuracy                           0.98       200
  macro avg       0.98       0.98      0.98       200
weighted avg       0.98       0.98      0.98       200
```

---

## File Structure
```
project-folder/
|— dataset/
|   |— Crop_recommendation.csv
|— crop_recommendation.py
|— README.md
```

---

## Future Enhancements

1. **Expand Dataset:**
   - Include data from diverse regions and crops.

2. **Advanced Models:**
   - Experiment with other machine learning algorithms like XGBoost or Neural Networks.

3. **Web Application:**
   - Develop a user-friendly web interface for farmers.

4. **Multi-Language Support:**
   - Provide recommendations in regional languages.

5. **Real-Time Data Integration:**
   - Incorporate live weather and soil sensor data.

---


## Acknowledgments
- [Scikit-Learn](https://scikit-learn.org/): Machine learning library.
- [Seaborn](https://seaborn.pydata.org/): Data visualization library.
- Dataset source: Publicly available agricultural dataset.

---

## Contact
For questions or suggestions, feel free to contact:
- **Name**: Dharshan B
- **Email**: dharshanb0025@gmail.com


