# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The goal is to develop a model that can accurately distinguish between legitimate and fraudulent transactions, helping financial institutions mitigate fraud.

## Features
- **Machine Learning Model**: Trained using real-world credit card transaction data.
- **Jupyter Notebook**: For data analysis, model training, and evaluation.
- **Web Interface**: A simple UI to test and visualize predictions.
- **Pre-trained Model**: A trained model saved as `fraud_model.joblib` for quick deployment.
- **Dataset**: Includes `credit_card_transactions.csv` for model training and testing.

## Project Structure
```
Credit-Card-Fraud-Detection/
│-- FraudDetection_Colab.ipynb    # Jupyter Notebook for analysis and training
│-- UI/
│   ├── app.py                   # Main script for UI
│   ├── credit_card_transactions.csv  # Transaction dataset
│   ├── fraud_model.joblib        # Trained model file
│   ├── static/                   # Static assets
│   ├── templates/                # HTML templates for the UI
│-- README.md                     # Project documentation
│-- .gitignore                     # Git ignore rules
```

## Installation & Setup
### Prerequisites
Ensure you have Python installed along with the required libraries.

### Steps
1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-repo/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook (optional for model training):**
   ```sh
   jupyter notebook FraudDetection_Colab.ipynb
   ```

4. **Run the web application:**
   ```sh
   python UI/app.py
   ```

## Usage
- **Model Training:** The Jupyter Notebook `FraudDetection_Colab.ipynb` can be used to train and evaluate different machine learning models.
- **Web Interface:** The UI allows users to input transaction details and get fraud detection predictions.
- **Pre-trained Model:** The saved `fraud_model.joblib` can be used for quick inference without retraining.

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, Joblib, Flask)
- **Jupyter Notebook** for data analysis
- **Flask** for the web interface

## Future Improvements
- Deploy the model as a REST API.
- Improve feature engineering for better accuracy.
- Implement real-time fraud detection with streaming data.

## Contributing
Feel free to fork the repository and submit pull requests with improvements or new features.

## License
This project is licensed under the MIT License.


