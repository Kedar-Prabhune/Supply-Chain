# Supply-Chain
# Late Delivery Prediction Dashboard

This project aims to predict the likelihood of late delivery for a given order using a machine learning model. The project includes a web-based dashboard built with Dash, where users can input features related to an order, and the model predicts whether the delivery will be late or not.

## Getting Started

To run the dashboard locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required Python packages listed in `requirements.txt`.
3. Run the Dash app by executing the `Notebook1.ipynb` file.

git clone https://github.com/Kedar-Prabhune/late-delivery-prediction.git
cd late-delivery-prediction
pip install -r requirements.txt
python Notebook1.ipynb


## Usage

- Enter the required features related to an order into the input fields.
- Click the "Predict" button to see the prediction for late delivery.

## Model

The machine learning model used in this project is a Random Forest classifier, trained on a dataset containing historical order data. The model predicts the likelihood of late delivery based on features such as shipping mode, days for shipping, customer segment, etc.
