import gradio as gr
import pandas as pd
import pickle

# Load the trained model
with open('catboost_model.pkl', 'rb') as model_file:
    regressor = pickle.load(model_file)

# Define the prediction function
def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame(
        [[age, sex, bmi, children, smoker, region]],
        columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    )
    
    # Make prediction
    prediction = regressor.predict(input_data)
    return f'The insurance cost is USD {prediction[0]:.2f}'

# Set up the Gradio interface
inputs = [
    gr.Slider(minimum=18, maximum=100,step=1, value=31, label="Age"),
    gr.Radio(choices=['Female', 'Male']),
    gr.Slider(minimum=10.0, maximum=50.0, step=0.1, value=25.74, label="BMI"),
    gr.Slider(minimum=0, maximum=10, value=0,step=1, label="Children"),
    gr.Radio(choices=['NO', 'Yes']),
    gr.Dropdown(choices=['Southwest', 'Southeast',  'Northwest', 'Northeast'], label="Region (Southwest, Southeast, Northwest, Northeast)")
]

output = gr.Textbox(label="Predicted Insurance Cost")

# Create the Gradio interface
gr.Interface(fn=predict_insurance_cost, inputs=inputs, outputs=output, title="Medical Insurance Cost Predictor", description="Predict the insurance cost based on various parameters.").launch()
