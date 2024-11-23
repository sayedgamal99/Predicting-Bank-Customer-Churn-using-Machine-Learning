from flask import Flask, render_template, request, jsonify
from utils import *


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    set_config(transform_output='pandas')
    if request.method == 'POST':
        try:
            input_data = {
                'id': 0,
                'CustomerId': int(request.form.get('CustomerId', 0)),
                'Surname': request.form.get('Surname', ''),
                'CreditScore': float(request.form['CreditScore']),
                'Geography': request.form['Geography'],
                'Gender': request.form['Gender'],
                'Age': float(request.form['Age']),
                'Tenure': float(request.form['Tenure']),
                'Balance': float(request.form['Balance']),
                'NumOfProducts': float(request.form['NumOfProducts']),
                'HasCrCard': float(request.form['HasCrCard']),
                'IsActiveMember': float(request.form['IsActiveMember']),
                'EstimatedSalary': float(request.form['EstimatedSalary']),
            }

            input_df = pd.DataFrame([input_data])

            # Call predict_churn
            churn_prob, prediction = predict_churn(
                input_df, preprocessor, model)
            logger.info(
                f"Churn probability: {churn_prob}, Prediction: {prediction}")

            return render_template('result.html', prediction=prediction,
                                   churn_prob=round(churn_prob * 100, 2),
                                   data=input_data)
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return render_template('error.html', error=str(e))
    return render_template('form.html', sample_data={})


@app.route('/use_sample', methods=['GET'])
def use_sample():
    """Endpoint for fetching a sample from the test dataset."""
    try:
        sample = test_data.sample(1).iloc[0].to_dict()
        return jsonify(sample)
    except Exception as e:
        logger.error(f"Error fetching sample data: {str(e)}")
        return jsonify({'error': str(e)})


def main():
    global preprocessor, model, test_data

    preprocessor, model = load_models()
    test_data = load_test_data()

    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
