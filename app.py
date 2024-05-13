from flask import Flask, request
from flask_cors import CORS
import json
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get code and number of results from request
    code = request.json.get('code')
    num_results = min(request.json.get('numResults', 5), 15)  # Ensure num_results <= num_beams

    # Summarize the code
    summarizer = pipeline("summarization")
    summaries = summarizer(code, max_length=150, min_length=30, num_beams=15, length_penalty=2.0, early_stopping=True, num_return_sequences=num_results)

    # Extract summaries
    summary_texts = [summary['summary_text'] for summary in summaries]

    # Create JSON response manually
    response_data = {'summaries': summary_texts}
    response = app.response_class(
        response=json.dumps(response_data),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    app.run(debug=True)