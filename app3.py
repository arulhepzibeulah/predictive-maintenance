from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained model and vectorizer
best_model = joblib.load("best_model.joblib")  # Ensure this file is in the same directory
vectorizer = joblib.load("vectorizer.joblib")  # Ensure this file is in the same directory

# Define intents with correct formatting
intents = {
    "intents": [
        {
            "tag": "MachineConditionCheck",
            "patterns": ["How's the machine looking?", "Check machine condition", "Machine status?"],
            "responses": ["The machine is currently in good condition.", "All performance metrics are normal."],
            "context_set": ""
        },
        {
            "tag": "ProblemDetected",
            "patterns": ["What should we do if there’s a problem?", "What if there’s an issue?", "How to handle machine problems?"],
            "responses": [
                "If you see any issues, here's what to do:", 
                "Unusual Noises: Check for loose parts or wear. Tighten or replace parts as needed.", 
                "Temperature Spikes: Ensure cooling systems are working. Clean or repair cooling components if necessary.", 
                "Vibration Issues: Inspect for misalignment or imbalance. Adjust or replace parts to correct the problem.", 
                "Error Messages: Follow the error code instructions. Consult the manual or contact support if needed."
            ],
            "context_set": ""
        },
        {
            "tag": "UnclearProblem",
            "patterns": ["What if the problem isn’t clear?", "What if I can’t identify the issue?", "How to handle unclear problems?"],
            "responses": [
                "If the issue isn’t obvious, perform a diagnostic check.",
                "Look at system logs for more details.",
                "If needed, escalate the issue to a senior technician."
            ],
            "context_set": ""
        },
        {
            "tag": "PreventFutureProblems",
            "patterns": ["How can we prevent future problems?", "What to do to avoid issues in the future?", "How to prevent machine failures?"],
            "responses": [
                "After fixing the issue, review the cause and update maintenance practices.",
                "Adjust schedules or improve monitoring based on new data."
            ],
            "context_set": ""
        },
        {
            "tag": "GeneralHelp",
            "patterns": ["Thanks", "Thank you", "I need more help"],
            "responses": ["You’re welcome! If you need more help, just ask."],
            "context_set": ""
        }
    ]
}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        sensor_data = data.get('sensor_data', [])
        if not sensor_data:
            return jsonify({"message": "No sensor data provided."}), 400

        response_message = f"Received sensor data: {sensor_data}. Predicting no immediate failure."
        return jsonify({"message": response_message})

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    try:
        data = request.get_json()
        user_input = data.get('message', '')

        if user_input:
            # Transform the user's input using the vectorizer
            input_text = vectorizer.transform([user_input])
            # Predict the intent using the loaded model
            predicted_intent = best_model.predict(input_text)[0]

            # Find the appropriate response from the intents
            response = "Sorry, I couldn't understand that. Please try again."
            for intent in intents['intents']:
                if intent['tag'] == predicted_intent:
                    response = random.choice(intent['responses'])
                    break

            return jsonify({"response": response})
        else:
            return jsonify({"response": "No message provided."}), 400

    except Exception as e:
        return jsonify({"response": "Sorry, there was an issue processing your request."})

if __name__ == '__main__':
    app.run(debug=True)
