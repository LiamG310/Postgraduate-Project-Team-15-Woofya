import openai
import os
import pandas as pd
from openai import OpenAI
from flask import Flask, request, jsonify

# Load your data (same as before)
test_df = pd.read_csv('woofya_db_2.csv')  # Place data
usr_data_df = pd.read_csv('usr_data.csv')  # User data

# Set up OpenAI API key securely
client = OpenAI(api_key=('replace with your api key'))

# Initialize the Flask app
app = Flask(__name__)

# Define a function to interact with the ChatGPT API
def chat_with_gpt(messages):
    completion = client.chat.completions.create(
        model="gpt-4",  # Correct model
        messages=messages
    )
    return completion.choices[0].message.content

# Define the route for the chatbot interaction
@app.route("/chat", methods=["POST"])
def chatbot():
    # Get user input from the request
    user_input = request.json
    user_description = user_input.get("description", "")
    user_location = user_input.get("location", "")
    
    # Validate user input
    if not user_description or not user_location:
        return jsonify({"error": "Please provide both description and location."}), 400
    
    # Filter places based on user location
    filtered_df = test_df[test_df['vicinity'].str.contains(user_location, na=False, case=False)]
    
    # Filter based on user preferences (e.g., 'off-leash')
    if 'off-leash' in user_description.lower() or 'off leash' in user_description.lower():
        filtered_df = filtered_df[filtered_df['combined'].str.contains('off-leash', na=False, case=False)]

    # Rank by location (first filter by location proximity) and rating
    sorted_df = filtered_df.sort_values(by=['vicinity', 'rating'], ascending=[True, False])
    
    recommendations = sorted_df[['name', 'vicinity', 'rating', 'types', 'review_text','opening_hours.periods']].head(3).to_dict(orient='records')

    # Format recommendations for GPT, including the 'combined' information
    recommendations_text = "\n".join([
        f"{i+1}. {place['name']} - {place['vicinity']} (Rating: {place['rating']}, Types: {place['types']}, Hours: {place['opening_hours.periods']})\nReviews: {place['review_text']}"
        for i, place in enumerate(recommendations)
    ])
    
    # Prepare the conversation context for GPT with the recommendations
    messages = [
        {"role": "system", "content": "You are a dog-friendly recommendation assistant."},
        {"role": "user", "content": f"Based on the following places assume they are dog-friendly, please provide a detailed response:\n{recommendations_text}"}
    ]
    
    # Get the response from GPT
    gpt_response = chat_with_gpt(messages)
    
    # Send GPT response and recommendations back to the user
    return jsonify({"gpt_response": gpt_response, "recommendations": recommendations})


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

