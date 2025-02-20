import os
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import pandas as pd
import torch
import io
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for Flutter requests
CORS(app)

PORT = 5000
# Load YOLO model for fruit detection
yolo_model = YOLO("models/best.pt")

# Load the 100-fruit dataset (CSV file with "Fruit Name" and "Information" columns)
dataset_path = "models/fruits_dataset.csv"
df = pd.read_csv(dataset_path)

# Debugging: Print columns to check if 'Fruit Name' and 'Information' exist
print("Dataset columns:", df.columns)

# Ensure the correct columns are being used
if "Fruit Name" not in df.columns or "Information" not in df.columns:
    raise ValueError("Dataset must have 'Fruit Name' and 'Information' columns.")

# Load GPT-2 model for additional fruit descriptions
gpt2_model_name = "gpt2"  # Using larger GPT-2 model
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt_pipeline = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer)

@app.route('/')
def index():
    return "Fruit Detection API is running!"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))

        # Run YOLO on image and extract detected fruits
        results = yolo_model(img)

        detected_fruit = None
        for result in results:
            # Iterate over each detection and get the fruit label (adjust based on actual detection)
            for box in result.boxes:
                fruit_class = int(box.cls[0])  # YOLO gives a class ID
                detected_fruit = result.names[fruit_class].lower()  # Get class name from YOLO
                break  # Exit after first detection (assuming one fruit per image)

        # Handle case where no fruit was detected
        if not detected_fruit:
            return jsonify({"error": "No fruit detected"}), 400

        # Search the dataset for fruit information using correct column names
        fruit_info = df[df["Fruit Name"].str.lower() == detected_fruit]["Information"].values

        # Fallback if fruit information is not found in the dataset
        fruit_description = fruit_info[0] if len(fruit_info) > 0 else f"{detected_fruit} is a nutritious fruit."

        # ✅ Improved structured prompt
        input_text = (
            f"The {detected_fruit} is a nutritious fruit. "
            f"It belongs to the family {fruit_description}. "
            f"Some key benefits of {detected_fruit} include:"
        )

        # ✅ Improved text generation settings
        gpt_output = gpt_pipeline(
            input_text,
            num_return_sequences=1,
            do_sample=True,  # Sampling ON for variety
            top_p=0.85,  # Balanced randomness
            temperature=0.4,  # Lower temp to prevent loops
            max_length=min(len(input_text) + 50, 200),  # Limits length dynamically
            repetition_penalty=1.2  # Prevents repeated words
        )[0]["generated_text"]

        # ✅ Post-processing to remove duplicate phrases
        gpt_output = gpt_output.replace("\n", " ").strip()

        # ✅ Remove exact repeating sentences
        sentences = gpt_output.split(". ")
        cleaned_sentences = []
        seen_sentences = set()
        for sentence in sentences:
            if sentence.lower() not in seen_sentences:
                cleaned_sentences.append(sentence)
                seen_sentences.add(sentence.lower())
        gpt_output = ". ".join(cleaned_sentences)

        # ✅ Final JSON response
        return jsonify({
            "fruit": detected_fruit,
            "info": fruit_description,
            "gpt_extra": gpt_output
        })

    except Exception as e:
        # Log the error (you can use logging module here for production)
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
