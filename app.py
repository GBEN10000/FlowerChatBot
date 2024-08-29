from flask import Flask, request, render_template
import openai
import aiohttp
import asyncio
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = "Api-Key"

# Load the model
model = load_model('flower_model.h5')

async def search_openai_api(user_query):
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {openai.api_key}'},
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': user_query}]
                }
            )
            data = await response.json()
            return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/', methods=['GET'])
def upload_image():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def handle_image_upload():
    file = request.files['file']
    if file:
        img_path = 'uploaded_image.jpg'
        file.save(img_path)
        
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])  # Get the index of the highest probability
        
        # Map class indices to class names
        class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        flower_type = class_names[predicted_class]
        
        # Remove the uploaded image file
        os.remove(img_path)
        
        return render_template('result.html', flower_type=flower_type)

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    user_query = request.form.get('user_query', '')
    flower_type = request.form.get('flower_type', 'Unknown')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ai_response = loop.run_until_complete(search_openai_api(user_query))
    
    return render_template('result.html', flower_type=flower_type, ai_response=ai_response)


if __name__ == '__main__':
    app.run(debug=True)
