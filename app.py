# Importing essential libraries and modules

from flask import Flask, render_template, request, session, redirect, url_for, jsonify, abort, flash
from markupsafe import Markup
from flask_babel import Babel, gettext as _
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from werkzeug.utils import secure_filename
import logging
import traceback

import numpy as np
import pandas as pd
from utils.disease import disease_dic

import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import base64
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

import threading
from io import BytesIO
import matplotlib.pyplot as plt
import uuid

import requests as pyrequests  # avoid conflict with Flask's requests
from utils import xai
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to INFO to show server status
    format='%(message)s'  # Simplified format
)
logger = logging.getLogger(__name__)

# Suppress Flask/Werkzeug logs except server status
logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('flask').setLevel(logging.WARNING)

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Define disease classes
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

try:
    # Loading plant disease classification model
    disease_model_path = 'plant-disease-model.pth'
    disease_model = ResNet9(3, len(disease_classes))
    disease_model.load_state_dict(torch.load(
        disease_model_path, 
        map_location=torch.device('cpu'),
        weights_only=True  # Add this to fix the warning
    ))
    disease_model.eval()
    logger.info("Disease model loaded successfully")

    # Loading crop recommendation model with version compatibility
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
    model = pickle.load(open('model1.pkl','rb'))
    sc = pickle.load(open('standscaler.pkl','rb'))
    mx = pickle.load(open('minmaxscaler.pkl','rb'))
    logger.info("Crop recommendation models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# =========================================================================================

# Custom functions for calculations

def allowed_file(filename):
    """Check if the file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    """Validate image file size and type"""
    if not file:
        return False, _('No file selected')
    if not allowed_file(file.filename):
        return False, _('Invalid file type. Please upload PNG, JPG, or JPEG')
    if file.content_length and file.content_length > 5 * 1024 * 1024:  # 5MB
        return False, _('File size exceeds 5MB limit')
    return True, None

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    try:
        api_key = config.weather_api_key
        base_url = "http://api.openweathermap.org/data/2.5/weather?"

        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        x = response.json()

        if x["cod"] != "404":
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2)
            humidity = y["humidity"]
            return temperature, humidity
        else:
            return None
    except Exception as e:
        logger.error(f"Error fetching weather: {str(e)}")
        return None

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        image = Image.open(io.BytesIO(img))
        img_t = transform(image)
        img_u = torch.unsqueeze(img_t, 0)

        # Get predictions from model
        yb = model(img_u)
        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)
        prediction = disease_classes[preds[0].item()]
        # Retrieve the class label
        return prediction
    except Exception as e:
        logger.error(f"Error predicting image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Required for session
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['LANGUAGES'] = {
    'en': 'English',
    'hi': 'Hindi'
}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize Babel with locale selector function
def get_locale():
    if 'language' in session:
        return session['language']
    return request.accept_languages.best_match(['en', 'hi'])

babel = Babel(app, locale_selector=get_locale)

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    logger.error(f"File too large: {str(error)}")
    return render_template('error.html', message=_('File too large')), 413

@app.errorhandler(429)
def ratelimit_handler(error):
    logger.error(f"Rate limit exceeded: {str(error)}")
    return render_template('error.html', message=_('Too many requests. Please try again later.')), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    logger.error(traceback.format_exc())
    return render_template('error.html', message=_('An internal error occurred. Please try again.')), 500

# Language route
@app.route('/language/<language>')
def set_language(language):
    session['language'] = language
    return redirect(request.referrer or url_for('home'))

# render home page
@app.route('/')
def home():
    title = 'Agrivision - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'Agrivision - Crop Recommendation'
    return render_template('crop1.html', title=title)

# render disease prediction result page
@app.route('/disease-predict', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def disease_prediction():
    title = 'Agrivision - Disease Detection'

    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('disease.html', title=title, error=_('No file selected'))
            
            file = request.files.get('file')
            is_valid, error_message = validate_image(file)
            
            if not is_valid:
                return render_template('disease.html', title=title, error=error_message)
            
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            logger.error(f"Error in disease prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return render_template('disease.html', title=title, error=_('Error processing image'))
    
    return render_template('disease.html', title=title)

@app.route('/gradcam', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def gradcam():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return render_template('gradcam.html', error=_('No file uploaded'))
            file = request.files['image']
            if not file or not file.filename:
                return render_template('gradcam.html', error=_('No file selected'))
            if not allowed_file(file.filename):
                return render_template('gradcam.html', error=_('Invalid file type. Please upload PNG, JPG, or JPEG'))
            img_bytes = file.read()
            if len(img_bytes) > 5 * 1024 * 1024:
                return render_template('gradcam.html', error=_('File size exceeds 5MB limit'))
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
            except Exception as e:
                return render_template('gradcam.html', error=_('Error processing image. Invalid image file.'))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(pil_img).unsqueeze(0)
            try:
                with torch.no_grad():
                    outputs = disease_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    pred_class = outputs.argmax(dim=1).item()
                    confidence = probabilities[pred_class].item()
            except Exception as e:
                return render_template('gradcam.html', error=_('Error during prediction. Please try again.'))
            try:
                disease_name = disease_classes[pred_class]
                disease_info = disease_dic.get(disease_name, '')
            except IndexError:
                return render_template('gradcam.html', error=_('Error: Invalid prediction result'))
            prediction = Markup(f"""
                <h4 class='alert-heading'>Disease Prediction</h4>
                <p><strong>Disease:</strong> {disease_name}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <hr>
                <p class='mb-0'>{disease_info}</p>
            """)
            try:
                img_np = np.array(pil_img).astype(np.float32) / 255.0
                target_layer = disease_model.conv4
                cam = GradCAM(model=disease_model, target_layers=[target_layer])
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
                cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                _, buffer = cv2.imencode('.png', cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
                gradcam_img = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                return render_template('gradcam.html', prediction=prediction, error='Error generating visualization. Prediction only.', title='Agrivision - GradCAM Visualization')
            return render_template('gradcam.html', prediction=prediction, gradcam_img=gradcam_img, title='Agrivision - GradCAM Visualization')
        except Exception as e:
            return render_template('gradcam.html', error='An unexpected error occurred. Please try again.', title='Agrivision - GradCAM Visualization')
    return render_template('gradcam.html', title='Agrivision - GradCAM Visualization')

lime_jobs = {}  # job_id: {'progress': 0, 'result': None, 'error': None}
from flask import send_file

@app.route('/lime', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def lime():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return render_template('lime.html', error=_('No file uploaded'))
            file = request.files['image']
            if file.filename == '':
                return render_template('lime.html', error=_('No file selected'))
            if not allowed_file(file.filename):
                return render_template('lime.html', error=_('Invalid file type. Please upload PNG, JPG, or JPEG'))
            if len(file.read()) > 5 * 1024 * 1024:
                return render_template('lime.html', error=_('File size exceeds 5MB limit'))
            file.seek(0)
            img_bytes = file.read()
            job_id = str(uuid.uuid4())
            session['lime_job_id'] = job_id
            lime_jobs[job_id] = {'progress': 0, 'result': None, 'error': None}

            num_samples = int(request.form.get('num_samples', 1000))
            num_features = int(request.form.get('num_features', 10))
            # Default positive_only to False for debugging
            positive_only = request.form.get('positive_only', None)
            if positive_only is None:
                positive_only = False
            else:
                positive_only = positive_only == 'on'
            def run_lime(img_bytes, job_id):
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    img_np = np.array(img)
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        output = disease_model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        predicted_class = torch.argmax(probabilities).item()
                        confidence = probabilities[0][predicted_class].item()
                    disease_name = disease_classes[predicted_class]
                    prediction_html = f"""
                        <div class='alert alert-info'>
                            <h5 class='mb-2'>Predicted Disease: {disease_name}</h5>
                            <p class='mb-1'>Confidence: {confidence:.2%}</p>
                            <p class='mb-0'>Class ID: {predicted_class}</p>
                        </div>
                    """
                    from lime import lime_image
                    explainer = lime_image.LimeImageExplainer()
                    orig_fn = explainer.data_labels
                    def progress_data_labels(*args, **kwargs):
                        if 'num_samples' in kwargs:
                            ns = kwargs['num_samples']
                        else:
                            ns = num_samples
                        for i in range(ns):
                            lime_jobs[job_id]['progress'] = int((i+1)/ns*90)
                        return orig_fn(*args, **kwargs)
                    explainer.data_labels = progress_data_labels
                    def predict_fn(images):
                        tensors = [transform(Image.fromarray(img.astype(np.uint8))) for img in images]
                        batch = torch.stack(tensors)
                        with torch.no_grad():
                            outputs = disease_model(batch)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                        return probs.numpy()
                    explanation = explainer.explain_instance(
                        img_np,
                        predict_fn,
                        top_labels=1,
                        hide_color=0,
                        num_samples=num_samples
                    )
                    lime_jobs[job_id]['progress'] = 95
                    temp, mask = explanation.get_image_and_mask(
                        predicted_class,
                        positive_only=positive_only,
                        num_features=num_features,
                        hide_rest=True
                    )
                    import logging
                    logging.warning(f'LIME temp shape: {temp.shape}, mask unique: {np.unique(mask)}')
                    # Ensure temp is float in [0, 1]
                    if temp.max() > 1.0:
                        temp = temp / 255.0
                    buffered = BytesIO()
                    plt.imsave(buffered, mark_boundaries(temp, mask))
                    lime_img = base64.b64encode(buffered.getvalue()).decode()
                    lime_jobs[job_id]['progress'] = 100
                    lime_jobs[job_id]['result'] = {'prediction': prediction_html, 'lime_img': lime_img}
                except Exception as e:
                    lime_jobs[job_id]['progress'] = -1
                    lime_jobs[job_id]['error'] = str(e)
            thread = threading.Thread(target=run_lime, args=(img_bytes, job_id))
            thread.start()
            return render_template('lime.html', job_id=job_id)
        except Exception as e:
            return render_template('lime.html', error=_('An error occurred while processing the image'))
    return render_template('lime.html')

@app.route('/lime-progress')
def lime_progress():
    job_id = session.get('lime_job_id')
    if not job_id or job_id not in lime_jobs:
        return jsonify({'progress': 0, 'error': 'No job found'})
    job = lime_jobs[job_id]
    resp = {'progress': job['progress']}
    if job['progress'] == 100:
        resp['result'] = job['result']
    if job['progress'] == -1:
        resp['error'] = job['error']
    return jsonify(resp)

crop_label_map = {
    1: 'rice',
    2: 'maize',
    3: 'jute',
    4: 'cotton',
    5: 'coconut',
    6: 'papaya',
    7: 'orange',
    8: 'apple',
    9: 'muskmelon',
    10: 'watermelon',
    11: 'grapes',
    12: 'mango',
    13: 'banana',
    14: 'pomegranate',
    15: 'lentil',
    16: 'blackgram',
    17: 'mungbean',
    18: 'mothbeans',
    19: 'pigeonpeas',
    20: 'kidneybeans',
    21: 'chickpea',
    22: 'coffee'
}

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            N = float(request.form.get('Nitrogen', 0))
            P = float(request.form.get('Phosphorus', 0))
            K = float(request.form.get('Potassium', 0))
            temperature = float(request.form.get('Temperature', 0))
            humidity = float(request.form.get('Humidity', 0))
            ph = float(request.form.get('pH', 0))
            rainfall = float(request.form.get('Rainfall', 0))
            state = request.form.get('state', '')
            season = request.form.get('season', '')

            # Validate input ranges
            if not (0 <= N <= 140):
                raise ValueError("Nitrogen value must be between 0 and 140")
            if not (0 <= P <= 140):
                raise ValueError("Phosphorus value must be between 0 and 140")
            if not (0 <= K <= 200):
                raise ValueError("Potassium value must be between 0 and 200")
            if not (0 <= temperature <= 50):
                raise ValueError("Temperature must be between 0 and 50°C")
            if not (0 <= humidity <= 100):
                raise ValueError("Humidity must be between 0 and 100%")
            if not (3 <= ph <= 10):
                raise ValueError("pH must be between 3 and 10")
            if not (0 <= rainfall <= 400):
                raise ValueError("Rainfall must be between 0 and 400mm")

            # Prepare input features
            input_features = [[N, P, K, temperature, humidity, ph, rainfall]]
            
            # Get probabilities for all crops
            probs = model.predict_proba(input_features)[0]
            
            # Get top 3 crop indices
            top_indices = np.argsort(probs)[::-1][:3]
            
            recommendations = []
            for idx in top_indices:
                crop = crop_label_map.get(idx+1, "Unknown")
                score = probs[idx]
                
                # Generate explanation based on input parameters
                explanation = f"{crop.capitalize()} is recommended because:"
                if N > 50:
                    explanation += " High nitrogen levels favor leaf growth."
                if P > 50:
                    explanation += " Good phosphorus levels support root development."
                if K > 50:
                    explanation += " Adequate potassium enhances disease resistance."
                if 20 <= temperature <= 30:
                    explanation += " Temperature is optimal for growth."
                if 60 <= humidity <= 90:
                    explanation += " Humidity levels are suitable."
                if 6 <= ph <= 7.5:
                    explanation += " Soil pH is in the ideal range."
                if 100 <= rainfall <= 300:
                    explanation += " Rainfall amount is appropriate."
                
                recommendations.append({
                    'crop': crop,
                    'score': score,
                    'explanation': explanation
                })

            return render_template('crop1.html', 
                                recommendations=recommendations, 
                                title='AgriXNet - Crop Recommendation',
                                state=state,
                                season=season)

        except ValueError as ve:
            logger.error(f"Validation error in crop recommendation: {str(ve)}")
            return render_template('crop1.html', 
                                error=str(ve),
                                title='AgriXNet - Crop Recommendation')
        except Exception as e:
            logger.error(f"Error in crop recommendation: {str(e)}")
            logger.error(traceback.format_exc())
            return render_template('crop1.html', 
                                error="An error occurred while processing your request. Please try again.",
                                title='AgriXNet - Crop Recommendation')
    
    return render_template('crop1.html', title='AgriXNet - Crop Recommendation')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        if username:
            session['username'] = username
            session['chat_history'] = []
            return redirect(url_for('home'))
        return render_template('login.html', error='Please enter a username.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

def local_ai_response(user_message, history=None):
    msg = user_message.lower()
    if 'feature' in msg or 'what can you do' in msg or 'app has' in msg:
        return ("This app offers: \n"
                "- Crop recommendation based on soil and climate\n"
                "- Plant disease detection from images\n"
                "- Explainable AI (XAI) with GradCAM and LIME\n"
                "- Multilingual support (English/Hindi)\n"
                "- Interactive support chat\n"
                "- And more! Ask about any feature for details.")
    if 'lime' in msg:
        return ("LIME (Local Interpretable Model-agnostic Explanations) helps explain model predictions by highlighting which parts of an image or which features most influenced the result. "
                "In this app, LIME shows you which regions of your plant image were most important for the disease prediction.")
    if 'gradcam' in msg:
        return ("GradCAM (Gradient-weighted Class Activation Mapping) is a visualization technique that highlights important regions in an image for a model's prediction. "
                "In this app, GradCAM helps you see which parts of your plant image led to the disease diagnosis.")
    if 'crop recommend' in msg or 'recommend crop' in msg:
        return ("The crop recommendation feature suggests the best crops for your land based on soil nutrients, temperature, humidity, pH, and rainfall. Just fill out the form and get instant suggestions!")
    if 'disease' in msg and 'upload' in msg:
        return ("To detect plant disease, go to the Disease Prediction section and upload a clear image of your plant leaf. The app will analyze it and tell you the likely disease.")
    if 'xai' in msg or 'explainable' in msg:
        return ("XAI (Explainable AI) helps you understand how AI models make decisions. This app uses GradCAM and LIME to visually explain predictions, making the results more transparent and trustworthy.")
    if 'language' in msg or 'hindi' in msg or 'english' in msg:
        return ("You can change the app language using the Language menu in the navigation bar. Currently, English and Hindi are supported.")
    if 'who built' in msg or 'developer' in msg or 'your creator' in msg:
        return ("This app was built by a team of agri-tech enthusiasts and AI developers to help farmers and researchers with crop and plant health management.")
    if 'contact' in msg or 'support' in msg:
        return ("For immediate support, you can reach our admins at:\nEmail: rohinsabharwal88@gmail.com or sachitsharma996@gmail.com")
    if 'weather' in msg:
        return "I can fetch weather info for you. Please provide your city name!"
    if 'recommend' in msg and 'crop' in msg:
        return "Based on your soil and climate, I recommend maize, millet, or pulses for summer. Want more details?"
    if 'thank' in msg:
        return "You're welcome! If you have more questions, just ask."
    if 'who are you' in msg or 'your name' in msg:
        return "I'm your virtual agri-support assistant, here to help you with crops, diseases, and more!"
    if 'bye' in msg or 'goodbye' in msg:
        return "Goodbye! Feel free to return if you need more help."
    if 'disease' in msg:
        return "You can upload a plant image for disease prediction, or describe the symptoms for advice."
    if 'help' in msg:
        return ("I can help with crop recommendations, disease prediction, XAI, language settings, and more. Ask me anything!")
    if 'hi' in msg or 'hello' in msg or 'hey' in msg:
        return f"Hello {session.get('username', 'there')}! How can I assist you today?"
    # Fallback with admin contact information
    return ("I apologize, but I'm not able to provide a specific answer to your question. For detailed assistance, please contact our admins:\n"
            "Email: rohinsabharwal88@gmail.com or sachitsharma996@gmail.com\n"
            "Meanwhile, I can help you with features, LIME, GradCAM, crop recommendation, or language settings!")

@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify({'reply': 'Please log in to use the chat.', 'history': []})
    data = request.get_json()
    user_message = data.get('message', '').strip()
    from datetime import datetime
    now = datetime.now().strftime('%H:%M')
    # Store in session chat history
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({'sender': 'user', 'text': user_message, 'time': now})
    # AI logic
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key:
        # Use OpenAI API for reply
        try:
            headers = {
                'Authorization': f'Bearer {openai_api_key}',
                'Content-Type': 'application/json'
            }
            messages = [
                {"role": "system", "content": "You are a helpful agricultural support assistant."}
            ]
            for m in session['chat_history'][-10:]:
                role = 'user' if m['sender'] == 'user' else 'assistant'
                messages.append({"role": role, "content": m['text']})
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": 120
            }
            response = pyrequests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                reply = response.json()['choices'][0]['message']['content'].strip()
            else:
                reply = local_ai_response(user_message, session['chat_history'])
        except Exception as e:
            reply = local_ai_response(user_message, session['chat_history'])
    else:
        reply = local_ai_response(user_message, session['chat_history'])
    session['chat_history'].append({'sender': 'bot', 'text': reply, 'time': now})
    session.modified = True
    return jsonify({'reply': reply, 'history': session['chat_history']})

@app.before_request
def require_login():
    allowed_routes = ['login', 'static', 'logout']
    if not session.get('username') and not any(request.endpoint and request.endpoint.startswith(route) for route in allowed_routes):
        return redirect(url_for('login'))

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return redirect(url_for('home'))
    
    # Here you can implement your search logic
    # For now, we'll just redirect to home with a flash message
    flash(f'Search functionality coming soon! You searched for: {query}', 'info')
    return redirect(url_for('home'))

@app.route('/cam', methods=['GET', 'POST'])
def cam():
    cam_img = None
    error = None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            error = 'No file selected.'
        else:
            file = request.files['file']
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    file.save(tmp.name)
                    cam_result = xai.generate_cam(disease_model, tmp.name)
                    _, buffer = cv2.imencode('.png', cam_result)
                    cam_img = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                error = f'Error generating CAM: {e}'
    return render_template('cam.html', cam_img=cam_img, error=error)

@app.route('/saliency', methods=['GET', 'POST'])
def saliency():
    saliency_img = None
    error = None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            error = 'No file selected.'
        else:
            file = request.files['file']
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    file.save(tmp.name)
                    saliency_result = xai.generate_saliency_map(disease_model, tmp.name)
                    _, buffer = cv2.imencode('.png', saliency_result)
                    saliency_img = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                error = f'Error generating saliency map: {e}'
    return render_template('saliency.html', saliency_img=saliency_img, error=error)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
