import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import requests # <--- NEW IMPORT
import json
import os
from dotenv import load_dotenv
# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app)
load_dotenv()
# API CONFIGURATION
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # <--- PASTE KEY HERE

# --- 1. DEFINE MODEL ARCHITECTURE ---
class CNN(nn.Module):
    def __init__(self, NUMBER_OF_CLASSES):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUMBER_OF_CLASSES),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        return x

# --- 2. LOAD MODEL ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
model = CNN(NUMBER_OF_CLASSES=4)

try:
    model.load_state_dict(torch.load('eye_disease_model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
# --- 3. HELPER DATA (Expanded) ---
DISEASE_INFO = {
    "Cataract": {
        "description": "A cataract is a clouding of the normally clear lens of the eye. It is like looking through a frosty or fogged-up window.",
        "related": "Age-related Macular Degeneration (AMD), Diabetic Retinopathy, Lens-Induced Glaucoma",
        "tips": [
            "Wear UV-protective sunglasses to slow progression.",
            "Quit smoking, as it increases cataract risk.",
            "Increase intake of antioxidants (Vitamin C and E) found in leafy greens.",
            "Schedule surgery if vision loss affects daily activities."
        ]
    },
    "Diabetic Retinopathy": {
        "description": "Diabetic retinopathy (DR) is a strong indicator of widespread blood vessel damage throughout the body.",
        "related": "It's increased risk of developing other serious systemic diseases, particularly affecting the kidneys, heart, and brain",
        "tips": [
            "Strictly control blood sugar (HbA1c) levels.",
            "Monitor blood pressure and cholesterol to protect blood vessels.",
            "Get a comprehensive dilated eye exam at least once a year.",
            "Report sudden vision changes (spots, blurriness) to a doctor immediately."
        ]
    },
    "Glaucoma": {
        "description": "A group of eye conditions that damage the optic nerve, the health of which is vital for good vision. This damage is often caused by an abnormally high pressure in your eye.",
        "related": "Cataracts,Cardiovascular Disease,Diabetes, Neurodegenerative Diseases,Sleep Apnea,Migraines and Vasospasm",
        "tips": [
            "Adhere strictly to prescribed eye drop schedules to lower pressure.",
            "Exercise regularly but avoid head-down positions (like certain yoga poses).",
            "Limit caffeine intake as it can temporarily raise eye pressure.",
            "Wear protective eyewear during sports or home improvement projects."
        ]
    },
    "Normal": {
        "description": "The retina appears healthy with no visible signs of disease. The optic nerve, blood vessels, and macula look normal.",
        "related": "Refractive Errors (Myopia, Hyperopia) - though not diseases, they affect vision.",
        "tips": [
            "Follow the 20-20-20 rule: Every 20 mins, look 20 feet away for 20 seconds.",
            "Wear sunglasses that block 99-100% of UVA and UVB radiation.",
            "Eat a diet rich in Omega-3 fatty acids and dark leafy greens.",
            "Don't smoke; smoking damages the optic nerve."
        ]
    }
}
# --- 4. ROUTES ---
@app.route('/')
def health_check():
    return {
        "status": "online",
        "message": "Eye Disease AI Backend is running",
        "version": "1.0.0"
    }, 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        # Preprocess
        img = Image.open(file).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
            label = CLASS_NAMES[predicted_idx.item()]
            conf_score = confidence.item() * 100
            
            # Get Info
            info = DISEASE_INFO.get(label, {})

        return jsonify({
            "class": label,
            "confidence": f"{conf_score:.2f}%",
            "description": info.get("description", ""),
            "related_diseases": info.get("related", "N/A"),
            "tips": info.get("tips", [])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    context = data.get('context', 'General Eye Health')

    if not user_message:
        return jsonify({"reply": "Please say something."})

    # 1. Prepare Prompt
    prompt_text = f"""
    You are an expert Ophthalmologist AI assistant. 
    The user is currently viewing a result for: {context}.
    User Question: {user_message}
    
    Provide a helpful, empathetic, and medical-fact-based answer. 
    Keep it concise (max 3 sentences).
    """

    # 2. Prepare API Request (REST API)
    # We use gemini-1.5-flash because it is faster and cheaper for chat
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }

    # 3. Send Request using 'requests' library
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result_json = response.json()
            # Extract text from the complex JSON response
            bot_reply = result_json['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"reply": bot_reply})
        else:
            print("API Error:", response.text)
            return jsonify({"reply": "I'm having trouble thinking right now. Please try again."})
            
    except Exception as e:
        print(f"Connection Error: {e}")
        return jsonify({"reply": "I am having trouble connecting to the internet."})
    

PORT = int(os.getenv("PORT",7860))
if __name__ == '__main__':
    app.run(debug=True, port=PORT)