const API_URL = "http://127.0.0.1:5000";
let currentDiseaseContext = "General Eye Health";

// --- Upload Logic ---
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('imagePreview').src = e.target.result;
            document.getElementById('previewContainer').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
});

async function analyzeImage() {
    const file = fileInput.files[0];
    if (!file) return alert("Please select an image first.");

    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    document.querySelector('.btn').innerText = "Analyzing...";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        // Update UI with results
        document.getElementById('resultSection').classList.remove('hidden');
        document.getElementById('resultTitle').innerText = data.class;
        document.getElementById('confidenceText').innerText = `Confidence: ${data.confidence}`;
        document.getElementById('confidenceLevel').style.width = data.confidence;
        document.getElementById('relatedDiseases').innerText = data.related_diseases;

        const tipsList = document.getElementById('tipsList');
        tipsList.innerHTML = "";
        data.tips.forEach(tip => {
            const li = document.createElement('li');
            li.innerText = tip;
            tipsList.appendChild(li);
        });

        // Set Context for Chatbot
        currentDiseaseContext = data.class;
        addBotMessage(`I've detected ${data.class}. Feel free to ask me questions about it!`);
        openChat();

    } catch (error) {
        alert("Error: " + error.message);
    } finally {
        document.querySelector('.btn').innerText = "🔍 Analyze Scan";
    }
}

// --- Chatbot Logic ---
function toggleChat() {
    const body = document.getElementById('chatBody');
    const input = document.querySelector('.chat-input-area');
    const icon = document.getElementById('toggleIcon');
    
    body.classList.toggle('open');
    input.classList.toggle('open');
    icon.classList.toggle('fa-chevron-down');
    icon.classList.toggle('fa-chevron-up');
}

function openChat() {
    const body = document.getElementById('chatBody');
    const input = document.querySelector('.chat-input-area');
    if (!body.classList.contains('open')) {
        body.classList.add('open');
        input.classList.add('open');
    }
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    // Add User Message
    const chatBody = document.getElementById('chatBody');
    chatBody.innerHTML += `<div class="message user">${message}</div>`;
    input.value = "";
    chatBody.scrollTop = chatBody.scrollHeight;

    // Call API
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: message, 
                context: currentDiseaseContext 
            })
        });
        const data = await response.json();
        addBotMessage(data.reply);
    } catch (error) {
        addBotMessage("Sorry, I can't connect right now.");
    }
}

function addBotMessage(text) {
    const chatBody = document.getElementById('chatBody');
    chatBody.innerHTML += `<div class="message bot">${text}</div>`;
    chatBody.scrollTop = chatBody.scrollHeight;
}