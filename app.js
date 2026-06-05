const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const flash = document.getElementById('flash');
const statusText = document.getElementById('status-text');
const cameraInput = document.getElementById('camera-input');

// Tarot variables
const questionInput = document.getElementById('question-input');
const sendQuestionBtn = document.getElementById('send-question-btn');
const viewResultBtn = document.getElementById('view-result-btn');
const resultModal = document.getElementById('result-modal');
const closeResult = document.querySelector('.close-result');
const resultText = document.getElementById('result-text');

// Settings toggles
const useSelfModelToggle = document.getElementById('use-self-model-toggle');
const useSplitInferenceToggle = document.getElementById('use-split-inference-toggle');
const useSplitInferenceLabel = document.getElementById('use-split-inference-label');

let capturedPhotos = [];
let isCompatibilityMode = false;

// Initialize Camera
async function initCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        switchToCompatibilityMode("瀏覽器限制相機存取 (非安全連線)");
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            },
            audio: false
        });
        video.srcObject = stream;
        statusText.innerText = '相機已就緒';
    } catch (err) {
        console.error("Error accessing camera: ", err);
        switchToCompatibilityMode("無法開啟即時預覽，切換至相容模式");
    }
}

function switchToCompatibilityMode(msg) {
    isCompatibilityMode = true;
    statusText.innerText = msg;
    statusText.style.color = '#fbbf24'; // Amber
    video.style.display = 'none';
    const placeholder = document.getElementById('compatibility-placeholder');
    if (placeholder) {
        placeholder.style.display = 'flex';
    }
}

// Handle System Camera Input
cameraInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = async (event) => {
            const imageData = event.target.result;
            processCapturedImage(imageData);
        };
        reader.readAsDataURL(file);
    }
});

// Capture Photo
async function capturePhoto() {
    if (isCompatibilityMode) {
        cameraInput.click();
        return;
    }

    // Flash effect
    flash.classList.remove('flash-animation');
    void flash.offsetWidth; // trigger reflow
    flash.classList.add('flash-animation');

    // Draw to canvas
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to image
    const imageData = canvas.toDataURL('image/png');
    processCapturedImage(imageData);
}

async function processCapturedImage(imageData) {
    // Save to local list
    capturedPhotos.push(imageData);
    
    // Upload to Server
    await uploadToServer(imageData);
}

async function uploadToServer(dataUrl) {
    statusText.innerText = '正在同步至 VS Code...';
    statusText.style.color = '#fbbf24';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataUrl }),
        });

        const result = await response.json();
        if (result.status === 'success') {
            statusText.innerText = '同步成功！照片已儲存';
            statusText.style.color = '#10b981';
        } else {
            throw new Error('Upload failed');
        }
    } catch (err) {
        console.error("Upload error: ", err);
        statusText.innerText = '連線失敗 (請檢查伺服器)';
        statusText.style.color = '#ef4444';
    }
}

// Event Listeners
captureBtn.addEventListener('click', capturePhoto);

// Tarot Event Listeners
sendQuestionBtn.addEventListener('click', async () => {
    const question = questionInput.value.trim();
    if (!question) {
        alert("請先輸入問題！");
        return;
    }
    statusText.innerText = '正在傳送問題...';
    statusText.style.color = '#fbbf24';
    try {
        const response = await fetch('/question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        const result = await response.json();
        if (result.status === 'success') {
            statusText.innerText = '問題已送出！';
            statusText.style.color = '#10b981';
        } else {
            throw new Error('Failed to send question');
        }
    } catch (err) {
        console.error(err);
        statusText.innerText = '傳送問題失敗';
        statusText.style.color = '#ef4444';
    }
});

viewResultBtn.addEventListener('click', async () => {
    resultModal.style.display = 'block';
    resultText.innerText = '正在獲取解答，請稍候...';
    try {
        const response = await fetch('/result?t=' + Date.now());
        const data = await response.json();
        if (data.status === 'success') {
            resultText.innerText = data.data;
        } else {
            resultText.innerText = '解答尚未產生，請稍後再試。';
        }
    } catch (err) {
        console.error(err);
        resultText.innerText = '無法連線到伺服器獲取解答。';
    }
});

closeResult.addEventListener('click', () => {
    resultModal.style.display = 'none';
});

window.onclick = (event) => {
    if (event.target == resultModal) {
        resultModal.style.display = 'none';
    }
};

// Settings Sync
async function initSettings() {
    try {
        const response = await fetch('/settings?t=' + Date.now());
        const data = await response.json();
        
        useSelfModelToggle.checked = !!data.use_self_model;
        useSplitInferenceToggle.checked = !!data.use_split_inference;
        
        updateToggleStates();
    } catch (err) {
        console.error("Failed to load settings:", err);
    }
}

function updateToggleStates() {
    const isSelfModelEnabled = useSelfModelToggle.checked;
    
    if (!isSelfModelEnabled) {
        useSplitInferenceToggle.disabled = true;
        useSplitInferenceToggle.checked = false;
        useSplitInferenceLabel.classList.add('disabled');
    } else {
        useSplitInferenceToggle.disabled = false;
        useSplitInferenceLabel.classList.remove('disabled');
    }
}

async function saveSettings() {
    updateToggleStates();
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                use_self_model: useSelfModelToggle.checked,
                use_split_inference: useSplitInferenceToggle.checked
            })
        });
    } catch (err) {
        console.error("Failed to save settings:", err);
    }
}

useSelfModelToggle.addEventListener('change', saveSettings);
useSplitInferenceToggle.addEventListener('change', saveSettings);

// Start the app
initCamera();
initSettings();
