const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const flash = document.getElementById('flash');
const lastShotPreview = document.getElementById('last-shot');
const galleryTrigger = document.getElementById('gallery-trigger');
const galleryModal = document.getElementById('gallery-modal');
const closeModal = document.querySelector('.close');
const galleryGrid = document.getElementById('gallery-grid');
const statusText = document.getElementById('status-text');
const cameraInput = document.getElementById('camera-input');

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
    
    // Show a placeholder in the viewfinder
    const viewfinder = document.querySelector('.viewfinder-container');
    const msgEl = document.createElement('div');
    msgEl.innerHTML = '<p style="text-align:center; padding:20px;">點擊下方快門按鈕<br>啟動手機內建相機</p>';
    viewfinder.appendChild(msgEl);
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
    
    // Update preview
    lastShotPreview.src = imageData;
    lastShotPreview.style.display = 'block';
    const placeholder = document.querySelector('.placeholder-icon');
    if (placeholder) placeholder.style.display = 'none';

    // Add to gallery
    addToGallery(imageData);

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

function addToGallery(dataUrl) {
    const item = document.createElement('div');
    item.className = 'gallery-item';
    const img = document.createElement('img');
    img.src = dataUrl;
    item.appendChild(img);
    galleryGrid.prepend(item);
}

// Event Listeners
captureBtn.addEventListener('click', capturePhoto);

galleryTrigger.addEventListener('click', () => {
    galleryModal.style.display = 'block';
});

closeModal.addEventListener('click', () => {
    galleryModal.style.display = 'none';
});

window.onclick = (event) => {
    if (event.target == galleryModal) {
        galleryModal.style.display = 'none';
    }
};

// Start the app
initCamera();
