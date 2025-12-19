const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const predictionText = document.getElementById('prediction');

let isRunning = false;
let intervalId = null;

// 1. Start Webcam
async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = stream;
        isRunning = true;
        
        // Wait for video to load data before setting canvas size
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // Start the loop
            intervalId = setInterval(processFrame, 200); // 5 FPS (Adjust for speed vs performance)
        };
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please allow permissions.");
    }
}

// 2. Stop Webcam
function stopVideo() {
    isRunning = false;
    clearInterval(intervalId);
    
    const stream = video.srcObject;
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
    }
    video.srcObject = null;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictionText.innerText = "--";
}

// 3. Process Frame & Send to Backend
async function processFrame() {
    if (!isRunning) return;

    // Draw current video frame to a temporary canvas to send it
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    // Convert to Blob (File object)
    tempCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            // Clear previous drawings
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (data.prediction && data.prediction !== "No Hand Detected") {
                // Update Text
                predictionText.innerText = data.prediction;
                
                // Draw Fancy Box (Since we don't have bbox from backend yet, we just show text)
                // If you want the box, we need to update backend to return bbox coordinates too.
                // For now, let's just show a global indicator.
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 4;
                ctx.strokeRect(50, 50, canvas.width - 100, canvas.height - 100);
            } else {
                predictionText.innerText = "...";
            }

        } catch (err) {
            console.error("Backend Error:", err);
        }
    }, 'image/jpeg');
}