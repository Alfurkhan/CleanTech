
let selectedFile = null;

document.getElementById('imageInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('predictBtn').disabled = false;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('previewImg').src = e.target.result;
            document.getElementById('imagePreview').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

function predictWaste() {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    document.getElementById('predictBtn').disabled = true;
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Display results
        const wasteType = document.getElementById('wasteType');
        wasteType.textContent = data.predicted_class;
        wasteType.className = 'waste-type ' + data.predicted_class;
        
        document.getElementById('confidence').textContent = 
            `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        
        // Show breakdown
        const breakdown = document.getElementById('breakdown');
        breakdown.innerHTML = '<h4>Prediction Breakdown:</h4>';
        
        Object.entries(data.all_predictions).forEach(([category, confidence]) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <span>${category}</span>
                <div class="prediction-bar">
                    <div class="prediction-fill" style="width: ${confidence * 100}%"></div>
                </div>
                <span>${(confidence * 100).toFixed(1)}%</span>
            `;
            breakdown.appendChild(item);
        });
        
        document.getElementById('results').style.display = 'block';
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
        alert('Error: ' + error);
    });
}

function trainModel() {
    if (confirm('This will retrain the model with synthetic data. Continue?')) {
        const button = event.target;
        button.textContent = '🧠 Training...';
        button.disabled = true;
        
        fetch('/train', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
            } else {
                alert('Model training completed successfully!');
            }
            button.textContent = '🧠 Retrain Model';
            button.disabled = false;
        })
        .catch(error => {
            alert('Error: ' + error);
            button.textContent = '🧠 Retrain Model';
            button.disabled = false;
        });
    }
}
