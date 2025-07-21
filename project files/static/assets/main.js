// CleanTech Waste Management - Main JavaScript Functions

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupFileUpload();
    setupDragAndDrop();
    setupImagePreview();
    setupFormValidation();
    setupAnimations();
    console.log('CleanTech Waste Management App Initialized');
}

// File Upload Functionality
function setupFileUpload() {
    const fileInput = document.getElementById('file');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadText = document.getElementById('uploadText');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                updateUploadText(file.name);
                previewImage(file);
                if (uploadBtn) uploadBtn.disabled = false;
            }
        });
    }
}

// Drag and Drop Functionality
function setupDragAndDrop() {
    const uploadArea = document.querySelector('.upload-area');
    
    if (!uploadArea) return;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    uploadArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    const uploadArea = document.querySelector('.upload-area');
    uploadArea.classList.add('dragover');
}

function unhighlight(e) {
    const uploadArea = document.querySelector('.upload-area');
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const file = files[0];
        if (isValidFileType(file)) {
            const fileInput = document.getElementById('file');
            if (fileInput) {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
                
                updateUploadText(file.name);
                previewImage(file);
                
                const uploadBtn = document.getElementById('uploadBtn');
                if (uploadBtn) uploadBtn.disabled = false;
            }
        } else {
            showAlert('Please upload a valid image file (PNG, JPG, JPEG, GIF)', 'danger');
        }
    }
}

// Image Preview
function setupImagePreview() {
    // Image preview is handled by the previewImage function
}

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewContainer = document.getElementById('imagePreview');
        if (previewContainer) {
            previewContainer.innerHTML = `
                <img src="${e.target.result}" alt="Preview" class="image-preview img-fluid">
            `;
            previewContainer.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
}

// Form Validation
function setupFormValidation() {
    const form = document.getElementById('uploadForm');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file');
            
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                showAlert('Please select an image file to classify', 'warning');
                return false;
            }
            
            const file = fileInput.files[0];
            if (!isValidFileType(file)) {
                e.preventDefault();
                showAlert('Please upload a valid image file (PNG, JPG, JPEG, GIF)', 'danger');
                return false;
            }
            
            if (file.size > 16 * 1024 * 1024) { // 16MB limit
                e.preventDefault();
                showAlert('File size must be less than 16MB', 'danger');
                return false;
            }
            
            showLoadingSpinner();
        });
    }
}

// Utility Functions
function updateUploadText(filename) {
    const uploadText = document.getElementById('uploadText');
    if (uploadText) {
        uploadText.innerHTML = `
            <i class="fas fa-check-circle text-success"></i>
            <strong>File Selected:</strong> ${filename}
        `;
    }
}

function isValidFileType(file) {
    const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/gif'];
    return allowedTypes.includes(file.type);
}

function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alertContainer') || createAlertContainer();
    
    const alertHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            <i class="fas fa-${getIconForAlert(type)}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    alertContainer.innerHTML = alertHTML;
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = alertContainer.querySelector('.alert');
        if (alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
}

function createAlertContainer() {
    const container = document.createElement('div');
    container.id = 'alertContainer';
    container.className = 'position-fixed top-0 end-0 p-3';
    container.style.zIndex = '1055';
    document.body.appendChild(container);
    return container;
}

function getIconForAlert(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function showLoadingSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'block';
    }
    
    // Disable form submission button
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
}

// Animations
function setupAnimations() {
    // Animate cards on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe all cards and sections
    const elementsToAnimate = document.querySelectorAll('.card, .feature-card, .process-step');
    elementsToAnimate.forEach(element => {
        observer.observe(element);
    });
}

// Prediction Results Display
function displayPredictionResults(results) {
    const resultsContainer = document.getElementById('predictionResults');
    if (!resultsContainer) return;
    
    const { predicted_class, confidence, all_probabilities } = results;
    
    let resultsHTML = `
        <div class="prediction-result slide-up">
            <h4 class="mb-3"><i class="fas fa-brain"></i> AI Classification Results</h4>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="alert alert-prediction">
                        <h5 class="alert-heading">
                            <i class="fas fa-tag"></i> Predicted Category
                        </h5>
                        <h3 class="class-${predicted_class.toLowerCase()}">${predicted_class}</h3>
                        <div class="confidence-bar" style="width: ${(confidence * 100).toFixed(1)}%"></div>
                        <small class="text-muted">Confidence: ${(confidence * 100).toFixed(1)}%</small>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <h6>All Probabilities:</h6>
                    <div class="probability-breakdown">
    `;
    
    // Add probability bars for all classes
    Object.entries(all_probabilities).forEach(([className, probability]) => {
        const percentage = (probability * 100).toFixed(1);
        resultsHTML += `
            <div class="mb-2">
                <div class="d-flex justify-content-between">
                    <span>${className}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="progress progress-sm">
                    <div class="progress-bar class-${className.toLowerCase()}" 
                         style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    });
    
    resultsHTML += `
                    </div>
                </div>
            </div>
            
            <div class="mt-3">
                <small class="text-muted">
                    <i class="fas fa-info-circle"></i> 
                    This prediction is made using a VGG16 deep learning model trained on waste classification data.
                </small>
            </div>
        </div>
    `;
    
    resultsContainer.innerHTML = resultsHTML;
}

// API Functions for AJAX requests (if needed)
async function classifyImage(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error('Error classifying image:', error);
        throw error;
    }
}

// Smooth scrolling for navigation links
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const targetId = e.target.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        
        if (targetElement) {
            targetElement.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

// Initialize tooltips and popovers if Bootstrap is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (typeof bootstrap !== 'undefined') {
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
        
        // Initialize popovers
        const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
        popoverTriggerList.map(function(popoverTriggerEl) {
            return new bootstrap.Popover(popoverTriggerEl);
        });
    }
});

// Console welcome message
console.log(`
🌱 CleanTech: Waste Management with AI
🤖 Powered by VGG16 Transfer Learning
♻️  Classify: Biodegradable | Recyclable | Trash
`);
