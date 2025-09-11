document.addEventListener('DOMContentLoaded', function() {
    const uploadSection = document.getElementById('uploadSection');
    const fileInput = document.getElementById('fileInput');
    const previewSection = document.getElementById('previewSection');
    const uploadForm = document.getElementById('uploadForm');
    
    // Drag and drop functionality
    uploadSection.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadSection.classList.add('dragover');
    });
    
    uploadSection.addEventListener('dragleave', () => {
        uploadSection.classList.remove('dragover');
    });
    
    uploadSection.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadSection.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            const imagePreview = document.getElementById('imagePreview');
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview" class="preview-image">`;
            previewSection.style.display = 'block';
            
            // Show loading state
            const overallStatus = document.getElementById('overallStatus');
            overallStatus.className = 'overall-status';
            overallStatus.innerHTML = '<div class="loading-spinner"></div> Analyzing image...';
            
            // Clear previous results
            document.getElementById('spotifyResults').innerHTML = '';
            document.getElementById('appleResults').innerHTML = '';
            document.getElementById('specGrid').innerHTML = '';
            
            // Upload and analyze the image
            uploadAndAnalyze(file);
        };
        reader.readAsDataURL(file);
    }
    
    function uploadAndAnalyze(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            const overallStatus = document.getElementById('overallStatus');
            overallStatus.className = 'overall-status status-fail';
            overallStatus.innerHTML = `❌ Error: ${error.message}`;
        });
    }
    
    function displayResults(results) {
        // Display overall status
        const overallStatus = document.getElementById('overallStatus');
        if (results.overall_status === 'pass') {
            overallStatus.className = 'overall-status status-pass';
            overallStatus.innerHTML = '✅ Artwork meets all requirements for both platforms!';
        } else if (results.overall_status === 'warning') {
            overallStatus.className = 'overall-status status-warning';
            overallStatus.innerHTML = '⚠️ Artwork has some issues that need attention';
        } else {
            overallStatus.className = 'overall-status status-fail';
            overallStatus.innerHTML = '❌ Artwork requires significant changes';
        }
        
        // Display platform results
        displayPlatformResults('spotifyResults', results.spotify_results);
        displayPlatformResults('appleResults', results.apple_results);
        
        // Display specifications
        displaySpecifications(results.specifications);
    }
    
    function displayPlatformResults(containerId, results) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        results.forEach(result => {
            const item = document.createElement('div');
            item.className = 'check-item';
            
            const icon = result.status === 'pass' ? '✓' : '✗';
            
            item.innerHTML = `
                <div class="check-icon ${result.status}">${icon}</div>
                <div>
                    <div style="font-weight: 600;">${result.test}</div>
                    <div style="font-size: 0.9em; color: #666;">${result.message}</div>
                </div>
            `;
            
            container.appendChild(item);
        });
    }
    
    function displaySpecifications(specs) {
        const specGrid = document.getElementById('specGrid');
        const specItems = [
            { label: 'Dimensions', value: specs.dimensions },
            { label: 'File Size', value: specs.file_size },
            { label: 'Format', value: specs.format },
            { label: 'Aspect Ratio', value: specs.aspect_ratio }
        ];
        
        specGrid.innerHTML = specItems.map(spec => `
            <div class="spec-item">
                <div class="spec-label">${spec.label}</div>
                <div class="spec-value">${spec.value}</div>
            </div>
        `).join('');
    }
});