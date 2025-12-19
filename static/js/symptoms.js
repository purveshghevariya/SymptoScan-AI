// Symptom Checker JavaScript

let selectedSymptoms = new Set();
let predictionData = null;
let userData = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initSymptomSelection();
    initSearch();
    initPredictButton();
    initModal();
});

// Initialize symptom selection
function initSymptomSelection() {
    const checkboxes = document.querySelectorAll('.symptom-check');
    
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedSymptoms.add(this.value);
            } else {
                selectedSymptoms.delete(this.value);
            }
            updateSelectedDisplay();
        });
    });
}

// Update selected display
function updateSelectedDisplay() {
    const count = document.getElementById('count');
    const selectedList = document.getElementById('selectedList');
    const selectedTags = document.getElementById('selectedTags');
    
    count.textContent = selectedSymptoms.size;
    
    if (selectedSymptoms.size > 0) {
        selectedList.style.display = 'block';
        selectedTags.innerHTML = Array.from(selectedSymptoms).map(symptom => `
            <span class="tag">
                ${symptom.replace(/_/g, ' ')}
                <span class="remove" onclick="removeSymptom('${symptom}')">×</span>
            </span>
        `).join('');
    } else {
        selectedList.style.display = 'none';
    }
}

// Remove symptom
function removeSymptom(symptom) {
    const checkbox = document.querySelector(`input[value="${symptom}"]`);
    if (checkbox) {
        checkbox.checked = false;
        selectedSymptoms.delete(symptom);
        updateSelectedDisplay();
    }
}

// Initialize search
function initSearch() {
    const searchInput = document.getElementById('searchInput');
    const symptomCards = document.querySelectorAll('.symptom-card');
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        
        symptomCards.forEach(card => {
            const text = card.textContent.toLowerCase();
            card.style.display = text.includes(searchTerm) ? 'block' : 'none';
        });
    });
}

// Initialize predict button
function initPredictButton() {
    const predictBtn = document.getElementById('predictBtn');
    
    predictBtn.addEventListener('click', function() {
        if (selectedSymptoms.size === 0) {
            alert('Please select at least one symptom!');
            return;
        }
        showModal();
    });
}

// Initialize modal
function initModal() {
    const modal = document.getElementById('userModal');
    const closeBtn = document.querySelector('.close');
    const form = document.getElementById('userForm');
    
    closeBtn.addEventListener('click', hideModal);
    
    window.addEventListener('click', function(e) {
        if (e.target === modal) {
            hideModal();
        }
    });
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        userData = {
            name: document.getElementById('userName').value,
            age: document.getElementById('userAge').value,
            gender: document.getElementById('userGender').value,
            contact: document.getElementById('userContact').value
        };
        
        hideModal();
        predictDisease();
    });
}

// Show/Hide modal
function showModal() {
    document.getElementById('userModal').style.display = 'block';
}

function hideModal() {
    document.getElementById('userModal').style.display = 'none';
}

// Show/Hide loading
function showLoading(show) {
    const loading = document.getElementById('loading');
    loading.style.display = show ? 'flex' : 'none';
}

// Predict disease
async function predictDisease() {
    showLoading(true);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symptoms: Array.from(selectedSymptoms)
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            predictionData = data;
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Display results
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    const html = `
        <div class="result-card">
            <div class="result-header">
                <h2>Diagnosis Results</h2>
                <h3>${data.disease}</h3>
                <div class="confidence">
                    <strong>Confidence:</strong> ${data.confidence}%
                </div>
            </div>
            
            <div class="result-section">
                <h3>Description</h3>
                <p>${data.description}</p>
            </div>
            
            <div class="result-section">
                <h3>Precautions</h3>
                <ul>
                    ${data.precautions.map(p => `<li>• ${p}</li>`).join('')}
                </ul>
            </div>
            
            <div class="result-section">
                <h3>Medications</h3>
                <p><strong>Suggested:</strong> ${data.medications.medications.join(', ')}</p>
                <p><strong>Dosage:</strong> ${data.medications.dosage}</p>
                <p><strong>Duration:</strong> ${data.medications.duration}</p>
            </div>
            
            <div class="result-section">
                <h3>Exercise Plan</h3>
                <p><strong>Intensity:</strong> ${data.workout_plan.intensity}</p>
                <ul>
                    ${data.workout_plan.exercises.map(ex => `
                        <li>${ex.name} - ${ex.duration} (${ex.frequency})</li>
                    `).join('')}
                </ul>
            </div>
            
            <div class="result-section">
                <h3>Diet Recommendations</h3>
                <p><strong>Foods to Eat:</strong> ${data.diet_plan.foods_to_eat.join(', ')}</p>
                <p><strong>Foods to Avoid:</strong> ${data.diet_plan.foods_to_avoid.join(', ')}</p>
            </div>
            
            <div style="text-align: center; margin-top: 2rem;">
                <button class="btn-download" onclick="downloadReport()">
                    <i class="fas fa-download"></i> Download PDF Report
                </button>
                <button class="btn-primary" onclick="location.reload()">
                    <i class="fas fa-redo"></i> New Prediction
                </button>
            </div>
            
            <div style="margin-top: 2rem; padding: 1rem; background: #fee2e2; border-radius: 5px; text-align: center;">
                <p style="color: #991b1b; margin: 0;">
                    ⚠️ This is an AI prediction. Always consult healthcare professionals.
                </p>
            </div>
        </div>
    `;
    
    resultsDiv.innerHTML = html;
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Download report
async function downloadReport() {
    if (!predictionData || !userData) {
        alert('Missing data for report');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/generate-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prediction_data: predictionData,
                user_data: userData
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Report_${userData.name}_${Date.now()}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            alert('Report downloaded successfully!');
        } else {
            alert('Error generating report');
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        showLoading(false);
    }
}