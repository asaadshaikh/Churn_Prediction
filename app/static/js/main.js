// Initialize charts when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupPredictionForm();
    setupTooltips();
});

// Initialize Bootstrap tooltips
function setupTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Setup prediction form submission
function setupPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show loading state
            const submitButton = form.querySelector('button[type="submit"]');
            const originalText = submitButton.innerHTML;
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
            submitButton.disabled = true;

            try {
                const formData = new FormData(form);
                const data = Object.fromEntries(formData.entries());
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                
                if (response.ok) {
                    displayPredictionResult(result);
                } else {
                    throw new Error(result.error || 'Prediction failed');
                }
            } catch (error) {
                showError('Error making prediction: ' + error.message);
            } finally {
                // Restore button state
                submitButton.innerHTML = originalText;
                submitButton.disabled = false;
            }
        });
    }
}

// Display prediction results
function displayPredictionResult(result) {
    const resultDiv = document.getElementById('prediction-result');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="card animate-fade-in">
                <div class="card-body text-center">
                    <h3 class="prediction-score">${(result.churn_probability * 100).toFixed(1)}%</h3>
                    <p class="prediction-label">Churn Probability</p>
                    <div class="alert ${result.churn_probability > 0.5 ? 'alert-danger' : 'alert-success'} mt-3">
                        <strong>${result.prediction}</strong>
                    </div>
                    ${renderRecommendations(result.recommendations)}
                </div>
            </div>
        `;
    }
}

// Render recommendations
function renderRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) return '';
    
    return `
        <div class="mt-4">
            <h5>Recommendations</h5>
            <ul class="list-group">
                ${recommendations.map(rec => `
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        ${rec.action}
                        <span class="badge bg-${rec.impact === 'high' ? 'danger' : 'warning'} rounded-pill">
                            ${rec.impact}
                        </span>
                    </li>
                `).join('')}
            </ul>
        </div>
    `;
}

// Initialize dashboard charts
function initializeCharts() {
    initializeChurnRateChart();
    initializeMonthlyTrendChart();
}

// Initialize churn rate by contract type chart
function initializeChurnRateChart() {
    const ctx = document.getElementById('churnRateChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Month-to-month', '1 year', '2 year'],
                datasets: [{
                    label: 'Churn Rate (%)',
                    data: [42, 15, 8],
                    backgroundColor: [
                        'rgba(231, 76, 60, 0.8)',
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(46, 204, 113, 0.8)'
                    ],
                    borderColor: [
                        'rgba(231, 76, 60, 1)',
                        'rgba(52, 152, 219, 1)',
                        'rgba(46, 204, 113, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Churn Rate by Contract Type'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }
}

// Initialize monthly trend chart
function initializeMonthlyTrendChart() {
    const ctx = document.getElementById('monthlyTrendChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Churn Rate',
                    data: [15, 18, 16, 14, 13, 12],
                    borderColor: 'rgba(52, 152, 219, 1)',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Monthly Churn Trend'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 30,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
    errorDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const form = document.getElementById('prediction-form');
    form.insertAdjacentElement('afterend', errorDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Update dashboard metrics
function updateDashboardMetrics(metrics) {
    Object.entries(metrics).forEach(([key, value]) => {
        const element = document.getElementById(`${key}-metric`);
        if (element) {
            element.textContent = value;
        }
    });
}

// Fetch and update dashboard data
async function refreshDashboardData() {
    try {
        const response = await fetch('/api/dashboard/metrics');
        const data = await response.json();
        
        if (response.ok) {
            updateDashboardMetrics(data.metrics);
            // You can also update charts here if needed
        } else {
            console.error('Failed to fetch dashboard data:', data.error);
        }
    } catch (error) {
        console.error('Error fetching dashboard data:', error);
    }
} 