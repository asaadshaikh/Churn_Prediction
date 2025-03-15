# test_api.ps1
# Script to test the Churn Prediction API

Write-Host "Testing Churn Prediction API..." -ForegroundColor Cyan

# Test the health endpoint
Write-Host "`nTesting health endpoint..." -ForegroundColor Yellow
$healthResponse = Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET
Write-Host "Health Status:" -ForegroundColor Green
$healthResponse | ConvertTo-Json

# Get sample data
Write-Host "`nGetting sample data..." -ForegroundColor Yellow
$sampleData = Invoke-RestMethod -Uri "http://localhost:5000/sample" -Method GET
Write-Host "Sample Data:" -ForegroundColor Green
$sampleData | ConvertTo-Json -Depth 5

# Make a prediction
Write-Host "`nMaking prediction with sample data..." -ForegroundColor Yellow
$predictionResponse = Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body ($sampleData | ConvertTo-Json) -ContentType "application/json"
Write-Host "Prediction Result:" -ForegroundColor Green
$predictionResponse | ConvertTo-Json

Write-Host "`nAPI testing completed." -ForegroundColor Cyan 