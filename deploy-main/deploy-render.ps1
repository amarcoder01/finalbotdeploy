# Render Deployment Guide for TradeAI Companion (PowerShell)
# Since Render doesn't have an official CLI for deployments,
# this script helps with the deployment process

Write-Host "=== Render Deployment Setup ===" -ForegroundColor Green

# Ensure all files are committed
Write-Host "Checking git status..." -ForegroundColor Cyan
git status

Write-Host "`nMake sure all changes are committed and pushed to GitHub." -ForegroundColor Yellow
Write-Host "`nTo deploy on Render:" -ForegroundColor White
Write-Host "1. Go to https://dashboard.render.com" -ForegroundColor White
Write-Host "2. Click 'New +' -> 'Web Service'" -ForegroundColor White
Write-Host "3. Connect your GitHub repository: amarcoder01/deploy" -ForegroundColor White
Write-Host "4. Use these settings:" -ForegroundColor White
Write-Host "   - Name: tradeai-companion" -ForegroundColor Gray
Write-Host "   - Runtime: Python 3" -ForegroundColor Gray
Write-Host "   - Build Command: pip install -r requirements.txt" -ForegroundColor Gray
Write-Host "   - Start Command: python -m aiohttp.web -H 0.0.0.0 -P `$PORT main:app" -ForegroundColor Gray
Write-Host "`nOr use the render.yaml file for automatic configuration." -ForegroundColor Cyan
Write-Host "`nEnvironment variables to set:" -ForegroundColor Yellow
Write-Host "- TELEGRAM_API_TOKEN" -ForegroundColor Gray
Write-Host "- OPENAI_API_KEY" -ForegroundColor Gray
Write-Host "- ALPACA_API_KEY" -ForegroundColor Gray
Write-Host "- ALPACA_API_SECRET" -ForegroundColor Gray
Write-Host "- CHART_IMG_API_KEY" -ForegroundColor Gray

Write-Host "`n=== Auto-deployment is enabled via render.yaml ===" -ForegroundColor Green
Write-Host "Any push to main branch will trigger automatic deployment." -ForegroundColor Green

# Quick deployment check
Write-Host "`nWould you like to push current changes to trigger deployment? (y/n): " -ForegroundColor Cyan -NoNewline
$response = Read-Host
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "Pushing to GitHub..." -ForegroundColor Green
    git add -A
    git commit -m "Deploy via render.yaml configuration"
    git push origin main
    Write-Host "Deployment triggered! Check Render dashboard for status." -ForegroundColor Green
}