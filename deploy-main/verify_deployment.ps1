# Deployment Verification Script for TradeAI Companion

Write-Host "=== TradeAI Companion Deployment Verification ===" -ForegroundColor Green

# Check main.py for app object
Write-Host ""
Write-Host "Checking main.py configuration..." -ForegroundColor Cyan
if (Test-Path "main.py") {
    $mainContent = Get-Content "main.py" -Raw
    if ($mainContent -match "app") {
        Write-Host "✓ main.py found and contains app reference" -ForegroundColor Green
    } else {
        Write-Host "✗ main.py missing app object" -ForegroundColor Red
    }
} else {
    Write-Host "✗ main.py not found!" -ForegroundColor Red
}

# Check requirements.txt
Write-Host ""
Write-Host "Checking requirements.txt..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    Write-Host "✓ requirements.txt found" -ForegroundColor Green
    $reqContent = Get-Content "requirements.txt"
    Write-Host "Dependencies count: $($reqContent.Count)" -ForegroundColor Gray
} else {
    Write-Host "✗ requirements.txt not found!" -ForegroundColor Red
}

# Check render.yaml configuration
Write-Host ""
Write-Host "Checking render.yaml configuration..." -ForegroundColor Cyan
if (Test-Path "render.yaml") {
    Write-Host "✓ render.yaml found" -ForegroundColor Green
    $renderContent = Get-Content "render.yaml" -Raw
    
    if ($renderContent -match "main:app") {
        Write-Host "✓ Start command correctly references main:app" -ForegroundColor Green
    } else {
        Write-Host "✗ Start command may be incorrect" -ForegroundColor Red
    }
    
    if ($renderContent -match "python3") {
        Write-Host "✓ Runtime set to python3" -ForegroundColor Green
    } else {
        Write-Host "✗ Runtime configuration may be incorrect" -ForegroundColor Red
    }
} else {
    Write-Host "✗ render.yaml not found!" -ForegroundColor Red
}

# Check git status
Write-Host ""
Write-Host "Checking git status..." -ForegroundColor Cyan
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "⚠ Uncommitted changes detected" -ForegroundColor Yellow
} else {
    Write-Host "✓ All changes committed" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Deployment Status ===" -ForegroundColor Yellow
Write-Host "Latest commit: $(git log -1 --oneline)" -ForegroundColor Gray
Write-Host "Branch: $(git branch --show-current)" -ForegroundColor Gray

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Yellow
Write-Host "1. Visit Render Dashboard: https://dashboard.render.com" -ForegroundColor White
Write-Host "2. Check build logs for errors" -ForegroundColor White
Write-Host "3. Verify environment variables are set" -ForegroundColor White
Write-Host "4. Test the bot once deployment completes" -ForegroundColor White

Write-Host ""
Write-Host "Verification complete!" -ForegroundColor Green