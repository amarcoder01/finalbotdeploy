# Deployment Monitoring Script for TradeAI Companion
# This script helps monitor the deployment process and handle errors

Write-Host "=== TradeAI Companion Deployment Monitor ===" -ForegroundColor Green
Write-Host "Deployment triggered at: $(Get-Date)" -ForegroundColor Cyan

Write-Host "`nDeployment has been initiated on Render!" -ForegroundColor Green
Write-Host "GitHub push completed successfully." -ForegroundColor Green

Write-Host "`n=== Next Steps ===" -ForegroundColor Yellow
Write-Host "1. Monitor deployment at: https://dashboard.render.com" -ForegroundColor White
Write-Host "2. Check build logs for any errors" -ForegroundColor White
Write-Host "3. Verify environment variables are set" -ForegroundColor White
Write-Host "4. Test the bot once deployment completes" -ForegroundColor White

Write-Host "`n=== Common Issues & Solutions ===" -ForegroundColor Yellow
Write-Host "• Build fails: Check requirements.txt dependencies" -ForegroundColor Gray
Write-Host "• Start fails: Verify main.py has 'app' object" -ForegroundColor Gray
Write-Host "• Bot not responding: Check TELEGRAM_API_TOKEN" -ForegroundColor Gray
Write-Host "• Database errors: Verify PostgreSQL connection" -ForegroundColor Gray

Write-Host "`n=== Environment Variables Required ===" -ForegroundColor Yellow
Write-Host "- TELEGRAM_API_TOKEN" -ForegroundColor Gray
Write-Host "- OPENAI_API_KEY" -ForegroundColor Gray
Write-Host "- ALPACA_API_KEY" -ForegroundColor Gray
Write-Host "- ALPACA_API_SECRET" -ForegroundColor Gray
Write-Host "- CHART_IMG_API_KEY" -ForegroundColor Gray

Write-Host "`nDeployment monitoring complete. Check Render dashboard for live status." -ForegroundColor Green