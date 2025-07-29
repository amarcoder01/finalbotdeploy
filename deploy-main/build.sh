#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # Exit on error

echo "🚀 Starting build process..."

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install system dependencies if needed
echo "🔧 Installing system dependencies..."
# Note: Render provides most system packages, but we can install additional ones if needed

# Install Python dependencies with specific flags for Render
echo "📚 Installing Python dependencies..."
pip install --no-cache-dir --upgrade -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p temp
mkdir -p .config/matplotlib
mkdir -p qlib_data

# Set matplotlib backend for headless environment
echo "🎨 Configuring matplotlib..."
echo "backend: Agg" > .config/matplotlib/matplotlibrc

# Set environment variables for production
echo "⚙️ Setting production environment..."
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

# Set proper permissions
echo "🔐 Setting permissions..."
chmod +x main.py
chmod +x *.py

# Verify critical files exist
echo "✅ Verifying deployment files..."
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

echo "✅ Build completed successfully!"
echo "📋 Build summary:"
echo "   - Python dependencies installed"
echo "   - Directories created"
echo "   - Matplotlib configured for headless mode"
echo "   - Permissions set"
echo "   - Ready for deployment"