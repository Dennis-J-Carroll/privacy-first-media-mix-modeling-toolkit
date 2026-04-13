#!/bin/bash
# Quick launcher for Privacy-First MMM Mobile App
# Usage: ./run_mobile_app.sh

echo "🔒 Privacy-First MMM - Mobile Web App Launcher"
echo "=============================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
    echo "✅ Streamlit installed!"
fi

# Get local IP for mobile testing
echo "📱 Mobile Testing Instructions:"
echo "--------------------------------"
echo ""
echo "To test on your phone (same Wi-Fi network):"
echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    LOCAL_IP=$(hostname -I | awk '{print $1}')
else
    # Windows or other
    LOCAL_IP="YOUR-LOCAL-IP"
fi

echo "  1. On your phone's browser, go to:"
echo "     http://$LOCAL_IP:8501"
echo ""
echo "  2. Test the sliders and interactive features"
echo ""
echo "For cloud deployment instructions, see MOBILE_DEPLOYMENT_GUIDE.md"
echo ""
echo "=============================================="
echo "Starting Streamlit app..."
echo ""

# Run Streamlit
streamlit run mmm_mobile_app.py
