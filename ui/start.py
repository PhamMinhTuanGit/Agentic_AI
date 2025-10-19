#!/usr/bin/env python3
"""
Quick start script for RAG Pipeline Streamlit UI
Checks dependencies and starts the application
"""

import subprocess
import sys
from pathlib import Path

def check_ollama():
    """Check if Ollama is running"""
    print("🔍 Checking Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except:
        pass
    
    print("❌ Ollama is not running!")
    print("   Start it with: ollama serve")
    return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required = {
        'streamlit': 'streamlit',
        'rag': 'parent package',
        'plotly': 'plotly',
        'pandas': 'pandas'
    }
    
    missing = []
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install -r ui/requirements.txt")
        return False
    
    return True

def main():
    """Main startup"""
    print("🚀 RAG Pipeline Streamlit UI - Quick Start")
    print("=" * 50)
    
    # Check Ollama
    if not check_ollama():
        print("\n⚠️ Please start Ollama first:")
        print("   ollama serve")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️ Please install missing dependencies:")
        print("   pip install -r ui/requirements.txt")
        return 1
    
    print("\n✅ All checks passed!")
    print("\n🎉 Starting Streamlit UI...")
    print("   Open: http://localhost:8501")
    print("\n" + "=" * 50)
    
    # Start Streamlit
    ui_dir = Path(__file__).parent
    app_file = ui_dir / "app.py"
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_file)],
            cwd=ui_dir.parent
        )
    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting Streamlit: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
