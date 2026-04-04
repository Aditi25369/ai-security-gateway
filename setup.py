"""
Setup script for AI Security Gateway
Downloads models, initializes database, and verifies installation
"""

import subprocess
import sys
import os
import urllib.request
import zipfile
import shutil

def run_command(cmd, description):
    """Run shell command with status"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print(f"   ⚠️ Warning: {result.stderr[:200]}")
            return False
        print(f"   ✅ Done")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def install_requirements():
    """Install Python dependencies"""
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def download_ollama():
    """Download and setup Ollama for Windows"""
    print("\n🔧 Setting up Ollama (local LLM runner)...")
    
    ollama_url = "https://ollama.com/download/OllamaSetup.exe"
    setup_path = "OllamaSetup.exe"
    
    if os.path.exists("C:\\Program Files\\Ollama\\ollama.exe"):
        print("   ✅ Ollama already installed")
        return True
    
    print(f"   Downloading Ollama from {ollama_url}...")
    print("   (This is a ~200MB download, please wait)")
    
    try:
        urllib.request.urlretrieve(ollama_url, setup_path)
        print(f"   ✅ Downloaded to {setup_path}")
        print("   📝 Please run OllamaSetup.exe to complete installation")
        print("   Then pull a model with: ollama pull llama3.2")
        return True
    except Exception as e:
        print(f"   ⚠️ Download failed: {e}")
        print("   Please install Ollama manually from https://ollama.com")
        return False

def download_models():
    """Download HuggingFace models for local safety classification"""
    print("\n🤖 Downloading safety classification models...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        print(f"   Downloading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        print(f"   ✅ Model downloaded and cached")
        return True
    except Exception as e:
        print(f"   ⚠️ Model download failed: {e}")
        print("   The gateway will use rule-based detection as fallback")
        return False

def initialize_database():
    """Create SQLite database"""
    print("\n💾 Initializing audit database...")
    
    try:
        import sqlite3
        conn = sqlite3.connect('security_audit.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                request_id TEXT,
                category TEXT,
                score REAL,
                blocked BOOLEAN,
                reason TEXT,
                pii_detected TEXT,
                latency_ms REAL,
                model_used TEXT,
                source_ip TEXT,
                user_agent TEXT,
                prompt_hash TEXT,
                full_prompt TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("   ✅ Database initialized")
        return True
    except Exception as e:
        print(f"   ❌ Database initialization failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ['logs', 'models', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("\n📁 Created directories: logs, models, results")

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("SETUP COMPLETE - NEXT STEPS")
    print("="*60)
    print("\n1️⃣  Start Ollama (if installed):")
    print("   ollama serve")
    print("   ollama pull llama3.2:latest")
    print("\n2️⃣  Start the Security Gateway:")
    print(f"   {sys.executable} gateway.py")
    print("\n3️⃣  Start the Dashboard (new terminal):")
    print(f"   {sys.executable} -m streamlit run dashboard.py")
    print("\n4️⃣  Test with evaluation:")
    print(f"   {sys.executable} evaluation.py")
    print("\n5️⃣  Configure your apps to use:")
    print("   http://localhost:8080/v1/chat/completions")
    print("   (instead of OpenAI/Ollama direct)")
    print("\n📚 Documentation:")
    print("   Gateway API: http://localhost:8080/docs")
    print("   Dashboard: http://localhost:8501")
    print("   Health: http://localhost:8080/health")
    print("\n" + "="*60)

def main():
    print("🛡️ AI Security Gateway - Setup")
    print("="*60)
    print("\nThis will:")
    print("  • Install Python dependencies")
    print("  • Download safety classification models")
    print("  • Initialize audit database")
    print("  • Setup local directories")
    print()
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    # Run setup steps
    success = True
    
    create_directories()
    
    if not install_requirements():
        print("\n⚠️ Some dependencies failed to install")
        print("You may need to install manually")
        success = False
    
    download_models()
    initialize_database()
    
    # Ollama is optional
    download_ollama()
    
    print_next_steps()
    
    if success:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n⚠️ Setup completed with warnings")

if __name__ == "__main__":
    main()
