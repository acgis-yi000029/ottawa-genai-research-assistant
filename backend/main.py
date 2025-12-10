"""
🚀 Ottawa GenAI Research Assistant - Application Entry Point

This is the main entry point for running the FastAPI application with uvicorn.
The actual FastAPI app is defined in app/main.py
"""

import os
import sys
from pathlib import Path

import uvicorn


def setup_python_path():
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent 


    if current_dir.name == "backend":
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        os.chdir(current_dir)
        app_module = "app.main:app"
        
        print(f" Startup directory: backend/")
        print(f" Working directory: {current_dir}")
        
    else:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
        os.chdir(project_root)
        app_module = "backend.app.main:app"
        
        print(f" Startup directory: Project root directory")
        print(f" Working directory: {project_root}")
    
    return app_module

if __name__ == "__main__":
    app_module = setup_python_path()
    
    try:
        if "backend" in str(Path.cwd()):
            from app.core.config import get_settings
        else:
            from backend.app.core.config import get_settings
        
        settings = get_settings()
        
        print(f" Start the server...")
        print(f" Module path: {app_module}")
        print(f" Debug mode: {settings.DEBUG}")
        
        uvicorn.run(
            app_module,
            host="0.0.0.0",
            port=8000,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower(),
        )
        
    except ImportError as e:
        print(f" Configuration import failed: {e}")
        print(" Start with default configuration...")
        
        uvicorn.run(
            app_module,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
