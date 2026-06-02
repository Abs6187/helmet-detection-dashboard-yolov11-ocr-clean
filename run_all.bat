@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ==================================================================
echo        Intelligence System Hub - Automated Setup ^& Launch
echo ==================================================================
echo.

:: Define the list of projects and their corresponding ports
set "projects=Helmet-License-Plate-Detection:7861 roadsentinel-helmet-monitor:7862 Vehicle_Speed_Estimation_and_Counting:7863"
set /a count=1

:: Set UV cache to the local drive to enable insanely fast hardlinking!
set "UV_CACHE_DIR=%~dp0.uv_cache"

for %%p in (%projects%) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%p") do (
        set "folder=%%a"
        set "port=%%b"
        
        echo ==================================================================
        echo [!count!/3] Processing: !folder!
        echo ==================================================================
        
        if exist "!folder!" (
            cd "!folder!"
            
            :: Check for virtual environment
            if not exist "venv\Scripts\python.exe" (
                echo [*] Virtual environment missing. Creating 'venv'...
                echo [LOADING] Setting up environment. Please wait...
                python -m venv venv
                if !errorlevel! neq 0 (
                    echo [ERROR] Failed to create virtual environment for !folder!
                    pause
                    exit /b !errorlevel!
                )
                echo [+] Virtual environment created successfully.
            ) else (
                echo [+] Virtual environment found.
            )
            
            set "VENV_PYTHON=venv\Scripts\python.exe"
            set "VENV_PIP=venv\Scripts\pip.exe"
            
            :: Check dependencies by attempting to import a key package
            echo [*] Verifying dependencies...
            !VENV_PYTHON! -c "import gradio" >nul 2>&1
            if !errorlevel! neq 0 (
                echo [!] Missing key dependencies.
                echo [LOADING] Installing 'uv' ^(Lightning-fast package manager^)...
                !VENV_PYTHON! -m pip install uv >nul 2>&1
                echo [LOADING] Installing from requirements.txt using 'uv'...
                echo [LOADING] This may take a moment. DO NOT CLOSE THIS WINDOW...
                !VENV_PYTHON! -m uv pip install -r requirements.txt
                if !errorlevel! neq 0 (
                    echo [ERROR] Failed to install dependencies for !folder!
                    pause
                    exit /b !errorlevel!
                )
                echo [+] Dependencies installed successfully.
            ) else (
                echo [+] Dependencies are verified and ready.
            )
            
            :: Start the application in a new detached window
            echo [*] Launching application on port !port!...
            start "!folder! (Port !port!)" cmd /c "call venv\Scripts\activate.bat && title !folder! && python app.py --port !port!"
            
            cd ..
            echo.
        ) else (
            echo [!] Directory !folder! not found! Skipping...
            echo.
        )
        set /a count+=1
    )
)

echo ==================================================================
echo [SUCCESS] All systems have been checked, configured, and launched!
echo [INFO] The separate console windows contain the server logs.
echo [INFO] You can now securely open 'index.html' to view the dashboard.
echo ==================================================================
pause
