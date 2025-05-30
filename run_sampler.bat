@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Image Sampler Tool Setup
echo ========================================
echo.

REM Function to check if Python is available
:check_python
echo Checking for Python installation...

REM Try different Python commands
set PYTHON_CMD=
for %%i in (python py python3) do (
    %%i --version >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_CMD=%%i
        goto :python_found
    )
)

echo ERROR: Python not found in PATH!
echo Please install Python 3.7+ and ensure it's added to your PATH.
echo Download from: https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during installation.
goto :error_exit

:python_found
echo Found Python: !PYTHON_CMD!
!PYTHON_CMD! --version

REM Check if venv module is available
echo.
echo Checking if venv module is available...
!PYTHON_CMD! -m venv --help >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python venv module not available!
    echo Please ensure you have a complete Python installation.
    goto :error_exit
)

REM Check if pip is available
echo Checking if pip is available...
!PYTHON_CMD! -m pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo WARNING: pip not available via module, trying to repair...
    
    REM Try to install/repair pip using ensurepip
    echo Attempting to install pip using ensurepip...
    !PYTHON_CMD! -m ensurepip --upgrade >nul 2>&1
    if !errorlevel! equ 0 (
        echo Successfully installed pip via ensurepip!
        !PYTHON_CMD! -m pip --version
    ) else (
        echo ensurepip failed, trying alternative method...
        
        REM Try downloading get-pip.py
        echo Downloading get-pip.py...
        powershell -Command "try { Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py' -UseBasicParsing } catch { exit 1 }"
        if !errorlevel! equ 0 (
            echo Installing pip using get-pip.py...
            !PYTHON_CMD! get-pip.py
            if !errorlevel! equ 0 (
                echo Successfully installed pip!
                del get-pip.py
                !PYTHON_CMD! -m pip --version
            ) else (
                echo ERROR: Failed to install pip using get-pip.py!
                del get-pip.py 2>nul
                goto :pip_manual_instructions
            )
        ) else (
            echo ERROR: Failed to download get-pip.py!
            goto :pip_manual_instructions
        )
    )
    
    REM Verify pip is now working
    !PYTHON_CMD! -m pip --version >nul 2>&1
    if !errorlevel! neq 0 (
        goto :pip_manual_instructions
    )
) else (
    echo pip is available!
    !PYTHON_CMD! -m pip --version
)

REM Check if venv exists, if not, create it
if not exist venv (
    echo.
    echo Creating virtual environment...
    !PYTHON_CMD! -m venv venv
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create virtual environment!
        goto :error_exit
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

REM Activate the virtual environment
echo.
echo Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment activation script not found!
    echo Trying to recreate virtual environment...
    rmdir /s /q venv
    !PYTHON_CMD! -m venv venv
    if !errorlevel! neq 0 (
        echo ERROR: Failed to recreate virtual environment!
        goto :error_exit
    )
    call venv\Scripts\activate.bat
)

REM Verify we're in the virtual environment
echo Verifying virtual environment...
where python
python --version

REM Upgrade pip in virtual environment
echo.
echo Upgrading pip in virtual environment...
python -m pip install --upgrade pip
if !errorlevel! neq 0 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

REM Install dependencies
echo.
echo Installing dependencies...
if exist requirements.txt (
    echo Installing from requirements.txt...
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install requirements!
        goto :error_exit
    )
) else (
    echo requirements.txt not found, installing individual packages...
)

REM Ensure PyQt5 is installed (critical dependency)
echo.
echo Ensuring PyQt5 is installed...
python -m pip install PyQt5
if !errorlevel! neq 0 (
    echo ERROR: Failed to install PyQt5!
    echo Trying alternative installation method...
    python -m pip install --upgrade --force-reinstall PyQt5
    if !errorlevel! neq 0 (
        echo ERROR: PyQt5 installation failed completely!
        echo Trying with --user flag...
        python -m pip install --user PyQt5
        if !errorlevel! neq 0 (
            echo ERROR: All PyQt5 installation methods failed!
            goto :error_exit
        )
    )
)

REM Ensure Pillow is installed
echo.
echo Ensuring Pillow is installed...
python -m pip install Pillow
if !errorlevel! neq 0 (
    echo WARNING: Failed to install Pillow, but continuing...
)

REM Verify installations
echo.
echo Verifying installations...
python -c "import PyQt5.QtWidgets; print('PyQt5 successfully imported')" 2>nul
if !errorlevel! neq 0 (
    echo WARNING: PyQt5.QtWidgets verification failed!
    echo Trying alternative verification...
    python -c "import PyQt5; print('PyQt5 basic import successful')" 2>nul
    if !errorlevel! neq 0 (
        echo ERROR: PyQt5 import completely failed!
        echo.
        echo Would you like to run the detailed test script? (y/n)
        set /p choice="Enter choice: "
        if /i "!choice!"=="y" (
            python test_pyqt5.py
            pause
        )
        goto :error_exit
    ) else (
        echo PyQt5 basic import works, continuing anyway...
    )
) else (
    echo PyQt5 verification successful!
)

python -c "import PIL; print('Pillow version:', PIL.__version__)" 2>nul
if !errorlevel! neq 0 (
    echo WARNING: Pillow verification failed, but continuing...
) else (
    echo Pillow verification successful!
)

echo.
echo ========================================
echo   Setup completed successfully!
echo ========================================
echo.

REM Start the Image Sampler
echo Starting Image Sampler Tool...
echo If the tool fails to start, you can run 'python test_pyqt5.py' for detailed diagnostics.
echo.
python image_sampler_tool.py

REM Check if the tool started successfully
if !errorlevel! neq 0 (
    echo.
    echo ========================================
    echo   Tool failed to start!
    echo ========================================
    echo.
    echo The Image Sampler Tool encountered an error.
    echo Would you like to run the diagnostic test? (y/n)
    set /p choice="Enter choice: "
    if /i "!choice!"=="y" (
        echo.
        echo Running PyQt5 diagnostic test...
        python test_pyqt5.py
    )
    goto :error_exit
)

goto :normal_exit

:pip_manual_instructions
echo.
echo ========================================
echo   Manual pip installation required
echo ========================================
echo.
echo Automatic pip installation failed. Please try these manual steps:
echo.
echo Option 1 - Reinstall Python:
echo 1. Go to https://www.python.org/downloads/
echo 2. Download Python 3.10 or newer
echo 3. During installation, check "Add Python to PATH"
echo 4. Check "pip" in optional features
echo 5. Run this script again
echo.
echo Option 2 - Manual pip installation:
echo 1. Download get-pip.py from https://bootstrap.pypa.io/get-pip.py
echo 2. Save it to this folder
echo 3. Run: !PYTHON_CMD! get-pip.py
echo 4. Run this script again
echo.
echo Option 3 - Use system package manager:
echo 1. If you have Chocolatey: choco install python
echo 2. If you have Scoop: scoop install python
echo.
pause
exit /b 1

:error_exit
echo.
echo ========================================
echo   Setup failed!
echo ========================================
echo.
echo Please check the error messages above and try again.
echo If problems persist, try:
echo 1. Reinstalling Python from https://www.python.org/downloads/
echo 2. Ensuring "Add Python to PATH" is checked during installation
echo 3. Running this script as Administrator
echo 4. Running the diagnostic script: python check_python.py
echo.
pause
exit /b 1

:normal_exit
echo.
echo Press any key to exit...
pause > nul
