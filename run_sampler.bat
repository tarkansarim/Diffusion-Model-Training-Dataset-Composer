@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Starting Image Sampler...
python image_sampler_tool.py

echo.
echo Press any key to exit...
pause > nul
