@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python image_sampler_tool.py
pause 