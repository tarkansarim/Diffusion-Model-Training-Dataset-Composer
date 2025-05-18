@echo off
REM Create virtual environment if it doesn't exist
if not exist .venv (
    python -m venv .venv
)

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

REM Launch the GUI
python dataset_sampler.py
