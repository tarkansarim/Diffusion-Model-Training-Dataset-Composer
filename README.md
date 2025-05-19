# Dataset Sampler

This tool provides a simple GUI for sampling images and captions from multiple folders.

## Features

- Add and remove multiple source folders
- Specify sampling percentage for each folder
- Configure caption style percentages (blank, basic, detailed, structured)
- Two sampling modes: Classic and Total Count
- Destination folder selection
- Progress bar and log output

The application is implemented with PyQt5 and stores the last used destination
folder via `QSettings`.

Run with:

```bash
python image_sampler_tool.py
```

Create and activate a virtual environment, then install requirements:

```bash
python -m venv venv
source venv/bin/activate  # on Windows use venv\Scripts\activate
pip install -r requirements.txt
```

You can also use `run_sampler.bat` on Windows.
