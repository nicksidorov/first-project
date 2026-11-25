$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if ($env:OS -ne 'Windows_NT') { throw 'Build this application on Windows.' }
python -m venv .build-venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create build environment.' }
$python = Join-Path $PSScriptRoot '.build-venv\Scripts\python.exe'
& $python -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install dependencies.' }
& $python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $python -m PyInstaller --noconfirm --clean --onefile --windowed --noupx --name MIX_QC_1.1 --collect-data plotly --collect-data openpyxl --exclude-module PyQt5.QtWebEngine --exclude-module PyQt5.QtWebEngineWidgets --exclude-module PyQt5.QtWebEngineCore --exclude-module matplotlib --exclude-module IPython --exclude-module notebook pipp.py
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
Write-Host 'Ready: dist\MIX_QC_1.1.exe'
