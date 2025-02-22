@echo off
REM Change directory to the location of this batch file.
cd /d %~dp0

echo Setting up the Python virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing numpy separately to satisfy Aeneas dependency...
python -m pip install numpy

echo Installing wheel (required for building wheels)...
python -m pip install wheel

echo Installing remaining dependencies from requirements.txt...
python -m pip install -r requirements.txt

echo Disabling the Aeneas C extension (cew) to avoid linking errors with eSpeak...
set AENEAS_WITH_CEW=False

echo Installing Aeneas without build isolation...
python -m pip install --no-build-isolation aeneas

echo Setup complete!
pause
