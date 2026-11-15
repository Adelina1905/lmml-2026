@echo off
rem Step 03: train a fresh local detector in a separate rerun folder.
rem The passing results and test-output.txt are preserved. No API requests are made.
title Bad BOMB (Laos) - Local Detector Training
cd /d "%~dp0.."
".venv\Scripts\python.exe" -u "scripts\step_03_train_local_detector.py" --run-name local_detector_rerun
echo.
echo Local run stopped or completed. See results\local_detector_rerun\status.json and review.html.
pause
