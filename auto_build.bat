@echo off
REM Auto-wait and build script for C++ YOLOv8

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   YOLOv8 C++ - Auto Build Monitor
echo ========================================
echo.
echo Monitoring vcpkg installation...
echo This script will automatically build when dependencies are ready.
echo.
echo Elapsed Time  Progress
echo ============  ========================================================
echo.

set /a elapsed=0
set /a total_wait=3600

:wait_loop
set /a elapsed=%elapsed%+60

REM Format elapsed time (MM:SS)
set /a minutes=%elapsed% / 60
set /a seconds=%elapsed% %% 60
if %seconds% lss 10 set seconds=0%seconds%

REM Check if OpenCV is installed
if exist "c:\OPEN CV\yolo_detection\vcpkg\installed\x64-windows\lib\cmake\OpenCV" (
    if exist "c:\OPEN CV\yolo_detection\vcpkg\installed\x64-windows\lib\cmake\onnxruntime" (
        echo Dependencies ready after !minutes!:!seconds!
        goto build_ready
    )
)

REM Check folder size for progress indication
for /f %%A in ('powershell -Command "([math]::Round(((Get-ChildItem 'c:\OPEN CV\yolo_detection\vcpkg' -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB)))"') do (
    set size=%%A
)

cls
echo.
echo ========================================
echo   YOLOv8 C++ - Auto Build Monitor
echo ========================================
echo.
echo Monitoring vcpkg installation...
echo This script will automatically build when dependencies are ready.
echo.
echo Elapsed Time: !minutes!:!seconds!
echo Folder Size: !size! MB
echo Status: Still compiling OpenCV and ONNX Runtime...
echo.

REM Wait 60 seconds before checking again
timeout /t 60 /nobreak >nul

if %elapsed% lss %total_wait% goto wait_loop

echo.
echo WARNING: Vcpkg installation taking longer than expected (60 minutes)
echo Please check manually or try again later.
pause
exit /b 1

:build_ready
echo.
echo ========================================
echo Dependencies installed successfully!
echo ========================================
echo.
echo Now configuring and building C++ project...
echo.

call build.bat
