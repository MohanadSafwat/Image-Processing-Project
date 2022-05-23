@echo off

set inputPath = %1
set outputPath = %2
set type = %3
python carDetection.py %inputPath % %outputPath % %type %

