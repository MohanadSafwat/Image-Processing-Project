@echo off

set inputPath = %1
set outputPath = %2
set type = %3
set debug = %4
python car_lanes.py %inputPath % %outputPath % %type % %debug %

