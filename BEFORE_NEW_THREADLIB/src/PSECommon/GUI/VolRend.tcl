#texslice demo

after 1000 {

set m01 [addModuleAtPosition ScalarFieldReader 25 10]
set m02 [addModuleAtPosition SFRGfile 25 20]
set m03 [addModuleAtPosition VolRendTexSlices 25 30]
set m04 [addModuleAtPosition Salmon 25 40]

addConnection $m01 0 $m02 0 
addConnection $m02 0 $m03 0
addConnection $m03 0 $m04 0

set $m01-filename /projects/scirun/scirundemo/data/BrainFE/brain.64

$m03 ui
$m04 ui

set $m03-accum 0.17
set $m03-bright 0.30
set $m02-haveOutVoxelTCL 1
set $m02-outVoxelTCL 5
set $m04-Roe_0-view-fov 20
set $m04-Roe_0-view-eyep-x 1.95
set $m04-Roe_0-view-eyep-y -3.33
set $m04-Roe_0-view-eyep-z 3.22
set $m04-Roe_0-view-lookat-x 0.50
set $m04-Roe_0-view-lookat-y 0.50
set $m04-Roe_0-view-lookat-z 0.50
set $m04-Roe_0-view-up-x 1.00
set $m04-Roe_0-view-up-y 0.00
set $m04-Roe_0-view-up-z 0.00

after 1000 {
  set m05 [addModuleAtPosition GenAxes 215 30]
  addConnection $m05 0 $m04 1
  $m01-c needexecute
}
