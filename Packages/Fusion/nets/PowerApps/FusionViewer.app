# SCI Network 1.0
#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

setProgressText "Loading FusionViewer Modules..."


#######################################################################
# Check environment variables.  Ask user for input if not set:
init_DATADIR_and_DATASET
############# NET ##############

::netedit dontschedule

global userName
set userName "ars"

global runDate
set runDate " Tue  Jan 1 2002"

global runTime
set runTime " 14:25:13"

global animate_lock
set animate_lock 1

#######################################################################
#######################################################################

# Create a DataIO->Readers->HDF5DataReader Module
set m99 [addModuleAtPosition "DataIO" "Readers" "HDF5DataReader" 0 0]

# Create a DataIO->Readers->HDF5DataReader Module
set m1 [addModuleAtPosition "DataIO" "Readers" "HDF5DataReader" 325 0]

# Create a DataIO->Readers->HDF5DataReader Module
set m2 [addModuleAtPosition "DataIO" "Readers" "HDF5DataReader" 1275 100]



# Create a DataIO->Readers->MDSPlusDataReader Module
set m3 [addModuleAtPosition "DataIO" "Readers" "MDSPlusDataReader" 25 75]

# Create a DataIO->Readers->MDSPlusDataReader Module
set m4 [addModuleAtPosition "DataIO" "Readers" "MDSPlusDataReader" 350 75]

# Create a DataIO->Readers->MDSPlusDataReader Module
set m5 [addModuleAtPosition "DataIO" "Readers" "MDSPlusDataReader" 1300 175]



# Create a Teem->NrrdData->ChooseNrrd Module
set m6 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 0 175]

# Create a Teem->NrrdData->ChooseNrrd Module
set m7 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 325 175]

# Create a Teem->NrrdData->ChooseNrrd Module
set m8 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 1275 275]



# Create a Teem->Converters->NrrdToField Module
set m11 [addModuleAtPosition "Teem" "Converters" "NrrdToField" 250 500]

# Create a Teem->Converters->NrrdToField Module
set m12 [addModuleAtPosition "Teem" "Converters" "NrrdToField" 1250 400]



# Create a SCIRun->FieldsCreate->FieldSubSample Module
set m20 [addModuleAtPosition "SCIRun" "FieldsCreate" "FieldSubSample" 250 600]

# Create a SCIRun->FieldsCreate->FieldSlicer Module
set m21 [addModuleAtPosition "SCIRun" "FieldsCreate" "FieldSlicer" 450 700]

# Create a SCIRun->FieldsCreate->FieldSlicer Module
set m22 [addModuleAtPosition "SCIRun" "FieldsCreate" "FieldSlicer" 650 700]


# Create a SCIRun->Visualization->Isosurface Module
set m23 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 250 900]

# Create a SCIRun->Visualization->Isosurface Module
set m24 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 450 900]

# Create a SCIRun->Visualization->Isosurface Module
set m25 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 650 900]


# Create a SCIRun->FieldsCreate->GatherFields Module
set m26 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherFields" 450 1000]

# Create a SCIRun->Visualization->ShowField Module
set m27 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 250 1100]

# Create a SCIRun->Visualization->ShowField Module
set m28 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 450 1100]


# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m29 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 0 600]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m30 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 0 700]



# Create a SCIRun->FieldsCreate->SampleField Module
set m40 [addModuleAtPosition "SCIRun" "FieldsCreate" "SampleField" 1275 500]

# Create a SCIRun->Visualization->StreamLines Module
set m41 [addModuleAtPosition "SCIRun" "Visualization" "StreamLines" 1250 600]

# Create a SCIRun->FieldsData->DirectInterpolate Module
set m42 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1025 700]

# Create a SCIRun->Visualization->ShowField Module
set m43 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 991 1000]

# Create a SCIRun->Visualization->ShowField Module
set m44 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1250 1000]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m45 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 900 800]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m46 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1350 800]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m47 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 900 900]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m48 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1350 900]

# Create a SCIRun->FieldsData->VectorMagnitude Module
set m49 [addModuleAtPosition "SCIRun" "FieldsData" "VectorMagnitude" 925 500]

# Create a SCIRun->FieldsOther->ChooseField Module
set m50 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 900 600]



# Create a SCIRun->Render->SynchronizeGeometry Module
set m100 [addModuleAtPosition "SCIRun" "Render" "SynchronizeGeometry" 0 1400]
# Create a SCIRun->Render->Viewer Module
set m101 [addModuleAtPosition "SCIRun" "Render" "Viewer" 481 1600]



# Create the Connections between Modules
#set c0 [addConnection $m1 8 $m2 0]
set c1 [addConnection $m99 0 $m6 0]
set c2 [addConnection $m1 0 $m7 0]
set c3 [addConnection $m2 0 $m8 0]

set c4 [addConnection $m3 0 $m6 1]
set c5 [addConnection $m4 0 $m7 1]
set c6 [addConnection $m5 0 $m8 1]

set c7 [addConnection $m6 0 $m11 0]
set c8 [addConnection $m6 0 $m12 0]
set c9 [addConnection $m7 0 $m11 2]
set c10 [addConnection $m8 0 $m12 2]


set c11 [addConnection $m11 0 $m20 0]
set c12 [addConnection $m11 0 $m50 0]

set c13 [addConnection $m12 0 $m40 0]
set c14 [addConnection $m12 0 $m41 0]
set c15 [addConnection $m12 0 $m49 0]




set c20 [addConnection $m20 0 $m21 0]
set c21 [addConnection $m20 0 $m22 0]
set c22 [addConnection $m20 0 $m23 0]
set c23 [addConnection $m20 0 $m30 1]
set c24 [addConnection $m21 0 $m24 0]
set c25 [addConnection $m22 0 $m25 0]
set c26 [addConnection $m23 0 $m27 0]
set c27 [addConnection $m24 0 $m26 0]
set c28 [addConnection $m25 0 $m26 1]
set c29 [addConnection $m26 0 $m28 0]
set c30 [addConnection $m27 0 $m100 0]
set c31 [addConnection $m28 0 $m100 1]
set c32 [addConnection $m29 0 $m30 0]
set c33 [addConnection $m30 0 $m27 1]
set c34 [addConnection $m30 0 $m28 1]

set c41 [addConnection $m40 0 $m41 1]
set c42 [addConnection $m41 0 $m42 1]
set c43 [addConnection $m41 0 $m48 1]
set c44 [addConnection $m41 0 $m44 0]
set c45 [addConnection $m42 0 $m43 0]
set c46 [addConnection $m43 0 $m100 2]
set c47 [addConnection $m44 0 $m100 3]
set c48 [addConnection $m45 0 $m47 0]
set c49 [addConnection $m46 0 $m48 0]
set c50 [addConnection $m47 0 $m43 1]
set c51 [addConnection $m48 0 $m44 1]
set c52 [addConnection $m49 0 $m50 1]
set c53 [addConnection $m50 0 $m42 0]
set c54 [addConnection $m50 0 $m47 1]

set c100 [addConnection $m100 0 $m101 0]
set c101 [addConnection $m40 1 $m101 1]


# Set GUI variables for the DataIO->Readers->HDF5DataReader Module
set $m99-filename "$DATADIR/$DATASET/phi.h5"
set $m99-datasets {{/ GRID X} {/ GRID Y} {/ GRID Z}}
set $m99-dumpname {/tmp/qwall.h5.dump}
set $m99-ports {   0   0   0}
set $m99-ndims {3}
set $m99-0-dim {101}
set $m99-0-count {101}
set $m99-1-dim {61}
set $m99-1-count {61}
set $m99-2-dim {101}
set $m99-2-count {101}

# Set GUI variables for the DataIO->Readers->HDF5DataReader Module
set $m1-selectable_max {115.0}
set $m1-range_max {115}
set $m1-current {87}
set $m1-execmode {current}
set $m1-filename "$DATADIR/$DATASET/phi.h5"
set $m1-datasets {{/ step_0000000 T_e} {/ step_0000050 T_e} {/ step_0000100 T_e} {/ step_0000150 T_e} {/ step_0000200 T_e} {/ step_0000250 T_e} {/ step_0000300 T_e} {/ step_0000350 T_e} {/ step_0000400 T_e} {/ step_0000450 T_e} {/ step_0000500 T_e} {/ step_0000550 T_e} {/ step_0000600 T_e} {/ step_0000650 T_e} {/ step_0000700 T_e} {/ step_0000750 T_e} {/ step_0000800 T_e} {/ step_0000850 T_e} {/ step_0000900 T_e} {/ step_0000950 T_e} {/ step_0001000 T_e} {/ step_0001050 T_e} {/ step_0001100 T_e} {/ step_0001150 T_e} {/ step_0001200 T_e} {/ step_0001250 T_e} {/ step_0001300 T_e} {/ step_0001350 T_e} {/ step_0001384 T_e} {/ step_0001400 T_e} {/ step_0001450 T_e} {/ step_0001500 T_e} {/ step_0001550 T_e} {/ step_0001600 T_e} {/ step_0001650 T_e} {/ step_0001700 T_e} {/ step_0001750 T_e} {/ step_0001776 T_e} {/ step_0001800 T_e} {/ step_0001850 T_e} {/ step_0001900 T_e} {/ step_0001950 T_e} {/ step_0002000 T_e} {/ step_0002050 T_e} {/ step_0002100 T_e} {/ step_0002150 T_e} {/ step_0002200 T_e} {/ step_0002250 T_e} {/ step_0002300 T_e} {/ step_0002350 T_e} {/ step_0002400 T_e} {/ step_0002450 T_e} {/ step_0002465 T_e} {/ step_0002500 T_e} {/ step_0002550 T_e} {/ step_0002600 T_e} {/ step_0002650 T_e} {/ step_0002700 T_e} {/ step_0002750 T_e} {/ step_0002783 T_e} {/ step_0002800 T_e} {/ step_0002850 T_e} {/ step_0002900 T_e} {/ step_0002950 T_e} {/ step_0003000 T_e} {/ step_0003050 T_e} {/ step_0003100 T_e} {/ step_0003150 T_e} {/ step_0003200 T_e} {/ step_0003250 T_e} {/ step_0003300 T_e} {/ step_0003350 T_e} {/ step_0003400 T_e} {/ step_0003450 T_e} {/ step_0003487 T_e} {/ step_0003500 T_e} {/ step_0003550 T_e} {/ step_0003600 T_e} {/ step_0003650 T_e} {/ step_0003700 T_e} {/ step_0003750 T_e} {/ step_0003800 T_e} {/ step_0003850 T_e} {/ step_0003900 T_e} {/ step_0003950 T_e} {/ step_0004000 T_e} {/ step_0004050 T_e} {/ step_0004100 T_e} {/ step_0004150 T_e} {/ step_0004200 T_e} {/ step_0004250 T_e} {/ step_0004300 T_e} {/ step_0004350 T_e} {/ step_0004400 T_e} {/ step_0004450 T_e} {/ step_0004500 T_e} {/ step_0004550 T_e} {/ step_0004600 T_e} {/ step_0004650 T_e} {/ step_0004700 T_e} {/ step_0004750 T_e} {/ step_0004800 T_e} {/ step_0004850 T_e} {/ step_0004900 T_e} {/ step_0004950 T_e} {/ step_0005000 T_e} {/ step_0005050 T_e} {/ step_0005100 T_e} {/ step_0005150 T_e} {/ step_0005200 T_e} {/ step_0005250 T_e} {/ step_0005300 T_e} {/ step_0005350 T_e} {/ step_0005400 T_e} {/ step_0005450 T_e} {/ step_0005487 T_e}}
set $m1-dumpname {/tmp/phi.h5.dump}
set $m1-ports {   0   0   0   1}
set $m1-ndims {3}
set $m1-animate {1}
set $m1-0-dim {101}
set $m1-0-count {101}
set $m1-1-dim {61}
set $m1-1-count {61}
set $m1-2-dim {101}
set $m1-2-count {101}


# Set GUI variables for the DataIO->Readers->HDF5DataReader Module
set $m2-selectable_max {115.0}
set $m2-range_max {115}
set $m2-current {87}
set $m2-execmode {current}
set $m2-filename "$DATADIR/$DATASET/phi.h5"
set $m2-datasets {{/ step_0000000 B X} {/ step_0000000 B Y} {/ step_0000000 B Z} {/ step_0000050 B X} {/ step_0000050 B Y} {/ step_0000050 B Z} {/ step_0000100 B X} {/ step_0000100 B Y} {/ step_0000100 B Z} {/ step_0000150 B X} {/ step_0000150 B Y} {/ step_0000150 B Z} {/ step_0000200 B X} {/ step_0000200 B Y} {/ step_0000200 B Z} {/ step_0000250 B X} {/ step_0000250 B Y} {/ step_0000250 B Z} {/ step_0000300 B X} {/ step_0000300 B Y} {/ step_0000300 B Z} {/ step_0000350 B X} {/ step_0000350 B Y} {/ step_0000350 B Z} {/ step_0000400 B X} {/ step_0000400 B Y} {/ step_0000400 B Z} {/ step_0000450 B X} {/ step_0000450 B Y} {/ step_0000450 B Z} {/ step_0000500 B X} {/ step_0000500 B Y} {/ step_0000500 B Z} {/ step_0000550 B X} {/ step_0000550 B Y} {/ step_0000550 B Z} {/ step_0000600 B X} {/ step_0000600 B Y} {/ step_0000600 B Z} {/ step_0000650 B X} {/ step_0000650 B Y} {/ step_0000650 B Z} {/ step_0000700 B X} {/ step_0000700 B Y} {/ step_0000700 B Z} {/ step_0000750 B X} {/ step_0000750 B Y} {/ step_0000750 B Z} {/ step_0000800 B X} {/ step_0000800 B Y} {/ step_0000800 B Z} {/ step_0000850 B X} {/ step_0000850 B Y} {/ step_0000850 B Z} {/ step_0000900 B X} {/ step_0000900 B Y} {/ step_0000900 B Z} {/ step_0000950 B X} {/ step_0000950 B Y} {/ step_0000950 B Z} {/ step_0001000 B X} {/ step_0001000 B Y} {/ step_0001000 B Z} {/ step_0001050 B X} {/ step_0001050 B Y} {/ step_0001050 B Z} {/ step_0001100 B X} {/ step_0001100 B Y} {/ step_0001100 B Z} {/ step_0001150 B X} {/ step_0001150 B Y} {/ step_0001150 B Z} {/ step_0001200 B X} {/ step_0001200 B Y} {/ step_0001200 B Z} {/ step_0001250 B X} {/ step_0001250 B Y} {/ step_0001250 B Z} {/ step_0001300 B X} {/ step_0001300 B Y} {/ step_0001300 B Z} {/ step_0001350 B X} {/ step_0001350 B Y} {/ step_0001350 B Z} {/ step_0001384 B X} {/ step_0001384 B Y} {/ step_0001384 B Z} {/ step_0001400 B X} {/ step_0001400 B Y} {/ step_0001400 B Z} {/ step_0001450 B X} {/ step_0001450 B Y} {/ step_0001450 B Z} {/ step_0001500 B X} {/ step_0001500 B Y} {/ step_0001500 B Z} {/ step_0001550 B X} {/ step_0001550 B Y} {/ step_0001550 B Z} {/ step_0001600 B X} {/ step_0001600 B Y} {/ step_0001600 B Z} {/ step_0001650 B X} {/ step_0001650 B Y} {/ step_0001650 B Z} {/ step_0001700 B X} {/ step_0001700 B Y} {/ step_0001700 B Z} {/ step_0001750 B X} {/ step_0001750 B Y} {/ step_0001750 B Z} {/ step_0001776 B X} {/ step_0001776 B Y} {/ step_0001776 B Z} {/ step_0001800 B X} {/ step_0001800 B Y} {/ step_0001800 B Z} {/ step_0001850 B X} {/ step_0001850 B Y} {/ step_0001850 B Z} {/ step_0001900 B X} {/ step_0001900 B Y} {/ step_0001900 B Z} {/ step_0001950 B X} {/ step_0001950 B Y} {/ step_0001950 B Z} {/ step_0002000 B X} {/ step_0002000 B Y} {/ step_0002000 B Z} {/ step_0002050 B X} {/ step_0002050 B Y} {/ step_0002050 B Z} {/ step_0002100 B X} {/ step_0002100 B Y} {/ step_0002100 B Z} {/ step_0002150 B X} {/ step_0002150 B Y} {/ step_0002150 B Z} {/ step_0002200 B X} {/ step_0002200 B Y} {/ step_0002200 B Z} {/ step_0002250 B X} {/ step_0002250 B Y} {/ step_0002250 B Z} {/ step_0002300 B X} {/ step_0002300 B Y} {/ step_0002300 B Z} {/ step_0002350 B X} {/ step_0002350 B Y} {/ step_0002350 B Z} {/ step_0002400 B X} {/ step_0002400 B Y} {/ step_0002400 B Z} {/ step_0002450 B X} {/ step_0002450 B Y} {/ step_0002450 B Z} {/ step_0002465 B X} {/ step_0002465 B Y} {/ step_0002465 B Z} {/ step_0002500 B X} {/ step_0002500 B Y} {/ step_0002500 B Z} {/ step_0002550 B X} {/ step_0002550 B Y} {/ step_0002550 B Z} {/ step_0002600 B X} {/ step_0002600 B Y} {/ step_0002600 B Z} {/ step_0002650 B X} {/ step_0002650 B Y} {/ step_0002650 B Z} {/ step_0002700 B X} {/ step_0002700 B Y} {/ step_0002700 B Z} {/ step_0002750 B X} {/ step_0002750 B Y} {/ step_0002750 B Z} {/ step_0002783 B X} {/ step_0002783 B Y} {/ step_0002783 B Z} {/ step_0002800 B X} {/ step_0002800 B Y} {/ step_0002800 B Z} {/ step_0002850 B X} {/ step_0002850 B Y} {/ step_0002850 B Z} {/ step_0002900 B X} {/ step_0002900 B Y} {/ step_0002900 B Z} {/ step_0002950 B X} {/ step_0002950 B Y} {/ step_0002950 B Z} {/ step_0003000 B X} {/ step_0003000 B Y} {/ step_0003000 B Z} {/ step_0003050 B X} {/ step_0003050 B Y} {/ step_0003050 B Z} {/ step_0003100 B X} {/ step_0003100 B Y} {/ step_0003100 B Z} {/ step_0003150 B X} {/ step_0003150 B Y} {/ step_0003150 B Z} {/ step_0003200 B X} {/ step_0003200 B Y} {/ step_0003200 B Z} {/ step_0003250 B X} {/ step_0003250 B Y} {/ step_0003250 B Z} {/ step_0003300 B X} {/ step_0003300 B Y} {/ step_0003300 B Z} {/ step_0003350 B X} {/ step_0003350 B Y} {/ step_0003350 B Z} {/ step_0003400 B X} {/ step_0003400 B Y} {/ step_0003400 B Z} {/ step_0003450 B X} {/ step_0003450 B Y} {/ step_0003450 B Z} {/ step_0003487 B X} {/ step_0003487 B Y} {/ step_0003487 B Z} {/ step_0003500 B X} {/ step_0003500 B Y} {/ step_0003500 B Z} {/ step_0003550 B X} {/ step_0003550 B Y} {/ step_0003550 B Z} {/ step_0003600 B X} {/ step_0003600 B Y} {/ step_0003600 B Z} {/ step_0003650 B X} {/ step_0003650 B Y} {/ step_0003650 B Z} {/ step_0003700 B X} {/ step_0003700 B Y} {/ step_0003700 B Z} {/ step_0003750 B X} {/ step_0003750 B Y} {/ step_0003750 B Z} {/ step_0003800 B X} {/ step_0003800 B Y} {/ step_0003800 B Z} {/ step_0003850 B X} {/ step_0003850 B Y} {/ step_0003850 B Z} {/ step_0003900 B X} {/ step_0003900 B Y} {/ step_0003900 B Z} {/ step_0003950 B X} {/ step_0003950 B Y} {/ step_0003950 B Z} {/ step_0004000 B X} {/ step_0004000 B Y} {/ step_0004000 B Z} {/ step_0004050 B X} {/ step_0004050 B Y} {/ step_0004050 B Z} {/ step_0004100 B X} {/ step_0004100 B Y} {/ step_0004100 B Z} {/ step_0004150 B X} {/ step_0004150 B Y} {/ step_0004150 B Z} {/ step_0004200 B X} {/ step_0004200 B Y} {/ step_0004200 B Z} {/ step_0004250 B X} {/ step_0004250 B Y} {/ step_0004250 B Z} {/ step_0004300 B X} {/ step_0004300 B Y} {/ step_0004300 B Z} {/ step_0004350 B X} {/ step_0004350 B Y} {/ step_0004350 B Z} {/ step_0004400 B X} {/ step_0004400 B Y} {/ step_0004400 B Z} {/ step_0004450 B X} {/ step_0004450 B Y} {/ step_0004450 B Z} {/ step_0004500 B X} {/ step_0004500 B Y} {/ step_0004500 B Z} {/ step_0004550 B X} {/ step_0004550 B Y} {/ step_0004550 B Z} {/ step_0004600 B X} {/ step_0004600 B Y} {/ step_0004600 B Z} {/ step_0004650 B X} {/ step_0004650 B Y} {/ step_0004650 B Z} {/ step_0004700 B X} {/ step_0004700 B Y} {/ step_0004700 B Z} {/ step_0004750 B X} {/ step_0004750 B Y} {/ step_0004750 B Z} {/ step_0004800 B X} {/ step_0004800 B Y} {/ step_0004800 B Z} {/ step_0004850 B X} {/ step_0004850 B Y} {/ step_0004850 B Z} {/ step_0004900 B X} {/ step_0004900 B Y} {/ step_0004900 B Z} {/ step_0004950 B X} {/ step_0004950 B Y} {/ step_0004950 B Z} {/ step_0005000 B X} {/ step_0005000 B Y} {/ step_0005000 B Z} {/ step_0005050 B X} {/ step_0005050 B Y} {/ step_0005050 B Z} {/ step_0005100 B X} {/ step_0005100 B Y} {/ step_0005100 B Z} {/ step_0005150 B X} {/ step_0005150 B Y} {/ step_0005150 B Z} {/ step_0005200 B X} {/ step_0005200 B Y} {/ step_0005200 B Z} {/ step_0005250 B X} {/ step_0005250 B Y} {/ step_0005250 B Z} {/ step_0005300 B X} {/ step_0005300 B Y} {/ step_0005300 B Z} {/ step_0005350 B X} {/ step_0005350 B Y} {/ step_0005350 B Z} {/ step_0005400 B X} {/ step_0005400 B Y} {/ step_0005400 B Z} {/ step_0005450 B X} {/ step_0005450 B Y} {/ step_0005450 B Z} {/ step_0005487 B X} {/ step_0005487 B Y} {/ step_0005487 B Z}}
set $m2-dumpname {/tmp/phi.h5.dump}
set $m2-ports {   0   0   0   1}
set $m2-ndims {3}
set $m2-animate {1}
set $m2-0-dim {101}
set $m2-0-count {101}
set $m2-1-dim {61}
set $m2-1-count {61}
set $m2-2-dim {101}
set $m2-2-count {101}


# Set GUI variables for the DataIO->Readers->MDSPlusDataReader Module
set $m3-server {}
set $m3-tree {}
set $m3-shot {}
set $m3-load-server {}
set $m3-load-tree {}
set $m3-load-shot {}
set $m3-load-signal {}
set $m3-search-server {}
set $m3-search-tree {}
set $m3-search-shot {}
set $m3-search-signal {}

# Set GUI variables for the DataIO->Readers->MDSPlusDataReader Module
set $m4-server {}
set $m4-tree {}
set $m4-shot {}
set $m4-load-server {}
set $m4-load-tree {}
set $m4-load-shot {}
set $m4-load-signal {}
set $m4-search-server {}
set $m4-search-tree {}
set $m4-search-shot {}
set $m4-search-signal {}

# Set GUI variables for the DataIO->Readers->MDSPlusDataReader Module
set $m5-server {}
set $m5-tree {}
set $m5-shot {}
set $m5-load-server {}
set $m5-load-tree {}
set $m5-load-shot {}
set $m5-load-signal {}
set $m5-search-server {}
set $m5-search-tree {}
set $m5-search-shot {}
set $m5-search-signal {}


# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m6-usefirstvalid {1}

# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m7-usefirstvalid {1}

# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m8-usefirstvalid {1}



# Set GUI variables for the Teem->Converters->NrrdToField Module
set $m11-datasets {{Points : -GRID-X-Y-Z:Vector} {Connections : (none)} {Data : -step_0004100-T_e:Scalar} {Original Field : (none)} }

# Set GUI variables for the Teem->Converters->NrrdToField Module
set $m12-datasets {{Points : -GRID-X-Y-Z:Vector} {Connections : (none)} {Data : -step_0004100-B-X-Y-Z:Vector} {Original Field : (none)} }




# Set GUI variables for the SCIRun->FieldsCreate->FieldSubSample Module
set $m20-wrap {1}
set $m20-i-dim {101}
set $m20-j-dim {61}
set $m20-k-dim {101}
set $m20-i-stop {100}
set $m20-j-stop {60}
set $m20-k-stop {67}

# Set GUI variables for the SCIRun->FieldsCreate->FieldSlicer Module
set $m21-i-dim {101}
set $m21-j-dim {61}
set $m21-k-dim {68}
set $m21-i-index {0}
set $m21-j-index {0}
set $m21-k-index {0}

# Set GUI variables for the SCIRun->FieldsCreate->FieldSlicer Module
set $m22-i-dim {101}
set $m22-j-dim {61}
set $m22-k-dim {68}
set $m22-i-index {0}
set $m22-j-index {0}
set $m22-k-index {67}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m23-isoval {1000}
set $m23-isoval-min {60.1251335144}
set $m23-isoval-max {4761.27148438}
set $m23-isoval-quantity 1
set $m23-isoval-list {1000 4000 7000 12000 13160}
set $m23-active-isoval-selection-tab {2}
set $m23-active_tab {}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m24-isoval {1000}
set $m24-isoval-min {71.6849060059}
set $m24-isoval-max {4749.19091797}
set $m24-isoval-quantity 1
set $m24-isoval-list {1000 4000 7000 12000 13160}
set $m24-active-isoval-selection-tab {2}
set $m24-active_tab {}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m25-isoval {1000}
set $m25-isoval-min {61.2371101379}
set $m25-isoval-max {4728.31152344}
set $m25-isoval-quantity 1
set $m25-isoval-list {1000 4000 7000 12000 13160}
set $m25-active-isoval-selection-tab {2}
set $m25-active_tab {}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m27-nodes-on {0}
set $m27-edges-on {0}
set $m27-use-normals {1}
set $m27-use-transparency {1}
set $m27-normalize-vectors {}
set $m27-has_scalar_data {1}
set $m27-active_tab {Faces}
set $m27-scalars_scale {0.3}
set $m27-show_progress {}
set $m27-field-name {-step_0004100-T_e:Scalar}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m28-nodes-on {0}
set $m28-faces-on {0}
set $m28-normalize-vectors {}
set $m28-has_scalar_data {1}
set $m28-def-color-r {0.0}
set $m28-def-color-g {0.0}
set $m28-def-color-b {0.0}
set $m28-active_tab {Edges}
set $m28-scalars_scale {0.3}
set $m28-show_progress {}
set $m28-field-name {-step_0004100-T_e:Scalar}

# Set GUI variables for the SCIRun->Visualization->GenStandardColorMaps Module
set $m29-positionList {{355 2}}
set $m29-nodeList {257}
set $m29-width {390}
set $m29-height {40}
set $m29-gamma {0.0}

# Set GUI variables for the SCIRun->Visualization->RescaleColorMap Module
set $m30-isFixed {1}
set $m30-min {50}
set $m30-max {13214.7080078}




# Set GUI variables for the SCIRun->FieldsCreate->SampleField Module
set $m40-endpoints {1}
set $m40-endpoint0x {-0.854404350943}
set $m40-endpoint0y {1.35035004817}
set $m40-endpoint0z {-0.0364978830565}
set $m40-endpoint1x {-0.864402193716}
set $m40-endpoint1y {1.33871534477}
set $m40-endpoint1z {-0.023278780019}
set $m40-widgetscale {0.112657141788}
set $m40-ringstate {}
set $m40-framestate {}
set $m40-maxseeds {2}
set $m40-autoexecute {0}

# Set GUI variables for the SCIRun->Visualization->StreamLines Module
set $m41-stepsize {0.2}
set $m41-tolerance {1e-05}
set $m41-maxsteps {504}
set $m41-direction {2}
set $m41-method {0}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m43-nodes-on {0}
set $m43-edges-on {1}
set $m43-faces-on {0}
set $m43-normalize-vectors {}
set $m43-has_scalar_data {1}
set $m43-edge_display_type {Cylinders}
set $m43-active_tab {Edges}
set $m43-scalars_scale {0.3}
set $m43-show_progress {}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m44-nodes-on {1}
set $m44-edges-on {0}
set $m44-faces-on {0}
set $m44-normalize-vectors {}
set $m44-has_scalar_data {1}
set $m44-node_display_type {Spheres}
set $m44-node_scale {0.02}
set $m44-scalars_scale {0.3}
set $m44-show_progress {}
set $m44-node-resolution {7}

# Set GUI variables for the SCIRun->Visualization->RescaleColorMap Module
set $m47-isFixed {1}
set $m47-min {50}
set $m47-max {13214.7080078}

# Set GUI variables for the SCIRun->Visualization->RescaleColorMap Module
set $m48-min {0.0}
set $m48-max {504.0}

# Set GUI variables for the SCIRun->FieldsOther->ChooseField Module
set $m50-port-index {0}
set $m50-usefirstvalid {0}



# Set GUI variables for the SCIRun->Render->SynchronizeGeometry Module
set $m100-enforce {0}

# Set GUI variables for the SCIRun->Render->Viewer Module
set $m101-ViewWindow_0-view-eyep-x {3.03303338901}
set $m101-ViewWindow_0-view-eyep-y {18.8977389132}
set $m101-ViewWindow_0-view-eyep-z {1.05385048457}
set $m101-ViewWindow_0-view-lookat-x {0.000236034393311}
set $m101-ViewWindow_0-view-lookat-y {-0.185139656067}
set $m101-ViewWindow_0-view-lookat-z {0.00331425666809}
set $m101-ViewWindow_0-view-up-x {-0.140932556522}
set $m101-ViewWindow_0-view-up-y {-0.0235129816983}
set $m101-ViewWindow_0-view-up-z {0.989739942714}
set $m101-ViewWindow_0-view-fov {20.0}
set $m101-ViewWindow_0-view-eyep_offset-x {}
set $m101-ViewWindow_0-view-eyep_offset-y {}
set $m101-ViewWindow_0-view-eyep_offset-z {}
set $m101-ViewWindow_0-sr {1}
set $m101-ViewWindow_0-do_stereo {0}
set $m101-ViewWindow_0-ortho-view {0}
set $m101-ViewWindow_0-trackViewWindow0 {1}
set $m101-ViewWindow_0-raxes {0}
set $m101-ViewWindow_0-ambient-scale {1.0}
set $m101-ViewWindow_0-diffuse-scale {1.0}
set $m101-ViewWindow_0-specular-scale {1.0}
set $m101-ViewWindow_0-emission-scale {1.0}
set $m101-ViewWindow_0-shininess-scale {1.0}
set $m101-ViewWindow_0-polygon-offset-factor {1.0}
set $m101-ViewWindow_0-polygon-offset-units {0.0}
set $m101-ViewWindow_0-point-size {1.0}
set $m101-ViewWindow_0-line-width {1.0}
set $m101-ViewWindow_0-sbase {0.4}
set $m101-ViewWindow_0-bgcolor-r {0}
set $m101-ViewWindow_0-bgcolor-g {0}
set $m101-ViewWindow_0-bgcolor-b {0}
set $m101-ViewWindow_0-fogusebg {1}
set $m101-ViewWindow_0-fogcolor-r {0.0}
set $m101-ViewWindow_0-fogcolor-g {0.0}
set $m101-ViewWindow_0-fogcolor-b {1.0}
set $m101-ViewWindow_0-fog-start {0.0}
set $m101-ViewWindow_0-fog-end {0.714265}
set $m101-ViewWindow_0-currentvisual {0}
set $m101-ViewWindow_0-caxes {0}
set $m101-ViewWindow_0-pos {y1_z1}
set $m101-ViewWindow_0-clip-num {}
set $m101-ViewWindow_0-clip-visible {}
set $m101-ViewWindow_0-clip-selected {}
set $m101-ViewWindow_0-clip-visible-1 {}
set $m101-ViewWindow_0-clip-normal-x-1 {}
set $m101-ViewWindow_0-clip-normal-y-1 {}
set $m101-ViewWindow_0-clip-normal-z-1 {}
set $m101-ViewWindow_0-clip-normal-d-1 {}
set $m101-ViewWindow_0-clip-visible-2 {}
set $m101-ViewWindow_0-clip-normal-x-2 {}
set $m101-ViewWindow_0-clip-normal-y-2 {}
set $m101-ViewWindow_0-clip-normal-z-2 {}
set $m101-ViewWindow_0-clip-normal-d-2 {}
set $m101-ViewWindow_0-clip-visible-3 {}
set $m101-ViewWindow_0-clip-normal-x-3 {}
set $m101-ViewWindow_0-clip-normal-y-3 {}
set $m101-ViewWindow_0-clip-normal-z-3 {}
set $m101-ViewWindow_0-clip-normal-d-3 {}
set $m101-ViewWindow_0-clip-visible-4 {}
set $m101-ViewWindow_0-clip-normal-x-4 {}
set $m101-ViewWindow_0-clip-normal-y-4 {}
set $m101-ViewWindow_0-clip-normal-z-4 {}
set $m101-ViewWindow_0-clip-normal-d-4 {}
set $m101-ViewWindow_0-clip-visible-5 {}
set $m101-ViewWindow_0-clip-normal-x-5 {}
set $m101-ViewWindow_0-clip-normal-y-5 {}
set $m101-ViewWindow_0-clip-normal-z-5 {}
set $m101-ViewWindow_0-clip-normal-d-5 {}
set $m101-ViewWindow_0-clip-visible-6 {}
set $m101-ViewWindow_0-clip-normal-x-6 {}
set $m101-ViewWindow_0-clip-normal-y-6 {}
set $m101-ViewWindow_0-clip-normal-z-6 {}
set $m101-ViewWindow_0-clip-normal-d-6 {}
set $m101-ViewWindow_0-global-light0 {1}
set $m101-ViewWindow_0-global-light1 {0}
set $m101-ViewWindow_0-global-light2 {0}
set $m101-ViewWindow_0-global-light3 {0}
set $m101-ViewWindow_0-lightColors {{1.0 1.0 1.0} {0.75 0.75 0.75} {0.75 0.75 0.75} {1.0 1.0 1.0}}
set $m101-ViewWindow_0-lightVectors {{ 0 0 1 } {-0.888888888889 0.0444444444444 0.455961878416} {0.888888888889 0.0222222222222 0.457583561822} {-0.133333333333 0.355555555556 0.925095924289}}
set $m101-ViewWindow_0-global-light {1}
set $m101-ViewWindow_0-global-fog {0}
set $m101-ViewWindow_0-global-debug {0}
set $m101-ViewWindow_0-global-clip {1}
set $m101-ViewWindow_0-global-cull {0}
set $m101-ViewWindow_0-global-dl {0}
set $m101-ViewWindow_0-global-type {Gouraud}
set "$m101-ViewWindow_0-SampleField Rake (2)" {1}
set "$m101-ViewWindow_0--step_0004100-T_e:Scalar Transparent Faces (1) (1)" {1}
set "$m101-ViewWindow_0--step_0004100-T_e:Scalar Edges (2) (1)" {1}
set "$m101-ViewWindow_0--step_0004100-T_e:Scalar Edges (3) (1)" {1}
set "$m101-ViewWindow_0-Edges (4) (1)" {1}
set "$m101-ViewWindow_0-Nodes (5) (1)" {1}


#######################################################################
#######################################################################


::netedit scheduleok


# global array indexed by module name to keep track of modules
global mods

set mods(HDF5-Grid) $m99
set mods(HDF5-Scalar) $m1
set mods(HDF5-Vector) $m2

set mods(MDSPlus-Grid) $m3
set mods(MDSPlus-Scalar) $m4
set mods(MDSPlus-Vector) $m5

set mods(ChooseNrrd-Grid) $m6

set mods(NrrdToField-Scalar) $m11
set mods(NrrdToField-Vector) $m12


set mods(SubSample) $m20
set mods(Slicer-Low) $m21
set mods(Slicer-High) $m22

set mods(Isosurface-Surface) $m23
set mods(Isosurface-Contour-Low) $m24
set mods(Isosurface-Contour-High) $m25

set mods(ShowField-Isosurface-Surface) $m27
set mods(ShowField-Isosurface-Contour) $m28


set mods(StreamLines-rake) $m40
set mods(StreamLines) $m41

set mods(ShowField-StreamLines-Vector) $m43
set mods(ShowField-StreamLines-Scalar) $m44

set mods(ChooseField-Interpolate) $m50


set mods(ColorMap-Isosurfaces) $m29
set mods(ColorMap-Streamlines) $m45
set mods(ColorMap-Other) $m46

set mods(RescaleColorMap-Isosurfaces) $m30
set mods(RescaleColorMap-Streamlines) $m47
set mods(RescaleColorMap-Other) $m48

set mods(Viewer) $m101

#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

setProgressText "Creating FusionViewer GUI..."

set auto_index(::PowerAppBase) "source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/PowerAppBase.app"

class FusionViewerApp {
    inherit ::PowerAppBase
    
    method appname {} {
	return "FusionViewer"
    }
    
    constructor {} {
	toplevel .standalone
	wm title .standalone "FusionViewer"	 
	set win .standalone
	
	set viewer_width 800
	set viewer_height 800
	
	set notebook_width 290
	set notebook_height [expr $viewer_height - 160]
	
	set vis_width [expr $notebook_width + 60]
	set vis_height $viewer_height

        set initialized 0


        set i_width 300
        set i_height 20
        set stripes 10

        set vis_frame_tab1 ""
        set vis_frame_tab2 ""
	set c_left_tab ""
     
        set error_module ""

        set data_tab1 ""
        set data_tab2 ""

        set animate_tab1 ""
        set animate_tab2 ""

        # colormaps
        set colormap_width 100
        set colormap_height 15
        set colormap_res 64

        set indicatorID 0

	### Define Tooltips
	##########################
	# General
	global tips

	# Data Acquisition Tab
        set tips(Execute-DataAcquisition) "Select to execute the\nData Acquisition step"
	set tips(Next-DataAcquisition) "Select to proceed to\nthe Registration step"

	global filename_grid filename_scalar filename_vector

	set filename_grid   "No Data Selected"
	set filename_scalar "No Data Selected"
	set filename_vector "No Data Selected"

	global shot_grid shot_scalar shot_vector

	set shot_grid   "No Data Selected"
	set shot_scalar "No Data Selected"
	set shot_vector "No Data Selected"

	set valid_scalar -1
	set valid_vector -1
    }
    

    destructor {
	destroy $this
    }

    
    method build_app {} {
	global mods
	
	# Embed the Viewer
	set eviewer [$mods(Viewer) ui_embedded]
	$eviewer setWindow $win.viewer $viewer_width $viewer_height
	#set_dataset 0

	### Menu
	build_menu $win


	### Vis Part
	#####################
	### Create a Detached Vis Part
	toplevel $win.detachedV
	frame $win.detachedV.f -relief flat
	pack $win.detachedV.f -side left -anchor n

	wm title $win.detachedV "Visualization Window"

	wm sizefrom $win.detachedV user
	wm positionfrom $win.detachedV user
	
	wm withdraw $win.detachedV

	### Create Attached Vis Part
	frame $win.attachedV
	frame $win.attachedV.f -relief flat
	pack $win.attachedV.f -side left -anchor n -fill both

	set IsVAttached 1

	### set frame data members
	set detachedVFr $win.detachedV
	set attachedVFr $win.attachedV
	
	init_Vframe $detachedVFr.f 0
	init_Vframe $attachedVFr.f 1

	# call back to re-configure isosurface slider
	global $mods(Isosurface-Surface)-isoval-max
	trace variable $mods(Isosurface-Surface)-isoval-max w "$this set_minmax_callback"

	global $mods(HDF5-Grid)-filename
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename
	trace variable $mods(HDF5-Grid)-filename w "$this update_filename_callback"
	trace variable $mods(HDF5-Scalar)-filename w "$this update_filename_callback"
	trace variable $mods(HDF5-Vector)-filename w "$this update_filename_callback"

	global $mods(MDSPlus-Grid)-num-entries
	global $mods(MDSPlus-Scalar)-num-entries
	global $mods(MDSPlus-Vector)-num-entries
	trace variable $mods(MDSPlus-Grid)-num-entries w "$this update_entries_callback"
	trace variable $mods(MDSPlus-Scalar)-num-entries w "$this update_entries_callback"
	trace variable $mods(MDSPlus-Vector)-num-entries w "$this update_entries_callback"

	global $mods(HDF5-Scalar)-animate
	global $mods(HDF5-Vector)-animate
	trace variable $mods(HDF5-Scalar)-animate w "$this update_animate_callback"
	trace variable $mods(HDF5-Vector)-animate w "$this update_animate_callback"

	global $mods(HDF5-Scalar)-current
	global $mods(HDF5-Vector)-current
	trace variable $mods(HDF5-Scalar)-current w "$this update_current_callback"
	trace variable $mods(HDF5-Vector)-current w "$this update_current_callback"

	global $mods(Slicer-High)-i-dim
	global $mods(Slicer-High)-j-dim
	global $mods(Slicer-High)-k-dim

	trace variable $mods(Slicer-High)-i-dim w "$this update_slicer_callback"
	trace variable $mods(Slicer-High)-j-dim w "$this update_slicer_callback"
	trace variable $mods(Slicer-High)-k-dim w "$this update_slicer_callback"

	### pack 3 frames
	pack $win.viewer $attachedVFr -side left \
	    -anchor n -fill both -expand 1

	set total_width [expr $viewer_width + $vis_width]

	set total_height $viewer_height

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]
	set pos_y [expr [expr $screen_height / 2] - [expr $total_height / 2]]

	append geom $total_width x $total_height + $pos_x + $pos_y
	wm geometry .standalone $geom
	update	

	$vis_frame_tab1 select "Data Selection"
	$vis_frame_tab2 select "Data Selection"

	$data_tab1 select "HDF5"
	$data_tab2 select "HDF5"

	$vis_tab1 select "Scalar Fields"
	$vis_tab2 select "Scalar Fields"

	$animate_tab1 select "Scalar"
	$animate_tab2 select "Scalar"

	$color_tab1 select "Isosurfaces"
	$color_tab2 select "Isosurfaces"

	set show_contours 1
	set show_integration 1

        set initialized 1

	global $mods(Isosurface-Surface)-active-isoval-selection-tab
	change_iso_tab [set $mods(Isosurface-Surface)-active-isoval-selection-tab]

	update_filename_callback 0 0 0
	update_entries_callback 0 0 0
	update_animate_callback 0 0 0
	update_current_callback 0 0 0

	global PowerAppSession
	if {[info exists PowerAppSession] && [set PowerAppSession] != ""} { 
	    set saveFile $PowerAppSession
	    wm title .standalone "FusionViewer - [getFileName $saveFile]"
	    $this load_session
	} 

    }

    method set_dataset { andexec } {
	global mods
	global DATADIR
	global DATASET

	source $DATADIR/$DATASET/$DATASET.settings

	#Fix up global scale.
	global global_scale

	global $mods(ShowField-StreamLines-Scalar)-node_scale
	global $mods(ShowField-StreamLines-Vector)-edge_scale
	global $mods(StreamLines)-stepsize
	global $mods(StreamLines)-tolerance

	set $mods(ShowField-StreamLines-Scalar)-node_scale [expr 0.01 * ${global-scale}]
	set $mods(ShowField-StreamLines-Vector)-edge_scale [expr 0.01 * ${global-scale}]
	set $mods(StreamLines)-stepsize [expr 0.004 * ${global-scale}]
	set $mods(StreamLines)-tolerance [expr 0.004 * ${global-scale}]

	global $mods(Viewer)-ViewWindow_0-view-eyep-x
	global $mods(Viewer)-ViewWindow_0-view-eyep-y
	global $mods(Viewer)-ViewWindow_0-view-eyep-z
	global $mods(Viewer)-ViewWindow_0-view-lookat-x
	global $mods(Viewer)-ViewWindow_0-view-lookat-y
	global $mods(Viewer)-ViewWindow_0-view-lookat-z
	global $mods(Viewer)-ViewWindow_0-view-up-x
	global $mods(Viewer)-ViewWindow_0-view-up-y
	global $mods(Viewer)-ViewWindow_0-view-up-z
	global $mods(Viewer)-ViewWindow_0-view-fov

	set $mods(Viewer)-ViewWindow_0-view-eyep-x ${view-eyep-x}
	set $mods(Viewer)-ViewWindow_0-view-eyep-y ${view-eyep-y}
	set $mods(Viewer)-ViewWindow_0-view-eyep-z ${view-eyep-z}
	set $mods(Viewer)-ViewWindow_0-view-lookat-x ${view-lookat-x}
	set $mods(Viewer)-ViewWindow_0-view-lookat-y ${view-lookat-y}
	set $mods(Viewer)-ViewWindow_0-view-lookat-z ${view-lookat-z}
	set $mods(Viewer)-ViewWindow_0-view-up-x ${view-up-x}
	set $mods(Viewer)-ViewWindow_0-view-up-y ${view-up-y}
	set $mods(Viewer)-ViewWindow_0-view-up-z ${view-up-z}
	set $mods(Viewer)-ViewWindow_0-view-fov ${view-fov}

	if {$andexec} { $this execute_Data }
    }


    method update_slicer_callback {varname varele varop} {
	global mods

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $mods(Slicer-High)-$index-dim
	    global $mods(Slicer-High)-$index-index
	    global $mods(Slicer-High)-$index-index2

	    set $mods(Slicer-High)-$index-index  [expr [set $mods(Slicer-High)-$index-dim] - 1]
	    set $mods(Slicer-High)-$index-index2 [expr [set $mods(Slicer-High)-$index-dim] - 1]
	}
    }


    method update_filename_callback {varname varele varop} {
	global mods
	global $mods(HDF5-Grid)-filename
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename

	global filename_grid
	global filename_scalar
	global filename_vector

	set tmp [set $mods(HDF5-Grid)-filename]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set filename_grid [string range $tmp $pos end]
	} else {
	    set filename_grid $tmp
	}

	if { [string length $filename_grid] == 0 } {
	    set filename_grid "No Data Selected"
	}

	set tmp [set $mods(HDF5-Scalar)-filename]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set filename_scalar [string range $tmp $pos end]
	} else {
	    set filename_scalar $tmp
	}

	if { [string length $filename_scalar] == 0 } {
	    set filename_scalar "No Data Selected"
	}

	set tmp [set $mods(HDF5-Vector)-filename]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set filename_vector [string range $tmp $pos end]
	} else {
	    set filename_vector $tmp
	}

	if { [string length $filename_vector] == 0 } {
	    set filename_vector "No Data Selected"
	}

	update_animate_callback 0 0 0
	update_module_status
    }


    method update_animate_callback {varname varele varop} {

	global mods
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename
	global $mods(HDF5-Scalar)-animate
	global $mods(HDF5-Vector)-animate

        if {$initialized != 0} {
	    if { [string length [set $mods(HDF5-Scalar)-filename]] &&
		 [set $mods(HDF5-Scalar)-animate] == 1 } {
		foreach w [winfo children $animate_scalar_tab1] {
		    enable_widget $w
		}
		foreach w [winfo children $animate_scalar_tab2] {
		    enable_widget $w
		}
	    } else {
		foreach w [winfo children $animate_scalar_tab1] {
		    disable_widget $w
		}
		foreach w [winfo children $animate_scalar_tab2] {
		    disable_widget $w
		}
	    }

	    if { [string length [set $mods(HDF5-Vector)-filename]] &&
		 [set $mods(HDF5-Vector)-animate] == 1 } {
		foreach w [winfo children $animate_vector_tab1] {
		    enable_widget $w
		}
		foreach w [winfo children $animate_vector_tab2] {
		    enable_widget $w
		}
	    } else {
		foreach w [winfo children $animate_vector_tab1] {
		    disable_widget $w
		}
		foreach w [winfo children $animate_vector_tab2] {
		    disable_widget $w
		}
	    }
	}
    }

    method update_current_callback {varname varele varop} {

	global mods
	global $mods(HDF5-Scalar)-current
	global $mods(HDF5-Vector)-current

	global animate_lock

        if { $animate_lock != 0 } {
	    if { [string first "$mods(HDF5-Scalar)-current" "$varname"] != -1 } {
		if { [set $mods(HDF5-Vector)-current] !=
		     [set $mods(HDF5-Scalar)-current] } {
		    set $mods(HDF5-Vector)-current [set $mods(HDF5-Scalar)-current]
		}
	    } elseif { [string first "$mods(HDF5-Vector)-current" "$varname"] != -1 } {
		if { [set $mods(HDF5-Scalar)-current] !=
		     [set $mods(HDF5-Vector)-current] } {
		    set $mods(HDF5-Scalar)-current [set $mods(HDF5-Vector)-current]
		}
	    }
	}
    }

    method update_entries_callback {varname varele varop} {
	global mods
	global $mods(MDSPlus-Grid)-shot
	global $mods(MDSPlus-Scalar)-shot
	global $mods(MDSPlus-Vector)-shot

	global shot_grid
	global shot_scalar
	global shot_vector

	set tmp [set $mods(MDSPlus-Grid)-shot]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set shot_grid [string range $tmp $pos end]
	} else {
	    set shot_grid $tmp
	}

	if { [string length $shot_grid] == 0 } {
	    set shot_grid "No Data Selected"
	}

	set tmp [set $mods(MDSPlus-Scalar)-shot]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set shot_scalar [string range $tmp $pos end]
	} else {
	    set shot_scalar $tmp
	}

	if { [string length $shot_scalar] == 0 } {
	    set shot_scalar "No Data Selected"
	}

	set tmp [set $mods(MDSPlus-Vector)-shot]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set shot_vector [string range $tmp $pos end]
	} else {
	    set shot_vector $tmp
	}

	if { [string length $shot_vector] == 0 } {
	    set shot_vector "No Data Selected"
	}

	update_module_status
    }

    method update_module_status { } {
	global mods 

	global $mods(HDF5-Grid)-filename
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename

	global $mods(MDSPlus-Grid)-num-entries
	global $mods(MDSPlus-Scalar)-num-entries
	global $mods(MDSPlus-Vector)-num-entries


	# This must be before the the next checks otherwise the modules will
	# back on.
	if { [string length [set $mods(HDF5-Grid)-filename]] == 0 &&
	     [set $mods(MDSPlus-Grid)-num-entries] == 0 } {
	    disableModule $mods(ChooseNrrd-Grid) 1
	} else {
	    disableModule $mods(ChooseNrrd-Grid) 0
	}

	if { [string length [set $mods(HDF5-Grid)-filename]] == 0 } {
	    disableModule $mods(HDF5-Grid) 1
	} else {
	    disableModule $mods(HDF5-Grid) 0
	}

	if { [set $mods(MDSPlus-Grid)-num-entries] == 0 } {
	    disableModule $mods(MDSPlus-Grid) 1
	} else {
	    disableModule $mods(MDSPlus-Grid) 0
	}


	if { [string length [set $mods(HDF5-Scalar)-filename]] == 0 } {
	    disableModule $mods(HDF5-Scalar) 1
	} else {
	    disableModule $mods(HDF5-Scalar) 0
	}

	if { [set $mods(MDSPlus-Scalar)-num-entries] == 0 } {
	    disableModule $mods(MDSPlus-Scalar) 1
	} else {
	    disableModule $mods(MDSPlus-Scalar) 0
	}

	if { [string length [set $mods(HDF5-Scalar)-filename]] == 0 &&
	     [set $mods(MDSPlus-Scalar)-num-entries] == 0 } {

	    set valid_scalar 0

	    global $mods(ShowField-Isosurface-Surface)-faces-on
	    set $mods(ShowField-Isosurface-Surface)-faces-on 0

	    toggle_isosurface

	    disable_widget $isosurfaces_frame1.show
	    disable_widget $isosurfaces_frame2.show

	    disableModule $mods(NrrdToField-Scalar) 1
	    disableModule $mods(SubSample) 1

	    global $mods(ChooseField-Interpolate)-port-index
	    set $mods(ChooseField-Interpolate)-port-index 1

	    disable_widget $streamlines_frame1.cm.l
	    disable_widget $streamlines_frame1.cm.scalar
	    disable_widget $streamlines_frame1.cm.vector

	    disable_widget $streamlines_frame2.cm.l
	    disable_widget $streamlines_frame2.cm.scalar
	    disable_widget $streamlines_frame2.cm.vector

	} else {

	    if {$valid_scalar == 0} {

		set valid_scalar 1
		
		global $mods(ShowField-Isosurface-Surface)-faces-on
		set $mods(ShowField-Isosurface-Surface)-faces-on 1

		toggle_isosurface

		disableModule $mods(NrrdToField-Scalar) 0
		disableModule $mods(SubSample) 0

		enable_widget $streamlines_frame1.cm.l
		enable_widget $streamlines_frame1.cm.scalar
		enable_widget $streamlines_frame1.cm.vector

		enable_widget $streamlines_frame2.cm.l
		enable_widget $streamlines_frame2.cm.scalar
		enable_widget $streamlines_frame2.cm.vector
	    }
	}


	if { [string length [set $mods(HDF5-Vector)-filename]] == 0 } {
	    disableModule $mods(HDF5-Vector) 1
	} else {
	    disableModule $mods(HDF5-Vector) 0
	}

	if { [set $mods(MDSPlus-Vector)-num-entries] == 0 } {
	    disableModule $mods(MDSPlus-Vector) 1
	} else {
	    disableModule $mods(MDSPlus-Vector) 0
	}

	if { [string length [set $mods(HDF5-Vector)-filename]] == 0 &&
	     [set $mods(MDSPlus-Vector)-num-entries] == 0 } {

	    set valid_vector 0

	    global $mods(ShowField-StreamLines-Vector)-edges-on
	    set $mods(ShowField-StreamLines-Vector)-edges-on 0

	    toggle_streamlines

	    disable_widget $streamlines_frame1.show
	    disable_widget $streamlines_frame2.show

	    disableModule $mods(NrrdToField-Vector) 1
	    disableModule $mods(ChooseField-Interpolate) 1

	} else {

	    if {$valid_vector == 0} {

		set valid_vector 1

		global $mods(ShowField-StreamLines-Vector)-edges-on
		set $mods(ShowField-StreamLines-Vector)-edges-on 1

		toggle_streamlines

		disableModule $mods(NrrdToField-Vector) 0
		disableModule $mods(ChooseField-Interpolate) 0
	    }
	}
    }


    method init_Vframe { m case} {
	global mods tips
	if { [winfo exists $m] } {
	    ### Visualization Frame
	    
	    iwidgets::labeledframe $m.vis \
		-labelpos n -labeltext "Visualization" 
	    pack $m.vis -side right -anchor n 
	    
	    set vis [$m.vis childsite]
	    
	    ### Tabs
	    iwidgets::tabnotebook $vis.tnb -width $notebook_width \
		-height $notebook_height -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            if {$case == 0} {
		set vis_frame_tab1 $vis.tnb
            } else {
		set vis_frame_tab2 $vis.tnb	    
            }


	    ### Data Tab
	    set data [$vis.tnb add -label "Data Selection" \
			  -command "$this change_vis_frame 0"]

	    ### Data Source Frame
	    iwidgets::labeledframe $data.source -labelpos nw \
		-labeltext "Data Source" 
	    
	    set source [$data.source childsite]

	    build_data_source_frame $source $case

	    pack $data.source -padx 4 -pady 4 -fill x 
	    

	    ### Data Animate Frame
	    iwidgets::labeledframe $data.animate -labelpos nw \
		-labeltext "Data Animation" 

	    set animate [$data.animate childsite]

	    build_animate_frame $animate $case

	    pack $data.animate -padx 4 -pady 4 -fill x 


	    ### Data Subsample Frame
	    iwidgets::labeledframe $data.subsample -labelpos nw \
		-labeltext "Data Subsample" 

	    set subsample [$data.subsample childsite]

	    build_subsample_frame $subsample $case
	    
            pack $data.subsample -padx 4 -pady 4 -fill x

            if {$case == 0} {
		set subsample_frame1 $subsample
            } else {
		set subsample_frame2 $subsample
            }



	    ### Vis Options Tab
	    set page [$vis.tnb add -label "Vis Options" \
			  -command "$this change_vis_frame 1"]



	    ### Tabs
	    iwidgets::tabnotebook $page.tnb -width $notebook_width \
		-height [expr $notebook_height-100] -tabpos n
	    pack $page.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            if {$case == 0} {
		set vis_tab1 $page.tnb
            } else {
		set vis_tab2 $page.tnb	    
            }


	    set scalar [$page.tnb add -label "Scalar Fields" \
			    -command "$this change_option_tab 0"]

            if {$case == 0} {
		set vis_scalar_tab1 $scalar
            } else {
		set vis_scalar_tab2 $scalar	    
            }


	    ### Vis Isosurface
	    iwidgets::labeledframe $scalar.isoframe -labelpos nw \
		-labeltext "Isosurface"

	    set iso [$scalar.isoframe childsite]
	    
	    build_isosurface_frame $iso $case
	     
            pack $scalar.isoframe -padx 4 -pady 4 -fill x

            if {$case == 0} {
		set isosurfaces_frame1 $iso
            } else {
		set isosurfaces_frame2 $iso
            }
	    


	    set vector [$page.tnb add -label "Vector Fields" \
			    -command "$this change_option_tab 1"]

            if {$case == 0} {
		set vis_vector_tab1 $vector
            } else {
		set vis_vector_tab2 $vector	    
            }


	    ### Vis StreamLines
	    iwidgets::labeledframe $vector.slframe -labelpos nw \
		-labeltext "Field Lines"

	    set sl [$vector.slframe childsite]
	    
	    build_streamlines_frame $sl
	    
            pack $vector.slframe -padx 4 -pady 4 -fill x

            if {$case == 0} {
		set streamlines_frame1 $sl
            } else {
		set streamlines_frame2 $sl
            }



	    set misc [$page.tnb add -label "Misc" \
			  -command "$this change_option_tab 2"]

            if {$case == 0} {
		set vis_misc_tab1 $misc
            } else {
		set vis_misc_tab2 $misc	    
            }

	    ### ColorMaps
	    iwidgets::labeledframe $misc.colorframe -labelpos nw \
		-labeltext "Color Maps"

	    set color [$misc.colorframe childsite]
	    
	    build_colormap_frame $color $case
	    
            pack $misc.colorframe -padx 4 -pady 4 -fill x

	    
	    ### Renderer Options Tab
	    create_viewer_tab $vis
	    
	    
	    # Execute Button
            frame $vis.last
            pack $vis.last -side bottom -anchor ne \
		-padx 5 -pady 5
	    
            button $vis.last.ex -text "Execute" \
		-background $execute_color \
		-activebackground $execute_color \
		-width 8 \
		-command "$this execute_Data"
	    Tooltip $vis.last.ex $tips(Execute-DataAcquisition)

            pack $vis.last.ex -side right -anchor ne \
		-padx 2 -pady 0

	    if {$case == 0} {
		set data_ex_button1 $vis.last.ex
	    } else {
		set data_ex_button2 $vis.last.ex
	    }

            ### Indicator
	    frame $vis.indicator -relief sunken -borderwidth 2
            pack $vis.indicator -side bottom -anchor s -padx 3 -pady 5
	    
	    canvas $vis.indicator.canvas -bg "white" -width $i_width \
	        -height $i_height 
	    pack $vis.indicator.canvas -side top -anchor n -padx 3 -pady 3
	    
            bind $vis.indicator <Button> {app display_module_error} 
	    
            label $vis.indicatorL -text "Press Execute to Load Data..."
            pack $vis.indicatorL -side bottom -anchor sw -padx 5 -pady 3
	    
	    
            if {$case == 0} {
		set indicator1 $vis.indicator.canvas
		set indicatorL1 $vis.indicatorL
            } else {
		set indicator2 $vis.indicator.canvas
		set indicatorL2 $vis.indicatorL
            }
	    
            construct_indicator $vis.indicator.canvas
	    

	    ### Attach/Detach button
            frame $m.d 
	    pack $m.d -side left -anchor e
            for {set i 0} {$i<34} {incr i} {
                button $m.d.cut$i -text " | " -borderwidth 0 \
                    -foreground "gray25" \
                    -activeforeground "gray25" \
                    -command "$this switch_V_frames" 
	        pack $m.d.cut$i -side top -anchor se -pady 0 -padx 0
		if {$case == 0} {
		    Tooltip $m.d.cut$i $tips(VisAttachHashes)
		} else {
		    Tooltip $m.d.cut$i $tips(VisDetachHashes)
		}
            }

	    wm protocol .standalone WM_DELETE_WINDOW { NiceQuit }  
	}
    }
    

# Setting the frame to attached or detached.
    method switch_V_frames {} {
	set c_width [winfo width $win]
	set c_height [winfo height $win]

      	set x [winfo x $win]
	set y [expr [winfo y $win] - 20]

	if { $IsVAttached } {
	    pack forget $attachedVFr
	    set new_width [expr $c_width - $vis_width]
	    append geom1 $new_width x $c_height
            wm geometry $win $geom1
	    set move [expr $c_width - $vis_width]
	    append geom2 $vis_width x $c_height + [expr $x + $move + 20] + $y
	    wm geometry $detachedVFr $geom2
	    wm deiconify $detachedVFr
	    set IsVAttached 0
	} else {
	    wm withdraw $detachedVFr
	    pack $attachedVFr -anchor n -side left -after $win.viewer \
	       -fill both -expand 1
	    set new_width [expr $c_width + $vis_width]
            append geom $new_width x $c_height
	    wm geometry $win $geom
	    set IsVAttached 1
	}
    }

    method save_session {} {
	global mods
	
	if {$saveFile == ""} {
	    
	    set types {
		{{App Settings} {.ses} }
		{{Other} { * } }
	    } 
	    set saveFile [ tk_getSaveFile -defaultextension {.ses} \
			       -filetypes $types ]
	}	

	if { $saveFile != "" } {
	    # configure title
	    wm title .standalone "FusionViewer - [getFileName $saveFile]" 

	    set fileid [open $saveFile w]
	    
	    # Save out data information 
	    puts $fileid "# FusionViewer Session\n"
	    puts $fileid "set app_version 1.0"

	    save_module_variables $fileid
	    save_class_variables $fileid

	    close $fileid

	    global NetworkChanged
	    set NetworkChanged 0
	}
    }


    #########################
    ### save_class_variables
    #########################
    # Save out all of the class variables 
    method save_class_variables { fileid} {
	puts $fileid "\n# Class Variables\n"
	foreach v [info variable] {
	    set var [get_class_variable_name $v]
	    if {$var != "this" } {
		puts $fileid "set $var \{[set $var]\}"
	    }
	}
	puts $fileid "set loading 1"
    }
    
    
    method load_session {} {	
	set types {
	    {{App Settings} {.ses} }
	    {{Other} { * }}
	}
	
	if {$saveFile == ""} {
	    set saveFile [tk_getOpenFile -filetypes $types]
	}
	
	if {$saveFile != ""} {
	    
	    # Reset application 
	    reset_app
	    
	    foreach g [info globals] {
		global $g
	    }
	    
	    source $saveFile
	    

	    # set a few variables that need to be reset
	    set indicate 0
	    set cycle 0
	    set IsVAttached 1
	    set executing_modules 0
	    
	    # configure all tabs by calling all configure functions
	    if {$c_left_tab != ""} {
		$vis_frame_tab1 view $c_left_tab
		$vis_frame_tab2 view $c_left_tab
	    }

	    change_indicator_labels "Press Execute to Load Data..."
	}	
    }

    ##############################
    ### save_image
    ##############################
    # To be filled in by child class. It should save out the
    # viewer image.
    method save_image {} {
	global mods
	$mods(Viewer)-ViewWindow_0 makeSaveImagePopup
    }

    
    method show_help {} {
	tk_messageBox -message "Please refer to the online FusionViewer Tutorial\nhttp://software.sci.utah.edu/doc/User/FusionViewerTutorial" -type ok -icon info -parent .standalone
    }
    
    method show_about {} {
	tk_messageBox -message "FusionViewer About Box" -type ok -icon info -parent .standalone
    }
    

    method indicate_dynamic_compile { which mode } {
	if {$mode == "start"} {
	    change_indicate_val 1
	    change_indicator_labels "Dynamically Compiling Code..."
        } else {
	    change_indicate_val 2
	    
	    change_indicator_labels "Visualization..."
	}
    }
    
    
    method update_progress { which state } {
	global mods
	if  {$which == $mods(ShowField-Isosurface-Surface) && $state == "JustStarted"} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Isosurface-Surface) && $state == "Completed"} {
	    change_indicate_val 2

	} elseif {$which == $mods(ShowField-StreamLines-Vector) && $state == "JustStarted"} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-StreamLines-Vector) && $state == "Completed"} {
	    change_indicate_val 2
	}
    }

    
    method indicate_error { which msg_state } {
	if {$msg_state == "Error"} {
	    if {$error_module == ""} {
		set error_module $which
		# turn progress graph red
		change_indicator_labels "E R R O R !"
		change_indicate_val 3
	    }
	} else {
	    if {$which == $error_module} {
		set error_module ""
		change_indicator_labels "Visualization..."
		change_indicate_val 0
	    }
	}
    }
	
	
    method execute_Data {} {
	global mods 

	global $mods(HDF5-Grid)-filename
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename

	global $mods(MDSPlus-Grid)-num-entries
	global $mods(MDSPlus-Scalar)-num-entries
	global $mods(MDSPlus-Vector)-num-entries

	if { [string length [set $mods(HDF5-Grid)-filename]] != 0 } {
	    $mods(HDF5-Grid)-c needexecute
	}

	if { [string length [set $mods(HDF5-Scalar)-filename]] != 0 } {
	    $mods(HDF5-Scalar)-c needexecute
	}

	if { [string length [set $mods(HDF5-Vector)-filename]] != 0 } {
	    $mods(HDF5-Vector)-c needexecute
	}

	if { [set $mods(MDSPlus-Grid)-num-entries] != 0 } {
	    $mods(MDSPlus-Grid)-c needexecute
	}
	if { [set $mods(MDSPlus-Scalar)-num-entries] != 0 } {
	    $mods(MDSPlus-Scalar)-c needexecute
	}

	if { [set $mods(MDSPlus-Vector)-num-entries] != 0 } {
	    $mods(MDSPlus-Vector)-c needexecute
	}
    }
    

    method build_data_source_frame { f case } {
	global mods tips

	if { [winfo exists $f] } {
	    
	    ### Tabs
	    iwidgets::tabnotebook $f.tnb -width $notebook_width \
		-height 150 -tabpos n
	    pack $f.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            if {$case == 0} {
		set data_tab1 $f.tnb
            } else {
		set data_tab2 $f.tnb	    
            }

	    ### Data HDF5
	    set hdf5 [$f.tnb add -label "HDF5" -command "$this change_data_tab 0"]

            if {$case == 0} {
		set data_hdf5_tab1 $hdf5
            } else {
		set data_hdf5_tab2 $hdf5
            }
	    
	    $mods(HDF5-Grid)   set_power_app "$this update_filename_callback 0 0 0"
	    $mods(HDF5-Scalar) set_power_app "$this update_filename_callback 0 0 0"
	    $mods(HDF5-Vector) set_power_app "$this update_filename_callback 0 0 0"

	    frame $hdf5.grid
	    button $hdf5.grid.button -text " Grid " -command "$mods(HDF5-Grid)   initialize_ui"
	    pack $hdf5.grid.button -side left -anchor nw -padx 3 -pady 3
	    label $hdf5.grid.label -textvariable filename_grid
	    pack $hdf5.grid.label  -side right -anchor nw -padx 3 -pady 3



	    frame $hdf5.scalar
	    button $hdf5.scalar.button -text "Scalar" \
		-command "$mods(HDF5-Scalar) initialize_ui"
	    pack $hdf5.scalar.button -side left -anchor nw -padx 3 -pady 3
	    label $hdf5.scalar.label -textvariable filename_scalar
	    pack $hdf5.scalar.label  -side left -anchor nw -padx 3 -pady 3


	    frame $hdf5.vector
	    button $hdf5.vector.button -text "Vector" \
		-command "$mods(HDF5-Vector) initialize_ui"
	    pack $hdf5.vector.button -side left -anchor nw -padx 3 -pady 3
	    label $hdf5.vector.label -textvariable filename_vector
	    pack $hdf5.vector.label  -side left -anchor nw -padx 3 -pady 3


	    pack $hdf5.grid   -side top -anchor nw -padx 3 -pady 3
	    pack $hdf5.scalar -side top -anchor nw -padx 3 -pady 3
	    pack $hdf5.vector -side top -anchor nw -padx 3 -pady 3


	    ### Data MDSPlus
	    set mdsplus [$f.tnb add -label "MDSPlus" -command "$this change_data_tab 1"]

            if {$case == 0} {
		set data_mdsplus_tab1 $mdsplus
            } else {
		set data_mdsplus_tab2 $mdsplus
            }
	    

	    $mods(MDSPlus-Grid)   set_power_app "$this update_entries_callback 0 0 0"
	    $mods(MDSPlus-Scalar) set_power_app "$this update_entries_callback 0 0 0"
	    $mods(MDSPlus-Vector) set_power_app "$this update_entries_callback 0 0 0"


	    frame $mdsplus.grid
	    button $mdsplus.grid.button -text " Grid " \
		-command "$mods(MDSPlus-Grid)   initialize_ui"
	    pack $mdsplus.grid.button   -side left -anchor nw -padx 3 -pady 3
	    label $mdsplus.grid.label -textvariable shot_grid
	    pack $mdsplus.grid.label   -side right -anchor nw -padx 3 -pady 3

	    frame $mdsplus.scalar
	    button $mdsplus.scalar.button -text "Scalar" \
		-command "$mods(MDSPlus-Scalar) initialize_ui"
	    pack $mdsplus.scalar.button -side left -anchor nw -padx 3 -pady 3
	    label $mdsplus.scalar.label -textvariable shot_scalar
	    pack $mdsplus.scalar.label   -side right -anchor nw -padx 3 -pady 3

	    frame $mdsplus.vector
	    button $mdsplus.vector.button -text "Vector" \
		-command "$mods(MDSPlus-Vector) initialize_ui"
	    pack $mdsplus.vector.button -side left -anchor nw -padx 3 -pady 3
	    label $mdsplus.vector.label -textvariable shot_vector
	    pack $mdsplus.vector.label   -side right -anchor nw -padx 3 -pady 3


	    pack $mdsplus.grid   -side top -anchor nw -padx 3 -pady 3
	    pack $mdsplus.scalar -side top -anchor nw -padx 3 -pady 3
	    pack $mdsplus.vector -side top -anchor nw -padx 3 -pady 3
	}	
    }
    
    method build_subsample_frame { f case } {
	global mods

	$mods(SubSample) set_power_app "$this update_subsample_frame"

	button $f.button -text "SubSample UI" -command "$mods(SubSample) initialize_ui"
	pack $f.button -side left -anchor nw -padx 3 -pady 3
    }

    method update_subsample_frame {} {
    }


    method build_animate_frame { f case } {
	global mods tips

	if { [winfo exists $f] } {
	    
	    ### Tabs
	    iwidgets::tabnotebook $f.tnb -width $notebook_width \
		-height 250 -tabpos n
	    pack $f.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            if {$case == 0} {
		set animate_tab1 $f.tnb
            } else {
		set animate_tab2 $f.tnb	    
            }

	    ### Animate Scalar
	    set scalar [$f.tnb add -label "Scalar" -command "$this change_animate_tab 0"]

            if {$case == 0} {
		set animate_scalar_tab1 $scalar
            } else {
		set animate_scalar_tab2 $scalar
            }

	    $mods(HDF5-Scalar) build_animate_ui $scalar
	    

	    ### Animate Vector
	    set vector [$f.tnb add -label "Vector" -command "$this change_animate_tab 1"]

            if {$case == 0} {
		set animate_vector_tab1 $vector
            } else {
		set animate_vector_tab2 $vector
	    }	    

	    $mods(HDF5-Vector) build_animate_ui $vector

	    ### Animate Lock
	    set lock [$f.tnb add -label "Locking" -command "$this change_animate_tab 2"]

            if {$case == 0} {
		set animate_lock_tab1 $lock
            } else {
		set animate_lock_tab2 $lock
	    }	    

	    global animate_lock
	    checkbutton $lock.show -text "Lock Animation" \
		-variable animate_lock \
		-command "$this update_current_callback $mods(HDF5-Scalar)-current 0 0"
	    pack $lock.show -side top -anchor nw -padx 3 -pady 3

	}
    }


    method toggle_isosurface {} {
	global mods
	global $mods(ShowField-Isosurface-Surface)-faces-on
	global $mods(ShowField-Isosurface-Contour)-edges-on

	if {[set $mods(ShowField-Isosurface-Surface)-faces-on] == 1} {
	    disableModule $mods(Isosurface-Surface) 0
	    disableModule $mods(Isosurface-Contour-Low) 0
	    disableModule $mods(Isosurface-Contour-High) 0

	    foreach w [winfo children $isosurfaces_frame1] {
		enable_widget $w
	    }
	    foreach w [winfo children $isosurfaces_frame2] {
		enable_widget $w
	    }

	    foreach w [winfo children $subsample_frame1] {
		enable_widget $w
	    }
	    foreach w [winfo children $subsample_frame2] {
		enable_widget $w
	    }

	    bind $iso_slider_tab1.isoval.s <ButtonRelease> "$this update_isovals"
	    bind $iso_slider_tab2.isoval.s <ButtonRelease> "$this update_isovals"
	    bind $iso_slider_tab1.isoval.val <Return> "$this update_isovals"
	    bind $iso_slider_tab2.isoval.val <Return> "$this update_isovals"

	    set $mods(ShowField-Isosurface-Contour)-edges-on $show_contours

	} else {
	    disableModule $mods(Isosurface-Surface) 1
	    disableModule $mods(Isosurface-Contour-Low) 1
	    disableModule $mods(Isosurface-Contour-High) 1

	    foreach w [winfo children $isosurfaces_frame1] {
		disable_widget $w
	    }
	    foreach w [winfo children $isosurfaces_frame2] {
		disable_widget $w
	    }

	    foreach w [winfo children $subsample_frame1] {
		disable_widget $w
	    }
	    foreach w [winfo children $subsample_frame2] {
		disable_widget $w
	    }

	    bind $iso_slider_tab1.isoval.s <ButtonRelease> ""
	    bind $iso_slider_tab2.isoval.s <ButtonRelease> ""
	    bind $iso_slider_tab1.isoval.val <Return> ""
	    bind $iso_slider_tab2.isoval.val <Return> ""

	    set show_contours [set $mods(ShowField-Isosurface-Contour)-edges-on]
	    set $mods(ShowField-Isosurface-Contour)-edges-on 0
	}
	
	enable_widget $isosurfaces_frame1.show
	enable_widget $isosurfaces_frame2.show

	$mods(ShowField-Isosurface-Surface)-c toggle_display_faces
	$mods(ShowField-Isosurface-Contour)-c toggle_display_edges
    }

    method update_isovals {} {
	global mods
	global $mods(Isosurface-Surface)-isoval
	global $mods(Isosurface-Contour-Low)-isoval
	global $mods(Isosurface-Contour-High)-isoval

	global $mods(Isosurface-Surface)-isoval-quantity
	global $mods(Isosurface-Contour-Low)-isoval-quantity
	global $mods(Isosurface-Contour-High)-isoval-quantity

	global $mods(Isosurface-Surface)-isoval-list
	global $mods(Isosurface-Contour-Low)-isoval-list
	global $mods(Isosurface-Contour-High)-isoval-list

	set $mods(Isosurface-Contour-Low)-isoval \
	    [set $mods(Isosurface-Surface)-isoval]
	set $mods(Isosurface-Contour-High)-isoval \
	    [set $mods(Isosurface-Surface)-isoval]

	set $mods(Isosurface-Contour-Low)-isoval-quantity \
	    [set $mods(Isosurface-Surface)-isoval-quantity]
	set $mods(Isosurface-Contour-High)-isoval-quantity \
	    [set $mods(Isosurface-Surface)-isoval-quantity]

	set $mods(Isosurface-Contour-Low)-isoval-list \
	    [set $mods(Isosurface-Surface)-isoval-list]
	set $mods(Isosurface-Contour-High)-isoval-list \
	    [set $mods(Isosurface-Surface)-isoval-list]

	$mods(Isosurface-Surface)-c needexecute
	$mods(Isosurface-Contour-Low)-c needexecute
	$mods(Isosurface-Contour-High)-c needexecute
    }

    method build_isosurface_frame { f case } {
	global mods
	global $mods(ShowField-Isosurface-Surface)-faces-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Isosurface" \
		-variable $mods(ShowField-Isosurface-Surface)-faces-on \
		-command "$this toggle_isosurface"
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    

	    ### Tabs
	    iwidgets::tabnotebook $f.tnb -width $notebook_width \
		-height 75 -tabpos n
	    pack $f.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            if {$case == 0} {
		set iso_tab1 $f.tnb
            } else {
		set iso_tab2 $f.tnb	    
            }

	    # Isovalue Slider
	    set slider [$f.tnb add -label "Slider" -command "$this change_iso_tab 0"]

            if {$case == 0} {
		set iso_slider_tab1 $slider
            } else {
		set iso_slider_tab2 $slider
            }

	    frame $slider.isoval
	    
	    global $mods(Isosurface-Surface)-isoval
	    global $mods(Isosurface-Surface)-isoval-min
	    global $mods(Isosurface-Surface)-isoval-max

	    label $slider.isoval.l -text "Isovalue:"
	    scale $slider.isoval.s \
		-from [set $mods(Isosurface-Surface)-isoval-min] \
		-to   [set $mods(Isosurface-Surface)-isoval-max] \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution 1 \
		-variable $mods(Isosurface-Surface)-isoval \
		-showvalue false \
		-orient horizontal

	    bind $slider.isoval.s <ButtonRelease> "$this update_isovals"

	    entry $slider.isoval.val -width 5 -relief flat \
		-textvariable $mods(Isosurface-Surface)-isoval

	    bind $slider.isoval.val <Return> "$this update_isovals"

	    pack $slider.isoval.l $slider.isoval.s $slider.isoval.val \
		-side left -anchor nw -padx 3

	    pack $slider.isoval -side top -anchor nw -padx 3 -pady 3


	    # Isovalue Quantity
	    set quantity [$f.tnb add -label "Quantity" -command "$this change_iso_tab 1"]

            if {$case == 0} {
		set iso_quantity_tab1 $quantity
            } else {
		set iso_quantity_tab2 $quantity
            }


	    frame $quantity.isoquant
	    
	    global $mods(Isosurface-Surface)-isoval-quantity

	    label $quantity.isoquant.l -text "Number of Isovalues:"
	    scale $quantity.isoquant.s \
		-from 1 -to 15 \
		-length 50 -width 15 \
		-sliderlength 15 \
		-resolution 1 \
		-variable $mods(Isosurface-Surface)-isoval-quantity \
		-showvalue false \
		-orient horizontal

	    bind $quantity.isoquant.s <ButtonRelease> "$this update_isovals"

	    entry $quantity.isoquant.val -width 5 -relief flat \
		-textvariable $mods(Isosurface-Surface)-isoval-quantity

	    bind $quantity.isoquant.val <Return> "$this update_isovals"

	    pack $quantity.isoquant.l $quantity.isoquant.s $quantity.isoquant.val \
		-side left -anchor nw -padx 3

	    pack $quantity.isoquant -side top -anchor nw -padx 3 -pady 3


	    # Isovalue List
	    set list [$f.tnb add -label "List" -command "$this change_iso_tab 2"]

            if {$case == 0} {
		set iso_list_tab1 $list
            } else {
		set iso_list_tab2 $list
            }


	    frame $list.isolist
	    
	    global $mods(Isosurface-Surface)-isoval-list

	    label $list.isolist.l -text "List of Isovals:"
	    entry $list.isolist.e -width 40 -text $mods(Isosurface-Surface)-isoval-list
	    bind $list.isolist.e <Return> "$this-c update_isovals"
	    pack $list.isolist.l $list.isolist.e -side left -anchor nw -padx 3 -fill both -expand 1
	    pack $list.isolist -side top -anchor nw -padx 3 -pady 3



	    checkbutton $f.normals -text "Render Smooth Faces" \
		    -variable $mods(ShowField-Isosurface-Surface)-use-normals \
		    -command "$mods(ShowField-Isosurface-Surface)-c rerender_faces"

	    pack $f.normals -side top -anchor w -padx 20

	    checkbutton $f.transparency -text "Render Transparent" \
		    -variable $mods(ShowField-Isosurface-Surface)-use-transparency \
		    -command "$mods(ShowField-Isosurface-Surface)-c rerender_faces"

	    pack $f.transparency -side top -anchor w -padx 20

	    checkbutton $f.isocontours -text "Show Isocontours" \
		-variable $mods(ShowField-Isosurface-Contour)-edges-on \
		-command "$mods(ShowField-Isosurface-Contour)-c toggle_display_edges"
	    pack $f.isocontours -side top -anchor w -padx 20
	}
    }	 

    method set_minmax_callback {varname varele varop} {
	global mods
 	global $mods(Isosurface-Surface)-isoval-min
	global $mods(Isosurface-Surface)-isoval-max
 	set min [set $mods(Isosurface-Surface)-isoval-min]
 	set max [set $mods(Isosurface-Surface)-isoval-max]

	set w $isosurfaces_frame1.isoval.s
 	if [ expr [winfo exists $w] ] {
 	    $w configure -from $min -to $max
 	    $w configure -resolution [expr ($max - $min)/100.]
 	}

	set w $isosurfaces_frame2.isoval.s
 	if [ expr [winfo exists $w] ] {
 	    $w configure -from $min -to $max
 	    $w configure -resolution [expr ($max - $min)/100.]
 	}
    }
	


    method toggle_streamlines {} {
	global mods
	global $mods(ShowField-StreamLines-Vector)-edges-on
	global $mods(ShowField-StreamLines-Scalar)-nodes-on

	if { [set $mods(ShowField-StreamLines-Vector)-edges-on] } {
	    disableModule $mods(StreamLines-rake) 0
	    set "$eviewer-StreamLines rake (5)" 1
	    $eviewer-c redraw

	    foreach w [winfo children $streamlines_frame1] {
		enable_widget $w
	    }
	    foreach w [winfo children $streamlines_frame2] {
		enable_widget $w
	    }

	    bind $streamlines_frame1.seeds.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame2.seeds.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame1.seeds.val <Return> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame2.seeds.val <Return> \
		"$mods(StreamLines-rake)-c needexecute"

	    bind $streamlines_frame1.stepsize.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame2.stepsize.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame1.stepsize.val <Return> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame2.stepsize.val <Return> \
		"$mods(StreamLines-rake)-c needexecute"

	    bind $streamlines_frame1.steps.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame2.steps.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame1.steps.val <Return> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_frame2.steps.val <Return> \
		"$mods(StreamLines-rake)-c needexecute"

	    set $mods(ShowField-StreamLines-Scalar)-nodes-on $show_integration

	} else {
	    disableModule $mods(StreamLines-rake) 1
	    set "$eviewer-StreamLines rake (5)" 0
	    $eviewer-c redraw

	    foreach w [winfo children $streamlines_frame1] {
		disable_widget $w
	    }
	    foreach w [winfo children $streamlines_frame2] {
		disable_widget $w
	    }

	    bind $streamlines_frame1.seeds.s <ButtonRelease> ""
	    bind $streamlines_frame2.seeds.s <ButtonRelease> ""
	    bind $streamlines_frame1.seeds.val <Return> ""
	    bind $streamlines_frame2.seeds.val <Return> ""

	    bind $streamlines_frame1.stepsize.s <ButtonRelease> ""
	    bind $streamlines_frame2.stepsize.s <ButtonRelease> ""
	    bind $streamlines_frame1.stepsize.val <Return> ""
	    bind $streamlines_frame2.stepsize.val <Return> ""

	    bind $streamlines_frame1.steps.s <ButtonRelease> ""
	    bind $streamlines_frame2.steps.s <ButtonRelease> ""
	    bind $streamlines_frame1.steps.val <Return> ""
	    bind $streamlines_frame2.steps.val <Return> ""

	    set show_integration [set $mods(ShowField-StreamLines-Scalar)-nodes-on]
	    set $mods(ShowField-StreamLines-Scalar)-nodes-on 0
	}

	enable_widget $streamlines_frame1.show
	enable_widget $streamlines_frame2.show

	$mods(ShowField-StreamLines-Vector)-c toggle_display_edges
	$mods(ShowField-StreamLines-Scalar)-c toggle_display_nodes

	global $mods(HDF5-Scalar)-filename
	global $mods(MDSPlus-Scalar)-num-entries

	if { [string length [set $mods(HDF5-Scalar)-filename]] == 0 &&
	     [set $mods(MDSPlus-Scalar)-num-entries] == 0 } {
	    disable_widget $streamlines_frame1.cm.l
	    disable_widget $streamlines_frame1.cm.scalar
	    disable_widget $streamlines_frame1.cm.vector

	    disable_widget $streamlines_frame2.cm.l
	    disable_widget $streamlines_frame2.cm.scalar
	    disable_widget $streamlines_frame2.cm.vector

	} else {
	    enable_widget $streamlines_frame1.cm.l
	    enable_widget $streamlines_frame1.cm.scalar
	    enable_widget $streamlines_frame1.cm.vector

	    enable_widget $streamlines_frame2.cm.l
	    enable_widget $streamlines_frame2.cm.scalar
	    enable_widget $streamlines_frame2.cm.vector
	}
    }

    method build_streamlines_frame { f } {
	global mods
	global $mods(ShowField-StreamLines-Vector)-edges-on
	global $mods(ShowField-StreamLines-Scalar)-nodes-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Field Lines" \
		-variable $mods(ShowField-StreamLines-Vector)-edges-on \
		-command "$this toggle_streamlines"
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    # seeds
	    frame $f.seeds
	    pack $f.seeds -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.seeds.l -text "Field Lines:"
	    scale $f.seeds.s -from 1 -to 10 \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution 1 \
		-variable $mods(StreamLines-rake)-maxseeds \
		-showvalue false \
		-orient horizontal
	    
	    bind $f.seeds.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    
	    entry $f.seeds.val -width 3 -relief flat \
		-textvariable $mods(StreamLines-rake)-maxseeds
	    
	    bind $f.seeds.val <Return> "$mods(StreamLines-rake)-c needexecute"

	    pack $f.seeds.l $f.seeds.s $f.seeds.val \
		-side left -anchor n -padx 3      

	    
	    # stepsize
	    frame $f.stepsize
	    pack $f.stepsize -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.stepsize.l -text "Step size:"
	    scale $f.stepsize.s -from .05 -to 10 \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution .05 \
		-variable $mods(StreamLines)-stepsize \
		-showvalue false \
		-orient horizontal
	    
	    bind $f.stepsize.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    
	    entry $f.stepsize.val -width 3 -relief flat \
		-textvariable $mods(StreamLines)-stepsize
	    
	    bind $f.stepsize.val <Return> "$mods(StreamLines-rake)-c needexecute"

	    pack $f.stepsize.l $f.stepsize.s $f.stepsize.val \
		-side left -anchor n -padx 3      


	    # steps
	    frame $f.steps
	    pack $f.steps -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.steps.l -text "Integration Steps:"
	    scale $f.steps.s -from 1 -to 1000 \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution 1 \
		-variable $mods(StreamLines)-maxsteps \
		-showvalue false \
		-orient horizontal
	    
	    bind $f.steps.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    
	    entry $f.steps.val -width 3 -relief flat \
		-textvariable $mods(StreamLines)-maxsteps
	    
	    bind $f.steps.val <Return> "$mods(StreamLines-rake)-c needexecute"

	    pack $f.steps.l $f.steps.s $f.steps.val \
		-side left -anchor n -padx 3      


	    
#	    radiobutton $f.fast -text "Fast Tracking" \
#		-variable $mods(StreamLines)-method -value 5 \
#		-command "$mods(StreamLines-rake)-c needexecute"
#	    radiobutton $f.adapt -text "Adaptive Tracking" \
#		-variable $mods(StreamLines)-method -value 4 \
#		-command "$mods(StreamLines-rake)-c needexecute"

#	    pack $f.fast $f.adapt -side top -anchor w -padx 20

	    frame $f.cm

	    label $f.cm.l -text "Color using:"
	    radiobutton $f.cm.scalar -text "Scalar Values" \
		-variable $mods(ChooseField-Interpolate)-port-index -value 0 \
		-command "$mods(ChooseField-Interpolate)-c needexecute"
	    radiobutton $f.cm.vector -text "Vector Values" \
		-variable $mods(ChooseField-Interpolate)-port-index -value 1 \
		-command "$mods(ChooseField-Interpolate)-c needexecute"

	    pack $f.cm.l $f.cm.scalar $f.cm.vector -side left -anchor w -padx 5

	    pack $f.cm -side top -anchor w -padx 5 -pady 3

	    checkbutton $f.integration -text "Show Scalar Integration Points" \
		-variable $mods(ShowField-StreamLines-Scalar)-nodes-on \
		-command "$mods(ShowField-StreamLines-Scalar)-c toggle_display_nodes"
	    pack $f.integration -side top -anchor w -padx 20 -pady 3
	}
    }


    method build_colormap_frame { f case } {

	global mods

	### Tabs
	iwidgets::tabnotebook $f.tnb -width $notebook_width \
	    -height 300 -tabpos n
	pack $f.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1
	
	if {$case == 0} {
	    set color_tab1 $f.tnb
	} else {
	    set color_tab2 $f.tnb	    
	}
	
	# Isosurface 
	set iso [$f.tnb add -label "Isosurfaces" -command "$this change_color_tab 0"]
	
	if {$case == 0} {
	    set color_isosurfaces_tab1 $iso
	} else {
	    set color_isosurfaces_tab2 $iso
	}

	build_colormap_tab $iso $mods(ColorMap-Isosurfaces) $mods(RescaleColorMap-Isosurfaces)

	# Streamlines 
	set iso [$f.tnb add -label "Streamlines" -command "$this change_color_tab 1"]
	
	if {$case == 0} {
	    set color_streamlines_tab1 $iso
	} else {
	    set color_streamlines_tab2 $iso
	}

	build_colormap_tab $iso $mods(ColorMap-Streamlines) $mods(RescaleColorMap-Streamlines)

	# Other 
	set iso [$f.tnb add -label "Other" -command "$this change_color_tab 2"]
	
	if {$case == 0} {
	    set color_other_tab1 $iso
	} else {
	    set color_other_tab2 $iso
	}

	build_colormap_tab $iso $mods(ColorMap-Other) $mods(RescaleColorMap-Other)
    }

    method build_colormap_tab { f cmapmod rscapmod} {

	if {![winfo exists $f.show]} {

	    iwidgets::labeledframe $f.colormaps -labelpos nw \
		-labeltext "Color Maps"

	    set cmf [$f.colormaps childsite]
	    
	    build_colormap_canvas $cmf $cmapmod "Gray"  0
	    build_colormap_canvas $cmf $cmapmod "Inverse Rainbow"  3
	    build_colormap_canvas $cmf $cmapmod "Rainbow"  2
	    build_colormap_canvas $cmf $cmapmod "Darkhue"  5
	    build_colormap_canvas $cmf $cmapmod "Blackbody" 7
	    build_colormap_canvas $cmf $cmapmod "Blue-to-Red" 17

	    iwidgets::labeledframe $f.rescaling -labelpos nw \
		-labeltext "Color Map Rescalings"

	    set cmrs [$f.rescaling childsite]
	    
	    $rscapmod build_ui $cmrs

	    pack $f.colormaps $f.rescaling -padx 4 -pady 4 -fill x
	}
    }
    
    
    method build_colormap_canvas { f cmapmod cmapname val } {
	set maps $f
	global $cmapmod-mapType

	frame $maps.cm-$val
	pack $maps.cm-$val -side top -anchor nw -padx 3 -pady 1 \
	    -fill x -expand 1
	radiobutton $maps.cm-$val.b -text "$cmapname" \
	    -variable $cmapmod-mapType \
	    -value $val \
	    -command "$cmapmod-c needexecute"
	pack $maps.cm-$val.b -side left -anchor nw -padx 3 -pady 0
	
	frame $maps.cm-$val.f -relief sunken -borderwidth 2
	pack $maps.cm-$val.f -padx 2 -pady 0 -side right -anchor e
	canvas $maps.cm-$val.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	pack $maps.cm-$val.f.canvas -anchor e \
	    -fill both -expand 1

	draw_colormap $cmapname $maps.cm-$val.f.canvas
    }

    method change_vis_frame { which } {
	# change tabs for attached and detached

        if {$initialized != 0} {
	    if {$which == 0} {
		$vis_frame_tab1 view "Data Selection"
		$vis_frame_tab2 view "Data Selection"
		set c_left_tab "Data Selection"
	    } elseif {$which == 1} {
		# Data Vis
		$vis_frame_tab1 view "Vis Options"
		$vis_frame_tab2 view "Vis Options"
		set c_left_tab "Vis Options"
	    } else {
 		$vis_frame_tab1 view "Viewer Options"
 		$vis_frame_tab2 view "Viewer Options"
		set c_left_tab "Viewer Options"
	    }
	}
    }
    

    method change_option_tab { which } {
	# change option tab for attached/detached

	if {$initialized != 0} {
	    if {$which == 0} {
		$vis_tab1 view "Scalar Fields"
		$vis_tab2 view "Scalar Fields"

	    } elseif {$which == 1} {
		$vis_tab1 view "Vector Fields"
		$vis_tab2 view "Vector Fields"

	    } elseif {$which == 2} {
		$vis_tab1 view "Misc"
		$vis_tab2 view "Misc"
	    }
	}
    }

    method change_data_tab { which } {
	# change data tab for attached/detached

	if {$initialized != 0} {
	    if {$which == 0} {
		$data_tab1 view "HDF5"
		$data_tab2 view "HDF5"

	    } elseif {$which == 1} {
		$data_tab1 view "MDSPlus"
		$data_tab2 view "MDSPlus"

	    } elseif {$which == 2} {
		$data_tab1 view "Data Subsample"
		$data_tab2 view "Data Subsample"
	    }
	}
    }


    method change_animate_tab { which } {
	# change animate tab for attached/detached

	if {$initialized != 0} {
	    if {$which == 0} {
		$animate_tab1 view "Scalar"
		$animate_tab2 view "Scalar"

	    } elseif {$which == 1} {
		$animate_tab1 view "Vector"
		$animate_tab2 view "Vector"

	    } elseif {$which == 2} {
		$animate_tab1 view "Locking"
		$animate_tab2 view "Locking"
	    }
	}
    }


    method change_iso_tab { which } {
	# change iso tab for attached/detached

        if {$initialized != 0} {
	    if {$which == 0} {
		$iso_tab1 view "Slider"
		$iso_tab2 view "Slider"

	    } elseif {$which == 1} {
		$iso_tab1 view "Quantity"
		$iso_tab2 view "Quantity"

	    } elseif {$which == 2} {
		$iso_tab1 view "List"
		$iso_tab2 view "List"
	    }

	    global mods
	    global $mods(Isosurface-Surface)-active-isoval-selection-tab
	    global $mods(Isosurface-Contour-Low)-active-isoval-selection-tab
	    global $mods(Isosurface-Contour-High)-active-isoval-selection-tab

	    set $mods(Isosurface-Surface)-active-isoval-selection-tab $which
	    set $mods(Isosurface-Contour-Low)-active-isoval-selection-tab $which
	    set $mods(Isosurface-Contour-High)-active-isoval-selection-tab $which
	}
    }


    method change_color_tab { which } {
	# change tabs for attached and detached

        if {$initialized != 0} {
	    if {$which == 0} {
		$color_tab1 view "Isosurfaces"
		$color_tab2 view "Isosurfaces"
	    } elseif {$which == 1} {
		# Data Vis
		$color_tab1 view "Streamlines"
		$color_tab2 view "Streamlines"
	    } else {
 		$color_tab1 view "Other"
 		$color_tab2 view "Other"
	    }
	}
    }
    
    
    method change_indicate_val { v } {
	# only change an error state if it has been cleared (error_module empty)
	# it will be changed by the indicate_error method when fixed
	if {$indicate != 3 || $error_module == ""} {
	    if {$v == 3} {
		# Error
		set cycle 0
		set indicate 3
		change_indicator
	    } elseif {$v == 0} {
		# Reset
		set cycle 0
		set indicate 0
		change_indicator
	    } elseif {$v == 1} {
		# Start
		set executing_modules [expr $executing_modules + 1]
		set indicate 1
		change_indicator
	    } elseif {$v == 2} {
		# Complete
		set executing_modules [expr $executing_modules - 1]
		if {$executing_modules == 0} {
		    # only change indicator if progress isn't running
		    set indicate 2
		    change_indicator
		} elseif {$executing_modules < 0} {
		    # something wasn't caught, reset
		    set executing_modules 0
		    set indicate 2
		    change_indicator
		}
	    }
	}
    }
    

    method change_indicator_labels { msg } {
	$indicatorL1 configure -text $msg
	$indicatorL2 configure -text $msg
    }
    
    

    # Visualiztion frame tabnotebook
    variable vis_frame_tab1
    variable vis_frame_tab2
    variable c_left_tab

    # Data tabs notebook
    variable data_tab1
    variable data_tab2

    variable data_hdf5_tab1
    variable data_hdf5_tab2

    variable data_mdsplus_tab1
    variable data_mdsplus_tab2

    variable subsample_frame1
    variable subsample_frame2

    # Animate tabs notebook
    variable animate_tab1
    variable animate_tab2

    variable animate_scalar_tab1
    variable animate_scalar_tab2

    variable animate_vector_tab1
    variable animate_vector_tab2

    variable animate_lock_tab1
    variable animate_lock_tab2


    # Vis options tabs notebook
    variable vis_tab1
    variable vis_tab2

    variable vis_scalar_tab1
    variable vis_scalar_tab2

    variable vis_vector_tab1
    variable vis_vector_tab2

    variable vis_misc_tab1
    variable vis_misc_tab2

    # Isosurfaces
    variable isosurfaces_frame1
    variable isosurfaces_frame2

    variable iso_tab1
    variable iso_tab2

    variable iso_slider_tab1
    variable iso_slider_tab2

    variable iso_quantity_tab1
    variable iso_quantity_tab2

    variable iso_list_tab1
    variable iso_list_tab2

    # Streamlines
    variable streamlines_frame1
    variable streamlines_frame2

    # Colormaps
    variable color_tab1
    variable color_tab2

    variable color_isosurfaces_tab1
    variable color_isosurfaces_tab2

    variable color_streamlines_tab1
    variable color_streamlines_tab2

    variable color_other_tab1
    variable color_other_tab2

    # Application placing and size
    variable notebook_width
    variable notebook_height

    variable show_contours
    variable show_integration

    variable valid_scalar
    variable valid_vector
}

FusionViewerApp app

setProgressText "Displaying FusionViewer GUI..."

app build_app

hideProgress


### Bind shortcuts - Must be after instantiation of IsoApp
bind all <Control-s> {
    app save_session
}

bind all <Control-o> {
    app load_session
}

bind all <Control-q> {
    app exit_app
}

bind all <Control-v> {
    global mods
    $mods(Viewer)-ViewWindow_0-c autoview
}
