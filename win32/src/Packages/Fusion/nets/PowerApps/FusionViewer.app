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

global probe_lock
global probe_scalar
global probe_vector

set probe_lock 1
set probe_scalar 1
set probe_vector 1

global slice_direction
set slice_direction 0

#######################################################################
#######################################################################

# Create a DataIO->Readers->HDF5DataReader Module
set m0 [addModuleAtPosition "DataIO" "Readers" "HDF5DataReader" 0 0]

# Create a DataIO->Readers->HDF5DataReader Module
set m1 [addModuleAtPosition "DataIO" "Readers" "HDF5DataReader" 325 0]

# Create a DataIO->Readers->HDF5DataReader Module
set m2 [addModuleAtPosition "DataIO" "Readers" "HDF5DataReader" 725 0]

# Create a DataIO->Readers->HDF5DataReader Module
set m3 [addModuleAtPosition "DataIO" "Readers" "HDF5DataReader" 1675 100]



# Create a DataIO->Readers->MDSPlusDataReader Module
set m4 [addModuleAtPosition "DataIO" "Readers" "MDSPlusDataReader" 25 75]

# Create a DataIO->Readers->MDSPlusDataReader Module
set m5 [addModuleAtPosition "DataIO" "Readers" "MDSPlusDataReader" 350 75]

# Create a DataIO->Readers->MDSPlusDataReader Module
set m6 [addModuleAtPosition "DataIO" "Readers" "MDSPlusDataReader" 750 75]

# Create a DataIO->Readers->MDSPlusDataReader Module
set m7 [addModuleAtPosition "DataIO" "Readers" "MDSPlusDataReader" 1700 175]



# Create a Teem->NrrdData->ChooseNrrd Module
set m8 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 0 175]

# Create a Teem->NrrdData->ChooseNrrd Module
set m9 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 325 175]

# Create a Teem->NrrdData->ChooseNrrd Module
set m10 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 725 175]

# Create a Teem->NrrdData->ChooseNrrd Module
set m11 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 1675 275]



# Create a Teem->Converters->NrrdToField Module
set m12 [addModuleAtPosition "Teem" "Converters" "NrrdToField" 650 500]

# Create a Teem->Converters->NrrdToField Module
set m13 [addModuleAtPosition "Teem" "Converters" "NrrdToField" 1650 400]


# Create a SCIRun->FieldsCreate->Probe Module
set m14 [addModuleAtPosition "SCIRun" "FieldsCreate" "Probe" 950 600]

# Create a SCIRun->FieldsCreate->Probe Module
set m15 [addModuleAtPosition "SCIRun" "FieldsCreate" "Probe" 1950 500]

# Create a SCIRun->FieldsOther->FieldInfo Module
set m16 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldInfo" 250 600]

# Create a SCIRun->FieldsOther->FieldInfo Module
set m17 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldInfo"  1150 500]


# Create a SCIRun->FieldsOther->ChooseField Module
set m18 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 650 700]



# Create a SCIRun->FieldsCreate->FieldSubSample Module
set m20 [addModuleAtPosition "SCIRun" "FieldsCreate" "FieldSubSample" 650 600]

# Create a SCIRun->FieldsCreate->FieldSlicer Module
set m21 [addModuleAtPosition "SCIRun" "FieldsCreate" "FieldSlicer" 850 700]

# Create a SCIRun->FieldsCreate->FieldSlicer Module
set m22 [addModuleAtPosition "SCIRun" "FieldsCreate" "FieldSlicer" 1050 700]


# Create a SCIRun->Visualization->Isosurface Module
set m23 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 650 900]

# Create a SCIRun->Visualization->Isosurface Module
set m24 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 850 900]

# Create a SCIRun->Visualization->Isosurface Module
set m25 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1050 900]


# Create a SCIRun->FieldsCreate->GatherFields Module
set m26 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherFields" 850 1000]

# Create a SCIRun->Visualization->ShowField Module
set m27 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 650 1100]

# Create a SCIRun->Visualization->ShowField Module
set m28 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 850 1100]


# Create a SCIRun->FieldsData->TransformData Module
set m31 [addModuleAtPosition "SCIRun" "FieldsData" "TransformData" 450 600]

# Create a SCIRun->Visualization->Isosurface Module
set m32 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 450 700]

# Create a SCIRun->FieldsData->ApplyInterpMatrix Module
set m33 [addModuleAtPosition "SCIRun" "FieldsData" "ApplyInterpMatrix" 300 800]

# Create a SCIRun->FieldsData->ApplyInterpMatrix Module
set m34 [addModuleAtPosition "SCIRun" "FieldsData" "ApplyInterpMatrix" 300 900]

# Create a SCIRun->FieldsCreate->ClipByFunction Module
set m35 [addModuleAtPosition "SCIRun" "FieldsCreate" "ClipByFunction" 450 800]

# Create a SCIRun->Visualization->Isosurface Module
set m36 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 350 1000]

# Create a SCIRun->FieldsOther->ChooseField Module
set m37 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 300 1100]

# Create a SCIRun->Visualization->ShowField Module
set m38 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 350 1200]



# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m45 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 0 600]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m46 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 0 700]



# Create a SCIRun->FieldsCreate->SampleField Module
set m60 [addModuleAtPosition "SCIRun" "FieldsCreate" "SampleField" 1675 500]

# Create a SCIRun->Visualization->StreamLines Module
set m61 [addModuleAtPosition "SCIRun" "Visualization" "StreamLines" 1650 600]

# Create a SCIRun->FieldsData->DirectInterpolate Module
set m62 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1425 700]

# Create a SCIRun->Visualization->ShowField Module
set m63 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1400 1000]

# Create a SCIRun->Visualization->ShowField Module
set m64 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1650 1000]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m65 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1300 800]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m66 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1750 800]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m67 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1300 900]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m68 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1750 900]

# Create a SCIRun->FieldsData->VectorMagnitude Module
set m69 [addModuleAtPosition "SCIRun" "FieldsData" "VectorMagnitude" 1325 500]

# Create a SCIRun->FieldsOther->ChooseField Module
set m70 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1300 600]



# Create a SCIRun->Render->SynchronizeGeometry Module
set m100 [addModuleAtPosition "SCIRun" "Render" "SynchronizeGeometry" 600 1400]
# Create a SCIRun->Render->Viewer Module
set m101 [addModuleAtPosition "SCIRun" "Render" "Viewer" 900 1600]



# Create the Connections between Modules
set c0 [addConnection $m0 0 $m8 0]
set c1 [addConnection $m1 0 $m9 0]
set c2 [addConnection $m2 0 $m10 0]
set c3 [addConnection $m3 0 $m11 0]

set c4 [addConnection $m4 0 $m8 1]
set c5 [addConnection $m5 0 $m9 1]
set c6 [addConnection $m6 0 $m10 1]
set c7 [addConnection $m7 0 $m11 1]

set c8  [addConnection $m8  0 $m12 0]
set c9  [addConnection $m9  0 $m12 1]
set c10 [addConnection $m10 0 $m12 2]

set c11 [addConnection $m8  0 $m13 0]
set c12 [addConnection $m9  0 $m13 1]
set c13 [addConnection $m11 0 $m13 2]


set c14 [addConnection $m12 0 $m20 0]
set c15 [addConnection $m12 0 $m70 0]

set c16 [addConnection $m13 0 $m60 0]
set c17 [addConnection $m13 0 $m61 0]
set c18 [addConnection $m13 0 $m69 0]

set c19 [addConnection $m12 0 $m14 0]
set c20 [addConnection $m13 0 $m15 0]

set c21 [addConnection $m12 0 $m16 0]
set c22 [addConnection $m13 0 $m17 0]


set c23 [addConnection $m20 0 $m18 0]
set c24 [addConnection $m12 0 $m18 1]
set c25 [addConnection $m12 0 $m31 0]
set c26 [addConnection $m12 0 $m33 0]
set c27 [addConnection $m31 0 $m32 0]
set c28 [addConnection $m32 0 $m33 1]
set c29 [addConnection $m32 2 $m33 2]
set c30 [addConnection $m32 0 $m35 0]
set c31 [addConnection $m33 0 $m34 0]
set c32 [addConnection $m34 0 $m37 0]
set c33 [addConnection $m35 0 $m34 1]
set c34 [addConnection $m35 1 $m34 2]
set c35 [addConnection $m46 0 $m37 1]
set c36 [addConnection $m46 0 $m38 1]
set c37 [addConnection $m34 0 $m36 0]
set c38 [addConnection $m36 0 $m38 0]
set c39 [addConnection $m37 0 $m100 0]
set c40 [addConnection $m38 0 $m100 1]


set c50 [addConnection $m20 0 $m21 0]
set c51 [addConnection $m20 0 $m22 0]
set c52 [addConnection $m18 0 $m23 0]
set c53 [addConnection $m12 0 $m46 1]
set c54 [addConnection $m21 0 $m24 0]
set c55 [addConnection $m22 0 $m25 0]
set c56 [addConnection $m23 0 $m27 0]
set c57 [addConnection $m24 0 $m26 0]
set c58 [addConnection $m25 0 $m26 1]
set c59 [addConnection $m26 0 $m28 0]
set c60 [addConnection $m27 0 $m100 2]
set c61 [addConnection $m28 0 $m100 3]
set c62 [addConnection $m45 0 $m46 0]
set c63 [addConnection $m46 0 $m27 1]
set c64 [addConnection $m46 0 $m28 1]

set c71 [addConnection $m60 0 $m61 1]
set c72 [addConnection $m61 0 $m62 1]
set c73 [addConnection $m61 0 $m68 1]
set c74 [addConnection $m61 0 $m64 0]
set c75 [addConnection $m62 0 $m63 0]
set c76 [addConnection $m63 0 $m100 4]
set c77 [addConnection $m64 0 $m100 5]
set c78 [addConnection $m65 0 $m67 0]
set c79 [addConnection $m66 0 $m68 0]
set c80 [addConnection $m67 0 $m63 1]
set c81 [addConnection $m68 0 $m64 1]
set c82 [addConnection $m69 0 $m70 1]
set c83 [addConnection $m70 0 $m62 0]
set c84 [addConnection $m70 0 $m67 1]

set c100 [addConnection $m100 0 $m101 0]
set c101 [addConnection $m14 0 $m101 1]
set c102 [addConnection $m15 0 $m101 2]
set c103 [addConnection $m60 1 $m101 3]

# Set GUI variables for the DataIO->Readers->HDF5DataReader Module
set $m0-filename "$DATADIR/$DATASET/phi.h5"
set $m0-datasets {{/ GRID X} {/ GRID Y} {/ GRID Z}}
set $m0-dumpname {/tmp/qwall.h5.dump}
set $m0-ports {   0   0   0}
set $m0-ndims {3}
set $m0-0-dim {101}
set $m0-0-count {101}
set $m0-1-dim {61}
set $m0-1-count {61}
set $m0-2-dim {101}
set $m0-2-count {101}

# Set GUI variables for the DataIO->Readers->HDF5DataReader Module
set $m1-filename ""
set $m1-datasets {}
set $m1-dumpname {}
set $m1-ports {}
set $m1-ndims {0}

# Set GUI variables for the DataIO->Readers->HDF5DataReader Module
set $m2-selectable_max {115.0}
set $m2-range_max {115}
set $m2-current {87}
set $m2-execmode {current}
set $m2-filename "$DATADIR/$DATASET/phi.h5"
set $m2-datasets {{/ step_0000000 T_e}}
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


# Set GUI variables for the DataIO->Readers->HDF5DataReader Module
set $m3-selectable_max {115.0}
set $m3-range_max {115}
set $m3-current {87}
set $m3-execmode {current}
set $m3-filename "$DATADIR/$DATASET/phi.h5"
set $m3-datasets {{/ step_0000000 B X}}
set $m3-dumpname {/tmp/phi.h5.dump}
set $m3-ports {   0   0   0   1}
set $m3-ndims {3}
set $m3-animate {1}
set $m3-0-dim {101}
set $m3-0-count {101}
set $m3-1-dim {61}
set $m3-1-count {61}
set $m3-2-dim {101}
set $m3-2-count {101}


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

# Set GUI variables for the DataIO->Readers->MDSPlusDataReader Module
set $m6-server {}
set $m6-tree {}
set $m6-shot {}
set $m6-load-server {}
set $m6-load-tree {}
set $m6-load-shot {}
set $m6-load-signal {}
set $m6-search-server {}
set $m6-search-tree {}
set $m6-search-shot {}
set $m6-search-signal {}

# Set GUI variables for the DataIO->Readers->MDSPlusDataReader Module
set $m7-server {}
set $m7-tree {}
set $m7-shot {}
set $m7-load-server {}
set $m7-load-tree {}
set $m7-load-shot {}
set $m7-load-signal {}
set $m7-search-server {}
set $m7-search-tree {}
set $m7-search-shot {}
set $m7-search-signal {}

# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m8-usefirstvalid {1}

# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m9-usefirstvalid {1}

# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m10-usefirstvalid {1}

# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m11-usefirstvalid {1}



# Set GUI variables for the Teem->Converters->NrrdToField Module
set $m12-datasets {{Points : -GRID-X-Y-Z:Vector} {Connections : (none)} {Data : -step_0004100-T_e:Scalar} {Original Field : (none)} }

# Set GUI variables for the Teem->Converters->NrrdToField Module
set $m13-datasets {{Points : -GRID-X-Y-Z:Vector} {Connections : (none)} {Data : -step_0004100-B-X-Y-Z:Vector} {Original Field : (none)} }


# Set GUI variables for the SCIRun->FieldsCreate->Probe Module
set $m14-locx {0.0}
set $m14-locy {0.0}
set $m14-locz {0.0}
set $m14-value {0}
set $m14-node {[0,0,0]}
set $m14-cell {[0,0,0]}

# Set GUI variables for the SCIRun->FieldsCreate->Probe Module
set $m15-locx {0.0}
set $m15-locy {0.0}
set $m15-locz {0.0}
set $m15-value {[0 0 0]}
set $m15-node {[0,0,0]}
set $m15-cell {[0,0,0]}

# Set GUI variables for the SCIRun->FieldsOther->ChooseField Module
set $m18-port-index {0}
set $m18-usefirstvalid {1}


# Set GUI variables for the SCIRun->FieldsCreate->FieldSubSample Module
set $m20-wrap {1}
set $m20-dims {3}
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
set $m23-isoval-min {60.1251335144}
set $m23-isoval-max {6761.27148438}
set $m23-isoval {1000}
set $m23-isoval-quantity 1
set $m23-isoval-list {1000 4000 7000 12000 13160}
set $m23-build_geom {0}
set $m23-active-isoval-selection-tab {2}
set $m23-active_tab {}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m24-isoval-min {71.6849060059}
set $m24-isoval-max {6749.19091797}
set $m24-isoval {1000}
set $m24-isoval-quantity 1
set $m24-isoval-list {1000 4000 7000 12000 13160}
set $m25-build_geom {0}
set $m24-active-isoval-selection-tab {2}
set $m24-active_tab {}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m25-isoval-min {61.2371101379}
set $m25-isoval-max {6728.31152344}
set $m25-isoval {1000}
set $m25-isoval-quantity 1
set $m25-isoval-list {1000 4000 7000 12000 13160}
set $m25-build_geom {0}
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
set $m27-field-name {Isosurfaces}

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
set $m28-field-name {Isocontours}


# Set GUI variables for the SCIRun->FieldsData->TransformData Module
set $m31-function {result = atan2(x,y);}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m32-isoval-min {-3.1}
set $m32-isoval-max { 3.1}
set $m32-isoval {0.0}
set $m32-isoval-quantity 1
set $m32-isoval-list {0}
set $m32-build_geom {0}
set $m32-active-isoval-selection-tab {0}
set $m32-active_tab {}

# Set GUI variables for the SCIRun->FieldsCreate->ClipByFunction Module
set $m35-clipmode {allnodes}
set $m35-clipfunction {fabs( atan2(x,y) - v) < 1e-2}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m36-isoval-min {110.923950195}
set $m36-isoval-max {140186.9375}
set $m36-isoval {64990}
set $m36-isoval-quantity 1
set $m36-isoval-list {1000 4000 7000 12000 13160}
set $m36-build_geom {0}
set $m36-active-isoval-selection-tab {2}
set $m36-active_tab {}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m37-nodes-on {0}
set $m37-edges-on {0}
set $m37-faces-on {1}
set $m37-use-normals {1}
set $m37-normalize-vectors {}
set $m37-has_scalar_data {1}
set $m37-active_tab {Faces}
set $m37-scalars_scale {0.3}
set $m37-show_progress {}
set $m37-field-name {Slice}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m38-nodes-on {0}
set $m38-edges-on {1}
set $m38-faces-on {0}
set $m38-use-normals {1}
set $m38-normalize-vectors {}
set $m38-has_scalar_data {1}
set $m38-edge_scale 0.15
set $m38-edge_display_type {Lines}
set $m38-active_tab {Faces}
set $m38-scalars_scale {0.3}
set $m38-show_progress {}
set $m38-field-name {Slice}


# Set GUI variables for the SCIRun->Visualization->GenStandardColorMaps Module
set $m45-positionList {{355 2}}
set $m45-nodeList {257}
set $m45-width {390}
set $m45-height {40}
set $m45-gamma {0.0}

# Set GUI variables for the SCIRun->Visualization->RescaleColorMap Module
set $m46-isFixed {1}
set $m46-min {50}
set $m46-max {13214.7080078}




# Set GUI variables for the SCIRun->FieldsCreate->SampleField Module
set $m60-endpoints {1}
set $m60-endpoint0x {-0.854404350943}
set $m60-endpoint0y {1.35035004817}
set $m60-endpoint0z {-0.0364978830565}
set $m60-endpoint1x {-0.864402193716}
set $m60-endpoint1y {1.33871534677}
set $m60-endpoint1z {-0.023278780019}
set $m60-widgetscale {0.112657141788}
set $m60-ringstate {}
set $m60-framestate {}
set $m60-maxseeds {2}
set $m60-autoexecute {0}

# Set GUI variables for the SCIRun->Visualization->StreamLines Module
set $m61-stepsize {0.2}
set $m61-tolerance {1e-05}
set $m61-maxsteps {500}
set $m61-direction {2}
set $m61-method {0}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m63-nodes-on {0}
set $m63-edges-on {1}
set $m63-faces-on {0}
set $m63-normalize-vectors {}
set $m63-has_scalar_data {1}
set $m63-edge_display_type {Cylinders}
set $m63-active_tab {Edges}
set $m63-scalars_scale {0.3}
set $m63-show_progress {}
set $m63-field-name {Fieldlines}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m64-nodes-on {1}
set $m64-edges-on {0}
set $m64-faces-on {0}
set $m64-normalize-vectors {}
set $m64-has_scalar_data {1}
set $m64-node_display_type {Spheres}
set $m64-node_scale {0.02}
set $m64-scalars_scale {0.3}
set $m64-show_progress {}
set $m64-node-resolution {7}
set $m63-field-name {Field Integration}

# Set GUI variables for the SCIRun->Visualization->RescaleColorMap Module
set $m67-isFixed {1}
set $m67-min {50}
set $m67-max {13214.7080078}

# Set GUI variables for the SCIRun->Visualization->RescaleColorMap Module
set $m68-min {0.0}
set $m68-max {500.0}

# Set GUI variables for the SCIRun->FieldsOther->ChooseField Module
set $m70-port-index {0}
set $m70-usefirstvalid {0}



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
set "$m101-ViewWindow_0-Probe Selection Widget (2)" {1}
set "$m101-ViewWindow_0-Probe Selection Widget (3)" {1}
set "$m101-ViewWindow_0-SampleField Rake (4)" {1}
set "$m101-ViewWindow_0--step_0004100-T_e:Scalar Transparent Faces (1) (1)" {1}
set "$m101-ViewWindow_0--step_0004100-T_e:Scalar Edges (2) (1)" {1}
set "$m101-ViewWindow_0-Edges (3) (1)" {1}
set "$m101-ViewWindow_0-Nodes (4) (1)" {1}


#######################################################################
#######################################################################


::netedit scheduleok

global connections

set connections(hdf5_to_cp) $c0
set connections(hdf5_to_cc) $c1
set connections(hdf5_to_cs) $c2
set connections(hdf5_to_cv) $c3

set connections(mds_to_cp) $c4
set connections(mds_to_cc) $c5
set connections(mds_to_cs) $c6
set connections(mds_to_cv) $c7

set connections(cp_to_scalar) $c8
set connections(cp_to_vector) $c11

set connections(cc_to_scalar) $c9
set connections(cc_to_vector) $c12

set connections(cs_to_scalar) $c10
set connections(cv_to_vector) $c13

set connections(scalar_to_subsample)    $c14
set connections(scalar_to_choose)       $c15
set connections(scalar_to_probe)        $c19
set connections(scalar_to_info)         $c21
set connections(scalar_to_choose_iso)   $c24
set connections(scalar_to_transform)    $c25
set connections(scalar_to_matrix)       $c26
set connections(scalar_to_color)        $c53

set connections(subsample_to_choose)      $c23
set connections(subsample_to_slicer_low)  $c50
set connections(subsample_to_slicer_high) $c51


set connections(transform_to_isosurface)   $c27
set connections(isosurface_fld_to_matrix)  $c28
set connections(isosurface_mtx_to_matrix)  $c29
set connections(isosurface_to_clipfuction) $c30
set connections(matrix_to_matrix)          $c31
set connections(matrix_to_choose)          $c32
set connections(clipfuction_fld_to_matrix) $c33
set connections(clipfuction_mtx_to_matrix) $c34

set connections(matrix_to_showfield)       $c32
set connections(matrix_to_isosurface)      $c37
set connections(isosurface_to_showfield)   $c38
set connections(colormap_matrix_to_showfield)       $c35
set connections(colormap_isosurface_to_showfield)   $c36

set connections(choose_to_iso)       $c52
set connections(iso_to_showfield)    $c56
set connections(slicer_low_to_iso)   $c54
set connections(slicer_high_to_iso)  $c55
set connections(iso_low_to_gather)   $c57
set connections(iso_high_to_gather)  $c58
set connections(gather_to_showfield) $c59
set connections(colormap_iso_to_showfield)    $c63
set connections(colormap_gather_to_showfield) $c64



set connections(vector_to_sample)      $c16
set connections(vector_to_streamlines) $c17
set connections(vector_to_magnitude)   $c18
set connections(vector_to_probe)       $c20
set connections(vector_to_info)        $c22


set connections(sample_to_streamlines)       $c71
set connections(streamlines_to_interpolate)  $c72
set connections(streamlines_to_showfield)    $c73
set connections(streamlines_to_rescalecolor) $c74
set connections(interpolate_to_showfield)    $c75
set connections(rescalecolor_streamlines_to_showfield)    $c80
set connections(rescalecolor_interpolate_to_showfield)    $c81

set connections(magnitude_to_choose)    $c82
set connections(choose_to_interpolate)  $c83
set connections(choose_to_rescalecolor) $c84


set connections(showfield_scalarslice_face_to_sync) $c39
set connections(showfield_scalarslice_edge_to_sync) $c40
set connections(showfield_isosurfaces_to_sync) $c60
set connections(showfield_isocontours_to_sync) $c61
set connections(showfield_streamline_edges_to_sync) $c76
set connections(showfield_streamline_nodes_to_sync) $c77

set connections(probe_scalar_to_viewer) $c101
set connections(probe_vector_to_viewer) $c102
set connections(sample_to_viewer)       $c103



# global array indexed by module name to keep track of modules
global mods

set mods(HDF5-Points) $m0
set mods(HDF5-Connections) $m1
set mods(HDF5-Scalar) $m2
set mods(HDF5-Vector) $m3

set mods(MDSPlus-Points) $m4
set mods(MDSPlus-Connections) $m5
set mods(MDSPlus-Scalar) $m6
set mods(MDSPlus-Vector) $m7

set mods(ChooseNrrd-Points)      $m8
set mods(ChooseNrrd-Connections) $m9
set mods(ChooseNrrd-Scalar)      $m10
set mods(ChooseNrrd-Vector)      $m11

set mods(NrrdToField-Scalar) $m12
set mods(NrrdToField-Vector) $m13

set mods(Probe-Scalar) $m14
set mods(Probe-Vector) $m15

set mods(FieldInfo-Scalar) $m16
set mods(FieldInfo-Vector) $m17
set mods(ChooseField-Isosurface-Surface) $m18


set mods(SubSample)   $m20
set mods(Slicer-Low)  $m21
set mods(Slicer-High) $m22

set mods(Isosurface-Surface)      $m23
set mods(Isosurface-Contour-Low)  $m24
set mods(Isosurface-Contour-High) $m25

set mods(ShowField-Isosurface-Surface) $m27
set mods(ShowField-Isosurface-Contour) $m28

set mods(TransformData-Scalar-Slice)          $m31
set mods(Isosurface-Scalar-Slice)             $m32
set mods(ApplyInterpMatrix-Scalar-Slice-Iso)  $m33
set mods(ApplyInterpMatrix-Scalar-Slice-Clip) $m34
set mods(ClipField-Scalar-Slice)              $m35
set mods(Isosurface-Slice-Contours)           $m36
set mods(ShowField-Scalar-Slice-Face)         $m37
set mods(ShowField-Scalar-Slice-Edge)         $m38

set mods(StreamLines-rake) $m60
set mods(StreamLines) $m61

set mods(DirectInterpolate-StreamLines-Vector) $m62
set mods(ShowField-StreamLines-Vector) $m63
set mods(ShowField-StreamLines-Scalar) $m64

set mods(ChooseField-Interpolate) $m70


set mods(ColorMap-Isosurfaces) $m45
set mods(ColorMap-Streamlines) $m65
set mods(ColorMap-Other) $m66

set mods(RescaleColorMap-Isosurfaces) $m46
set mods(RescaleColorMap-Streamlines) $m67
set mods(RescaleColorMap-Other) $m68

set mods(Synchronize) $m100
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
	
	set viewer_width 700
	set viewer_height 775
	
	set notebook_width 290
	set notebook_height [expr $viewer_height - 160]
	
	set vis_width [expr $notebook_width + 60]
	set vis_height $viewer_height

        set initialized 0
        set allow_execution 0

        set i_width 300
        set i_height 20
        set stripes 10

        set vis_frame_tab0 ""
        set vis_frame_tab1 ""
	set c_left_tab ""
     
        set error_module ""

        set data_tab0 ""
        set data_tab1 ""

        set animate_tab0 ""
        set animate_tab1 ""

        set slice_tab0 ""
        set slice_tab1 ""

        set contour_tab0 ""
        set contour_tab1 ""

        set iso_tab0 ""
        set iso_tab1 ""

        # colormaps
        set colormap_width 100
        set colormap_height 15
        set colormap_res 64

        set indicatorID 0

	### Define Tooltips
	##########################
	# General
	global tips

	# Visualization Tab
        set tips(Execute) "Select to execute the changes"

	global filename_points filename_connections 
	global filename_scalar filename_vector

	set filename_points      "No Data Selected"
	set filename_connections "No Data Selected"
	set filename_scalar      "No Data Selected"
	set filename_vector      "No Data Selected"

	global shot_points shot_connections shot_scalar shot_vector

	set shot_points      "No Data Selected"
	set shot_connections "No Data Selected"
	set shot_scalar      "No Data Selected"
	set shot_vector      "No Data Selected"

	set valid_points -1
	set valid_connections -1
	set valid_scalar -1
	set valid_vector -1

	set have_scalarslice 0
	set have_isosurfaces 0
	set have_streamlines 0
    }
    

    destructor {
	destroy $this
    }

    
    method build_app {} {
	set DEBUG 0
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
	wm protocol $win.detachedV WM_DELETE_WINDOW \
	    { app hide_visualization_window }

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
	global $mods(Isosurface-Surface)-isoval-min
	global $mods(Isosurface-Surface)-isoval-max
	global $mods(Isosurface-Surface)-active-isoval-selection-tab
	trace variable $mods(Isosurface-Surface)-isoval-min w \
	    "$this update_minmax_callback"
	trace variable $mods(Isosurface-Surface)-isoval-max w \
	    "$this update_minmax_callback"
	trace variable \
	    $mods(Isosurface-Surface)-active-isoval-selection-tab w \
	    "$this update_isotab_callback"


	global $mods(Isosurface-Scalar-Slice)-isoval-min
	global $mods(Isosurface-Scalar-Slice)-isoval-min
	trace variable $mods(Isosurface-Scalar-Slice)-isoval-min w \
	    "$this update_minmax_callback"
	trace variable $mods(Isosurface-Scalar-Slice)-isoval-max w \
	    "$this update_minmax_callback"

	global $mods(HDF5-Points)-filename
	global $mods(HDF5-Connections)-filename
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename
	trace variable $mods(HDF5-Points)-filename w \
	    "$this update_hdf5_callback"
	trace variable $mods(HDF5-Connections)-filename w \
	    "$this update_hdf5_callback"
	trace variable $mods(HDF5-Scalar)-filename w \
	    "$this update_hdf5_callback"
	trace variable $mods(HDF5-Vector)-filename w \
	    "$this update_hdf5_callback"

	global $mods(MDSPlus-Points)-num-entries
	global $mods(MDSPlus-Connections)-num-entries
	global $mods(MDSPlus-Scalar)-num-entries
	global $mods(MDSPlus-Vector)-num-entries
	trace variable $mods(MDSPlus-Points)-num-entries w \
	    "$this update_mdsplus_callback"
	trace variable $mods(MDSPlus-Connections)-num-entries w \
	    "$this update_mdsplus_callback"
	trace variable $mods(MDSPlus-Scalar)-num-entries w \
	    "$this update_mdsplus_callback"
	trace variable $mods(MDSPlus-Vector)-num-entries w \
	    "$this update_mdsplus_callback"


	global $mods(HDF5-Scalar)-animate
	global $mods(HDF5-Vector)-animate
	trace variable $mods(HDF5-Scalar)-animate w \
	    "$this update_animate_callback"
	trace variable $mods(HDF5-Vector)-animate w \
	    "$this update_animate_callback"

	global $mods(HDF5-Scalar)-current
	global $mods(HDF5-Vector)-current
	trace variable $mods(HDF5-Scalar)-current w \
	    "$this update_current_callback"
	trace variable $mods(HDF5-Vector)-current w \
	    "$this update_current_callback"

	global $mods(Slicer-High)-i-dim
	global $mods(Slicer-High)-j-dim
	global $mods(Slicer-High)-k-dim

	trace variable $mods(Slicer-High)-i-dim w "$this update_slicer_callback"
	trace variable $mods(Slicer-High)-j-dim w "$this update_slicer_callback"
	trace variable $mods(Slicer-High)-k-dim w "$this update_slicer_callback"

	global $mods(Probe-Scalar)-locx
	global $mods(Probe-Scalar)-locy
	global $mods(Probe-Scalar)-locz
	trace variable $mods(Probe-Scalar)-locx w "$this update_probe_callback"
	trace variable $mods(Probe-Scalar)-locy w "$this update_probe_callback"
	trace variable $mods(Probe-Scalar)-locz w "$this update_probe_callback"

	global $mods(Probe-Vector)-locx
	global $mods(Probe-Vector)-locy
	global $mods(Probe-Vector)-locz
	trace variable $mods(Probe-Vector)-locx w "$this update_probe_callback"
	trace variable $mods(Probe-Vector)-locy w "$this update_probe_callback"
	trace variable $mods(Probe-Vector)-locz w "$this update_probe_callback"

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

	$vis_frame_tab0 select "Data Selection"
	$vis_frame_tab1 select "Data Selection"

	$data_tab0 select "HDF5"
	$data_tab1 select "HDF5"

	$vis_tab0 select "Scalar"
	$vis_tab1 select "Scalar"

	$animate_tab0 select "Scalar"
	$animate_tab1 select "Scalar"

	$color_tab0 select "Isosurfaces"
	$color_tab1 select "Isosurfaces"

	set show_faces 1
	set show_isocontours 1
	set show_integration 1

	set initialized 1
	set ignore_callbacks 0

	global PowerAppSession
	if {[info exists PowerAppSession] && [set PowerAppSession] != ""} { 
	    set saveFile $PowerAppSession

	    load_session_data
	} else {
#	    set ignore_callbacks 1
	    $mods(HDF5-Points) clear
	    $mods(HDF5-Connections) clear
	    $mods(HDF5-Scalar) clear
	    $mods(HDF5-Vector) clear
	    $mods(MDSPlus-Points) deleteEntry 1
	    $mods(MDSPlus-Connections) deleteEntry 1
	    $mods(MDSPlus-Scalar) deleteEntry 1
	    $mods(MDSPlus-Vector) deleteEntry 1
#	    set ignore_callbacks 0
	    
	    update_state
	}
    }
	    
    method update_state {} {

	if { $DEBUG == 1 } {
	    puts stderr "update_state"
	}

	set allow_execution 0

	global mods

	global $mods(Isosurface-Slice-Contours)-active-isoval-selection-tab
	change_iso_tab \
	    [set $mods(Isosurface-Slice-Contours)-active-isoval-selection-tab] \
	    $mods(Isosurface-Slice-Contours) contour

	global $mods(Isosurface-Scalar-Slice)-active-isoval-selection-tab
	change_iso_tab \
	    [set $mods(Isosurface-Scalar-Slice)-active-isoval-selection-tab] \
	    $mods(Isosurface-Scalar-Slice) slice

	global $mods(Isosurface-Surface)-active-isoval-selection-tab
	change_iso_tab \
	    [set $mods(Isosurface-Surface)-active-isoval-selection-tab] \
	    $mods(Isosurface-Surface) iso

	update_hdf5_callback 0 0 0
	update_mdsplus_callback 0 0 0
	update_current_callback 0 0 0
	update_slicer_callback 0 0 0
	update_probe_callback 0 0 0
	update_minmax_callback 0 0 0

	update_point_modules
	update_connection_modules
	update_scalar_modules
	update_vector_modules
	
	set allow_execution 1

	if { $DEBUG == 1 } {
	    puts stderr "out update_state"
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

	    set vis_frame_tab$case $vis.tnb

############### Data Tab
	    set data [$vis.tnb add -label "Data Selection" \
			  -command "$this change_vis_frame 0"]

############### Data Source Frame
	    iwidgets::labeledframe $data.source -labelpos nw \
		-labeltext "Data Source" 
	    
	    set source [$data.source childsite]

	    build_data_source_frame $source $case

	    pack $data.source -padx 4 -pady 4 -fill x 
	    

############### Data Animate Frame
	    iwidgets::labeledframe $data.animate -labelpos nw \
		-labeltext "Data Animation" 

	    set animate [$data.animate childsite]

	    build_animate_frame $animate $case

	    pack $data.animate -padx 4 -pady 4 -fill x 


############### Data Subsample Frame
	    iwidgets::labeledframe $data.subsample -labelpos nw \
		-labeltext "Data Subsample" 

	    set subsample [$data.subsample childsite]

	    build_subsample_frame $subsample $case
	    
            pack $data.subsample -padx 4 -pady 4 -fill x

	    set subsample_frame$case $subsample


############### Vis Options Tab
	    set page [$vis.tnb add -label "Vis Options" \
			  -command "$this change_vis_frame 1"]

############### Tabs
	    iwidgets::tabnotebook $page.tnb -width $notebook_width \
		-height [expr $notebook_height-100] -tabpos n
	    pack $page.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	    set vis_tab$case $page.tnb

################## Scalar Fields Tab
	    set fields [$page.tnb add -label "Scalar" \
			    -command "$this change_option_tab 0"]

	    set vis_scalar_tab$case $fields


############### Scalar Slice
	    iwidgets::labeledframe $fields.slice -labelpos nw \
		-labeltext "Scalar Slice"

	    set slice [$fields.slice childsite]
	    
	    build_scalarslice_frame $slice $case
	     
            pack $fields.slice -padx 4 -pady 4 -fill x

	    set scalarslice_frame$case $slice

	    
############### Isosurface
	    iwidgets::labeledframe $fields.isoframe -labelpos nw \
		-labeltext "Isosurface"

	    set iso [$fields.isoframe childsite]
	    
	    build_isosurface_frame $iso $case
	     
            pack $fields.isoframe -padx 4 -pady 4 -fill x

	    set isosurfaces_frame$case $iso
	    

################## Fields Tab
	    set fields [$page.tnb add -label "Vector" \
			    -command "$this change_option_tab 1"]

	    set vis_vector_tab$case $fields


################## StreamLines
	    iwidgets::labeledframe $fields.slframe -labelpos nw \
		-labeltext "Field Lines"

	    set sl [$fields.slframe childsite]
	    
	    build_streamlines_frame $sl
	    
            pack $fields.slframe -padx 4 -pady 4 -fill x

	    set streamlines_frame$case $sl


################## Probe Tab
	    set probes [$page.tnb add -label "Probes" \
			    -command "$this change_option_tab 2"]

            if {$case == 0} {
		set vis_probe_tab$case $probes
            } else {
		set vis_probe_tab1 $probes	    
            }


################## Scalar Probe
	    iwidgets::labeledframe $probes.sframe -labelpos nw \
		-labeltext "Scalar Probe"

	    set sprobe [$probes.sframe childsite]
	    
	    build_probe_frame $sprobe $mods(Probe-Scalar) probe_scalar
	     
            pack $probes.sframe -padx 4 -pady 4 -fill x

            if {$case == 0} {
		set probe_scalar_frame$case $sprobe
            } else {
		set probe_scalar_frame1 $sprobe
            }

################## Vector Probe
	    iwidgets::labeledframe $probes.vframe -labelpos nw \
		-labeltext "Vector Probe"

	    set vprobe [$probes.vframe childsite]
	    
	    build_probe_frame $vprobe $mods(Probe-Vector) probe_vector
	     
            pack $probes.vframe -padx 4 -pady 4 -fill x

            if {$case == 0} {
		set probe_vector_frame$case $vprobe
            } else {
		set probe_vector_frame1 $vprobe
            }

################## Misc Probe
	    iwidgets::labeledframe $probes.mframe -labelpos nw \
		-labeltext "Misc"

	    set misc [$probes.mframe childsite]

	    global probe_lock
	    global $mods(Probe-Scalar)-loc
	    checkbutton $misc.lock -text "Lock probes" \
		-variable probe_lock \
		-command "$this update_probe_callback $mods(Probe-Scalar)-locx 0 0"
	    pack $misc.lock -side top -anchor nw -padx 3 -pady 3

            pack $probes.mframe -padx 4 -pady 4 -fill x

################## Misc Page
	    set misc [$page.tnb add -label "Misc" \
			  -command "$this change_option_tab 3"]

	    set vis_misc_tab$case $misc


################## ColorMaps
	    iwidgets::labeledframe $misc.colorframe -labelpos nw \
		-labeltext "Color Maps"

	    set color [$misc.colorframe childsite]
	    
	    build_colormap_frame $color $case
	    
            pack $misc.colorframe -padx 4 -pady 4 -fill x

	    
################## Synchronize
	    iwidgets::labeledframe $misc.sync -labelpos nw \
		-labeltext "Synchronize"

	    set sync [$misc.sync childsite]

	    global $mods(Synchronize)-enforce
	    checkbutton $sync.enforce -text "Enforce" \
		-variable $mods(Synchronize)-enforce
	    pack $sync.enforce -side top -anchor nw -padx 3 -pady 3

            pack $misc.sync -padx 4 -pady 4 -fill x

################## Renderer Options Tab
	    create_viewer_tab $vis

	    $vis.tnb view "Vis Options"
	    
	    
################## Execute Button
            frame $vis.last
            pack $vis.last -side bottom -anchor ne \
		-padx 5 -pady 5
	    
            button $vis.last.ex -text "Execute" \
		-background $execute_color \
		-activebackground $execute_color \
		-width 8 \
		-command "$this execute_Data"
	    Tooltip $vis.last.ex $tips(Execute)

            pack $vis.last.ex -side right -anchor ne \
		-padx 2 -pady 0

	    set data_ex_button1 $vis.last.ex


            ### Indicator
	    frame $vis.indicator -relief sunken -borderwidth 2
            pack $vis.indicator -side bottom -anchor s -padx 3 -pady 5
	    
	    canvas $vis.indicator.canvas -bg "white" -width $i_width \
	        -height $i_height 
	    pack $vis.indicator.canvas -side top -anchor n -padx 3 -pady 3
	    
            bind $vis.indicator <Button> {app display_module_error} 
	    
            label $vis.indicatorL -text "Press Execute to update visualization ..."
            pack $vis.indicatorL -side bottom -anchor sw -padx 5 -pady 3
	    
	    
	    set indicator$case $vis.indicator.canvas
	    set indicatorL$case $vis.indicatorL
	    
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
    


    method build_data_source_frame { f case } {
	global mods tips

	### Tabs
	iwidgets::tabnotebook $f.tnb -width $notebook_width \
	    -height 175 -tabpos n
	pack $f.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	set data_tab$case $f.tnb

	### Data HDF5
	set hdf5 [$f.tnb add -label "HDF5" -command "$this change_data_tab 0"]
	set data_hdf5_tab$case $hdf5

	global $mods(HDF5-Points)-file

	$mods(HDF5-Points) \
	    set_power_app_cmd "$this update_hdf5_callback $mods(HDF5-Points)-filename 0 0"
	$mods(HDF5-Connections) \
	    set_power_app_cmd "$this update_hdf5_callback $mods(HDF5-Connections)-filename 0 0"
	$mods(HDF5-Scalar) \
	    set_power_app_cmd "$this update_hdf5_callback $mods(HDF5-Scalar)-filename 0 0"
	$mods(HDF5-Vector) \
	    set_power_app_cmd "$this update_hdf5_callback $mods(HDF5-Vector)-filename 0 0"

	frame $hdf5.points
	button $hdf5.points.button -text " Points " \
	    -command "$mods(HDF5-Points)   initialize_ui"
	pack $hdf5.points.button -side left -anchor nw -padx 3 -pady 3
	label $hdf5.points.label -textvariable filename_points
	pack $hdf5.points.label  -side right -anchor nw -padx 3 -pady 3


	frame $hdf5.connections
	button $hdf5.connections.button -text "Connections" \
	    -command "$mods(HDF5-Connections)   initialize_ui"
	pack $hdf5.connections.button -side left -anchor nw -padx 3 -pady 3
	label $hdf5.connections.label -textvariable filename_connections
	pack $hdf5.connections.label  -side right -anchor nw -padx 3 -pady 3



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


	pack $hdf5.points      -side top -anchor nw -padx 3 -pady 3
	pack $hdf5.connections -side top -anchor nw -padx 3 -pady 3
	pack $hdf5.scalar      -side top -anchor nw -padx 3 -pady 3
	pack $hdf5.vector       -side top -anchor nw -padx 3 -pady 3


	### Data MDSPlus
	set mdsplus [$f.tnb add -label "MDSPlus" \
			 -command "$this change_data_tab 1"]
	set data_mdsplus_tab$case $mdsplus
	

	$mods(MDSPlus-Points) set_power_app_cmd \
	    "$this update_mdsplus_callback $mods(MDSPlus-Points)-shot 0 0"
	$mods(MDSPlus-Connections) set_power_app_cmd \
	    "$this update_mdsplus_callback $mods(MDSPlus-Connections)-shot 0 0"
	$mods(MDSPlus-Scalar) set_power_app_cmd \
	    "$this update_mdsplus_callback $mods(MDSPlus-Scalar)-shot 0 0"
	$mods(MDSPlus-Vector) set_power_app_cmd \
	    "$this update_mdsplus_callback $mods(MDSPlus-Vector)-shot 0 0"


	frame $mdsplus.connections
	button $mdsplus.connections.button -text "Connections" \
	    -command "$mods(MDSPlus-Connections)   initialize_ui"
	pack $mdsplus.connections.button   -side left -anchor nw -padx 3 -pady 3
	label $mdsplus.connections.label -textvariable shot_connections
	pack $mdsplus.connections.label   -side right -anchor nw -padx 3 -pady 3

	frame $mdsplus.points
	button $mdsplus.points.button -text " Points " \
	    -command "$mods(MDSPlus-Points)   initialize_ui"
	pack $mdsplus.points.button   -side left -anchor nw -padx 3 -pady 3
	label $mdsplus.points.label -textvariable shot_points
	pack $mdsplus.points.label   -side right -anchor nw -padx 3 -pady 3

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


	pack $mdsplus.points      -side top -anchor nw -padx 3 -pady 3
	pack $mdsplus.connections -side top -anchor nw -padx 3 -pady 3
	pack $mdsplus.scalar      -side top -anchor nw -padx 3 -pady 3
	pack $mdsplus.vector      -side top -anchor nw -padx 3 -pady 3
    }
    

    method build_animate_frame { f case } {
	global mods tips

	### Tabs
	iwidgets::tabnotebook $f.tnb -width $notebook_width \
	    -height 250 -tabpos n
	pack $f.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	set animate_tab$case $f.tnb

	### Animate Scalar
	set scalar [$f.tnb add -label "Scalar" -command "$this change_animate_tab 0"]

	set animate_scalar_tab$case $scalar

	$mods(HDF5-Scalar) build_animate_ui $scalar
	

	### Animate Vector
	set vector [$f.tnb add -label "Vector" -command "$this change_animate_tab 1"]

	set animate_vector_tab$case $vector

	$mods(HDF5-Vector) build_animate_ui $vector

	### Animate Lock
	set lock [$f.tnb add -label "Locking" -command "$this change_animate_tab 2"]

	set animate_lock_tab$case $lock

	global animate_lock
	checkbutton $lock.show -text "Lock Animation" \
	    -variable animate_lock \
	    -command "$this update_current_callback $mods(HDF5-Scalar)-current 0 0"
	pack $lock.show -side top -anchor nw -padx 3 -pady 3
    }


    method build_subsample_frame { f case } {
	global mods

	$mods(SubSample) set_power_app_cmd "$this update_subsample_frame"

	button $f.button -text "SubSample UI" -command "$mods(SubSample) initialize_ui"
	pack $f.button -side left -anchor nw -padx 3 -pady 3
    }

    method update_subsample_frame {} {
    }


    method build_probe_frame { f probemod var } {

	global $var
	global $probemod-locx

	checkbutton $f.show -text "Show probe" \
	    -variable $var \
	    -command "$this toggle_probes $probemod"
	pack $f.show -side top -anchor nw -padx 3 -pady 3
	
	frame $f.ui	
	$probemod build_ui $f.ui

	pack $f.ui -pady 4 -fill x
    }


    method build_scalarslice_frame { f case } {
	global mods
	global $mods(ShowField-Scalar-Slice-Edge)-edges-on
	global $mods(ShowField-Scalar-Slice-Face)-faces-on

	checkbutton $f.show -text "Show Scalar Slice Contours" \
	    -variable $mods(ShowField-Scalar-Slice-Edge)-edges-on \
	    -command "$this toggle_scalarslice"
	pack $f.show -side top -anchor nw -padx 3 -pady 3


	global slice_direction

	frame $f.direction

	label $f.direction.l -text "Slice Direction:"
	radiobutton $f.direction.phi -text "Phi" \
	    -variable slice_direction -value 0 \
	    -command "$this update_slice_direction"
	radiobutton $f.direction.z -text "Z" \
	    -variable slice_direction -value 1 \
	    -command "$this update_slice_direction"

	pack $f.direction.l $f.direction.phi $f.direction.z \
	    -side left -anchor w -padx 5

	pack $f.direction -side top -anchor w -padx 3 -pady 3

	build_isosurface_tabs \
	    $f $case $mods(Isosurface-Scalar-Slice) \
	    "slice" update_slicevals

	frame $f.contours

	label $f.contours.l -text "Contours:"
	pack $f.contours.l -side top -anchor w -padx 5

	build_isosurface_tabs \
	    $f.contours $case $mods(Isosurface-Slice-Contours) \
	    "contour" update_contourvals

	pack $f.contours -side top -anchor nw -pady 3

	frame $f.faces

	checkbutton $f.faces.show -text "Show Scalar Slices as Faces" \
	    -variable $mods(ShowField-Scalar-Slice-Face)-faces-on \
	    -command "$this toggle_faces 0"

	pack $f.faces.show -side top -anchor nw -padx 3 -pady 3

	pack $f.faces -side top -anchor nw -pady 3
    }


    method build_isosurface_frame { f case } {
	global mods
	global $mods(ShowField-Isosurface-Surface)-faces-on

	checkbutton $f.show -text "Show Isosurface" \
	    -variable $mods(ShowField-Isosurface-Surface)-faces-on \
	    -command "$this toggle_isosurfaces"
	pack $f.show -side top -anchor nw -padx 3 -pady 3
	
	build_isosurface_tabs \
	    $f $case $mods(Isosurface-Surface) "iso" update_isovals

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
	    -command "$this toggle_isocontours 0"
	pack $f.isocontours -side top -anchor w -padx 20
    }	 


    method build_isosurface_tabs { f case isomod suffix cmd } {

############ Tabs
	iwidgets::tabnotebook $f.tnb -width $notebook_width \
	    -height 75 -tabpos n
	pack $f.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	set tt _tab
	set $suffix$tt$case $f.tnb

############ Isovalue Slider
	set slider [$f.tnb add -label "Slider" \
			-command "$this change_iso_tab 0 $isomod $suffix"]

	set tt _slider_tab
	set $suffix$tt$case $slider

	frame $slider.isoval
	
	global $isomod-isoval
	global $isomod-isoval-min
	global $isomod-isoval-max

 	set min [set $isomod-isoval-min]
 	set max [set $isomod-isoval-max]

	set lg [expr floor( log10($max-$min) ) ]
	set range [expr pow(10.0, $lg )]
	
	set scale 1.0
	
	if { $lg > 5.0 } {
	    set scale [expr pow(10.0, $lg-5 )]
	}

	set res [expr $range/(1.0e4*$scale)]

	label $slider.isoval.l -text "$suffix value:"
	scale $slider.isoval.s \
	    -from [set $isomod-isoval-min] \
	    -to   [set $isomod-isoval-max] \
	    -length 100 -width 15 \
	    -sliderlength 15 \
	    -resolution $res \
	    -variable $isomod-isoval \
	    -showvalue false \
	    -orient horizontal

	bind $slider.isoval.s <ButtonRelease> "$this $cmd"

	entry $slider.isoval.val -width 5 -textvariable $isomod-isoval

	bind $slider.isoval.val <Return> "$this $cmd"

	pack $slider.isoval.l $slider.isoval.s $slider.isoval.val \
	    -side left -anchor nw -padx 3

	pack $slider.isoval -side top -anchor nw -padx 3 -pady 3


########### Isovalue Quantity
	set quantity [$f.tnb add -label "Quantity" \
			  -command "$this change_iso_tab 1 $isomod $suffix"]

	set tt _quantity_tab
	set $suffix$tt$case $quantity

	set tab $suffix$tt

	global $isomod-isoval-quantity

	iwidgets::spinner $quantity.isoquant \
	    -labeltext "No. evenly-spaced $suffix values: " \
	    -width 5 -fixed 5 \
	    -validate "$this set-quantity %P $isomod-isoval-quantity]" \
	    -decrement "$this spin-quantity -1 \
                        $tab $isomod-isoval-quantity; \
                        $this $cmd" \
	    -increment "$this spin-quantity  1 \
                        $tab $isomod-isoval-quantity; \
                        $this $cmd" 

	$quantity.isoquant insert 1 [set $isomod-isoval-quantity]

	pack $quantity.isoquant -side top -anchor nw -padx 3 -pady 3

########### Isovalue List
	set list [$f.tnb add -label "List" \
		      -command "$this change_iso_tab 2 $isomod $suffix"]

	set tt _list_tab
	set $suffix$tt$case $quantity

	frame $list.isolist
	
	global $isomod-isoval-list

	label $list.isolist.l -text "List of $suffix values:"
	entry $list.isolist.e -width 40 -text $isomod-isoval-list
	bind $list.isolist.e <Return> "$this $cmd"
	pack $list.isolist.l $list.isolist.e \
	    -side left -anchor nw -padx 3 -fill both -expand 1
	pack $list.isolist -side top -anchor nw -padx 3 -pady 3
    }


    method build_streamlines_frame { f } {
	global mods
	global $mods(ShowField-StreamLines-Vector)-edges-on
	global $mods(ShowField-StreamLines-Scalar)-nodes-on

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
	
#	bind $f.seeds.s <ButtonRelease> "$mods(StreamLines-rake)-c needexecute"
	
	entry $f.seeds.val -width 3 -relief flat \
	    -textvariable $mods(StreamLines-rake)-maxseeds
	
#	bind $f.seeds.val <Return> "$mods(StreamLines-rake)-c needexecute"

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
	
#	bind $f.stepsize.s <ButtonRelease> "$mods(StreamLines-rake)-c needexecute"
	
	entry $f.stepsize.val -width 3 -relief flat \
	    -textvariable $mods(StreamLines)-stepsize
	
#	bind $f.stepsize.val <Return> "$mods(StreamLines-rake)-c needexecute"

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
	
#	bind $f.steps.s <ButtonRelease> "$mods(StreamLines-rake)-c needexecute"
	
	entry $f.steps.val -width 3 -relief flat \
	    -textvariable $mods(StreamLines)-maxsteps
	
#	bind $f.steps.val <Return> "$mods(StreamLines-rake)-c needexecute"

	pack $f.steps.l $f.steps.s $f.steps.val \
	    -side left -anchor n -padx 3      

	
	frame $f.cm

	label $f.cm.l -text "Color using:"
	radiobutton $f.cm.scalar -text "Scalar Values" \
	    -variable $mods(ChooseField-Interpolate)-port-index -value 0 
#	    -command "$mods(ChooseField-Interpolate)-c needexecute"
	radiobutton $f.cm.vector -text "Vector Values" \
	    -variable $mods(ChooseField-Interpolate)-port-index -value 1 
#	    -command "$mods(ChooseField-Interpolate)-c needexecute"

	pack $f.cm.l $f.cm.scalar $f.cm.vector -side left -anchor w -padx 5

	pack $f.cm -side top -anchor w -padx 5 -pady 3

	frame $f.integration -relief groove -borderwidth 2

	checkbutton $f.integration.show \
	    -text "Show Scalar Integration Points" \
	    -variable $mods(ShowField-StreamLines-Scalar)-nodes-on \
	    -command "$this toggle_integration 0"
	pack $f.integration.show -side top -anchor w -padx 20 -pady 3

	frame $f.integration.color
	label $f.integration.color.label -text "Color Style:"
	frame $f.integration.color.left
	frame $f.integration.color.right
	radiobutton $f.integration.color.left.const -text   "Seed Number" \
	    -variable $mods(StreamLines)-color -value 0
	radiobutton $f.integration.color.left.incr -text "Integration Step" \
	    -variable $mods(StreamLines)-color -value 1
	radiobutton $f.integration.color.right.delta \
	    -text "Distance from Seed" \
	    -variable $mods(StreamLines)-color -value 2
	radiobutton $f.integration.color.right.total \
	    -text "Streamline Length" \
	    -variable $mods(StreamLines)-color -value 3

	pack $f.integration.color.left.const \
	    $f.integration.color.left.incr -side top -anchor w
	pack $f.integration.color.right.delta \
	    $f.integration.color.right.total -side top -anchor w

	pack $f.integration.color.label -side top -anchor w
	pack $f.integration.color.left -side left -anchor w
	pack $f.integration.color.right -side left -padx 20

	pack $f.integration.color -side top -anchor w -padx 5 -pady 3 -fill x
	pack $f.integration -side top -anchor w -padx 5 -pady 3 -fill x
    }


    method build_colormap_frame { f case } {

	global mods

	### Tabs
	iwidgets::tabnotebook $f.tnb -width $notebook_width \
	    -height 300 -tabpos n
	pack $f.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1
	
	set color_tab$case $f.tnb
	
	# Isosurface 
	set iso [$f.tnb add -label "Isosurfaces" \
		     -command "$this change_color_tab 0"]
	set color_isosurfaces_tab$case $iso
	build_colormap_tab $iso $mods(ColorMap-Isosurfaces) \
	    $mods(RescaleColorMap-Isosurfaces)

	# Streamlines 
	set stream [$f.tnb add -label "Streamlines" \
			-command "$this change_color_tab 1"]	
	set color_streamlines_tab$case $stream
	build_colormap_tab $stream $mods(ColorMap-Streamlines) \
	    $mods(RescaleColorMap-Streamlines)

	# Other 
	set other [$f.tnb add -label "Other" \
		       -command "$this change_color_tab 2"]	
	set color_other_tab$case $other
	build_colormap_tab $other $mods(ColorMap-Other) \
	    $mods(RescaleColorMap-Other)
    }


    method build_colormap_tab { f cmapmod rscapmod} {

	iwidgets::labeledframe $f.colormaps -labelpos nw \
	    -labeltext "Color Maps"

	set cmf [$f.colormaps childsite]
	
	build_colormap_canvas $cmf $cmapmod "Gray"
	build_colormap_canvas $cmf $cmapmod "Rainbow"
	build_colormap_canvas $cmf $cmapmod "Darkhue"
	build_colormap_canvas $cmf $cmapmod "Lighthue"
	build_colormap_canvas $cmf $cmapmod "Blackbody"
	build_colormap_canvas $cmf $cmapmod "BP Seismic"

	iwidgets::labeledframe $f.rescaling -labelpos nw \
	    -labeltext "Color Map Rescalings"

	set cmrs [$f.rescaling childsite]

	$rscapmod build_ui $cmrs

	pack $f.colormaps $f.rescaling -padx 4 -pady 4 -fill x
    }
    
    
    method build_colormap_canvas { f cmapmod cmapname } {
	set maps $f
	global $cmapmod-mapType

	frame $maps.cm-$cmapname
	pack $maps.cm-$cmapname -side top -anchor nw -padx 3 -pady 1 \
	    -fill x -expand 1
	radiobutton $maps.cm-$cmapname.b -text "$cmapname" \
	    -variable $cmapmod-mapName \
	    -value "$cmapname" \
	    -command "$cmapmod change" 
#	    -command "$cmapmod SetColorMap" # for no execute, the above executes

	pack $maps.cm-$cmapname.b -side left -anchor nw -padx 3 -pady 0
	
	frame $maps.cm-$cmapname.f -relief sunken -borderwidth 2
	pack $maps.cm-$cmapname.f -padx 2 -pady 0 -side right -anchor e
	canvas $maps.cm-$cmapname.f.canvas -bg "#ffffff" \
	    -height $colormap_height -width $colormap_width
	pack $maps.cm-$cmapname.f.canvas -anchor e \
	    -fill both -expand 1

	draw_colormap $cmapname $maps.cm-$cmapname.f.canvas
    }


    method update_slice_direction {} {
	global mods
	global slice_direction

	global $mods(TransformData-Scalar-Slice)-function
	global $mods(ClipField-Scalar-Slice)-clipfunction

	if { $slice_direction == 0 } {
	    set $mods(TransformData-Scalar-Slice)-function \
		{result = atan2(x,y);}
	    set $mods(ClipField-Scalar-Slice)-clipfunction \
		{fabs(atan2(x,y) - v) < 1e-2}

	} elseif { $slice_direction == 1 } {
	    set $mods(TransformData-Scalar-Slice)-function {result = z;}
	    set $mods(ClipField-Scalar-Slice)-clipfunction {fabs(z - v) < 1e-2}
	}

	if { $allow_execution == 1 } {
#	    $mods(TransformData-Scalar-Slice)-c needexecute
	}
    }


    method update_slicevals {} {
	global mods

	if { $allow_execution == 1 } {
#	    $mods(Isosurface-Scalar-Slice)-c needexecute
	}
    }

    method update_contourvals {} {
	global mods

	if { $allow_execution == 1 } {
#	    $mods(Isosurface-Scalar-Contours)-c needexecute
	}
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

	if { $allow_execution == 1 } {
#	    $mods(Isosurface-Surface)-c needexecute
	    if { $valid_connections == 0 } {
#		$mods(Isosurface-Contour-Low)-c needexecute
#		$mods(Isosurface-Contour-High)-c needexecute
	    }
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

    method toggle_scalarslice {} {
	toggle_faces 1

	global mods connections
	global $mods(ShowField-Scalar-Slice-Face)-faces-on
	global $mods(ShowField-Scalar-Slice-Edge)-edges-on
	global $mods(ShowField-Scalar-Slice-Face)-port-index

	if { $DEBUG == 1 } {
	    puts stderr "toggle_scalarslice $have_scalarslice [set $mods(ShowField-Scalar-Slice-Face)-faces-on] [set $mods(ShowField-Scalar-Slice-Edge)-edges-on]"
	}

	if { [set $mods(ShowField-Scalar-Slice-Edge)-edges-on] == 1 } {
	    set on 1
	    set disable 0

	    foreach w [winfo children $scalarslice_frame0] {
		enable_widget $w
	    }
	    foreach w [winfo children $scalarslice_frame1] {
		enable_widget $w
	    }
	} else {
	    set on 0
	    set disable 1

	    foreach w [winfo children $scalarslice_frame0] {
		disable_widget $w
	    }
	    foreach w [winfo children $scalarslice_frame1] {
		disable_widget $w
	    }
	}
	
	disableConnectionID $connections(scalar_to_matrix)          $disable
	disableConnectionID $connections(scalar_to_transform)       $disable
	disableConnectionID $connections(transform_to_isosurface)   $disable
	disableConnectionID $connections(isosurface_fld_to_matrix)  $disable
	disableConnectionID $connections(isosurface_mtx_to_matrix)  $disable
	disableConnectionID $connections(isosurface_to_clipfuction) $disable
	disableConnectionID $connections(matrix_to_matrix)          $disable
	disableConnectionID $connections(matrix_to_showfield)       $disable
	disableConnectionID $connections(matrix_to_isosurface)      $disable
	disableConnectionID $connections(isosurface_to_showfield)   $disable
	disableConnectionID $connections(clipfuction_fld_to_matrix) $disable
	disableConnectionID $connections(clipfuction_mtx_to_matrix) $disable

	disableConnectionID $connections(colormap_matrix_to_showfield)       $disable
	disableConnectionID $connections(colormap_isosurface_to_showfield)   $disable

#       Disconnecting a dynamic port causes a hang
#	disableConnectionID $connections(showfield_scalarslice_face_to_sync) $disable
#	disableConnectionID $connections(showfield_scalarslice_edge_to_sync) $disable

	enable_widget $scalarslice_frame0.show
	enable_widget $scalarslice_frame1.show

	if { $have_scalarslice == 1 } {
	    $mods(ShowField-Scalar-Slice-Edge)-c toggle_display_edges
	}
    }


    method toggle_faces { update } {
	global mods connections
	global $mods(ShowField-Scalar-Slice-Face)-faces-on
	global $mods(ShowField-Scalar-Slice-Edge)-edges-on

	if { $DEBUG == 1 } {
	    puts stderr "in  toggle_faces [set $mods(ShowField-Scalar-Slice-Edge)-edges-on] [set $mods(ShowField-Scalar-Slice-Face)-faces-on] $have_scalarslice $show_faces"
	}

	if { $update == 1 } {
	    if {[set $mods(ShowField-Scalar-Slice-Edge)-edges-on] } {

		set $mods(ShowField-Scalar-Slice-Face)-faces-on \
		    $show_faces

		enable_widget $scalarslice_frame0.faces.show
		enable_widget $scalarslice_frame1.faces.show
	    } else {
		set show_faces \
		    [set $mods(ShowField-Scalar-Slice-Face)-faces-on]
		set $mods(ShowField-Scalar-Slice-Face)-faces-on 0

		disable_widget $scalarslice_frame0.faces.show
		disable_widget $scalarslice_frame1.faces.show
	    }
	}

	if {[set $mods(ShowField-Scalar-Slice-Face)-faces-on] } {
#	    set $mods(ShowField-Scalar-Slice-Edge)-edge_display_type Cylinders
#	    set $mods(ShowField-Scalar-Slice-Edge)-edge_scale 0.00
	    set disable 0
	} else {
#	    set $mods(ShowField-Scalar-Slice-Edge)-edge_display_type Lines
#	    set $mods(ShowField-Scalar-Slice-Edge)-edge_scale 0.01
	    set disable 1
	}

	if { $have_scalarslice == 1 } {
#	    $mods(ShowField-Scalar-Slice-Edge)-c edge_display_type
	    $mods(ShowField-Scalar-Slice-Face)-c toggle_display_faces
	}

	if { $DEBUG == 1 } {
	    puts stderr "out toggle_faces [set $mods(ShowField-Scalar-Slice-Face)-faces-on] [set $mods(ShowField-Scalar-Slice-Edge)-edges-on] $have_scalarslice $show_faces"
	}
    }


    method toggle_isosurfaces { } {
	toggle_isocontours 1

	global mods connections
	global $mods(ShowField-Isosurface-Surface)-faces-on
	global $mods(ShowField-Isosurface-Contour)-edges-on

	if { $DEBUG == 1 } {
	    puts stderr "toggle_isosurfaces [set $mods(ShowField-Isosurface-Surface)-faces-on] $valid_connections"
	}

	if {[set $mods(ShowField-Isosurface-Surface)-faces-on] == 1} {
	    set disable 0

	    foreach w [winfo children $isosurfaces_frame0] {
		enable_widget $w
	    }
	    foreach w [winfo children $isosurfaces_frame1] {
		enable_widget $w
	    }

	    if { $valid_connections == 0 } {
		foreach w [winfo children $subsample_frame0] {
		    enable_widget $w
		}
		foreach w [winfo children $subsample_frame1] {
		    enable_widget $w
		}
	    } else {
		foreach w [winfo children $subsample_frame0] {
		    disable_widget $w
		}
		foreach w [winfo children $subsample_frame1] {
		    disable_widget $w
		}
	    }

	    disableConnectionID $connections(scalar_to_subsample) $valid_connections
	    disableConnectionID $connections(subsample_to_choose) $valid_connections

	} else {
	    set disable 1

	    foreach w [winfo children $isosurfaces_frame0] {
		disable_widget $w
	    }
	    foreach w [winfo children $isosurfaces_frame1] {
		disable_widget $w
	    }

	    foreach w [winfo children $subsample_frame0] {
		disable_widget $w
	    }
	    foreach w [winfo children $subsample_frame1] {
		disable_widget $w
	    }

	    disableConnectionID $connections(scalar_to_subsample) 1
	    disableConnectionID $connections(subsample_to_choose) 1
	}


	disableConnectionID $connections(scalar_to_choose_iso) $disable
	disableConnectionID $connections(choose_to_iso)        $disable
	disableConnectionID $connections(iso_to_showfield)     $disable
	disableConnectionID $connections(colormap_iso_to_showfield)     $disable
#       Disconnecting a dynamic port causes a hang
#	disableConnectionID $connections(showfield_isosurfaces_to_sync) $disable

	enable_widget $isosurfaces_frame0.show
	enable_widget $isosurfaces_frame1.show

	if { $have_isosurfaces == 1 } {
	    $mods(ShowField-Isosurface-Surface)-c toggle_display_faces
	}
    }

    method toggle_isocontours { update } {
	global mods connections
	global $mods(ShowField-Isosurface-Surface)-faces-on
	global $mods(ShowField-Isosurface-Contour)-edges-on

	if { $DEBUG == 1 } {
	    puts stderr "in  toggle_isocontours [set $mods(ShowField-Isosurface-Surface)-faces-on] [set $mods(ShowField-Isosurface-Contour)-edges-on] $valid_connections $show_isocontours"
	}

	if { $update == 1 } {
	    if {[set $mods(ShowField-Isosurface-Surface)-faces-on] &&
		$valid_connections == 0 } {

		set $mods(ShowField-Isosurface-Contour)-edges-on \
		    $show_isocontours

		enable_widget $isosurfaces_frame0.isocontours
		enable_widget $isosurfaces_frame1.isocontours
	    } else {
		set show_isocontours \
		    [set $mods(ShowField-Isosurface-Contour)-edges-on]
		set $mods(ShowField-Isosurface-Contour)-edges-on 0

		disable_widget $isosurfaces_frame0.isocontours
		disable_widget $isosurfaces_frame1.isocontours
	    }
	}

	if {[set $mods(ShowField-Isosurface-Contour)-edges-on] } {
	    set disable 0
	} else {
	    set disable 1
	}

	disableConnectionID $connections(subsample_to_slicer_low)  $disable
	disableConnectionID $connections(subsample_to_slicer_high) $disable
	disableConnectionID $connections(slicer_low_to_iso)   $disable
	disableConnectionID $connections(slicer_high_to_iso)  $disable
	disableConnectionID $connections(iso_low_to_gather)   $disable
	disableConnectionID $connections(iso_high_to_gather)  $disable
	disableConnectionID $connections(gather_to_showfield) $disable
	disableConnectionID $connections(colormap_gather_to_showfield) $disable

#       Disconnecting a dynamic port causes a hang
#	disableConnectionID $connections(showfield_isocontours_to_sync) $disable
	
	if { $have_isosurfaces == 1 } {
	    $mods(ShowField-Isosurface-Contour)-c toggle_display_edges
	}
	
	if { $DEBUG == 1 } {
	    puts stderr "out toggle_isocontours [set $mods(ShowField-Isosurface-Surface)-faces-on] [set $mods(ShowField-Isosurface-Contour)-edges-on] $valid_connections $show_isocontours"
	}
    }


    method toggle_streamlines {} {
	toggle_integration 1

	global mods connections
	global $mods(ShowField-StreamLines-Vector)-edges-on
	global $mods(ShowField-StreamLines-Scalar)-nodes-on

	if { $DEBUG == 1 } {
	    puts stderr "toggle_streamlines [set $mods(ShowField-StreamLines-Vector)-edges-on] have_streamlines $have_streamlines"
	}

	if { [set $mods(ShowField-StreamLines-Vector)-edges-on] } {

	    set on 1
	    set disable 0

	    set cmd "$mods(StreamLines-rake)-c needexecute"

	    foreach w [winfo children $streamlines_frame0] {
		enable_widget $w
	    }
	    foreach w [winfo children $streamlines_frame1] {
		enable_widget $w
	    }

	    if { $valid_scalar  == 1 } {
		enable_widget $streamlines_frame0.cm.l
		enable_widget $streamlines_frame0.cm.scalar
		enable_widget $streamlines_frame0.cm.vector

		enable_widget $streamlines_frame1.cm.l
		enable_widget $streamlines_frame1.cm.scalar
		enable_widget $streamlines_frame1.cm.vector
	    } else {
		disable_widget $streamlines_frame0.cm.l
		disable_widget $streamlines_frame0.cm.scalar
		disable_widget $streamlines_frame0.cm.vector

		disable_widget $streamlines_frame1.cm.l
		disable_widget $streamlines_frame1.cm.scalar
		disable_widget $streamlines_frame1.cm.vector
	    }

	} else {
	    set on 0
	    set disable 1

	    set cmd ""

	    foreach w [winfo children $streamlines_frame0] {
		disable_widget $w
	    }
	    foreach w [winfo children $streamlines_frame1] {
		disable_widget $w
	    }
	}

	
	disableConnectionID $connections(vector_to_streamlines)       $disable
	disableConnectionID $connections(vector_to_sample)            $disable
	disableConnectionID $connections(sample_to_streamlines)       $disable
	disableConnectionID $connections(streamlines_to_interpolate)  $disable

	disableConnectionID $connections(vector_to_magnitude)    $disable
	disableConnectionID $connections(magnitude_to_choose)    $disable
	disableConnectionID $connections(choose_to_interpolate)  $disable
	disableConnectionID $connections(choose_to_rescalecolor) $disable
	disableConnectionID $connections(interpolate_to_showfield)    $disable
	disableConnectionID $connections(rescalecolor_interpolate_to_showfield)    $disable

#       Disconnecting a dynamic port causes a hang
#	disableConnectionID $connections(showfield_streamline_edges_to_sync) $disable
#	disableConnectionID $connections(sample_to_viewer) $disable

	set "$eviewer-StreamLines rake (4)" $on
	$eviewer-c redraw

#	    bind $streamlines_frame0.seeds.s <ButtonRelease> $cmd
#	    bind $streamlines_frame1.seeds.s <ButtonRelease> $cmd
#	    bind $streamlines_frame0.seeds.val <Return> $cmd
#	    bind $streamlines_frame1.seeds.val <Return> $cmd

#	    bind $streamlines_frame0.stepsize.s <ButtonRelease> $cmd
#	    bind $streamlines_frame1.stepsize.s <ButtonRelease> $cmd
#	    bind $streamlines_frame0.stepsize.val <Return> $cmd
#	    bind $streamlines_frame1.stepsize.val <Return> $cmd
	    
#	    bind $streamlines_frame0.steps.s <ButtonRelease> $cmd
#	    bind $streamlines_frame1.steps.s <ButtonRelease> $cmd
#	    bind $streamlines_frame0.steps.val <Return> $cmd
#	    bind $streamlines_frame1.steps.val <Return> $cmd

	enable_widget $streamlines_frame0.show
	enable_widget $streamlines_frame1.show

	if { $have_streamlines == 1 } {
	    $mods(ShowField-StreamLines-Vector)-c toggle_display_edges
	}
    }

    method toggle_integration { update } {
	global mods connections
	global $mods(ShowField-StreamLines-Vector)-edges-on
	global $mods(ShowField-StreamLines-Scalar)-nodes-on

	if { $DEBUG == 1 } {
	    puts stderr "toggle_integration [set $mods(ShowField-StreamLines-Vector)-edges-on] [set $mods(ShowField-StreamLines-Scalar)-nodes-on] have_streamlines $have_streamlines"
	}

	if { $update == 1 } {
	    if { [set $mods(ShowField-StreamLines-Vector)-edges-on] } {
		set $mods(ShowField-StreamLines-Scalar)-nodes-on \
		    $show_integration
	    } else {
		set show_integration \
		    [set $mods(ShowField-StreamLines-Scalar)-nodes-on]
		set $mods(ShowField-StreamLines-Scalar)-nodes-on 0
	    }
	}

	if {[set $mods(ShowField-StreamLines-Scalar)-nodes-on] == 0 } {
	    set disable 1

	    foreach w [winfo children $streamlines_frame0.integration] {
		disable_widget $w
	    }
	    foreach w [winfo children $streamlines_frame1.integration] {
		disable_widget $w
	    }

	    if { [set $mods(ShowField-StreamLines-Vector)-edges-on] } {
		enable_widget $streamlines_frame0.integration.show
		enable_widget $streamlines_frame1.integration.show
	    } else {
		disable_widget $streamlines_frame0.integration.show
		disable_widget $streamlines_frame1.integration.show
	    }
	} elseif { [set $mods(ShowField-StreamLines-Vector)-edges-on] } {
	    set disable 0
	    
	    foreach w [winfo children $streamlines_frame0.integration] {
		enable_widget $w
	    }
	    foreach w [winfo children $streamlines_frame1.integration] {
		enable_widget $w
	    }
	} else {
	    set disable 1
	}
	 
	disableConnectionID $connections(streamlines_to_showfield)    $disable
	disableConnectionID $connections(streamlines_to_rescalecolor) $disable
	disableConnectionID $connections(rescalecolor_streamlines_to_showfield)    $disable

#       Disconnecting a dynamic port causes a hang
#	disableConnectionID $connections(showfield_streamline_nodes_to_sync) $disable
	
	if { $have_streamlines == 1 } {
	    $mods(ShowField-StreamLines-Scalar)-c toggle_display_nodes
	}
    }


    method toggle_probes { probemod } {
	if { $DEBUG == 1 } {
	    puts stderr "toggle_probes $probemod"
	}
	global mods connections

	if { $probemod == $mods(Probe-Scalar) } {
	    global probe_scalar
	    if { $DEBUG == 1 } {
		puts stderr "Probe-Scalar $probe_scalar"
	    }
	    if { $probe_scalar == 1 } {
		set disable 0

		foreach w [winfo children $probe_scalar_frame0] {
		    enable_widget $w
		}
		foreach w [winfo children $probe_scalar_frame1] {
		    enable_widget $w
		}
	    } else {
		set disable 1

		foreach w [winfo children $probe_scalar_frame0] {
		    disable_widget $w
		}
		foreach w [winfo children $probe_scalar_frame1] {
		    disable_widget $w
		}
	    }

	    disableConnectionID $connections(scalar_to_probe)        $disable
#	    Disconnecting a dynamic port causes a hang
#	    disableConnectionID $connections(probe_scalar_to_viewer) $disable
	    set "$eviewer-Probe Selection Widget (2)" $probe_scalar
	    $eviewer-c redraw

	    if { $valid_scalar == 1 } {
		enable_widget $probe_scalar_frame0.show
		enable_widget $probe_scalar_frame1.show
	    }

	} elseif { $probemod == $mods(Probe-Vector) } {
	    global probe_vector
	    if { $DEBUG == 1 } {
		puts stderr "Probe-Vector $probe_vector"
	    }
	    if { $probe_vector == 1 } {
		set disable 0

		foreach w [winfo children $probe_vector_frame0] {
		    enable_widget $w
		}
		foreach w [winfo children $probe_vector_frame1] {
		    enable_widget $w
		}
	    } else {
		set disable 1

		foreach w [winfo children $probe_vector_frame0] {
		    disable_widget $w
		}
		foreach w [winfo children $probe_vector_frame1] {
		    disable_widget $w
		}
	    }

	    disableConnectionID $connections(vector_to_probe)        $disable
#	    Disconnecting a dynamic port causes a hang
#	    disableConnectionID $connections(probe_vector_to_viewer) $disable
	    set "$eviewer-Probe Selection Widget (3)" $probe_vector
	    $eviewer-c redraw

	    if { $valid_vector == 1 } {
		enable_widget $probe_vector_frame0.show
		enable_widget $probe_vector_frame1.show
	    }
	}
    }


    method change_vis_frame { which } {
	# change tabs for attached and detached

        if {$initialized != 0} {
	    if {$which == 0} {
		$vis_frame_tab0 view "Data Selection"
		$vis_frame_tab1 view "Data Selection"
		set c_left_tab "Data Selection"
	    } elseif {$which == 1} {
		# Data Vis
		$vis_frame_tab0 view "Vis Options"
		$vis_frame_tab1 view "Vis Options"
		set c_left_tab "Vis Options"
	    } else {
 		$vis_frame_tab0 view "Viewer Options"
 		$vis_frame_tab1 view "Viewer Options"
		set c_left_tab "Viewer Options"
	    }
	}
    }
    

    method change_data_tab { which } {
	# change data tab for attached/detached

	if {$initialized != 0} {
	    if {$which == 0} {
		$data_tab0 view "HDF5"
		$data_tab1 view "HDF5"

	    } elseif {$which == 1} {
		$data_tab0 view "MDSPlus"
		$data_tab1 view "MDSPlus"

	    } elseif {$which == 2} {
		$data_tab0 view "Data Subsample"
		$data_tab1 view "Data Subsample"
	    }
	}
    }


    method change_animate_tab { which } {
	# change animate tab for attached/detached

	if {$initialized != 0} {
	    if {$which == 0} {
		$animate_tab0 view "Scalar"
		$animate_tab1 view "Scalar"

	    } elseif {$which == 1} {
		$animate_tab0 view "Vector"
		$animate_tab1 view "Vector"

	    } elseif {$which == 2} {
		$animate_tab0 view "Locking"
		$animate_tab1 view "Locking"
	    }
	}
    }


    method change_option_tab { which } {
	# change option tab for attached/detached

	if {$initialized != 0} {
	    if {$which == 0} {
		$vis_tab0 view "Scalar"
		$vis_tab1 view "Scalar"

	    } elseif {$which == 1} {
		$vis_tab0 view "Vector"
		$vis_tab1 view "Vector"

	    } elseif {$which == 2} {
		$vis_tab0 view "Probes"
		$vis_tab1 view "Probes"

	    } elseif {$which == 3} {
		$vis_tab0 view "Misc"
		$vis_tab1 view "Misc"
	    }
	}
    }


    method change_iso_tab { which isomod suffix } {
	# change iso tab for attached/detached

	set tab0 _tab0
	set tab1 _tab1

	set st0 $suffix$tab0
	set st1 $suffix$tab1

        if {$initialized != 0} {
	    if {$which == 0} {
		[set $suffix$tab0] view "Slider"
		[set $suffix$tab1] view "Slider"
		
	    } elseif {$which == 1} {
		[set $suffix$tab0] view "Quantity"
		[set $suffix$tab1] view "Quantity"
		
	    } elseif {$which == 2} {
		[set $suffix$tab0] view "List"
		[set $suffix$tab1] view "List"
	    }
	}

	global mods
	global $isomod-active-isoval-selection-tab
	set $isomod-active-isoval-selection-tab $which
    }


    method change_color_tab { which } {
	# change tabs for attached and detached

        if {$initialized != 0} {
	    if {$which == 0} {
		$color_tab0 view "Isosurfaces"
		$color_tab1 view "Isosurfaces"
	    } elseif {$which == 1} {
		# Data Vis
		$color_tab0 view "Streamlines"
		$color_tab1 view "Streamlines"
	    } else {
 		$color_tab0 view "Other"
 		$color_tab1 view "Other"
	    }
	}
    }
    
    method update_hdf5_callback {varname varele varop} {

	if { $ignore_callbacks == 1 } {
	    return
	}

	if { $DEBUG == 1 } {
	    puts stderr "update_hdf5_callback"
	}
	global mods
	global $mods(HDF5-Points)-filename
	global $mods(HDF5-Connections)-filename
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename

	global filename_points
	global filename_connections
	global filename_scalar
	global filename_vector

	set tmp [set $mods(HDF5-Points)-filename]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set filename_points [string range $tmp $pos end]
	} else {
	    set filename_points $tmp
	}

	if { [string length $filename_points] == 0 } {
	    set filename_points "No Data Selected"
	}

	set tmp [set $mods(HDF5-Connections)-filename]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set filename_connections [string range $tmp $pos end]
	} else {
	    set filename_connections $tmp
	}

	if { [string length $filename_connections] == 0 } {
	    set filename_connections "No Data Selected"
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

	if { [string first "$mods(HDF5-Points)-filename" "$varname"] != -1 } {
	    update_point_modules
	} elseif { [string first "$mods(HDF5-Connections)-filename" "$varname"] != -1 } {
	    update_connection_modules
	} elseif { [string first "$mods(HDF5-Scalar)-filename" "$varname"] != -1 } {
	    update_scalar_modules
	} elseif { [string first "$mods(HDF5-Vector)-filename" "$varname"] != -1 } {
	    update_vector_modules
	}
    }


    method update_mdsplus_callback {varname varele varop} {
	if { $ignore_callbacks == 1 } {
	    return
	}

	if { $DEBUG == 1 } {
	    puts stderr "update_mdsplus_callback"
	}
	global mods
	global $mods(MDSPlus-Connections)-shot
	global $mods(MDSPlus-Points)-shot
	global $mods(MDSPlus-Scalar)-shot
	global $mods(MDSPlus-Vector)-shot

	global $mods(MDSPlus-Connections)-num-entries
	global $mods(MDSPlus-Points)-num-entries
	global $mods(MDSPlus-Scalar)-num-entries
	global $mods(MDSPlus-Vector)-num-entries

	global shot_points
	global shot_scalar
	global shot_vector

	set tmp [set $mods(MDSPlus-Points)-shot]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set shot_points [string range $tmp $pos end]
	} else {
	    set shot_points $tmp
	}

	if { [set $mods(MDSPlus-Points)-num-entries] == 0 ||
	     [string length $shot_points] == 0 } {
	    set shot_points "No Data Selected"
	}

	set tmp [set $mods(MDSPlus-Connections)-shot]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set shot_connections [string range $tmp $pos end]
	} else {
	    set shot_connections $tmp
	}

	if { [set $mods(MDSPlus-Connections)-num-entries] == 0 ||
	     [string length $shot_connections] == 0 } {
	    set shot_connections "No Data Selected"
	}

	set tmp [set $mods(MDSPlus-Scalar)-shot]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set shot_scalar [string range $tmp $pos end]
	} else {
	    set shot_scalar $tmp
	}

	if { [set $mods(MDSPlus-Scalar)-num-entries] == 0 ||
	     [string length $shot_scalar] == 0 } {
	    set shot_scalar "No Data Selected"
	}

	set tmp [set $mods(MDSPlus-Vector)-shot]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set shot_vector [string range $tmp $pos end]
	} else {
	    set shot_vector $tmp
	}

	if { [set $mods(MDSPlus-Vector)-num-entries] == 0 ||
	     [string length $shot_vector] == 0 } {
	    set shot_vector "No Data Selected"
	}


	if { [string first "$mods(MDSPlus-Points)-shot" "$varname"] != -1 } {
	    update_point_modules
	} elseif { [string first "$mods(MDSPlus-Connections)-shot" "$varname"] != -1 } {
	    update_connection_modules
	} elseif { [string first "$mods(MDSPlus-Scalar)-shot" "$varname"] != -1 } {
	    update_scalar_modules
	} elseif { [string first "$mods(MDSPlus-Vector)-shot" "$varname"] != -1 } {
	    update_vector_modules
	}
    }


    method update_animate_callback {varname varele varop} {
	if { $ignore_callbacks == 1 } {
	    return
	}

	if { $DEBUG == 1 } {
	    puts stderr "update_animate_callback"
	}
	global mods
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename
	global $mods(HDF5-Scalar)-animate
	global $mods(HDF5-Vector)-animate

        if {$initialized != 0} {
	    if { [string length [set $mods(HDF5-Scalar)-filename]] &&
		 [set $mods(HDF5-Scalar)-animate] == 1 } {
		foreach w [winfo children $animate_scalar_tab0] {
		    enable_widget $w
		}
		foreach w [winfo children $animate_scalar_tab1] {
		    enable_widget $w
		}
	    } else {
		foreach w [winfo children $animate_scalar_tab0] {
		    disable_widget $w
		}
		foreach w [winfo children $animate_scalar_tab1] {
		    disable_widget $w
		}
	    }

	    if { [string length [set $mods(HDF5-Vector)-filename]] &&
		 [set $mods(HDF5-Vector)-animate] == 1 } {
		foreach w [winfo children $animate_vector_tab0] {
		    enable_widget $w
		}
		foreach w [winfo children $animate_vector_tab1] {
		    enable_widget $w
		}
	    } else {
		foreach w [winfo children $animate_vector_tab0] {
		    disable_widget $w
		}
		foreach w [winfo children $animate_vector_tab1] {
		    disable_widget $w
		}
	    }
	}
    }


    method update_current_callback {varname varele varop} {
	if { $ignore_callbacks == 1 } {
	    return
	}

	if { $DEBUG == 1 } {
	    puts stderr "update_current_callback"
	}
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


    method update_slicer_callback {varname varele varop} {
	if { $ignore_callbacks == 1 } {
	    return
	}

	if { $DEBUG == 1 } {
	    puts stderr "update_slicer_callback"
	}
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

	    set $mods(Slicer-High)-$index-index  \
		[expr [set $mods(Slicer-High)-$index-dim] - 1]
	    set $mods(Slicer-High)-$index-index2 \
		[expr [set $mods(Slicer-High)-$index-dim] - 1]
	}
    }



    method update_isotab_callback {varname varele varop} {
	if { $ignore_callbacks == 1 } {
	    return
	}

	if { $DEBUG == 1 } {
	    puts stderr "update_isotab_callback"
	}
	global mods

	global $mods(Isosurface-Surface)-active-isoval-selection-tab
	global $mods(Isosurface-Contour-Low)-active-isoval-selection-tab
	global $mods(Isosurface-Contour-High)-active-isoval-selection-tab
	
	set $mods(Isosurface-Contour-Low)-active-isoval-selection-tab \
	    [set $mods(Isosurface-Surface)-active-isoval-selection-tab]
	set $mods(Isosurface-Contour-High)-active-isoval-selection-tab \
	    [set $mods(Isosurface-Surface)-active-isoval-selection-tab]
    }

    method update_minmax_callback {varname varele varop} {

	if { $ignore_callbacks == 1 } {
	    return
	}

	if { $DEBUG == 1 } {
	    puts stderr "update_minmax_callback $varname $varele $varop"
	}
	global mods
 	global $mods(Isosurface-Surface)-isoval-min
	global $mods(Isosurface-Surface)-isoval-max
 	set min [set $mods(Isosurface-Surface)-isoval-min]
 	set max [set $mods(Isosurface-Surface)-isoval-max]

	if { $min < $max } { 
	    set lg [expr floor( log10($max-$min) ) ]
	    set range [expr pow(10.0, $lg )]
	    
	    set scale 1.0
	    
	    if { $lg > 5.0 } {
		set scale [expr pow(10.0, $lg-5 )]
	    }

	    set res [expr $range/(1.0e4*$scale)]

	    set w $iso_slider_tab0.isoval.s
	    if [ expr [winfo exists $w] ] {
		$w configure -from $min -to $max
		$w configure -resolution $res
	    }

	    set w $iso_slider_tab1.isoval.s
	    if [ expr [winfo exists $w] ] {
		$w configure -from $min -to $max
		$w configure -resolution $res
	    }
	}
	
 	global $mods(Isosurface-Scalar-Slice)-isoval-min
	global $mods(Isosurface-Scalar-Slice)-isoval-max
 	set min [set $mods(Isosurface-Scalar-Slice)-isoval-min]
 	set max [set $mods(Isosurface-Scalar-Slice)-isoval-max]

	if { $min < $max } { 
	    set res [expr ($max - $min)/100.]

	    set w $slice_slider_tab0.isoval.s
	    if [ expr [winfo exists $w] ] {
		$w configure -from $min -to $max
		$w configure -resolution $res
	    }

	    set w $slice_slider_tab1.isoval.s
	    if [ expr [winfo exists $w] ] {
		$w configure -from $min -to $max
		$w configure -resolution $res
	    }
	}
    }
	

    method update_probe_callback {varname varele varop} {
	if { $ignore_callbacks == 1 } {
	    return
	}

	global mods
	global $mods(Probe-Scalar)-locx
	global $mods(Probe-Scalar)-locy
	global $mods(Probe-Scalar)-locz
	global $mods(Probe-Vector)-locx
	global $mods(Probe-Vector)-locy
	global $mods(Probe-Vector)-locz

	global probe_lock

        if { $probe_lock != 0 } {
	    if { [string first "$mods(Probe-Scalar)-locx" "$varname"] != -1 ||
		 [string first "$mods(Probe-Scalar)-locy" "$varname"] != -1 ||
		 [string first "$mods(Probe-Scalar)-locz" "$varname"] != -1 } {
		if { [set $mods(Probe-Vector)-locx] !=
		     [set $mods(Probe-Scalar)-locx] } {
		    set $mods(Probe-Vector)-locx [set $mods(Probe-Scalar)-locx]
		}
		if { [set $mods(Probe-Vector)-locy] !=
		     [set $mods(Probe-Scalar)-locy] } {
		    set $mods(Probe-Vector)-locy [set $mods(Probe-Scalar)-locy]
		}
		if { [set $mods(Probe-Vector)-locz] !=
		     [set $mods(Probe-Scalar)-locz] } {
		    set $mods(Probe-Vector)-locz [set $mods(Probe-Scalar)-locz]
		}

		if {$allow_execution != 0} {
		    $mods(Probe-Vector) move_location
		}
	    } elseif { [string first "$mods(Probe-Vector)-locx" "$varname"] != -1 ||
		       [string first "$mods(Probe-Vector)-locy" "$varname"] != -1 ||
		       [string first "$mods(Probe-Vector)-locz" "$varname"] != -1 } {
		if { [set $mods(Probe-Scalar)-locx] !=
		     [set $mods(Probe-Vector)-locx] } {
		    set $mods(Probe-Scalar)-locx [set $mods(Probe-Vector)-locx]
		}
		if { [set $mods(Probe-Scalar)-locy] !=
		     [set $mods(Probe-Vector)-locy] } {
		    set $mods(Probe-Scalar)-locy [set $mods(Probe-Vector)-locy]
		}
		if { [set $mods(Probe-Scalar)-locz] !=
		     [set $mods(Probe-Vector)-locz] } {
		    set $mods(Probe-Scalar)-locz [set $mods(Probe-Vector)-locz]
		}

		if {$allow_execution != 0} {
		    $mods(Probe-Scalar) move_location
		}
	    }
	}
    }


############ Vector
    method update_vector_modules { } {
	global mods connections
	global $mods(HDF5-Vector)-filename
	global $mods(MDSPlus-Vector)-num-entries

	if { $DEBUG == 1 } {
	    puts stderr "update_vector_modules $valid_vector [set $mods(HDF5-Vector)-filename] [set $mods(MDSPlus-Vector)-num-entries]"
	}
	global probe_vector

	global $mods(ShowField-StreamLines-Vector)-edges-on

	set disable -1

	if { [string length [set $mods(HDF5-Vector)-filename]] == 0 &&
	     [set $mods(MDSPlus-Vector)-num-entries] == 0 } {

	    if {$valid_vector != 0} {
		set disable 1
		set valid_vector 0
		set probe_vector 0
		
		set $mods(ShowField-StreamLines-Vector)-edges-on 0
		
		toggle_streamlines
		toggle_probes $mods(Probe-Vector)

		disable_widget $streamlines_frame0.show
		disable_widget $streamlines_frame1.show

		disableConnectionID $connections(cv_to_vector) 1
		disableConnectionID $connections(cc_to_vector) 1
		disableConnectionID $connections(cp_to_vector) 1
	    }
	} else {

	    if {$valid_vector != 1} {
		set disable 0
		set valid_vector 1
		set probe_vector 1

#		set $mods(ShowField-StreamLines-Vector)-edges-on 1

		toggle_streamlines
		toggle_probes $mods(Probe-Vector)

		global $mods(ChooseField-Interpolate)-port-index
		set $mods(ChooseField-Interpolate)-port-index 1

		if { $valid_scalar } {
		    enable_widget $streamlines_frame0.cm.l
		    enable_widget $streamlines_frame0.cm.scalar
		    enable_widget $streamlines_frame0.cm.vector
		    
		    enable_widget $streamlines_frame1.cm.l
		    enable_widget $streamlines_frame1.cm.scalar
		    enable_widget $streamlines_frame1.cm.vector
		}

		disableConnectionID $connections(cv_to_vector) 0

		if {$valid_connections == 1} {
		    disableConnectionID $connections(cc_to_vector) 0
		}
		if {$valid_points == 1} {
		    disableConnectionID $connections(cp_to_vector) 0
		}
		if {$valid_scalar == 1} {
		    disableConnectionID $connections(scalar_to_choose) 0
		}
	    }
	}

	if { $disable != -1 } {
	    disableConnectionID $connections(vector_to_info)         $disable
	}

	if { [string length [set $mods(HDF5-Vector)-filename]] == 0 } {
	    disableConnectionID $connections(hdf5_to_cv) 1
	} else {
	    disableConnectionID $connections(hdf5_to_cv) 0
	}

	if { [set $mods(MDSPlus-Vector)-num-entries] == 0 } {
	    disableConnectionID $connections(mds_to_cv) 1
	} else {
	    disableConnectionID $connections(mds_to_cv) 0
	}
    }
    

############ Scalar
    method update_scalar_modules { } {
	global mods connections
	global $mods(HDF5-Scalar)-filename
	global $mods(MDSPlus-Scalar)-num-entries

	global probe_scalar

	if { $DEBUG == 1 } {
	    puts stderr "update_scalar_modules $valid_scalar [set $mods(HDF5-Scalar)-filename] [set $mods(MDSPlus-Scalar)-num-entries]"
	}
	
	global $mods(ShowField-Scalar-Slice-Edge)-edges-on
	global $mods(ShowField-Isosurface-Surface)-faces-on

	set disable -1

	if { [string length [set $mods(HDF5-Scalar)-filename]] == 0 &&
	     [set $mods(MDSPlus-Scalar)-num-entries] == 0 } {

	    if {$valid_scalar != 0} {
		set disable 1
		set valid_scalar 0
		set probe_scalar 0
		
		set $mods(ShowField-Scalar-Slice-Edge)-edges-on 0
		set $mods(ShowField-Isosurface-Surface)-faces-on 0
		
		toggle_scalarslice
		toggle_isosurfaces
		toggle_probes $mods(Probe-Scalar)
		
		disable_widget $isosurfaces_frame0.show
		disable_widget $isosurfaces_frame1.show
		disable_widget $scalarslice_frame0.show
		disable_widget $scalarslice_frame1.show
		

		global $mods(ChooseField-Interpolate)-port-index
		set $mods(ChooseField-Interpolate)-port-index 1

		disable_widget $streamlines_frame0.cm.l
		disable_widget $streamlines_frame0.cm.scalar
		disable_widget $streamlines_frame0.cm.vector
		
		disable_widget $streamlines_frame1.cm.l
		disable_widget $streamlines_frame1.cm.scalar
		disable_widget $streamlines_frame1.cm.vector

		disableConnectionID $connections(cs_to_scalar) 1
		disableConnectionID $connections(cc_to_scalar) 1
		disableConnectionID $connections(cp_to_scalar) 1

		disableConnectionID $connections(scalar_to_choose) 1
	    }
	} else {

	    if {$valid_scalar != 1} {
		set disable 0
		set valid_scalar 1
		set probe_scalar 1
		
#		set $mods(ShowField-Scalar-Slice-Edge)-edges-on 1
#		set $mods(ShowField-Isosurface-Surface)-faces-on 1
		
		toggle_scalarslice
		toggle_isosurfaces
		toggle_probes $mods(Probe-Scalar)

		if { $valid_vector } {
		    enable_widget $streamlines_frame0.cm.l
		    enable_widget $streamlines_frame0.cm.scalar
		    enable_widget $streamlines_frame0.cm.vector
		    
		    enable_widget $streamlines_frame1.cm.l
		    enable_widget $streamlines_frame1.cm.scalar
		    enable_widget $streamlines_frame1.cm.vector
		}

		disableConnectionID $connections(cs_to_scalar) 0

		if {$valid_connections == 1} {
		    disableConnectionID $connections(cc_to_scalar) 0
		}
		if {$valid_points == 1} {
		    disableConnectionID $connections(cp_to_scalar) 0
		}

		if {$valid_vector == 1} {
		    disableConnectionID $connections(scalar_to_choose) 0
		}
	    }
	}


	if { $disable != -1 } {
	    disableConnectionID $connections(scalar_to_color) $disable
	    disableConnectionID $connections(scalar_to_info)  $disable
	}
	
	if { [string length [set $mods(HDF5-Scalar)-filename]] == 0 } {
	    disableConnectionID $connections(hdf5_to_cs) 1
	} else {
	    disableConnectionID $connections(hdf5_to_cs) 0
	}

	if { [set $mods(MDSPlus-Scalar)-num-entries] == 0 } {
	    disableConnectionID $connections(mds_to_cs) 1
	} else {
	    disableConnectionID $connections(mds_to_cs) 0
	}
    }
    
############ Connections
    method update_connection_modules { } {
	if { $DEBUG == 1 } {
	    puts stderr "update_connection_modules"
	}
	global mods connections
	global $mods(HDF5-Connections)-filename
	global $mods(MDSPlus-Connections)-num-entries

	global $mods(ShowField-Isosurface-Contour)-edges-on

	if { [string length [set $mods(HDF5-Connections)-filename]] == 0 &&
	     [set $mods(MDSPlus-Connections)-num-entries] == 0 } {

	    if {$valid_connections != 0} {
		set valid_connections 0
		toggle_isocontours 1
	    }

	    disableConnectionID $connections(cc_to_scalar) 1
	    disableConnectionID $connections(cc_to_vector) 1

	} else {
	    if {$valid_connections != 1} {
		set valid_connections 1
		toggle_isocontours 1
	    }

	    if {$valid_scalar == 1} {
		disableConnectionID $connections(cc_to_scalar) 0
	    }

	    if {$valid_vector == 1} {
		disableConnectionID $connections(cc_to_vector) 0
	    }
	}

	if { [string length [set $mods(HDF5-Connections)-filename]] == 0 } {
	    disableConnectionID $connections(hdf5_to_cc) 1
	} else {
	    disableConnectionID $connections(hdf5_to_cc) 0
	}

	if { [set $mods(MDSPlus-Connections)-num-entries] == 0 } {
	    disableConnectionID $connections(mds_to_cc) 1
	} else {
	    disableConnectionID $connections(mds_to_cc) 0
	}
   }

    
############ Points
    method update_point_modules { } {
	if { $DEBUG == 1 } {
	    puts stderr "update_point_modules"
	}
	global mods connections
	global $mods(HDF5-Points)-filename
	global $mods(MDSPlus-Points)-num-entries

#	upvar \#0 $mods(MDSPlus-Points)-num-entries ne

	if { [string length [set $mods(HDF5-Points)-filename]] == 0 &&
	     [set $mods(MDSPlus-Points)-num-entries] == 0 } {
	    set valid_points 0

	    disableConnectionID $connections(cp_to_scalar) 1
	    disableConnectionID $connections(cp_to_vector) 1

	} else {
	    set valid_points 1

	    if {$valid_scalar == 1} {
		disableConnectionID $connections(cp_to_scalar) 0
	    }
	    if {$valid_vector == 1} {
		disableConnectionID $connections(cp_to_vector) 0
	    }
	}

	if { [string length [set $mods(HDF5-Points)-filename]] == 0 } {
	    disableConnectionID $connections(hdf5_to_cp) 1
	} else {
	    disableConnectionID $connections(hdf5_to_cp) 0
	}

	if { [set $mods(MDSPlus-Points)-num-entries] == 0 } {
	    disableConnectionID $connections(mds_to_cp) 1
	} else {
	    disableConnectionID $connections(mds_to_cp) 0
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


    method load_session_data {} {
	wm title .standalone "FusionViewer - [getFileName $saveFile]"

	# Because the HDF5/MDSPlus is dynamic the entries must be deleted before reseting.
	global mods
#	set ignore_callbacks 1
	$mods(HDF5-Points) clear
	$mods(HDF5-Connections) clear
	$mods(HDF5-Scalar) clear
	$mods(HDF5-Vector) clear
	$mods(MDSPlus-Points) deleteEntry 1
	$mods(MDSPlus-Connections) deleteEntry 1
	$mods(MDSPlus-Scalar) deleteEntry 1
	$mods(MDSPlus-Vector) deleteEntry 1
#	set ignore_callbacks 0

	update_state

#	set ignore_callbacks 1

	# reset defaults 
	reset_defaults
	
	foreach g [info globals] {
	    global $g
	}

	set debug $DEBUG

	if { $DEBUG == 1 } {
	    puts stderr "sourcing"
	}

	source $saveFile

	set DEBUG $debug

	if { $DEBUG == 1 } {
	    puts stderr "done sourcing"
	}

        # local state vars that must be reset
	set valid_points -1
	set valid_connections -1
	set valid_scalar -1
	set valid_vector -1

	set have_scalarslice 0
	set have_isosurfaces 0
	set have_streamlines 0

	global probe_scalar
	global probe_vector

	set probe_scalar 0
	set probe_vector 0
	
	set show_faces       \
	    [set $mods(ShowField-Scalar-Slice-Face)-faces-on]
	set show_isocontours \
	    [set $mods(ShowField-Isosurface-Contour)-edges-on]
	set show_integration \
	    [set $mods(ShowField-StreamLines-Scalar)-nodes-on]

	global $mods(ChooseNrrd-Points)-usefirstvalid 1
	global $mods(ChooseNrrd-Connections)-usefirstvalid 1
	global $mods(ChooseNrrd-Scalar)-usefirstvalid 1
	global $mods(ChooseNrrd-Vector)-usefirstvalid 1

	set $mods(ChooseNrrd-Points)-usefirstvalid 1
	set $mods(ChooseNrrd-Connections)-usefirstvalid 1
	set $mods(ChooseNrrd-Scalar)-usefirstvalid 1
	set $mods(ChooseNrrd-Vector)-usefirstvalid 1

	global $mods(ChooseField-Isosurface-Surface)-usefirstvalid
	set $mods(ChooseField-Isosurface-Surface)-usefirstvalid 1

	spin-quantity 0 slice_quantity_tab \
	    $mods(Isosurface-Scalar-Slice)-isoval-quantity

	spin-quantity 0 contour_quantity_tab \
	    $mods(Isosurface-Slice-Contours)-isoval-quantity

	spin-quantity 0 iso_quantity_tab \
	    $mods(Isosurface-Surface)-isoval-quantity


	# set a few variables that need to be reset
	set indicate 0
	set cycle 0
	set IsVAttached 1
	set executing_modules 0
	
	# configure all tabs by calling all configure functions
	if {$c_left_tab != ""} {
	    $vis_frame_tab0 view $c_left_tab
	    $vis_frame_tab1 view $c_left_tab
	}

	change_indicator_labels "Press Execute to Load Data..."

#	set ignore_callbacks 0

	update_state
    }	


    method save_session {} {
	global mods
	
	if {$saveFile == ""} {	    
	    save_session_as

	} else {
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

    ##########################
    ### save_module_variables
    ##########################
    # This method saves out the variables of all of the modules to the
    # specified file. It currently only saves out the variables for the
    # modules that the application has included in the global mods array.
    method save_module_variables { fileid } {
	# make globals accessible
	foreach g [info globals] {
	    global $g
	}

 	puts $fileid "global mods"
 	set script "\n"
	
	global mods

	$mods(HDF5-Points)      writeStateToScript script "\$mods(HDF5-Points)" ""
	$mods(HDF5-Connections) writeStateToScript script "\$mods(HDF5-Connections)" ""
	$mods(HDF5-Scalar)      writeStateToScript script "\$mods(HDF5-Scalar)" ""
	$mods(HDF5-Vector)      writeStateToScript script "\$mods(HDF5-Vector)" ""

	$mods(MDSPlus-Points)      writeStateToScript script "\$mods(MDSPlus-Points)" ""
	$mods(MDSPlus-Connections) writeStateToScript script "\$mods(MDSPlus-Connections)" ""
	$mods(MDSPlus-Scalar)      writeStateToScript script "\$mods(MDSPlus-Scalar)" ""
	$mods(MDSPlus-Vector)      writeStateToScript script "\$mods(MDSPlus-Vector)" ""

	$mods(ChooseNrrd-Points)      writeStateToScript script "\$mods(ChooseNrrd-Points)" ""
	$mods(ChooseNrrd-Connections) writeStateToScript script "\$mods(ChooseNrrd-Connections)" ""
	$mods(ChooseNrrd-Scalar)      writeStateToScript script "\$mods(ChooseNrrd-Scalar)" ""
	$mods(ChooseNrrd-Vector)      writeStateToScript script "\$mods(ChooseNrrd-Vector)" ""

	$mods(NrrdToField-Scalar) writeStateToScript script "\$mods(NrrdToField-Scalar)" ""
	$mods(NrrdToField-Vector) writeStateToScript script "\$mods(NrrdToField-Vector)" ""

	$mods(Probe-Scalar) writeStateToScript script "\$mods(Probe-Scalar)" ""
	$mods(Probe-Vector) writeStateToScript script "\$mods(Probe-Scalar)" ""

	$mods(FieldInfo-Scalar) writeStateToScript script "\$mods(FieldInfo-Scalar)" ""
	$mods(FieldInfo-Vector) writeStateToScript script "\$mods(FieldInfo-Vector)" ""
	$mods(ChooseField-Isosurface-Surface) writeStateToScript script "\$mods(ChooseField-Isosurface-Surface)" ""


	$mods(SubSample)   writeStateToScript script "\$mods(SubSample)" ""
	$mods(Slicer-Low)  writeStateToScript script "\$mods(Slicer-Low)" ""
	$mods(Slicer-High) writeStateToScript script "\$mods(Slicer-High)" ""

	$mods(Isosurface-Surface)       writeStateToScript script "\$mods(Isosurface-Surface)" ""
	$mods(Isosurface-Contour-Low)   writeStateToScript script "\$mods(Isosurface-Contour-Low)" ""
	$mods(Isosurface-Contour-High)  writeStateToScript script "\$mods(Isosurface-Contour-High)" ""

	$mods(ShowField-Isosurface-Surface)  writeStateToScript script "\$mods(ShowField-Isosurface-Surface)" ""
	$mods(ShowField-Isosurface-Contour)  writeStateToScript script "\$mods(ShowField-Isosurface-Contour)" ""

	$mods(TransformData-Scalar-Slice)           writeStateToScript script "\$mods(TransformData-Scalar-Slice)" ""
	$mods(Isosurface-Scalar-Slice)              writeStateToScript script "\$mods(Isosurface-Scalar-Slice)" ""
	$mods(ApplyInterpMatrix-Scalar-Slice-Iso)   writeStateToScript script "\$mods(ApplyInterpMatrix-Scalar-Slice-Iso)" ""
	$mods(ApplyInterpMatrix-Scalar-Slice-Clip)  writeStateToScript script "\$mods(ApplyInterpMatrix-Scalar-Slice-Clip)" ""
	$mods(ClipField-Scalar-Slice)               writeStateToScript script "\$mods(ClipField-Scalar-Slice)" ""
	$mods(Isosurface-Slice-Contours)            writeStateToScript script "\$mods(Isosurface-Slice-Contours)" ""
	$mods(ShowField-Scalar-Slice-Face)          writeStateToScript script "\$mods(ShowField-Scalar-Slice-Face)" ""
	$mods(ShowField-Scalar-Slice-Edge)          writeStateToScript script "\$mods(ShowField-Scalar-Slice-Edge)" ""

	$mods(StreamLines-rake)  writeStateToScript script "\$mods(StreamLines-rake)" ""
	$mods(StreamLines)  writeStateToScript script "\$mods(StreamLines)" ""

	$mods(DirectInterpolate-StreamLines-Vector)  writeStateToScript script "\$mods(DirectInterpolate-StreamLines-Vector)" ""
	$mods(ShowField-StreamLines-Vector)  writeStateToScript script "\$mods(ShowField-StreamLines-Vector)" ""
	$mods(ShowField-StreamLines-Scalar)  writeStateToScript script "\$mods(ShowField-StreamLines-Scalar)" ""

	$mods(ChooseField-Interpolate)  writeStateToScript script "\$mods(ChooseField-Interpolate)" ""


	$mods(ColorMap-Isosurfaces) writeStateToScript script "\$mods(ColorMap-Isosurfaces)" ""
	$mods(ColorMap-Streamlines) writeStateToScript script "\$mods(ColorMap-Streamlines)" ""
	$mods(ColorMap-Other)       writeStateToScript script "\$mods(ColorMap-Other)" ""

	$mods(RescaleColorMap-Isosurfaces) writeStateToScript script "\$mods(RescaleColorMap-Isosurfaces)" ""
	$mods(RescaleColorMap-Streamlines) writeStateToScript script "\$mods(RescaleColorMap-Streamlines)" ""
	$mods(RescaleColorMap-Other)       writeStateToScript script "\$mods(RescaleColorMap-Other)" ""

	$mods(Synchronize) writeStateToScript script "\$mods(Synchronize)" ""
	$mods(Viewer)      writeStateToScript script "\$mods(Viewer)" ""


# 	set searchID [array startsearch mods]
# 	while {[array anymore mods $searchID]} {
# 	    set m [array nextelement mods $searchID]
	    # Call same method called for writing networks
	    # which writes them out in consistent order and
	    # only if they differ from the default.
# 	    $mods($m) writeStateToScript script "\$mods($m)" ""
# 	}
# 	array donesearch mods $searchID
	
 	puts $fileid "$script"
	puts $fileid "::netedit scheduleok"
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

	} elseif {$which == $mods(ShowField-Scalar-Slice-Edge) && $state == "JustStarted"} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Scalar-Slice-Edge) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Scalar-Slice-Face) && $state == "JustStarted"} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Scalar-Slice-Face) && $state == "Completed"} {
	    change_indicate_val 2
	}
    }

    
    method indicate_error { which msg_state } {
	if {$msg_state == "Error"} {
	    if {$error_module == ""} {
		set error_module $which

		set pos [string last "RescaleColorMap" $which]
		if {$pos != -1} { return }

		set pos [string last "ChooseNrrd" $which]
		if {$pos != -1} { return }

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
#	update_isovals
	update_point_modules
	update_connection_modules
	update_scalar_modules
	update_vector_modules

	global mods 

	global $mods(HDF5-Points)-filename
	global $mods(HDF5-Connections)-filename
	global $mods(HDF5-Scalar)-filename
	global $mods(HDF5-Vector)-filename

	global $mods(MDSPlus-Points)-num-entries
	global $mods(MDSPlus-Connections)-num-entries
	global $mods(MDSPlus-Scalar)-num-entries
	global $mods(MDSPlus-Vector)-num-entries

	if { [string length [set $mods(HDF5-Points)-filename]] != 0 } {
	    $mods(HDF5-Points)-c needexecute
	}
	if { [string length [set $mods(HDF5-Connections)-filename]] != 0 } {
	    $mods(HDF5-Connections)-c needexecute
	}
	if { [string length [set $mods(HDF5-Scalar)-filename]] != 0 } {
	    $mods(HDF5-Scalar)-c needexecute
	}
	if { [string length [set $mods(HDF5-Vector)-filename]] != 0 } {
	    $mods(HDF5-Vector)-c needexecute
	}

	if { [set $mods(MDSPlus-Points)-num-entries] != 0 } {
	    $mods(MDSPlus-Points)-c needexecute
	}
	if { [set $mods(MDSPlus-Connections)-num-entries] != 0 } {
	    $mods(MDSPlus-Connections)-c needexecute
	}
	if { [set $mods(MDSPlus-Scalar)-num-entries] != 0 } {
	    $mods(MDSPlus-Scalar)-c needexecute
	}
	if { [set $mods(MDSPlus-Vector)-num-entries] != 0 } {
	    $mods(MDSPlus-Vector)-c needexecute
	}

	set have_scalarslice $valid_scalar
	set have_isosurfaces $valid_scalar
	set have_streamlines $valid_vector
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
	$indicatorL0 configure -text $msg
	$indicatorL1 configure -text $msg
    }
    
    
    method set-quantity {new quantity} {
	global $quantity
	if {! [regexp "\\A\\d*\\.*\\d+\\Z" $quantity]} {
	    return 0
	} elseif {$quantity < 1.0} {
	    return 0
	}
	set $quantity $new
	return 1
    }

    method spin-quantity {step tab quantity} {
	global $quantity
	set newquantity [expr [set $quantity] + $step]

	if {$newquantity < 1.0} {
	    set newquantity 0
	}   

	set $quantity $newquantity

	set c0 0
	set tab0 $tab$c0
	set spinner [set $tab0].isoquant
	enable_widget $spinner
	$spinner delete 0 end
	$spinner insert 1 [set $quantity]

	set c1 1
	set tab1 $tab$c1
	set spinner [set $tab1].isoquant
	enable_widget $spinner
	$spinner delete 0 end
	$spinner insert 1 [set $quantity]
    }

    # Visualiztion frame tabnotebook
    variable vis_frame_tab0
    variable vis_frame_tab1
    variable c_left_tab

    # Data tabs notebook
    variable data_tab0
    variable data_tab1

    variable data_hdf5_tab0
    variable data_hdf5_tab1

    variable data_mdsplus_tab0
    variable data_mdsplus_tab1

    variable subsample_frame0
    variable subsample_frame1

    # Animate tabs notebook
    variable animate_tab0
    variable animate_tab1

    variable animate_scalar_tab0
    variable animate_scalar_tab1

    variable animate_vector_tab0
    variable animate_vector_tab1

    variable animate_lock_tab0
    variable animate_lock_tab1


    # Vis options tabs notebook
    variable vis_tab0
    variable vis_tab1

    variable vis_scalar_tab0
    variable vis_scalar_tab1

    variable vis_vector_tab0
    variable vis_vector_tab1

    variable vis_probe_tab0
    variable vis_probe_tab1

    variable vis_misc_tab0
    variable vis_misc_tab1

    # Scalar Slice
    variable scalarslice_frame0
    variable scalarslice_frame1

    variable slice_tab0
    variable slice_tab1

    variable slice_slider_tab0
    variable slice_slider_tab1

    variable slice_quantity_tab0
    variable slice_quantity_tab1

    variable slice_list_tab0
    variable slice_list_tab1

    # Scalar Slice Contours
    variable contour_tab0
    variable contour_tab1

    variable contour_slider_tab0
    variable contour_slider_tab1

    variable contour_quantity_tab0
    variable contour_quantity_tab1

    variable contour_list_tab0
    variable contour_list_tab1

    # Isosurfaces
    variable isosurfaces_frame0
    variable isosurfaces_frame1

    variable iso_tab0
    variable iso_tab1

    variable iso_slider_tab0
    variable iso_slider_tab1

    variable iso_quantity_tab0
    variable iso_quantity_tab1

    variable iso_list_tab0
    variable iso_list_tab1

    # Streamlines
    variable streamlines_frame0
    variable streamlines_frame1

    # Vector Probe
    variable probe_scalar_frame0
    variable probe_scalar_frame1

    variable probe_vector_frame0
    variable probe_vector_frame1

    # Colormaps
    variable color_tab0
    variable color_tab1

    variable color_isosurfaces_tab0
    variable color_isosurfaces_tab1

    variable color_streamlines_tab0
    variable color_streamlines_tab1

    variable color_other_tab0
    variable color_other_tab1

    # Application placing and size
    variable notebook_width
    variable notebook_height

    variable show_faces
    variable show_isocontours
    variable show_integration

    variable ignore_callbacks 

    variable valid_points
    variable valid_connections
    variable valid_scalar
    variable valid_vector

    variable have_scalarslice
    variable have_isosurfaces
    variable have_streamlines

    variable DEBUG
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
