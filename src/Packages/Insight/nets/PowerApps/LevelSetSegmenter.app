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

setProgressText "Loading LevelSetSegmenter Modules..."


#######################################################################
# Check environment variables.  Ask user for input if not set:
init_DATADIR_and_DATASET
############# NET ##############

::netedit dontschedule

# Create a Teem->NrrdData->NrrdInfo Module
set m1 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 99 192]

# Create a Teem->UnuAtoM->UnuMinmax Module
set m2 [addModuleAtPosition "Teem" "UnuAtoM" "UnuMinmax" 123 129]

# Create a Insight->Filters->ThresholdSegmentationLevelSetImageFilter Module
set m3 [addModuleAtPosition "Insight" "Filters" "ThresholdSegmentationLevelSetImageFilter" 415 1009]

# Create a Insight->Filters->BinaryThresholdImageFilter Module
set m4 [addModuleAtPosition "Insight" "Filters" "BinaryThresholdImageFilter" 23 1116]

# Create a SCIRun->FieldsCreate->SeedPoints Module
set m5 [addModuleAtPosition "SCIRun" "FieldsCreate" "SeedPoints" 1137 505]

# Create a Insight->Filters->BinaryThresholdImageFilter Module
set m6 [addModuleAtPosition "Insight" "Filters" "BinaryThresholdImageFilter" 826 446]
set Notes($m6) {Thresholding
seed.}
set Notes($m6-Position) {def}
set Notes($m6-Color) {white}

# Create a Insight->Converters->ImageToField Module
set m7 [addModuleAtPosition "Insight" "Converters" "ImageToField" 1137 447]
set Notes($m7) {Positive
seed points.}
set Notes($m7-Position) {def}
set Notes($m7-Color) {white}

# Create a SCIRun->Visualization->RescaleColorMap Module
set m8 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 364 629]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m9 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 306 1354]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m10 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1370 1194]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m11 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 52 271]

# Create a SCIRun->Render->ViewSlices Module
set m12 [addModuleAtPosition "SCIRun" "Render" "ViewSlices" 16 333]

# Create a Insight->Converters->BuildSeedVolume Module
set m13 [addModuleAtPosition "Insight" "Converters" "BuildSeedVolume" 1155 564]

# Create a SCIRun->Render->Viewer Module
set m14 [addModuleAtPosition "SCIRun" "Render" "Viewer" 328 1673]

# Create a SCIRun->Visualization->ShowField Module
set m15 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 346 688]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m16 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 364 568]

# Create a Insight->Converters->ImageToField Module
set m17 [addModuleAtPosition "Insight" "Converters" "ImageToField" 346 498]

# Create a Insight->Converters->ImageToField Module
set m18 [addModuleAtPosition "Insight" "Converters" "ImageToField" 288 1234]
set Notes($m18) {Segmentation
result slice.}
set Notes($m18-Position) {def}
set Notes($m18-Color) {white}

# Create a SCIRun->Visualization->ShowField Module
set m19 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 288 1415]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m20 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 306 1294]

# Create a Insight->Converters->ImageToNrrd Module
set m21 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 826 507]

# Create a SCIRun->Visualization->GenTitle Module
set m22 [addModuleAtPosition "SCIRun" "Visualization" "GenTitle" 180 1576]

# Create a Insight->Converters->FloatToUChar Module
#set m23 [addModuleAtPosition "Insight" "Converters" "FloatToUChar" 23 1242]

# Create a Teem->Converters->NrrdToField Module
set m24 [addModuleAtPosition "Teem" "Converters" "NrrdToField" 1352 1073]
set Notes($m24) {Current
seed slice}
set Notes($m24-Position) {def}
set Notes($m24-Color) {white}

# Create a SCIRun->Visualization->ShowField Module
set m25 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1352 1253]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m26 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1370 1135]

# Create a Insight->Converters->ImageToField Module
set m27 [addModuleAtPosition "Insight" "Converters" "ImageToField" 1391 445]
set Notes($m27) {Negative 
seed points.}
set Notes($m27-Position) {def}
set Notes($m27-Color) {white}

# Create a SCIRun->FieldsCreate->SeedPoints Module
set m28 [addModuleAtPosition "SCIRun" "FieldsCreate" "SeedPoints" 1391 502]

# Create a Insight->Converters->BuildSeedVolume Module
set m29 [addModuleAtPosition "Insight" "Converters" "BuildSeedVolume" 1409 560]

# Create a Teem->NrrdData->ChooseNrrd Module
set m30 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 809 640]
set Notes($m30) {Change when
seeding method
changes (port 0
for prev).}
set Notes($m30-Position) {def}
set Notes($m30-Color) {white}

# Create a Teem->UnuAtoM->Unu2op Module
set m31 [addModuleAtPosition "Teem" "UnuAtoM" "Unu2op" 1155 720]
set Notes($m31) {Add positive seeds
to selected seed
method output.}
set Notes($m31-Position) {def}
set Notes($m31-Color) {white}

# Create a Teem->NrrdData->ChooseNrrd Module
set m32 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 1137 787]
set Notes($m32) {Change if using
seeds only or
combination.}
set Notes($m32-Position) {def}
set Notes($m32-Color) {white}

# Create a Teem->UnuAtoM->Unu2op Module
set m33 [addModuleAtPosition "Teem" "UnuAtoM" "Unu2op" 1137 847]
set Notes($m33) {Subtract
negative seeds.}
set Notes($m33-Position) {def}
set Notes($m33-Color) {white}

# Create a Insight->Converters->NrrdToImage Module
set m34 [addModuleAtPosition "Insight" "Converters" "NrrdToImage" 1137 908]

# Create a Insight->Converters->ImageToNrrd Module
set m35 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 1388 1014]

# Create a Teem->DataIO->AnalyzeNrrdReader Module
set m36 [addModuleAtPosition "Teem" "DataIO" "AnalyzeNrrdReader" 16 42]
set Notes($m36) {Low-Res Volume}
set Notes($m36-Position) {n}
set Notes($m36-Color) {white}

# Create a Insight->Filters->GradientAnisotropicDiffusionImageFilter Module
set m37 [addModuleAtPosition "Insight" "Filters" "GradientAnisotropicDiffusionImageFilter" 450 121]

# Create a Insight->Filters->CurvatureAnisotropicDiffusionImageFilter Module
set m38 [addModuleAtPosition "Insight" "Filters" "CurvatureAnisotropicDiffusionImageFilter" 468 180]

# Create a Insight->Filters->DiscreteGaussianImageFilter Module
set m39 [addModuleAtPosition "Insight" "Filters" "DiscreteGaussianImageFilter" 486 241]

# Create a Insight->DataIO->ChooseImage Module
set m40 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 346 439]
set Notes($m40) {Current slice (either
original or smoothed).}
set Notes($m40-Position) {def}
set Notes($m40-Color) {white}

# Create a Insight->DataIO->ChooseImage Module
set m41 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 432 307]
set Notes($m41) {Smoothing vs. not smoothing
and which smoothing filter}
set Notes($m41-Position) {def}
set Notes($m41-Color) {white}

# Create a Insight->Converters->ImageToField Module
set m42 [addModuleAtPosition "Insight" "Converters" "ImageToField" 929 1119]
set Notes($m42) {Speed Image}
set Notes($m42-Position) {def}
set Notes($m42-Color) {white}

# Create a SCIRun->Visualization->Isosurface Module
set m43 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 930 1202]
set Notes($m43) {Growing}
set Notes($m43-Position) {n}
set Notes($m43-Color) {white}

# Create a SCIRun->Visualization->Isosurface Module
set m44 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1107 1200]
set Notes($m44) {Shrinking}
set Notes($m44-Position) {n}
set Notes($m44-Color) {white}

# Create a SCIRun->Visualization->ShowField Module
set m45 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 930 1263]

# Create a SCIRun->Visualization->ShowField Module
set m46 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1107 1261]

# Create a Insight->DataIO->ImageFileWriter Module
set m47 [addModuleAtPosition "Insight" "DataIO" "ImageFileWriter" 23 1300]
set Notes($m47) {Writer for 
binary volume.}
set Notes($m47-Position) {def}
set Notes($m47-Color) {white}

# Create a Insight->Converters->ImageToField Module
set m48 [addModuleAtPosition "Insight" "Converters" "ImageToField" 654 1205]
set Notes($m48) {Commit to 3D
isocontours.}
set Notes($m48-Position) {def}
set Notes($m48-Color) {white}

# Create a SCIRun->Visualization->Isosurface Module
set m49 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 654 1267]

# Create a SCIRun->FieldsCreate->GatherFields Module
set m50 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherFields" 654 1328]

# Create a SCIRun->Visualization->ShowField Module
set m51 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 654 1387]

# Create a SCIRun->Render->Viewer Module
set m52 [addModuleAtPosition "SCIRun" "Render" "Viewer" 636 1468]

# Create a SCIRun->Visualization->GenTitle Module
set m53 [addModuleAtPosition "SCIRun" "Visualization" "GenTitle" 484 1386]

# Create a Insight->DataIO->SliceReader Module
set m54 [addModuleAtPosition "Insight" "DataIO" "SliceReader" 346 37]
set Notes($m54) {High-Res Volume}
set Notes($m54-Position) {n}
set Notes($m54-Color) {white}

# Create a SCIRun->FieldsOther->FieldInfo Module
set m55 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldInfo" 1587 558]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m56 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1099 1059]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m57 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 978 1337]

# Create a SCIRun->Visualization->ShowField Module
set m58 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 960 1396]

# Create a Insight->Converters->ImageToNrrd Module
set m59 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 809 579]

# Create a Insight->DataIO->ImageReaderFloat2D Module
set m60 [addModuleAtPosition "Insight" "DataIO" "ImageReaderFloat2D" 616 491]

# Create a SCIRun->FieldsGeometry->ChangeFieldBounds Module
set m61 [addModuleAtPosition "SCIRun" "FieldsGeometry" "ChangeFieldBounds" 826 1308]

# Create a SCIRun->Math->BuildTransform Module
set m62 [addModuleAtPosition "SCIRun" "Math" "BuildTransform" 745 1067]

# Create a SCIRun->FieldsGeometry->TransformField Module
set m63 [addModuleAtPosition "SCIRun" "FieldsGeometry" "TransformField" 727 1130]




# Create a Teem->UnuNtoZ->UnuSave Module
set m64 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSave" 717 721]

# Create a Teem->UnuNtoZ->UnuSave Module
set m65 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSave" 921 851]

# Create a Teem->UnuNtoZ->UnuSave Module
set m66 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSave" 920 950]

# Create a Teem->UnuNtoZ->UnuSave Module
set m67 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSave" 1422 778]

# Create a Insight->Converters->ImageToNrrd Module
set m68 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 842 246]

# Create a Teem->UnuNtoZ->UnuSave Module
set m69 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSave" 842 315]

# Create a Insight->Converters->ImageToField Module
set m70 [addModuleAtPosition "Insight" "Converters" "ImageToField" 810 101]

# Create a SCIRun->DataIO->FieldWriter Module
set m71 [addModuleAtPosition "SCIRun" "DataIO" "FieldWriter" 810 164]



# Create the Connections between Modules
set c1 [addConnection $m13 0 $m32 0]
set c2 [addConnection $m13 0 $m31 0]
#set c3 [addConnection $m23 0 $m47 0]
set c3 [addConnection $m4 0 $m47 0]
set c4 [addConnection $m7 0 $m5 0]
set c5 [addConnection $m17 0 $m15 0]
set c6 [addConnection $m18 0 $m19 0]
set c7 [addConnection $m27 0 $m28 0]
set c8 [addConnection $m42 0 $m43 0]
set c9 [addConnection $m42 0 $m44 0]
set c10 [addConnection $m48 0 $m49 0]
set c11 [addConnection $m59 0 $m30 0]
set c12 [addConnection $m34 0 $m35 0]
set c13 [addConnection $m34 0 $m3 0]
set c14 [addConnection $m40 0 $m17 0]
set c15 [addConnection $m41 0 $m7 0]
set c16 [addConnection $m41 0 $m27 0]
set c17 [addConnection $m41 0 $m6 0]
set c18 [addConnection $m54 0 $m40 0]
set c19 [addConnection $m54 0 $m41 0]
set c20 [addConnection $m54 0 $m38 0]
set c21 [addConnection $m54 0 $m39 0]
set c22 [addConnection $m54 0 $m37 0]
#set c23 [addConnection $m4 0 $m23 0]
set c24 [addConnection $m4 0 $m18 0]
set c25 [addConnection $m6 0 $m21 0]
set c26 [addConnection $m3 0 $m48 0]
set c27 [addConnection $m3 0 $m4 0]
set c28 [addConnection $m3 1 $m42 0]
set c29 [addConnection $m62 0 $m63 1]
set c30 [addConnection $m5 1 $m13 0]
set c31 [addConnection $m28 1 $m29 0]
set c32 [addConnection $m16 0 $m8 0]
set c33 [addConnection $m20 0 $m9 0]
set c34 [addConnection $m26 0 $m10 0]
set c35 [addConnection $m22 0 $m14 0]
set c36 [addConnection $m53 0 $m52 0]
set c37 [addConnection $m43 0 $m45 0]
set c38 [addConnection $m44 0 $m46 0]
set c39 [addConnection $m49 0 $m61 0]
set c40 [addConnection $m24 0 $m25 0]
set c41 [addConnection $m36 0 $m12 0]
set c42 [addConnection $m36 0 $m1 0]
set c43 [addConnection $m36 0 $m2 0]
set c44 [addConnection $m32 0 $m33 0]
set c45 [addConnection $m33 0 $m34 0]
set c46 [addConnection $m29 0 $m33 1]
set c47 [addConnection $m17 0 $m8 1]
set c48 [addConnection $m18 0 $m9 1]
set c49 [addConnection $m41 0 $m40 1]
set c50 [addConnection $m41 0 $m3 1]
set c51 [addConnection $m37 0 $m41 1]
set c52 [addConnection $m8 0 $m15 1]
set c53 [addConnection $m9 0 $m19 1]
set c54 [addConnection $m10 0 $m25 1]
set c55 [addConnection $m15 0 $m14 1]
set c56 [addConnection $m51 0 $m52 1]
set c57 [addConnection $m24 0 $m10 1]
set c58 [addConnection $m30 0 $m31 1]
set c59 [addConnection $m31 0 $m32 1]
set c60 [addConnection $m35 0 $m24 2]
set c61 [addConnection $m38 0 $m41 2]
set c62 [addConnection $m21 0 $m30 1]
set c63 [addConnection $m11 0 $m12 2]
set c64 [addConnection $m5 0 $m14 2]
set c65 [addConnection $m39 0 $m41 3]
set c66 [addConnection $m60 0 $m59 0]
set c67 [addConnection $m28 0 $m14 3]
set c68 [addConnection $m25 0 $m14 4]
set c69 [addConnection $m19 0 $m14 5]
set c70 [addConnection $m27 0 $m55 0]
set c71 [addConnection $m41 0 $m13 1]
set c72 [addConnection $m41 0 $m29 1]
set c73 [addConnection $m56 0 $m57 0]
set c74 [addConnection $m42 0 $m57 1]
set c75 [addConnection $m42 0 $m58 0]
set c76 [addConnection $m57 0 $m58 1]
set c77 [addConnection $m58 0 $m14 6]
set c78 [addConnection $m61 0 $m50 0]
set c79 [addConnection $m50 0 $m63 0]
set c80 [addConnection $m63 0 $m51 0]


set c100 [addConnection $m21 0 $m64 0]
set c101 [addConnection $m31 0 $m65 0]
set c102 [addConnection $m33 0 $m66 0]
set c102 [addConnection $m29 0 $m67 0]
set c102 [addConnection $m41 0 $m68 0]
set c102 [addConnection $m68 0 $m69 0]
set c103 [addConnection $m41 0 $m70 0]
set c104 [addConnection $m70 0 $m71 0]

setGlobal $m64-format {text}
setGlobal $m64-encoding {ascii}
setGlobal $m64-filename {/tmp/thresh.txt}

setGlobal $m65-format {text}
setGlobal $m65-encoding {ascii}
setGlobal $m65-filename {/tmp/2op-1.txt}

setGlobal $m66-format {text}
setGlobal $m66-encoding {ascii}
setGlobal $m66-filename {/tmp/2op-2.txt}

setGlobal $m67-format {text}
setGlobal $m67-encoding {ascii}
setGlobal $m67-filename {/tmp/neg-seeds.txt}

setGlobal $m69-format {text}
setGlobal $m69-encoding {ascii}
setGlobal $m69-filename {/tmp/smooth.txt}

setGlobal $m70-copy {1}
setGlobal $m71-filename {/tmp/smooth.fld}
setGlobal $m71-exporttype {SCIRun Field ASCII (*.fld)}



# Set GUI variables

# UnuMinmax_0
setGlobal $m2-min0 0
setGlobal $m2-max0 0

# Set GUI variables for the Insight->Filters->ThresholdSegmentationLevelSetImageFilter Module
setGlobal $m3-isovalue {0.5}
setGlobal $m3-curvature_scaling {1.0}
setGlobal $m3-propagation_scaling {1.0}
setGlobal $m3-edge_weight {1.0}
setGlobal $m3-reverse_expansion_direction 0
setGlobal $m3-smoothing_iterations 0
setGlobal $m3-smoothing_conductance {0.5}
setGlobal $m3-smoothing_time_step {0.1}
setGlobal $m3-maximum_iterations {0}
setGlobal $m3-update_OutputImage {1}
setGlobal $m3-update_iters_OutputImage {2}

# Set GUI variables for the Insight->Filters->BinaryThresholdImageFilter Module
set $m4-upper_threshold {100.0}

# Set GUI variables for the SCIRun->FieldsCreate->SeedPoints Module
set $m5-num_seeds {0}
set $m5-probe_scale {10}
set $m5-widget {1}
set $m5-green {0.0}
set $m5-blue {0.0}
set $m5-auto_execute {0}
set $m5-send {1}

# Set GUI variables for the Insight->Filters->BinaryThresholdImageFilter Module
set $m6-inside_value {0}
set $m6-outside_value {1}

# Set GUI variables for the Insight->Converters->ImageToField Module
set $m7-copy {1}

# Set GUI variables for the SCIRun->Visualization->GenStandardColorMaps Module
set $m11-width {552}
set $m11-height {40}
set $m11-mapName {Gray}

# Set GUI variables for the SCIRun->Render->ViewSlices Module
set $m12-clut_ww {1.0}
set $m12-clut_wl {0.0}
set $m12-probe {0}
set $m12-show_colormap2 {0}
set $m12-painting {0}
set $m12-crop {0}
set $m12-crop_minAxis0 {0}
set $m12-crop_minAxis1 {0}
set $m12-crop_minAxis2 {0}
set $m12-crop_maxAxis0 {0}
set $m12-crop_maxAxis1 {0}
set $m12-crop_maxAxis2 {0}
set $m12-crop_minPadAxis0 {0}
set $m12-crop_minPadAxis1 {0}
set $m12-crop_minPadAxis2 {0}
set $m12-crop_maxPadAxis0 {0}
set $m12-crop_maxPadAxis1 {0}
set $m12-crop_maxPadAxis2 {0}
set $m12-texture_filter {1}
set $m12-anatomical_coordinates {0}
set $m12-show_text {1}
set $m12-color_font-r {1.0}
set $m12-color_font-g {1.0}
set $m12-color_font-b {1.0}
set $m12-color_font-a {1.0}
set $m12-min {-1.0}
set $m12-max {-1.0}
set $m12-dim0 {0}
set $m12-dim1 {0}
set $m12-dim2 {0}
set $m12-geom_flushed {0}
set $m12-background_threshold {0.0}
set $m12-gradient_threshold {0.0}
set $m12-font_size {15.0}

# Set GUI variables for the Insight->Converters->BuildSeedVolume Module
set $m13-inside_value {0}
set $m13-outside_value {1}

# Set GUI variables for the SCIRun_Render_Viewer_0 Module
setGlobal $m14-ViewWindow_0-raxes 0
setGlobal $m14-ViewWindow_0-ortho-view 1
setGlobal $m14-ViewWindow_0-pos "z1_y1"
setGlobal $m14-ViewWindow_1-raxes 0
setGlobal $m14-ViewWindow_1-ortho-view 1
setGlobal $m14-ViewWindow_1-pos "z1_y1"
# setGlobal {$m14-ViewWindow_1-Transparent Faces (2)} 0
# setGlobal {$m14-ViewWindow_1-Transparent Faces (4)} 0
# setGlobal {$m14-ViewWindow_0-Title (2)} 0
# setGlobal {$m14-ViewWindow_1-Title (1)} 0

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m15-nodes-on {0}
set $m15-edges-on {0}
set $m15-faces-usetexture {1}

# Set GUI variables for the SCIRun->Visualization->GenStandardColorMaps Module
set $m16-width {552}
set $m16-height {40}
set $m16-mapName {Gray}

# Set GUI variables for the Insight->Converters->ImageToField Module
set $m17-copy {1}

# Set GUI variables for the Insight->Converters->ImageToField Module
set $m18-copy {1}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m19-nodes-on {0}
set $m19-edges-on {0}
set $m19-faces-on {0}
set $m19-use-transparency {1}
set $m19-faces-usetexture {1}

# Set GUI variables for the SCIRun->Visualization->GenStandardColorMaps Module
set $m20-positionList {{0 0} {273 00} {277 40} {552 40}}
set $m20-nodeList {3 4 5 6}
set $m20-width {552}
set $m20-height {40}
set $m20-mapName {BP Seismic}
set $m20-reverse {1}
set $m20-resolution {2}
set $m20-realres {2}

# Set GUI variables for the SCIRun->Visualization->GenTitle Module
set $m22-value {0.0}
set $m22-bbox {0}
set $m22-format {Segmentation}
set $m22-location {Top Center}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m25-nodes-on {0}
set $m25-edges-on {0}
set $m25-faces-on {0}
set $m25-use-transparency {1}
set $m25-faces-usetexture {1}

# Set GUI variables for the SCIRun->Visualization->GenStandardColorMaps Module
set $m26-positionList {{0 0} {273 0} {277 40} {552 40}}
set $m26-nodeList {3 4 5 6}
set $m26-width {552}
set $m26-height {40}
set $m26-mapName {BP Seismic}
set $m26-reverse {1}
set $m26-resolution {2}
set $m26-realres {2}

# Set GUI variables for the Insight->Converters->ImageToField Module
set $m27-copy {1}

# Set GUI variables for the SCIRun->FieldsCreate->SeedPoints Module
set $m28-num_seeds {0}
set $m28-probe_scale {10}
set $m28-widget {1}
set $m28-red {0.0}
set $m28-green {0.0}
set $m28-auto_execute {0}
set $m28-send {1}

# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m30-port-index {1}

# Set GUI variables for the Teem->UnuAtoM->Unu2op Module
set $m31-operator {min}

# Set GUI variables for the Teem->NrrdData->ChooseNrrd Module
set $m32-port-index {1}

# Set GUI variables for the Teem->UnuAtoM->Unu2op Module
set $m33-operator {max}

# Set GUI variables for the Teem->DataIO->AnalyzeNrrdReader Module
#setGlobal $m36-num-files {1}
#setGlobal $m36-filenames0 "/home/darbyb/work/data/SCIRunData/1.22.0/Test.hdr"

# GradientAnisotropicDiffusion_0
setGlobal $m37-time_step 0.0625
setGlobal $m37-iterations 5
setGlobal $m37-conductance 0.5

# CurvatureAnisotropicDiffusion_0
setGlobal $m38-time_step 0.0625
setGlobal $m38-iterations 5
setGlobal $m38-conductance 0.5

# Set GUI variables for the Insight->Converters->ImageToField Module
set $m42-copy {1}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m43-isoval {-0.0}
set $m43-isoval-typed {-0.1}
set $m43-build_geom {0}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m44-isoval {0.1}
set $m44-isoval-typed {0.1}
set $m44-build_geom {0}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m45-nodes-on {0}
set $m45-faces-on {0}
set $m45-def-color-r {0.8}
set $m45-def-color-g {0.0}
set $m45-def-color-b {0.0}
set $m45-edge_display_type {Cylinders}
set $m45-active_tab {Edges}
set $m45-edge_scale {1.0}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m46-nodes-on {0}
set $m46-faces-on {0}
set $m46-def-color-r {0.0}
set $m46-def-color-g {0.0}
set $m46-def-color-b {0.8}
set $m46-edge_display_type {Cylinders}
set $m46-active_tab {Edges}
set $m46-edge_scale {1.0}

# Set GUI variables for the Insight->Converters->ImageToField Module
set $m48-copy {1}

# Set GUI variables for the SCIRun->Visualization->Isosurface Module
set $m49-isoval {1.0}
set $m49-isoval-typed {1.0}
set $m49-build_geom {0}

# Set GUI variables for the SCIRun->FieldsCreate->GatherFields Module
setGlobal $m50-accumulating {1}

# Set GUI variables for the SCIRun->Visualization->ShowField Module
set $m51-nodes-on {0}
set $m51-faces-on {0}
set $m51-def-color-r {0.0}
set $m51-def-color-g {0.8}
set $m51-def-color-b {0.0}
set $m51-def-color-a {1.0}
set $m51-active_tab {Edges}
set $m51-edge_scale {1.0}

# Set GUI variables for the SCIRun->Visualization->GenTitle Module
set $m53-value {0.0}
set $m53-bbox {0}
set $m53-format {Isocontours}
set $m53-location {Top Center}

# Set GUI variables for the Insight->DataIO->SliceReader Module
#setGlobal $m54-filename \
#    "/home/darbyb/work/data/SCIRunData/1.22.0/smallcere.hdr"
setGlobal $m54-cast_output {1}
#setGlobal $m54-slice {0}

set $m56-mapName {BP Seismic}
set $m56-resolution {2}
set $m56-realres {2}

set $m58-nodes-on {0}
set $m58-edges-on {0}
set $m58-faces-on {0}
set $m58-use-transparency {1}

setGlobal $m61-useoutputcenter {1}
setGlobal $m61-outputcenterx {0.0}
setGlobal $m61-outputcentery {0.0}
setGlobal $m61-outputcenterz {0.0}

set $m62-rotate_z {1.0}
set $m62-translate_x {0.0}
set $m62-translate_y {0.0}
set $m62-translate_z {0.0}
set $m62-scale_z {0.0}
set $m62-shear_plane_c {1.0}
set $m62-shear_plane_d {1.0}
set $m62-which_transform {scale}
set $m62-widget_scale {1.0}


####################################################
# Determine if load file was passed in, 
# and if it is an analyze file, otherwise it
# defaults to use the nrrd reader (generic tab)
# global load_file
# set load_file ""
# global load_file_type
# set load_file_type "Generic"
# if {[netedit getenv LEVELSETSEGMENTER_LOAD_FILE] == ""} {
#     set load_file  "/usr/sci/data/Medical/ucsd/king_filt/king_filt-full.nhdr"
#     set load_file_type "Generic"
# } else {
#     # Determine which reader to use by looking at file extension
#     set index [string last "." [netedit getenv LEVELSETSEGMENTER_LOAD_FILE]]
#     if {$index > 0} {
# 	set ext [string range [netedit getenv LEVELSETSEGMENTER_LOAD_FILE] $index end]

# 	if {[string equal $ext ".hdr"] == 1} {
# 	    # Analyze file
# 	    set load_file [netedit getenv LEVELSETSEGMENTER_LOAD_FILE]
# 	    set load_file_type "Analyze"
# 	} else {
# 	    # Some other file, hopefully something the generic
# 	    # tab (NrrdReader) can handle -- DICOM can't be read
# 	    # in via command line
# 	    set load_file [netedit getenv LEVELSETSEGMENTER_LOAD_FILE]
# 	    set load_file_type "Generic"
# 	}
#     } else {
# 	set load_file  "/usr/sci/data/Medical/ucsd/king_filt/king_filt-full.nhdr"
# 	set load_file_type "Generic"
#     }
# }

# # NrrdReader_0
# if {[string equal $load_file_type "Generic"] == 1} {
#     setGlobal $m1-filename $load_file
# }


# # AnalyzeNrrdReader_0
# if {[string equal $load_file_type "Analyze"] == 1} {
#     setGlobal $m3-file $load_file
#     setGlobal $m3-num-files {1}
#     setGlobal $m3-filenames0 $load_file
# }



# # ImageFileWriter_0
# if {[netedit getenv LEVELSETSEGMENTER_SAVE_BINARY_FILE] == ""} {
#     setGlobal $m23-filename "/tmp/binary.mhd"
# } else {
#     setGlobal $m23-filename "[netedit getenv LEVELSETSEGMENTER_SAVE_BINARY_FILE]"
# }


# # RescaleColorMap_0
# setGlobal $m32-isFixed 1
# setGlobal $m32-min 0
# setGlobal $m32-max 1



# # ImageFileWriter_1
# if {[netedit getenv LEVELSETSEGMENTER_SAVE_FLOAT_FILE] == ""} {
#     setGlobal $m74-filename "/tmp/float.mhd"
# } else {
#     setGlobal $m74-filename "[netedit getenv LEVELSETSEGMENTER_SAVE_FLOAT_FILE]"
# }

::netedit scheduleok


# global array indexed by module name to keep track of modules
global mods

# Viewer Stuff
set mods(Viewer) $m14
set mods(Viewer-Vol) $m52
set mods(ViewSlices) $m12
set mods(GenTitle-Seg) $m22 
set mods(GenTitle-Vol) $m53
set mods(NrrdInfo-Slice) $m1

# Readers
set mods(AnalyzeNrrdReader) $m36 
set mods(SliceReader) $m54
set mods(ImageReaderFloat2D) $m60

# Writers
set mods(ImageFileWriter-Binary) $m47

# Smoothers
set mods(Smooth-Gradient) $m37
set mods(Smooth-Curvature) $m38
set mods(Smooth-Gaussian) $m39
set mods(ChooseImage-Smooth) $m41
set mods(ChooseImage-ShowSlice) $m40

# Speed
set mods(Isosurface-Grow) $m43
set mods(Isosurface-Shrink) $m44
set mods(ShowField-Speed) $m58
set mods(ImageToField-Speed) $m42
set mods(GenStandardColorMaps-Speed) $m56

# Segmentation Window
set mods(ShowField-Slice) $m15

# LevelSet Segmentation
set mods(LevelSet) $m3
set mods(BinaryThreshold-Seed) $m6
set mods(ImageToField-PosSeeds) $m7
set mods(ImageToField-NegSeeds) $m27
set mods(SeedPoints-PosSeeds) $m5
set mods(SeedPoints-NegSeeds) $m28
set mods(FieldInfo-Smoothed) $m55
set mods(ImageToField-Seg) $m18
set mods(ShowField-Seg) $m19

# Seeds
set mods(BuildSeedVolume-PosSeeds) $m13
set mods(BuildSeedVolume-NegSeeds) $m29
set mods(ImageToNrrd-CurSeed) $m35
set mods(ShowField-Seed) $m25
set mods(ChooseNrrd-Seeds) $m30
set mods(ChooseNrrd-Combine) $m32
set mods(ImageToNrrd-Prev) $m59

# Isocontours
set mods(ImageToField-Iso) $m48
set mods(GatherFields) $m50
set mods(ShowField-Iso) $m51
set mods(ChangeFieldBounds) $m61
set mods(BuildTransform) $m62


global axis
set axis 2

global to_smooth
set to_smooth 0

# global current_slice
# set current_slice 0

# global seed_method
# set seed_method "thresh"

global max_iter
set max_iter 100

# global slice
# set slice 0

# global image_dir no_seg_icon
# global old_seg_icon updated_seg_icon

# set image_dir [netedit getenv SCIRUN_SRCDIR]/Packages/Insight/Dataflow/GUI
# set no_seg_icon [image create photo -file ${image_dir}/no-seg.ppm]
# set old_seg_icon [image create photo -file ${image_dir}/old-seg.ppm]
# set updated_seg_icon [image create photo -file ${image_dir}/updated-seg.ppm]

global show_seeds
set show_seeds 1

global commit_dir
set commit_dir "/tmp"

global base_filename
set base_filename "out"

global spacing
set spacing 1.0

#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

setProgressText "Creating LevelSetSegmenter GUI..."

set auto_index(::PowerAppBase) "source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/PowerAppBase.app"

class LevelSetSegmenterApp {
    inherit ::PowerAppBase
    
    method appname {} {
	return "LevelSetSegmenter"
    }
    
    constructor {} {
	toplevel .standalone
	wm title .standalone "LevelSetSegmenter"	 
	set win .standalone

	set viewer_width 620
	set viewer_height 750

	set process_width 400
	set process_height $viewer_height

        set initialized 0

        set i_width [expr $process_width - 50]
        set i_height 20
        set stripes 10

        set error_module ""

        set indicatorID 0

 	set eviewer2 ""
 	set has_loaded 0
 	set 2D_fixed 0

 	set size0 0
 	set size1 0
 	set size2 0

# 	set range_min 0
# 	set range_max 0
# 	set slice_frame ""

 	set filter_menu1 ""
 	set filter_menu2 ""
	set filter_type "GradientAnisotropicDiffusion"

 	set seed_menu1 ""
 	set seed_menu2 ""
	set seed_type "Thresholds and Seed Points"
# #	set filter_enabled 0

# 	set next_load ""
# 	set next_smooth ""

# 	set execute_color "#008b45"
# 	set execute_active_color "#31a065"
 	set execute_color "#63c070"
	set execute_active_color $execute_color

	set pos_seeds_used 0
	set neg_seeds_used 0

# 	set has_smoothed 0

# 	set has_segmented 0
# 	set segmentation_initialized 0

# 	set updating_speed 0
 	set segmenting 0

	set committing 0

	set status_canvas1 ""
	set status_canvas2 ""
	set status_width 0

	set has_autoviewed 0
	
# 	set pasting_binary 0
# 	set pasting_float 0

# 	set current_slice 0

# 	set region_changed 0

# 	set updating_crop_widget 0

# 	set smoothing 0
# 	set smoothing_method "None"
# 	set smoothing_type "Reset"

# 	set segmenting_type "Reset"

 	set has_committed 0

	set reverse_changed 0


	### Define Tooltips
	##########################
	# General
	global tips

	$this initialize_blocks
    }
    

    destructor {
	destroy $this
    }

    #############################
    ### initialize_blocks
    #############################
    # Disable modules for any steps beyond first one.
    method initialize_blocks {} { 
	global mods

 	# Disable smoothers until needed
 	disableModule $mods(Smooth-Gradient) 1
 	disableModule $mods(Smooth-Curvature) 1
 	disableModule $mods(Smooth-Gaussian) 1

	# Disable Seeds
#	disableModule $mods(BuildSeedVolume-PosSeeds) 1
#	disableModule $mods(BuildSeedVolume-NegSeeds) 1
#	disableModule $mods(ImageToNrrd-CurSeed) 1
	disableModule $mods(ImageToNrrd-Prev) 1

	# Disable output segmentation
#	disableModule $mods(ImageToField-Seg) 1

	# Disable 3D Isocontours until commit
	disableModule $mods(ImageToField-Iso) 1

	# Disable writer
	disableModule $mods(ImageFileWriter-Binary) 1

	# Disable speed from Isos
	disableModule $mods(Isosurface-Grow) 1
	disableModule $mods(Isosurface-Shrink) 1
    }

    
    method build_app {} {
	global mods

	frame $win.viewers
	
	### Processing Part
	#########################
	### Create Detached Processing Part
	toplevel $win.detachedP
	frame $win.detachedP.f -relief flat
	pack $win.detachedP.f -side left -anchor n -fill both -expand 1
	
	wm title $win.detachedP "Processing Window"
	
	wm sizefrom $win.detachedP user
	wm positionfrom $win.detachedP user
	
	wm withdraw $win.detachedP

	### Create Attached Processing Part
	frame $win.attachedP 
	frame $win.attachedP.f -relief flat 
	pack $win.attachedP.f -side top -anchor n -fill both -expand 1

	set IsPAttached 1

	### set frame data members
	set detachedPFr $win.detachedP
	set attachedPFr $win.attachedP

	init_Pframe $detachedPFr.f 1
	init_Pframe $attachedPFr.f 2

	### create detached width and height
	append geomP $process_width x $process_height
	wm geometry $detachedPFr $geomP

	# add  viewers
	$this build_viewers $mods(Viewer) $mods(Viewer-Vol) $mods(ViewSlices)

	### pack 3 frames
	pack $attachedPFr -side left -anchor n -fill y 

	pack $win.viewers -side left -anchor n -fill both -expand 1 

	set total_width [expr $viewer_width + $process_width]

	set total_height $viewer_height

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]
	set pos_y [expr [expr $screen_height / 2] - [expr $total_height / 2]]

	append geom $total_width x $total_height + $pos_x + $pos_y
	wm geometry .standalone $geom
	update	

        set initialized 1

	global PowerAppSession
	if {[info exists PowerAppSession] && [set PowerAppSession] != ""} { 
	    set saveFile $PowerAppSession
	    wm title .standalone "LevelSetSegmenter - [getFileName $saveFile]"
	    $this load_session_data
	}

	# Execute GenTitle modules
	$mods(GenTitle-Seg)-c needexecute
	$mods(GenTitle-Vol)-c needexecute
	
    }

    #############################
    ### init_Pframe
    #############################
    # Initialize the processing frame on the left. For this app
    # that includes the Load Data, Restriation, and Build Tensors steps.
    # This method will call the base class build_menu method and sets 
    # the variables that point to the tabs and tabnotebooks.
    method init_Pframe { m case } {
        global mods
	global tips
        
	if { [winfo exists $m] } {

	    build_menu $m

 	    ### Processing Steps
 	    #####################
 	    iwidgets::labeledframe $m.p \
 		-labelpos n -labeltext "Processing Steps" 
 	    pack $m.p -side left -fill both -anchor n -expand 1
	    
 	    set process [$m.p childsite]

	    $this create_volumes_frame $process $case

	    $this create_smoothing_frame $process $case

	    $this create_speed_frame $process $case

	    $this create_seeds_frame $process $case

	    $this create_segmentation_frame $process $case

	    $this create_commit_frame $process $case

# #             iwidgets::tabnotebook $process.tnb \
# #                 -width [expr $process_width - 50] \
# #                 -height [expr $process_height - 140] \
# #                 -tabpos n -equaltabs 0
# # 	    pack $process.tnb -side top -anchor n 

# # 	    set proc_tab$case $process.tnb

# # 	    ############# Load ###########
# #             set step_tab [$process.tnb add -label "Load" -command "$this change_processing_tab Load"]

# # 	    iwidgets::labeledframe $step_tab.load \
# # 		-labeltext "Data File Format and Orientation" \
# # 		-labelpos nw
# # 	    pack $step_tab.load -side top -anchor nw -pady 3 \
# # 		-expand yes -fill x

# # 	    set load [$step_tab.load childsite]

# # 	    # Build data tabs
# # 	    iwidgets::tabnotebook $load.tnb \
# # 		-width [expr $process_width - 110] -height 75 \
# # 		-tabpos n -equaltabs false
# # 	    pack $load.tnb -side top -anchor n \
# # 		-padx 0 -pady 3

# # 	    set data_tab$case $load.tnb
	
# # 	    # Nrrd
# # 	    set page [$load.tnb add -label "Generic" \
# # 			  -command "$this set_curr_data_tab Generic; $this configure_readers Generic"]       
	    
# # 	    global $mods(NrrdReader)-filename
# # 	    frame $page.file
# # 	    pack $page.file -side top -anchor nw -padx 3 -pady 0 -fill x
	    
# # 	    label $page.file.l -text ".vol/.vff/.nrrd file:" 
# # 	    entry $page.file.e -textvariable $mods(NrrdReader)-filename \
# # 		-width [expr $process_width - 120]
# # 	    pack $page.file.l $page.file.e -side left -padx 3 -pady 0 -anchor nw \
# # 		-fill x 
	    
# # 	    bind $page.file.e <Return> "$this load_data"
	    
# # 	    button $page.load -text "Browse" \
# # 		-command "$this open_nrrd_reader_ui" \
# # 		-width 12
# # 	    pack $page.load -side top -anchor n -padx 3 -pady 1
	    
	    
# # 	    ### Dicom
# # 	    set page [$load.tnb add -label "Dicom" \
# # 			  -command "$this set_curr_data_tab Dicom; $this configure_readers Dicom"]
	    
# # 	    button $page.load -text "Dicom Loader" \
# # 		-command "$this dicom_ui"
	    
# # 	    pack $page.load -side top -anchor n \
# # 		-padx 3 -pady 10 -ipadx 2 -ipady 2
	    
# # 	    ### Analyze
# # 	    set page [$load.tnb add -label "Analyze" \
# # 			  -command "$this set_curr_data_tab Analyze; $this configure_readers Analyze"]
	    
# # 	    button $page.load -text "Analyze Loader" \
# # 		-command "$this analyze_ui"
	    
# # 	    pack $page.load -side top -anchor n \
# # 		-padx 3 -pady 10 -ipadx 2 -ipady 2

# # 	    global load_file_type
# # 	    $load.tnb view $load_file_type

# # 	    # Viewing slices axis
# # 	    frame $load.axis
# # 	    pack $load.axis -side top -anchor nw -expand yes -fill x -pady 2

# # 	    set a $load.axis
# # 	    label $a.label -text "View Slices Along:"
	    
# # 	    global axis
# # 	    radiobutton $a.x -text "X Axis" \
# # 		-variable axis -value 0 \
# # 		-command "$this change_axis"

# # 	    radiobutton $a.y -text "Y Axis" \
# # 		-variable axis -value 1 \
# # 		-command "$this change_axis"

# # 	    radiobutton $a.z -text "Z Axis" \
# # 		-variable axis -value 2 \
# # 		-command "$this change_axis"

# # 	    pack $a.label $a.x $a.y $a.z -side left -expand yes -fill x

# # 	    button $load.load -text "Load" \
# # 		-command "$this load_data" \
# # 		-background $execute_color \
# # 		-activebackground $execute_active_color \
# # 		-width 10
# # 	    pack $load.load -side top -anchor n -padx 3 -pady 4 \
# # 		-ipadx 2



# # 	    # Histogram

# # 	    iwidgets::labeledframe $step_tab.stats \
# # 		-labeltext "Volume Statistics" \
# # 		-labelpos nw
# # 	    pack $step_tab.stats -side top -anchor nw -expand yes -fill x
# # 	    set stats [$step_tab.stats childsite]

# # 	    label $stats.samples -text "Samples: $orig_size0, $orig_size1, $orig_size2"
# # 	    label $stats.range -text "Data Range: $range_min - $range_max"

# # 	    pack $stats.samples $stats.range -side top -anchor nw \
# # 		-expand no

# # 	    iwidgets::labeledframe $stats.histo \
# # 		-labelpos nw -labeltext "Histogram"
# # 	    pack $stats.histo -side top -fill x -anchor n -expand 1
	    
# # 	    set histo [$stats.histo childsite]
	    
# # 	    global $mods(ScalarFieldStats)-min
# # 	    global $mods(ScalarFieldStats)-max
# # 	    global $mods(ScalarFieldStats)-nbuckets
	    
# # 	    blt::barchart $histo.graph -title "" \
# # 		-height 220 \
# # 		-width [expr $process_width - 50] -plotbackground gray80
# # 	    pack $histo.graph

# # 	    pack $stats.histo -side top -anchor nw \
# # 		-fill x -expand yes
	    
# # 	    # Next button
# # 	    button $step_tab.next -text "Next" \
# #                 -command "$this change_processing_tab Smooth" -width 8 \
# #                 -state disabled
# # 	    pack $step_tab.next -side top -anchor ne \
# # 		-padx 3 -pady 3

# # 	    set next_load "f.p.childsite.tnb.canvas.notebook.cs.page1.cs.next"


# # 	    ############# Smooth ###########
# #             set step_tab [$process.tnb add -label "Smooth" -command "$this change_processing_tab Smooth"]

# # 	    iwidgets::labeledframe $step_tab.roi \
# # 		-labeltext "Region of Interest" \
# # 		-labelpos nw
# # 	    pack $step_tab.roi -side top -anchor nw -expand no -fill x -pady 3
	    
# # 	    set roi [$step_tab.roi childsite]
# # 	    global $mods(UnuCrop)-digits_only

# # 	    global show_roi
# # 	    checkbutton $roi.t -text "Show Crop Widget" \
# # 		-variable show_roi \
# # 		-command "$this toggle_show_roi"
# # 	    pack $roi.t -side top -anchor nw

# # 	    foreach l {{0 X} {1 Y} {2 Z}} {
# # 		set i [lindex $l 0]
# # 		set label [lindex $l 1]
# # 		global $mods(UnuCrop)-minAxis$i
# # 		global $mods(UnuCrop)-maxAxis$i

# # 		set $mods(UnuCrop)-minAxis$i 0
# # 		if {[set $mods(UnuCrop)-digits_only] == 1} {
# # 		    set $mods(UnuCrop)-maxAxis$i 0
# # 		} else {
# # 		    set $mods(UnuCrop)-maxAxis$i M
# # 		}

# # 		frame $roi.$i
# # 		pack $roi.$i -side top -anchor nw -expand yes -fill x \
# # 		    -padx 2 -pady 2

# # 		label $roi.$i.minl -text "Min Axis $label:"
# # 		entry $roi.$i.minv -textvariable $mods(UnuCrop)-minAxis$i \
# # 		    -width 6
# # 		label $roi.$i.maxl -text "Max Axis $label:"
# # 		entry $roi.$i.maxv -textvariable $mods(UnuCrop)-maxAxis$i \
# # 		    -width 6
# # 		grid configure $roi.$i.minl -row $i -column 0 -sticky "w"
# # 		grid configure $roi.$i.minv -row $i -column 1 -sticky "e"
# # 		grid configure $roi.$i.maxl -row $i -column 2 -sticky "w"
# # 		grid configure $roi.$i.maxv -row $i -column 3 -sticky "e"

# # 		bind $roi.$i.minv <ButtonPress-1> "$this start_crop"
# # 		bind $roi.$i.maxv <ButtonPress-1> "$this start_crop"
# # 		bind $roi.$i.minv <Return> "$this update_crop_widget min $i"
# # 		bind $roi.$i.maxv <Return> "$this update_crop_widget max $i"

# # 		global $mods(ViewSlices)-crop_minAxis$i $mods(ViewSlices)-crop_maxAxis$i
# # 		trace variable $mods(ViewSlices)-crop_minAxis$i w "$this update_crop_values"
# # 		trace variable $mods(ViewSlices)-crop_maxAxis$i w "$this update_crop_values"
# # 	    }

# # 	    button $roi.button -text "Crop" \
# # 		-background $execute_color \
# # 		-activebackground $execute_active_color \
# # 		-command "$this select_region_of_interest" \
# # 		-width 10
# # 	    pack $roi.button -side top -anchor n -padx 3 -pady 4 -ipadx 2

# # 	    # Smoothing Filters
# # 	    iwidgets::labeledframe $step_tab.smooth \
# # 		-labeltext "Smoothing" \
# # 		-labelpos nw
# # 	    pack $step_tab.smooth -side top -anchor nw -expand yes -fill x -pady 3
	    
# # 	    set smooth [$step_tab.smooth childsite]

# # 	    global smooth_region
# # 	    radiobutton $smooth.roi -text "Smooth Region of Interest" \
# # 		-variable smooth_region \
# # 		-value roi \
# # 		-command "$this change_smooth_region"

# # 	    radiobutton $smooth.vol -text "Smooth Entire Volume" \
# # 		-variable smooth_region \
# # 		-value vol \
# # 		-command "$this change_smooth_region"
# # 	    pack $smooth.roi $smooth.vol -side top -anchor nw

# # 	    iwidgets::optionmenu $smooth.filter -labeltext "Filter:" \
# # 		-labelpos w -command "$this change_filter $smooth.filter"
# # 	    pack $smooth.filter -side top -anchor nw 

# # 	    set filter_menu$case $smooth.filter

# # 	    $smooth.filter insert end "GradientAnisotropicDiffusion" \
# # 		"CurvatureAnisotropicDiffusion" \
# # 		"Gaussian" "None"
# # 	    $smooth.filter select "None"

	 

# # 	    # Smooth button
# # 	    frame $smooth.buttons
# # 	    pack $smooth.buttons -side bottom -anchor n -padx 3 \
# # 		-pady 4

# # 	    button $smooth.buttons.reset -text "Reset" \
# # 		-background $execute_color \
# # 		-activebackground $execute_active_color \
# # 		-command "$this smooth_data Reset" -width 10

# # 	    button $smooth.buttons.go -text "Go" \
# # 		-background $execute_color \
# # 		-activebackground $execute_active_color \
# # 		-command "$this smooth_data Go" -width 10

# # 	    pack $smooth.buttons.reset $smooth.buttons.go  \
# # 		-side left -anchor nw -padx 3 \
# # 		-pady 4 -ipadx 2   

# # 	    # ViewSlices toggle
# # 	    frame $smooth.toggle
# # 	    pack $smooth.toggle -side bottom -anchor n -pady 3

# # 	    global $mods(ChooseNrrd-2D)-port-index
# # 	    radiobutton $smooth.toggle.orig -text "Show Original" \
# # 		-variable $mods(ChooseNrrd-2D)-port-index -value 0 \
# # 		-command "$this update_ViewSlices_input"

# # 	    radiobutton $smooth.toggle.smooth -text "Show Smoothed" \
# # 		-variable $mods(ChooseNrrd-2D)-port-index -value 1 \
# # 		-command "$this update_ViewSlices_input"

# # 	    pack $smooth.toggle.orig $smooth.toggle.smooth -side left \
# # 		-padx 4



# # 	    # Next button
# # 	    button $step_tab.next -text "Next" \
# #                 -command "$this change_processing_tab Segment" -width 8 \
# # 		-activebackground $next_color \
# # 		-background $next_color 
# # 	    pack $step_tab.next -side top -anchor ne \
# # 		-padx 3 -pady 3

# # 	    set next_smooth "f.p.childsite.tnb.canvas.notebook.cs.page2.cs.next"
	    

# # 	    ############# Segment ###########
# #             set step_tab [$process.tnb add -label "Segment" -command "$this change_processing_tab Segment"]
	    
# # 	    # Current Slice
# # 	    frame $step_tab.slice
# # 	    pack $step_tab.slice

# # 	    global no_seg_icon
# # 	    button $step_tab.slice.status -relief flat -image $no_seg_icon
# # 	    pack $step_tab.slice.status -side left
	    
# # 	    global slice
# # 	    iwidgets::spinint $step_tab.slice.sp -labeltext "Segmenting Slice:" -width 6 \
# # 		-range {0 100} -step 1 -textvariable slice
# # 	    button $step_tab.slice.up -text "Change Slice" \
# # 		-command "$this current_slice_changed"

# # 	    pack $step_tab.slice.sp $step_tab.slice.up -side left

# # 	    # Level Set Parameters
# # 	    iwidgets::labeledframe $step_tab.params \
# # 		-labeltext "Tune Speed Image" \
# # 		-labelpos nw
# # 	    pack $step_tab.params -side top -anchor nw -expand no -fill x \
# # 		-pady 3

# # 	    set params [$step_tab.params childsite]

# # 	    global $mods(LevelSet)-lower_threshold
# # 	    global $mods(LevelSet)-upper_threshold
# # 	    global $mods(LevelSet)-curvature_scaling
# # 	    global $mods(LevelSet)-propagation_scaling
# # 	    global $mods(LevelSet)-edge_weight
# # 	    global $mods(LevelSet)-max_iterations
# # 	    global $mods(LevelSet)-max_rms_change
# # 	    global $mods(LevelSet)-reverse_expansion_direction
# # 	    global $mods(LevelSet)-smoothing_iterations
# # 	    global $mods(LevelSet)-smoothing_time_step
# # 	    global $mods(LevelSet)-smoothing_conductance

# # 	    trace variable $mods(LevelSet)-lower_threshold w "$this update_seed_binary_threshold"
# # 	    trace variable $mods(LevelSet)-upper_threshold w "$this update_seed_binary_threshold"

# # 	    # thresholds
# # 	    frame $params.lthresh
# # 	    pack $params.lthresh -side top -anchor nw -expand yes -fill x
# # 	    label $params.lthresh.l -text "Lower Threshold"
# # 	    scale $params.lthresh.s -variable $mods(LevelSet)-lower_threshold \
# # 		-from 0 -to 255 -width 15 \
# # 		-showvalue false -length 150 \
# # 		-orient horizontal
# # 	    entry $params.lthresh.e -textvariable $mods(LevelSet)-lower_threshold \
# # 		-width 6
# # 	    pack $params.lthresh.l $params.lthresh.s $params.lthresh.e -side left -pady 2

# # 	    frame $params.uthresh
# # 	    pack $params.uthresh -side top -anchor nw -expand yes -fill x
# # 	    label $params.uthresh.l -text "Upper Threshold"
# # 	    scale $params.uthresh.s -variable $mods(LevelSet)-upper_threshold \
# # 		-from 0 -to 255 -width 15 \
# # 		-showvalue false -length 150 \
# # 		-orient horizontal
# # 	    entry $params.uthresh.e -textvariable $mods(LevelSet)-upper_threshold \
# # 		-width 6
# # 	    pack $params.uthresh.l $params.uthresh.s $params.uthresh.e -side left -pady 2

# # 	    # Equation Term Weights
# # 	    iwidgets::labeledframe $params.terms \
# # 		-labeltext "Equation Term Weights" \
# # 		-labelpos nw
# # 	    pack $params.terms -side top -anchor n -padx 3
	    
# # 	    set terms [$params.terms childsite]
# # 	    frame $terms.scaling1
# # 	    pack $terms.scaling1 -side top -anchor nw \
# # 		-padx 3 -pady 1
# # 	    make_entry $terms.scaling1.curv "Curvature" $mods(LevelSet)-curvature_scaling
# # 	    make_entry $terms.scaling1.prop "Propagation" $mods(LevelSet)-propagation_scaling

# # 	    pack $terms.scaling1.curv $terms.scaling1.prop \
# # 		-side left -anchor ne \
# # 		-padx 3 -pady 1

# # 	    make_entry $terms.edge "Edge Weight (Laplacian)" $mods(LevelSet)-edge_weight
# # 	    checkbutton $terms.exp -text "Reverse Expansion Direction" \
# # 		-variable $mods(LevelSet)-reverse_expansion_direction

# # 	    pack $terms.edge $terms.exp -side top -anchor nw \
# # 		-padx 3 -pady 1


# # 	    button $params.button -text "Update Speed Image" \
# # 		-background $execute_color \
# # 		-activebackground $execute_active_color \
# # 		-command "$this update_speed_image" \
# # 		-width 20
# # 	    pack $params.button -side top -anchor n -padx 3 -pady 3 -ipadx 2


# # 	    # Seeding parameters
# # 	    iwidgets::labeledframe $step_tab.seeds \
# # 		-labeltext "Initial Segmentation" \
# # 		-labelpos nw
# # 	    pack $step_tab.seeds -side top -anchor nw -expand no -fill x \
# # 		-pady 3

# # 	    set seeds [$step_tab.seeds childsite]
	    

# # 	    global seed_method
# # 	    global $mods(SampleField-Seeds)-num_seeds
# # 	    global $mods(SampleField-SeedsNeg)-num_seeds
# # 	    trace variable $mods(SampleField-Seeds)-num_seeds w "$this seeds_changed"
# # 	    trace variable $mods(SampleField-SeedsNeg)-num_seeds w "$this seeds_changed"

# # 	    frame $seeds.options
# # 	    frame $seeds.options.a
# # 	    frame $seeds.options.b
# # 	    pack $seeds.options -side top -anchor n

# # 	    pack $seeds.options.a $seeds.options.b -side left -anchor nw

# # 	    # Previous
# # 	    radiobutton $seeds.options.a.prev \
# # 		-text "Use Previous Segmentation" \
# # 		-variable seed_method -value "prev" \
# # 		-command "$this change_seed_method"

# # 	    # Current
# # 	    radiobutton $seeds.options.a.curr \
# # 		-text "Use Current Segmentation" \
# # 		-variable seed_method -value "curr" \
# # 		-command "$this change_seed_method"

# # 	    # Current
# # 	    radiobutton $seeds.options.a.next \
# # 		-text "Use Next Segmentation" \
# # 		-variable seed_method -value "next" \
# # 		-command "$this change_seed_method"

# # 	    # Thresholds
# # 	    radiobutton $seeds.options.b.thresh \
# # 		-text "Use Thresholds" \
# # 		-variable seed_method -value "thresh" \
# # 		-command "$this change_seed_method"

# # 	    # Seeds
# # 	    radiobutton $seeds.options.b.point -text "Use Seed Points Only" \
# # 		-variable seed_method -value "points" \
# # 		-command "$this change_seed_method"

# # 	    pack $seeds.options.a.prev $seeds.options.a.curr \
# # 		$seeds.options.a.next -side top -anchor nw

# # 	    pack $seeds.options.b.thresh $seeds.options.b.point \
# # 		-side top -anchor nw


# # 	    frame $seeds.points -relief groove -borderwidth 2

# # 	    frame $seeds.points.pos
# # 	    set f $seeds.points.pos
# # 	    label $f.l -text "Positive Seed Points: " \
# # 		-foreground "#990000"
# # 	    button $f.decr -text "-" -command "$this change_number_of_seeds + -"
# # 	    entry $f.e -textvariable $mods(SampleField-Seeds)-num_seeds \
# # 		-width 4 -foreground "#990000"
# # 	    button $f.incr -text "+" -command "$this change_number_of_seeds + +"
# # 	    bind $f.e <Return> "$this change_number_of_seeds + ="

# # 	    pack $f.l $f.decr $f.e $f.incr -side left -anchor nw -expand yes -fill x

# # 	    frame $seeds.points.neg
# # 	    set f $seeds.points.neg
# # 	    label $f.l -text "Negative Seed Points: " \
# # 		-foreground "blue"
# # 	    button $f.decr -text "-" -command "$this change_number_of_seeds - -"
# # 	    entry $f.e -textvariable $mods(SampleField-SeedsNeg)-num_seeds \
# # 		-width 4 -foreground "blue"
# # 	    button $f.incr -text "+" -command "$this change_number_of_seeds - +"
# # 	    bind $f.e <Return> "$this change_number_of_seeds - ="

# # 	    pack $f.l $f.decr $f.e $f.incr -side left -anchor nw -expand yes -fill x

# # 	    global show_seeds
# # 	    checkbutton $seeds.points.toggle -text "Show Seeds" \
# # 		-variable show_seeds -command "$this seeds_changed 1 2 3"

# # 	    pack $seeds.points.pos $seeds.points.neg \
# # 		-side top -anchor ne
# # 	    pack $seeds.points.toggle -side top -anchor n

# # 	    button $seeds.generate -text "Update Initial Segmentation" \
# # 		-background $execute_color \
# # 		-activebackground $execute_active_color \
# # 		-command "$this create_seeds"


# # 	    pack $seeds.points $seeds.generate -side top -anchor n -pady 3 -ipadx 2

# # 	    # Segment frame

# # 	    iwidgets::labeledframe $step_tab.segment \
# # 		-labeltext "Segment" \
# # 		-labelpos nw
# # 	    pack $step_tab.segment -side top -anchor nw \
# # 		-expand no -fill x
# # 	    set segment [$step_tab.segment childsite]

# # 	    global max_iter
# # 	    global $mods(LevelSet)-max_rms_change

# # 	    frame $segment.params
# # 	    make_entry $segment.params.iter "Maximum Iterations:" max_iter 5
# # 	    make_entry $segment.params.rms "Maximum RMS:" $mods(LevelSet)-max_rms_change 5
# # 	    pack $segment.params.iter $segment.params.rms \
# # 		-side left -anchor nw
# # 	    frame $segment.buttons
# # 	    button $segment.buttons.reset -text "Reset" \
# # 		-background $execute_color \
# # 		-activebackground $execute_active_color \
# # 		-command "$this start_segmentation Reset"
# # 	    button $segment.buttons.go -text "Go" \
# # 		-background $execute_color \
# # 		-activebackground $execute_active_color \
# # 		-command "$this start_segmentation Go"
# # 	    button $segment.buttons.stop -text "Stop" \
# # 		-background "#990000" \
# # 		-activebackground "#CC0000" \
# # 		-command "$this stop_segmentation"
# # 	    button $segment.buttons.commit -text "Commit" \
# # 		-activebackground $next_color \
# # 		-background $next_color \
# # 		-command "$this commit_segmentation"
# # 	    pack $segment.buttons.reset $segment.buttons.go \
# # 		$segment.buttons.stop $segment.buttons.commit \
# # 		-side left -anchor n -padx 4 -pady 3 -expand yes \
# # 		-ipadx 2

# # 	    pack $segment.params -side top -anchor nw 
# # 	    pack $segment.buttons -side top -anchor n -expand yes -fill x
	    

# # 	    button $step_tab.volren -text "Update Volume Rendering" \
# # 		-state disabled \
# # 		-command "$this update_volume_rendering"
# # 	    pack $step_tab.volren -side top -anchor n -pady 3

# # # 	    # Radiobuttons for what is volume rendered
# #  	    frame $step_tab.whichvol 
# #  	    pack $step_tab.whichvol -side top -anchor n -pady 3
# #  	    global vol_foreground
# #  	    radiobutton $step_tab.whichvol.a -text "Show Segmentation" \
# #  		-variable vol_foreground -value 1 \
# #  		-command "$this toggle_volume_render_object"
# #  	    radiobutton $step_tab.whichvol.b -text "Show Background" \
# #  		-variable vol_foreground -value 0 \
# #  		-command "$this toggle_volume_render_object"
# #  	    pack $step_tab.whichvol.a $step_tab.whichvol.b -side left \
# #  		-anchor nw -pady 3

# # 	    frame $step_tab.savebin 
# # 	    frame $step_tab.savefl
# # 	    pack $step_tab.savebin $step_tab.savefl -side top -anchor n \
# # 		-pady 2
	    
# # 	    global $mods(ImageFileWriter-Binary)-filename
# # 	    button $step_tab.savebin.btn -text "Save Binary" \
# # 		-command "$this save_binary" -state disabled
# # 	    label $step_tab.savebin.l -text "File:"
# # 	    entry $step_tab.savebin.e \
# # 		-textvariable $mods(ImageFileWriter-Binary)-filename
# # 	    button $step_tab.savebin.browse -text "Browse" \
# # 		-command "$this open_save_binary_ui"
# # 	    pack $step_tab.savebin.btn $step_tab.savebin.l \
# # 		$step_tab.savebin.e $step_tab.savebin.browse -side left

# # 	    global $mods(ImageFileWriter-Float)-filename
# # 	    button $step_tab.savefl.btn -text " Save Float " \
# # 		-command "$this save_float" -state disabled
# # 	    label $step_tab.savefl.l -text "File:"
# # 	    entry $step_tab.savefl.e \
# # 		-textvariable $mods(ImageFileWriter-Float)-filename
# # 	    button $step_tab.savefl.browse -text "Browse" \
# # 		-command "$this open_save_float_ui"
# # 	    pack $step_tab.savefl.btn $step_tab.savefl.l \
# # 		$step_tab.savefl.e $step_tab.savefl.browse -side left
	    
	    
            ### Indicator
	    frame $process.indicator -relief sunken -borderwidth 2
            pack $process.indicator -side bottom -anchor s -padx 3 -pady 5
	    
	    canvas $process.indicator.canvas -bg "white" -width $i_width \
	        -height $i_height 
	    pack $process.indicator.canvas -side top -anchor n -padx 3 -pady 3
	    
            bind $process.indicator <Button> {app display_module_error} 
	    
            label $process.indicatorL -text "Press Load to Load Volume..."
            pack $process.indicatorL -side bottom -anchor sw -padx 5 -pady 3
	    	    
            # The new way of keeping track of tabs is to use 0 and 1.  The old
	    # old way was to use 1 and 2.  This app uses the old way except with
	    # regards to the indicator object, because it is used in the PowerAppBase
	    # class.
	    if {$case == 1} {
		set indicator0 $process.indicator.canvas
		set indicatorL0 $process.indicatorL
	    } else {
		set indicator1 $process.indicator.canvas
		set indicatorL1 $process.indicatorL
	    }


            construct_indicator $process.indicator.canvas
	    
#            $process.tnb view "Load"

	    ### Attach/Detach button
            frame $m.d 
	    pack $m.d -side left -anchor e
            for {set i 0} {$i<40} {incr i} {
                button $m.d.cut$i -text " | " -borderwidth 0 \
                    -foreground "gray25" \
                    -activeforeground "gray25" \
                    -command "$this switch_P_frames" 
	        pack $m.d.cut$i -side top -anchor se -pady 0 -padx 0
                if {$case == 0} {
		    Tooltip $m.d.cut$i $tips(ProcAttachHashes)
		} else {
		    Tooltip $m.d.cut$i $tips(ProcDetachHashes)
		}
            }
	    
	}
	
        wm protocol .standalone WM_DELETE_WINDOW { NiceQuit }  
    }

    method create_volumes_frame {process case} {
	global mods

	### Volumes
	iwidgets::labeledframe $process.volumes \
	    -labelpos nw -labeltext "1. Load Volumes"
	pack $process.volumes -side top -anchor nw -expand yes -fill x
	
	set data [$process.volumes childsite]
	frame $data.lowres
	frame $data.highres
	
	# lowres
	global $mods(AnalyzeNrrdReader)-filenames0
	label $data.lowres.l -text "Low-Res File: "
	entry $data.lowres.e -width 25 \
	    -textvariable $mods(AnalyzeNrrdReader)-filenames0
	button $data.lowres.b -text "Browse" \
	    -command "$this analyze_ui" -width 8
	button $data.lowres.go -text "Go" -width 3 \
	    -background $execute_color \
	    -activebackground $execute_active_color \
	    -command "$this go_lowres"
	pack $data.lowres.l $data.lowres.e $data.lowres.b \
	    -side left -padx 3
	pack $data.lowres.go \
	    -side right -padx 1
	
	# high res
	global $mods(SliceReader)-filename
	label $data.highres.l -text "High-Res File:"
	entry $data.highres.e -width 25 \
	    -textvariable $mods(SliceReader)-filename
	button $data.highres.b -text "Browse" \
	    -command "$this slice_reader_ui" -width 8
	pack $data.highres.l $data.highres.e $data.highres.b \
	    -side left -padx 3
	
	# high res slice
	frame $data.slice
	global $mods(SliceReader)-slice
	label $data.slice.l -text " Slice:"
 	scale $data.slice.sl -variable $mods(SliceReader)-slice \
 	    -from 0 -to 255 -width 15 \
 	    -showvalue false -length 200 \
 	    -orient horizontal
 	entry $data.slice.e -textvariable $mods(SliceReader)-slice \
 	    -width 4
	button $data.slice.next -text "Next" -width 4 \
	    -background $execute_color \
	    -activebackground $execute_active_color \
	    -command "$this next_highres"
	button $data.slice.go -text "Go" -width 3 \
	    -background $execute_color \
	    -activebackground $execute_active_color \
	    -command "$this go_highres"
#	bind $data.slice.e <Return> "$this go_highres"
	
 	pack $data.slice.l $data.slice.sl $data.slice.e \
 	    -side left 
	
	pack $data.slice.go $data.slice.next \
	    -side right -padx 1
	
	global $mods(SliceReader)-size_0
	global $mods(SliceReader)-size_1
	global $mods(SliceReader)-size_2
	
	trace variable $mods(SliceReader)-size_0 w \
	    "$this SliceReader_size_changed"
	trace variable $mods(SliceReader)-size_1 w \
	    "$this SliceReader_size_changed"
	trace variable $mods(SliceReader)-size_2 w \
	    "$this SliceReader_size_changed"

	global $mods(SliceReader)-spacing_0
	global $mods(SliceReader)-spacing_1
	global $mods(SliceReader)-spacing_2
	trace variable $mods(SliceReader)-spacing_0 w \
	    "$this SliceReader_spacing_changed"
	trace variable $mods(SliceReader)-spacing_1 w \
	    "$this SliceReader_spacing_changed"
	trace variable $mods(SliceReader)-spacing_2 w \
	    "$this SliceReader_spacing_changed"
	
	
	pack $data.lowres $data.highres $data.slice \
	    -side top -anchor nw -expand yes -fill x -padx 3
    }

    method create_smoothing_frame {process case} {
	global mods
	
	### Smoothing
	iwidgets::labeledframe $process.smoothing \
	    -labelpos nw -labeltext "2. Setup Smoothing"
	pack $process.smoothing -side top -anchor nw -expand yes -fill x
	
	set smooth [$process.smoothing childsite]
	
	frame $smooth.a
	
	global to_smooth
	checkbutton $smooth.a.toggle -text "Smooth Slice" \
	    -variable to_smooth \
	    -command "$this to_smooth_changed"
	
	iwidgets::optionmenu $smooth.a.filter -labeltext "Filter:" \
	    -labelpos w -command "$this change_filter $smooth.a.filter"
	
	pack $smooth.a.toggle -side left -anchor w -padx 3
	pack $smooth.a.filter -side right -anchor e -padx 3
	
	set filter_menu$case $smooth.a.filter
	
	$smooth.a.filter insert end "GradientAnisotropicDiffusion" \
	    "CurvatureAnisotropicDiffusion" \
	    "Gaussian" 
	$smooth.a.filter select "GradientAnisotropicDiffusion"
	
	pack $smooth.a -side top -anchor nw -expand yes -fill x

	# pack ui for GradientAnisotropic first
	# Gradient
	frame $smooth.gradient
	pack $smooth.gradient -side top -anchor n -pady 4 \
	    -expand yes -fill x
	
	global $mods(Smooth-Gradient)-time_step
	global $mods(Smooth-Gradient)-iterations
	global $mods(Smooth-Gradient)-conductance_parameter
	make_entry $smooth.gradient.time "Time Step" \
	    $mods(Smooth-Gradient)-time_step 8
	make_entry $smooth.gradient.iter "Iterations" \
	    $mods(Smooth-Gradient)-iterations 4 
	make_entry $smooth.gradient.cond "Conductance" \
	    $mods(Smooth-Gradient)-conductance_parameter 4
	button $smooth.gradient.b -text "Go" -width 3 \
	    -background $execute_color \
	    -activebackground $execute_active_color \
	    -command "$this smooth_slice"
	
	pack $smooth.gradient.time $smooth.gradient.iter \
	    $smooth.gradient.cond -side left
	pack $smooth.gradient.b -side right -padx 3
	
	# Curvature
	frame $smooth.curvature
	global $mods(Smooth-Curvature)-time_step
	global $mods(Smooth-Curvature)-iterations
	global $mods(Smooth-Curvature)-conductance_parameter
	make_entry $smooth.curvature.time "Time Step" \
	    $mods(Smooth-Curvature)-time_step 8
	make_entry $smooth.curvature.iter "Iterations" \
	    $mods(Smooth-Curvature)-iterations 4 
	make_entry $smooth.curvature.cond "Conductance" \
	    $mods(Smooth-Curvature)-conductance_parameter 4
	button $smooth.curvature.b -text "Go" -width 3 \
	    -background $execute_color \
	    -activebackground $execute_active_color \
	    -command "$this smooth_slice"
	    
	pack $smooth.curvature.time $smooth.curvature.iter \
	    $smooth.curvature.cond -side left
	pack $smooth.curvature.b -side right -padx 3

	# Gaussian
	frame $smooth.gaussian
	global $mods(Smooth-Gaussian)-variance
	global $mods(Smooth-Gaussian)-maximum_error
	global $mods(Smooth-Gaussian)-maximum_kernel_width
	make_entry $smooth.gaussian.var "Variance" \
	    $mods(Smooth-Gaussian)-variance 
	make_entry $smooth.gaussian.err "Max Error" \
	    $mods(Smooth-Gaussian)-maximum_error 
	make_entry $smooth.gaussian.kern "Kernel Width" \
	    $mods(Smooth-Gaussian)-maximum_kernel_width 
	button $smooth.gaussian.b -text "Go" -width 3 \
	    -background $execute_color \
	    -activebackground $execute_active_color \
	    -command "$this smooth_slice"
	
	pack $smooth.gaussian.var $smooth.gaussian.err $smooth.gaussian.kern \
	    -side left
	pack $smooth.gaussian.b -side right -padx 3

	frame $smooth.show 
	pack $smooth.show -side bottom -anchor s

	global $mods(ChooseImage-ShowSlice)-port-index
	radiobutton $smooth.show.orig -text "Show Original" \
	    -variable $mods(ChooseImage-ShowSlice)-port-index -value 0 \
	    -command "$mods(ChooseImage-ShowSlice)-c needexecute"
	
	radiobutton $smooth.show.smooth -text "Show Smoothed" \
	    -variable $mods(ChooseImage-ShowSlice)-port-index -value 1 \
	    -command "$mods(ChooseImage-ShowSlice)-c needexecute"
	
	pack $smooth.show.orig $smooth.show.smooth -side left \
	    -padx 4
    }

    method create_speed_frame {process case} {
	global mods
	
	### Tuning Speed Image
	iwidgets::labeledframe $process.speed \
	    -labelpos nw -labeltext "3. Tune Speed Image"
	pack $process.speed -side top -anchor nw -expand yes -fill x
	
	set speed [$process.speed childsite]

	global $mods(LevelSet)-lower_threshold
	global $mods(LevelSet)-upper_threshold
	global $mods(LevelSet)-curvature_scaling
	global $mods(LevelSet)-propagation_scaling
	global $mods(LevelSet)-edge_weight
	global $mods(LevelSet)-max_iterations
	global $mods(LevelSet)-max_rms_change
	global $mods(LevelSet)-reverse_expansion_direction
	global $mods(LevelSet)-smoothing_iterations
	global $mods(LevelSet)-smoothing_time_step
	global $mods(LevelSet)-smoothing_conductance
	
	trace variable $mods(LevelSet)-lower_threshold w \
	    "$this update_seed_binary_threshold"
	trace variable $mods(LevelSet)-upper_threshold w \
	    "$this update_seed_binary_threshold"
	
	# thresholds
	frame $speed.lthresh
	pack $speed.lthresh -side top -anchor nw -expand yes -fill x
	label $speed.lthresh.l -text "Lower Threshold:"
	scale $speed.lthresh.s -variable $mods(LevelSet)-lower_threshold \
	    -from 0 -to 255 -width 15 \
	    -showvalue false -length 200 \
	    -orient horizontal
	entry $speed.lthresh.e -textvariable $mods(LevelSet)-lower_threshold \
	    -width 6
	pack $speed.lthresh.l $speed.lthresh.s $speed.lthresh.e \
	    -side left -pady 2
	
	frame $speed.uthresh
	pack $speed.uthresh -side top -anchor nw -expand yes -fill x
	label $speed.uthresh.l -text "Upper Threshold:"
	scale $speed.uthresh.s -variable $mods(LevelSet)-upper_threshold \
	    -from 0 -to 255 -width 15 \
	    -showvalue false -length 200 \
	    -orient horizontal
	entry $speed.uthresh.e -textvariable $mods(LevelSet)-upper_threshold \
	    -width 6
	
	pack $speed.uthresh.l $speed.uthresh.s $speed.uthresh.e \
	    -side left -pady 2

	frame $speed.extra
	pack $speed.extra -side top -anchor n -expand yes -fill x

	checkbutton $speed.extra.exp -text "Reverse Expansion Direction" \
	    -variable $mods(LevelSet)-reverse_expansion_direction \
	    -command "$this reverse_expansion_direction_changed"

	button $speed.extra.b -text "Go" -width 3 \
	    -background $execute_color \
	    -activebackground $execute_active_color \
	    -command "$this update_speed_image"
	
	pack $speed.extra.exp -side left -anchor w \
	    -padx 3 -pady 1

	pack $speed.extra.b -side right -anchor e \
	    -padx 3 -pady 1
    }

    method create_seeds_frame {process case} {
	global mods
	
	### Creating Seeds
	iwidgets::labeledframe $process.seeds \
	    -labelpos nw -labeltext "4. Create Seeds"
	pack $process.seeds -side top -anchor nw -expand yes -fill x
	
	set seeds [$process.seeds childsite]

	iwidgets::optionmenu $seeds.method -labeltext "Seed Method:" \
	    -labelpos w -command "$this change_seed_method $seeds.method"
	pack $seeds.method -side top -anchor nw 
	
	set seed_menu$case $seeds.method
	
	$seeds.method insert end "Thresholds and Seed Points" \
	    "Previous Segmentation and Seed Points" "Seed Points Only"
	$seeds.method select $seed_type

	# Initially disable previous option
	$seeds.method disable "Previous Segmentation and Seed Points"

	frame $seeds.points 
	pack $seeds.points -side top -anchor nw -pady 2 \
	    -expand yes -fill x

	# positive
	global $mods(SeedPoints-PosSeeds)-num_seeds
	trace variable $mods(SeedPoints-PosSeeds)-num_seeds w \
	    "$this seeds_changed"

	frame $seeds.points.pos -relief groove -borderwidth 2
	set f $seeds.points.pos
	label $f.l -text "Positive Seeds: " \
	    -foreground "#990000"
	button $f.decr -text "-" -command "$this change_number_of_seeds + -"
	entry $f.e -textvariable $mods(SeedPoints-PosSeeds)-num_seeds \
	    -width 4 -foreground "#990000"
	button $f.incr -text "+" -command "$this change_number_of_seeds + +"
	bind $f.e <Return> "$this change_number_of_seeds + ="
	
	pack $f.l $f.decr $f.e $f.incr -side left -anchor nw \
	    -expand yes -fill x
	
	# negative
	global $mods(SeedPoints-NegSeeds)-num_seeds
	trace variable $mods(SeedPoints-NegSeeds)-num_seeds w \
	    "$this seeds_changed"

	frame $seeds.points.neg -relief groove -borderwidth 2
	set f $seeds.points.neg
	label $f.l -text "Negative Seeds: " \
	    -foreground "blue"
	button $f.decr -text "-" -command "$this change_number_of_seeds - -"
	entry $f.e -textvariable $mods(SeedPoints-NegSeeds)-num_seeds \
	    -width 4 -foreground "blue"
	button $f.incr -text "+" -command "$this change_number_of_seeds - +"
	bind $f.e <Return> "$this change_number_of_seeds - ="
	
	pack $f.l $f.decr $f.e $f.incr -side left -anchor nw \
	    -expand yes -fill x

	pack $seeds.points.pos $seeds.points.neg \
	    -side left -anchor w -padx 3 

	button $seeds.points.b -text "Go" -width 3 \
	    -background $execute_color \
	    -activebackground $execute_active_color \
	    -command "$this create_seeds"

	pack $seeds.points.b -side right -anchor e -padx 3
    }
    

    method create_segmentation_frame {process case} {
	global mods
	
	### Segmenting
	iwidgets::labeledframe $process.seg \
	    -labelpos nw -labeltext "5. Segment"
	pack $process.seg -side top -anchor nw -expand yes -fill x
	
	set segment [$process.seg childsite]

	global max_iter
	global $mods(LevelSet)-max_rms_change

	# Equation Term Weights
	iwidgets::labeledframe $segment.terms \
	    -labeltext "Equation Term Weights" \
	    -labelpos nw
	pack $segment.terms -side top -anchor n -padx 3
	
	set terms [$segment.terms childsite]
	frame $terms.scaling1
	pack $terms.scaling1 -side top -anchor nw \
	    -padx 3 -pady 1
	make_entry $terms.scaling1.curv "Curvature:" \
	    $mods(LevelSet)-curvature_scaling
	make_entry $terms.scaling1.prop "Propagation:" \
	    $mods(LevelSet)-propagation_scaling
	make_entry $terms.scaling1.edge "Edge Weight:" \
	    $mods(LevelSet)-edge_weight
	
	pack $terms.scaling1.curv $terms.scaling1.prop \
	    $terms.scaling1.edge -side left -anchor ne \
	    -padx 3 -pady 1
	
	frame $segment.params
	make_entry $segment.params.iter "Maximum Iterations:" max_iter 5
	make_entry $segment.params.rms "Maximum RMS:" \
	    $mods(LevelSet)-max_rms_change 5
	pack $segment.params.iter $segment.params.rms \
	    -side left -anchor w

	button $segment.params.stop -text "Stop" \
	    -command "$this stop_segmentation"
	button $segment.params.go -text "Go" \
	    -background $execute_color -width 3 \
	    -activebackground $execute_active_color \
	    -command "$this start_segmentation"
	pack $segment.params.go  $segment.params.stop \
	    -side right -anchor w -padx 3 
	
	pack $segment.params -side top -anchor n -pady 2 \
	    -expand yes -fill both
    }


    method create_commit_frame {process case} {
	global mods
	
	### Segmenting
	iwidgets::labeledframe $process.commit \
	    -labelpos nw -labeltext "6. Commit"
	pack $process.commit -side top -anchor nw -expand yes -fill both
	
	set commit [$process.commit childsite]

	frame $commit.params
	pack $commit.params -side top -anchor n -expand yes -fill x

 	global commit_dir
 	label $commit.params.cl -text "Commit Dir:"
 	entry $commit.params.ce -width 12 \
 	    -textvariable commit_dir 
 	button $commit.params.cb -text "Browse" \
 	    -command "$this change_commit_dir"
	global base_filename
	label $commit.params.bl -text "Base Filename:"
	entry $commit.params.be -width 8 \
	    -textvariable base_filename

 	pack $commit.params.cl $commit.params.ce \
 	    $commit.params.cb  $commit.params.bl \
	    $commit.params.be \
 	    -side left -anchor w -padx 0 -pady 3 -expand yes

 	button $commit.params.b -text "Go" \
	    -width 3 \
 	    -activebackground $execute_active_color \
 	    -background $execute_color \
 	    -command "$this commit_segmentation"
 	pack $commit.params.b  \
 	    -side right -anchor e -padx 3

	# Status canvas
	frame $commit.status 
	label $commit.status.l -text "Committed Slices"

	canvas $commit.status.canvas -bg "white" \
	    -width [expr $process_width - 75] \
	    -height 10
	pack $commit.status.l $commit.status.canvas -side top -anchor n \
	    -pady 3 -padx 3

	bind $commit.status.canvas <ButtonPress-1> "$this show_commit_status"
	
	set status_canvas$case $commit.status.canvas

	pack $commit.status -side top -anchor n
    }

    method show_commit_status {} {
	set w .standalone.commitstatus
	
	if {$has_loaded == 0} {
	    tk_messageBox -message "Please load a high-res dataset first." \
		-type ok -icon info -parent .standalone
	} elseif {![winfo exists $w]} {
	    toplevel $w 
	    wm title $w "Commit Status"

	    global axis
	    set s 0
	    if {$axis == 0} {
		set s $size0
	    } elseif {$axis == 1} {
		set s $size1
	    } else {
		set s $size2
	    }

	    # format in columns, 10 slices per column
	    set cols [expr int([expr $s/10])] 
	    if {[expr $s%10] > 0} {
		set cols [expr $cols + 1]
	    }

	    wm minsize $w [expr $cols * 67] 300

	    iwidgets::scrolledframe $w.sf \
		-width  [expr $cols * 20] \
		-height 50 -labeltext "Commit Status (Per Slice)"
	    pack $w.sf -side top -anchor n -expand yes -fill both

	    set sf [$w.sf childsite]

	    label $sf.info -text "Highlighted slices commited"
	    pack $sf.info -side top -anchor n -pady 3
	    
	    set which_col {-1}
 	    for {set i 0} {$i < $s} {incr i} {
		if {[expr $i%10] == 0} {
		    set which_col [expr $which_col + 1]
		    frame $sf.c$which_col -relief groove -borderwidth 2
		    pack $sf.c$which_col -side left -anchor nw -padx 3
		}
 		frame $sf.c$which_col.s$i
 		label $sf.c$which_col.s$i.l -text "Slice $i: $commits($i)"
		if {$commits($i) == 1} {
		    $sf.c$which_col.s$i.l configure -foreground "#cc0000"
		}
 		pack $sf.c$which_col.s$i.l -side left
 		pack $sf.c$which_col.s$i -side top -anchor nw
 	    }
	    
	    button $w.b -text "Close" \
		-command "destroy $w"
	    pack $w.b -side top -anchor n -pady 3 -ipadx 2
	} else {
	    SciRaise $w
	}
    }

    method update_commit_status_window {s} {
	set sf .standalone.commitstatus.sf.lwchildsite.clipper.canvas.sfchildsite
	if {[winfo exists $sf]} {
	    # update the new slice s

	    # determine which column its in
	    set col [expr int([expr $s/10])]

	    # change the label text and color
	    if {[winfo exists  $sf.c$col.s$s.l]} {
		$sf.c$col.s$s.l configure -text "Slice $s: $commits($s)" \
		    -foreground "#cc0000"
	    }
	}
    }

    method change_commit_dir {} {
	global commit_dir
	
	set dir [tk_chooseDirectory -mustexist 1 \
		     -parent . -title "Select Commit Directory"]
	if {$dir != ""} {
	    set commit_dir $dir
	} else {
	    tk_messageBox -message "Directory not specified" \
		-type ok -icon info -parent .standalone
	}
    }
    
    

    ##########################
    ### switch_P_frames
    ##########################
    # This method is called when the user wants to attach or detach
    # the processing frame.
    method switch_P_frames {} {
	set c_width [winfo width $win]
	set c_height [winfo height $win]
	
    	set x [winfo x $win]
	set y [expr [winfo y $win] - 20]
	
	if { $IsPAttached } {	    
	    pack forget $attachedPFr
	    set new_width [expr $c_width - $process_width]
	    append geom1 $new_width x $c_height + [expr $x+$process_width] + $y
            wm geometry $win $geom1 
	    append geom2 $process_width x $c_height + [expr $x-20] + $y
	    wm geometry $detachedPFr $geom2
	    wm deiconify $detachedPFr
	    set IsPAttached 0

	} else {
	    wm withdraw $detachedPFr
	    pack $attachedPFr -anchor n -side left -before $win.viewers
	    set new_width [expr $c_width + $process_width]
            append geom $new_width x $c_height + [expr $x - $process_width] + $y
	    wm geometry $win $geom
	    set IsPAttached 1
	}
    }

    method build_viewers {viewer1 viewer2 viewslices} {
	set w $win.viewers

	iwidgets::panedwindow $w.topbot -orient horizontal -thickness 0 \
	    -sashwidth 5000 -sashindent 0 -sashborderwidth 2 -sashheight 6 \
	    -sashcursor sb_v_double_arrow -width $viewer_width -height $viewer_height
	pack $w.topbot -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
	
	$w.topbot add top -margin 3 -minimum 0
	$w.topbot add bottom  -margin 0 -minimum 0

	set top [$w.topbot childsite top]
	set bot [$w.topbot childsite bottom]

	$w.topbot fraction 60 40

	# top
	iwidgets::panedwindow $top.lmr -orient vertical -thickness 0 \
	    -sashheight 5000 -sashwidth 6 -sashindent 0 -sashborderwidth 2 \
	    -sashcursor sb_h_double_arrow

	$top.lmr add left -margin 3 -minimum 0
	$top.lmr add right -margin 3 -minimum 0

	set topl [$top.lmr childsite left]
	set topr [$top.lmr childsite right]

	pack $top.lmr -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0

	$top.lmr fraction 70 30

	# bottom
	iwidgets::panedwindow $bot.lmr -orient vertical -thickness 0 \
	    -sashheight 5000 -sashwidth 6 -sashindent 0 -sashborderwidth 2 \
	    -sashcursor sb_h_double_arrow

	$bot.lmr add left -margin 3 -minimum 0
	$bot.lmr add right -margin 3 -minimum 0
	set botl [$bot.lmr childsite left]
	set botr [$bot.lmr childsite right]

	pack $bot.lmr -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0

	# add slice viewer on bottom left
 	$viewslices control_panel $w.cp
 	$viewslices add_nrrd_tab $w 1

	set slice_frame $botl.slice

 	create_2d_frame $botl "slice"
 	$viewslices gl_frame $botl.slice
 	pack $botl.slice -expand 1 -fill both \
 	    -side top -padx 0 -ipadx 0 -pady 0 -ipady 0

	# embed viewer on top 
	global mods

 	set eviewer [$mods(Viewer) ui_embedded]
 	$eviewer setWindow $topl $viewer_width \
 	    [expr $viewer_height/2] \

 	pack $topl -side top -anchor n \
 	    -expand 1 -fill both -padx 4 -pady 0

	# pack vis stuff into top right
	iwidgets::labeledframe $topr.vis \
	    -labelpos n -labeltext "Segmentation\nWindow Controls"
	pack $topr.vis -side top -anchor n 

	set vis [$topr.vis childsite]

	global mods(ShowField-Speed)-faces-on
	checkbutton $vis.speed -text "Show Speed Image" \
	    -variable $mods(ShowField-Speed)-faces-on \
	    -command "$this toggle_show_speed"

	global show_seeds
	checkbutton $vis.seeds -text "Show Seed Point Widgets" \
	    -variable show_seeds -command "$this seeds_changed 1 2 3"

 	global $mods(ShowField-Seed)-faces-on
 	checkbutton $vis.seed -text "Show Current Seed" \
 	    -variable $mods(ShowField-Seed)-faces-on \
	    -command "$this toggle_show_seed"

 	global $mods(ShowField-Seg)-faces-on
 	checkbutton $vis.seg -text "Show Segmentation" \
 	    -variable $mods(ShowField-Seg)-faces-on \
	    -command "$this toggle_show_segmentation"
	
	pack $vis.speed $vis.seeds $vis.seed \
	    $vis.seg -side top -anchor w

	iwidgets::labeledframe $topr.vis2 \
	    -labelpos n -labeltext "Isocontours\nWindow Controls"
	pack $topr.vis2 -side top -anchor n 

	set vis2 [$topr.vis2 childsite]
	frame $vis2.f
	pack $vis2.f -side top -anchor nw -expand yes -fill x

	global spacing
	label $vis2.f.l -text "Isocontour Spacing:"
	entry $vis2.f.e -textvariable spacing -width 3
	pack $vis2.f.l $vis2.f.e -side left -anchor w -padx 3
	scale $vis2.s -label "" \
	    -variable spacing \
	    -from 0 -to 10 -width 15 -length 200 \
	    -showvalue true -resolution 0.5  \
	    -orient vertical -showvalue false \
	    -command "$this update_spacing"
	pack $vis2.s -side top -anchor n 

 	pack $topr -side top -anchor n \
 	    -expand 1 -fill both -padx 4 -pady 0

	# embed volume viewer in bottom right
 	set eviewer2 [$mods(Viewer-Vol) ui_embedded]
 	$eviewer2 setWindow $botr $viewer_width \
 	    [expr $viewer_height/2] \

 	pack $botr -side top -anchor n \
 	    -expand 1 -fill both -padx 4 -pady 0

	# rebind middle mouse on ViewWindows so that it
	# is truly orthographic
	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane0.childsite <ButtonPress-2> \
	    "$this do_nothing"
	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane0.childsite <ButtonRelease-2> \
	    "$this do_nothing"
	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane0.childsite <Button2-Motion> \
	    "$this do_nothing"

# # 	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane1.childsite \
# # 	    <ButtonPress-2> "$this do_nothing"
# # 	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane1.childsite \
# # 	    <ButtonRelease-2> "$this do_nothing"
# # 	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane1.childsite \
# # 	    <Button2-Motion> "$this do_nothing"
    }

    method create_2d_frame { window axis } {
	global mods 

	# Modes for $axis
	frame $window.modes
	pack $window.modes -side bottom -padx 0 -pady 0 -expand 0 -fill x
	
	frame $window.modes.slider
	pack $window.modes.slider \
	    -side top -pady 0 -anchor n -expand yes -fill x
	
	# Initialize with slice scale visibe
	frame $window.modes.slider.slice
	pack $window.modes.slider.slice -side top -anchor n -expand 1 -fill x

	# dummy value label
	label $window.modes.slider.slice.dummy -text "Slice:" 

	# slice slider
	scale $window.modes.slider.slice.s \
	    -variable $mods(ViewSlices)-$axis-viewport0-slice \
	    -from 0 -to 20 -width 13 \
	    -showvalue false \
	    -orient horizontal \
	    -command "$mods(ViewSlices)-c rebind $window.$axis; \
                      $mods(ViewSlices)-c redrawall"

	# slice value label
	label $window.modes.slider.slice.l \
	    -textvariable $mods(ViewSlices)-$axis-viewport0-slice \
	    -justify left -width 3 -anchor w
	
	pack $window.modes.slider.slice.dummy -anchor w -side left \
	    -padx 0 -pady 0 -expand 0
	
	pack $window.modes.slider.slice.l -anchor e -side right \
	    -padx 0 -pady 0 -expand 0

	pack $window.modes.slider.slice.s -anchor n -side left \
	    -padx 0 -pady 0 -expand 1 -fill x

	
	# show/hide bar
	set img [image create photo -width 1 -height 1]
	button $window.modes.expand -height 4 -bd 2 \
	    -relief raised -image $img \
	    -cursor based_arrow_down \
	    -command "$this hide_vs_control_panel $window.modes"
	pack $window.modes.expand -side bottom -fill both
    }

    method show_vs_control_panel { w } {
	pack forget $w.expand
	pack $w.slider -side top -pady 0 -anchor nw -expand yes -fill x
	pack $w.expand -side bottom -fill both

	$w.expand configure -command "$this hide_vs_control_panel $w" \
	    -cursor based_arrow_down
    }

    method hide_vs_control_panel { w } {
	pack forget $w.slider
	pack $w.expand -side bottom -fill both

	$w.expand configure -command "$this show_vs_control_panel $w" \
	    -cursor based_arrow_up
    }

# #     method change_processing_tab {which} {
# # 	if {$initialized} {
# # 	    if {$which == "Segment" && !$loading} {
# # 		if {$has_smoothed == 0} {
# # 		    # $this smooth_data
# # 		}
# # 		# get speed image
# # 		$this initialize_segmentation
# # 	    }
	    
# # 	    $proc_tab1 view $which
# # 	    $proc_tab2 view $which
# # 	    set curr_proc_tab $which
# # 	}
# #     }


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
	    wm title .standalone "LevelSetSegmenter - [getFileName $saveFile]" 

	    set fileid [open $saveFile w]
	    
	    # Save out data information 
	    puts $fileid "# LevelSetSegmenter Session\n"
	    puts $fileid "set app_version 1.0"

	    save_module_variables $fileid

	    puts $fileid "set \$mods(ShowDipole)-num-dipoles {0}"
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
	
	set saveFile [tk_getOpenFile -filetypes $types]

	if {$saveFile != ""} {
	    load_session_data
	}
    }

    method load_session_data {} {
	
	wm title .standalone "LevelSetSegmenter - [getFileName $saveFile]"

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
	tk_messageBox -message "Please refer to the online LevelSetSegmenter Tutorial\nhttp://software.sci.utah.edu/doc/User/LevelSetSegmenterTutorial" -type ok -icon info -parent .standalone
    }
    
    method show_about {} {
	tk_messageBox -message "Future To Do List:\n---------------------\nLoading/Saving Sessions\nLoading Saved Binary/Float Segmentation\nFinal Antialiasing or Smoothing Step\n" -type ok -icon info -parent .standalone
    }
    

    method indicate_dynamic_compile { which mode } {
	global mods

	if {$mode == "start"} {
	    change_indicate_val 1
	    change_indicator_labels "Dynamically Compiling $which..."
        } else {
	    change_indicate_val 2
	    if {$has_loaded == 1} {
		change_indicator_labels "Done Loading Volume"
	    } else {
		change_indicator_labels "Loading Volume..."
	    }
	}
    }
    
    
    method update_progress { which state } {
	global mods

	if {$which == $mods(SliceReader) && \
		$state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(SliceReader) && \
		      $state == "Completed"} {
	    change_indicate_val 2
	    set has_loaded 1
	} elseif {$which == $mods(ViewSlices) && \
		      $state == "Completed"} {
	    if {$2D_fixed == 0} {
 		# setup correct axis
 		global $mods(ViewSlices)-slice-viewport0-axis axis
		
 		set $mods(ViewSlices)-slice-viewport0-axis $axis
		
 		# fix window width and level
 		upvar \#0 $mods(ViewSlices)-min val_min 
		upvar \#0 $mods(ViewSlices)-max val_max
		set ww [expr abs($val_max-$val_min)]
		set wl [expr ($val_min+$val_max)/2.0]
		
 		setGlobal $mods(ViewSlices)-clut_ww $ww
 		setGlobal $mods(ViewSlices)-clut_wl $wl
		
		$mods(ViewSlices)-c rebind $slice_frame
		
 		$mods(ViewSlices)-c setclut
		
		set 2D_fixed 1
 	    } 
 	} elseif {$which == $mods(NrrdInfo-Slice) && \
		      $state == "Completed"} {
 	    global $which-dimension
	    
 	    if {[set $which-dimension] != 3} {
 		tk_messageBox -message "Data must be 3 dimensional scalar data." -type ok -icon info -parent .standalone
 		return
 	    }

 	    global $which-size0 $which-size1 $which-size2
 	    set size0 [set $which-size0]
 	    set size1 [set $which-size1]
 	    set size2 [set $which-size2]

 	    configure_slice_sliders
 	} elseif {$which == $mods(ShowField-Slice) && \
		      $state == "JustStarted"} { 
 	    change_indicate_val 1
 	} elseif {$which == $mods(ShowField-Slice) && \
		      $state == "Completed"} { 
 	    change_indicate_val 2

 	    # Setup views of 2 View Windows
 	    global axis
 	    global $mods(Viewer)-ViewWindow_0-pos
 	    global $mods(Viewer)-ViewWindow_1-pos
 	    set $mods(Viewer)-ViewWindow_0-pos "z1_y1"
 	    set $mods(Viewer)-ViewWindow_1-pos "z1_y1"

  	    if {$axis == 0} {
  		set $mods(Viewer)-ViewWindow_0-pos "x0_y0"
  		set $mods(Viewer)-ViewWindow_1-pos "x0_y0"
  	    } elseif {$axis == 1} {
  		set $mods(Viewer)-ViewWindow_0-pos "y0_x0"
  		set $mods(Viewer)-ViewWindow_1-pos "y0_x0"
  	    } else {
  		set $mods(Viewer)-ViewWindow_0-pos "z1_y1"
  		set $mods(Viewer)-ViewWindow_1-pos "z1_y1"
  	    }

 	    after 100 "$mods(Viewer)-ViewWindow_0-c autoview; $mods(Viewer)-ViewWindow_0-c Views"
 	} elseif {$which == $mods(Smooth-Gradient) && \
		      $state == "JustStarted"} { 
 	    change_indicate_val 1
	    change_indicator_labels "Smoothing..."
	} elseif {$which == $mods(Smooth-Gradient) && \
		      $state == "Completed"} { 
 	    change_indicate_val 2
	    change_indicator_labels "Done Smoothing"
	} elseif {$which == $mods(Smooth-Curvature) && \
		      $state == "JustStarted"} { 
 	    change_indicate_val 1
	    change_indicator_labels "Smoothing..."
	} elseif {$which == $mods(Smooth-Curvature) && \
		      $state == "Completed"} { 
 	    change_indicate_val 2
	    change_indicator_labels "Done Smoothing"
	} elseif {$which == $mods(Smooth-Gaussian) && \
		      $state == "JustStarted"} { 
 	    change_indicate_val 1
	    change_indicator_labels "Smoothing..."
	} elseif {$which == $mods(Smooth-Gaussian) && \
		      $state == "Completed"} { 
 	    change_indicate_val 2
	    change_indicator_labels "Done Smoothing"
	} elseif {$which == $mods(FieldInfo-Smoothed) && \
		      $state == "Completed"} { 
	    global $which-datamin $which-datamax
	    upvar \#0 $which-datamin min
	    upvar \#0 $which-datamax max

	    if {$min != "---" && $max != "---"} {
		# reconfigure threshold sliders
		$attachedPFr.f.p.childsite.speed.childsite.lthresh.s \
		    configure  -from $min -to $max
		$attachedPFr.f.p.childsite.speed.childsite.lthresh.s \
		    configure -from $min -to $max
		$attachedPFr.f.p.childsite.speed.childsite.uthresh.s \
		    configure -from $min -to $max
		$attachedPFr.f.p.childsite.speed.childsite.uthresh.s \
		    configure -from $min -to $max
	    }

	    # if reconfiguring the slider causes the upper and lower
	    # threshold to be the same, fix the lower threshold
	    global $mods(LevelSet)-lower_threshold 
	    global $mods(LevelSet)-lower_threshold
	    upvar \#0 $mods(LevelSet)-lower_threshold lower
	    upvar \#0 $mods(LevelSet)-upper_threshold upper
	    
	    if {$lower == $upper} {
		set lower \
		    [expr int([expr $min + [expr [expr $max - $min]/2]])]
	    }

	} elseif {$which == $mods(ShowField-Speed) && \
		      $state == "JustStarted"} { 
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Speed) && \
		      $state == "Completed"} { 
	    change_indicate_val 2
	} elseif {$which == $mods(LevelSet) && \
		      $state == "JustStarted"} { 
	    if {$segmenting == 1} {
		change_indicator_labels "Segmenting..."
	    } 
	    change_indicate_val 1
	} elseif {$which == $mods(LevelSet) && \
		      $state == "Completed"} { 
	    if {$segmenting == 1} {
		change_indicator_labels "Done Segmenting"
#		set has_segmented 1
	    } 
	    change_indicate_val 2
	    set segmenting 0

	    after 500 "set $mods(LevelSet)-max_iterations 0"
	} elseif {$which == $mods(ImageFileWriter-Binary) && \
		      $state == "Completed"} { 
	    if {$committing == 1} {
		set committing 0
		disableModule $mods(ImageFileWriter-Binary) 1

		# indicate commit on status canvas
		global $mods(SliceReader)-slice
#		upvar \#0 $mods(SliceReader)-slice s
		set s [set $mods(SliceReader)-slice]

		set oldx [expr $s * $status_width]
		set newx [expr $oldx + $status_width]
		set r 0
		set g 139
		set b 69
		set c [format "#%02x%02x%02x" $r $g $b]

		$status_canvas1 create rectangle \
		    $oldx 0 $newx 10 \
		    -fill $c -outline "black" -tags completed
		$status_canvas2 create rectangle \
		    $oldx 0 $newx 10 \
		    -fill $c -outline "black" -tags completed

		set commits($s) 1
		$this update_commit_status_window $s
	    }
	} elseif {$which == $mods(GatherFields) && \
		      $state == "Completed"} {
	    # disable 3d stuff
	    disableModule $mods(ImageToField-Iso) 1
	} elseif {$which == $mods(ShowField-Iso) && \
		      $state == "Completed"} {
	    if {$has_autoviewed == 0} {
		set has_autoviewed 1
		after 100 "$mods(Viewer-Vol)-ViewWindow_0-c autoview; global $mods(Viewer-Vol)-ViewWindow_0-pos; set $mods(Viewer-Vol)-ViewWindow_0-pos \"z1_y1\"; $mods(Viewer-Vol)-ViewWindow_0-c Views;"
	    }
	}
	
# # 	if {$which == $mods(PasteImageFilter-Smooth) \
# # 		&& $state == "Completed"} {

# # 	    global axis

# # 	    # determine if this was the last run
# # 	    set last 0
# # 	    if {$axis == 0} {
# # 		set last [expr $size0 - 1]
# # 	    } elseif {$axis == 1} {
# # 		set last [expr $size1 - 1]
# # 	    } else {
# # 		set last [expr $size2 - 1]
# # 	    }
	    
# # 	    global $mods(PasteImageFilter-Smooth)-index
# # 	    if {[set $mods(PasteImageFilter-Smooth)-index] == $last} {
# # 		# set 2D viewer to use smoothed data as input
# # 		global $mods(ChooseNrrd-2D)-port-index
# # 		set $mods(ChooseNrrd-2D)-port-index 1
		
# # 		set which [$filter_menu1 get]
# # 		set has_smoothed 1
# # 		set region_changed 0
# # 		set smoothing 0

# #  		# enable modules downstream of smoothing and execute
# #  		# them
# # ###  		disableModule $mods(ImageToNrrd-ViewSlices) 0
# # 		after 500 "$mods(ImageToNrrd-ViewSlices)-c needexecute"
# # 	    } else {
# # 		# increment Extract values and Paste index
# # 		if {$axis == 0} {
# # 		    global $mods(Extract-Smooth)-minDim0
# # 		    global $mods(Extract-Smooth)-maxDim0

# # 		    set prev [expr [set $mods(Extract-Smooth)-minDim0] + 1]
# # 		    set $mods(Extract-Smooth)-minDim0 $prev
# # 		    set $mods(Extract-Smooth)-maxDim0 [expr $prev + 1]
# # 		} elseif {$axis == 1} {
# # 		    global $mods(Extract-Smooth)-minDim1
# # 		    global $mods(Extract-Smooth)-maxDim1

# # 		    set prev [expr [set $mods(Extract-Smooth)-minDim1] + 1]
# # 		    set $mods(Extract-Smooth)-minDim1 $prev
# # 		    set $mods(Extract-Smooth)-maxDim1 [expr $prev + 1]
# # 		} else {
# # 		    global $mods(Extract-Smooth)-minDim2
# # 		    global $mods(Extract-Smooth)-maxDim2

# # 		    set prev [expr [set $mods(Extract-Smooth)-minDim2] + 1]
# # 		    set $mods(Extract-Smooth)-minDim2 $prev
# # 		    set $mods(Extract-Smooth)-maxDim2 [expr $prev + 1]
# # 		}
		
# # 		global $mods(PasteImageFilter-Smooth)-index
# # 		set $mods(PasteImageFilter-Smooth)-index \
# # 		    [expr [set $mods(PasteImageFilter-Smooth)-index] + 1]

# # 		# Execute Extract Again
# # 		after 500 "$mods(Extract-Smooth)-c needexecute"
# # 	    }
# # 	} elseif {$which == $mods(ChooseImage-Hack) && \
# # 		      $state == "Completed"} {	    
# # 	    if {$smoothing_type == "Reset"} {
# # 		disableModule $mods(ChooseImage-SmoothInput) 1
# # 	    } else {
# # 		# Data has gotten to Choose module so disable it
# # 		# and then enable Extract and execute it
# # 		disableModule $mods(ChooseImage-SmoothInput) 1
# # 		disableModule $mods(Extract-Smooth) 0
		
# # 		$mods(Extract-Smooth)-c needexecute
# # 	    }
# # 	} elseif {$which == $mods(ChooseImage-Hack2) && \
# # 		      $state == "Completed"} {	    
# # 	    if {$segmenting_type == "Reset"} {
# # 		disableModule $mods(ChooseImage-SegInput) 1
# # 	    } else {
# # 		# Data has gotten to Choose module so disable it
# # 		# and then enable next Choose and execute it
# # 		disableModule $mods(ChooseImage-SegInput) 1
# # 		disableModule $mods(ChooseImage-Hack3) 0
		
# # 		$mods(ChooseImage-Hack3)-c needexecute
# # 	    }
# # 	} elseif {$which == $mods(ChooseNrrd-Reader) && $state == "JustStarted"} {
# # 	    change_indicate_val 1
# # 	    change_indicator_labels "Loading Volume..."
# # 	} elseif {$which == $mods(ChooseNrrd-Reader) && $state == "Completed"} {
# # 	    change_indicate_val 2
# # 	} elseif {$which == $mods(ScalarFieldStats) && $state == "JustStarted"} {
# # 	    change_indicate_val 1
# # 	    change_indicator_labels "Building Histogram..."
# # 	} elseif {$which == $mods(ScalarFieldStats) && $state == "Completed"} {
# # 	    change_indicate_val 2
# # 	    set has_loaded 1
# # 	    change_indicator_labels "Done Loading Volume"
# # 	} elseif {$which == $mods(NrrdInfo-Reader) && $state == "Completed"} {
# # 	    global $which-dimension

# # 	    if {[set $which-dimension] != 3} {
# # 		tk_messageBox -message "Data must be 3 dimensional scalar data." -type ok -icon info -parent .standalone
# # 		return
# # 	    }

# # 	    global $which-size0 $which-size1 $which-size2
# # 	    set size0 [set $which-size0]
# # 	    set size1 [set $which-size1]
# # 	    set size2 [set $which-size2]

# # 	    set orig_size0 $size0
# # 	    set orig_size1 $size1
# # 	    set orig_size2 $size2

# # 	    # Fix initial crop values
# # 	    if {!$loading} {
# # 		foreach i {0 1 2} {
# # 		    global $mods(UnuCrop)-minAxis$i $mods(UnuCrop)-maxAxis$i
# # 		    global $mods(ViewSlices)-crop_minAxis$i 
# # 		    global $mods(ViewSlices)-crop_maxAxis$i
# # 		    set $mods(UnuCrop)-minAxis$i 0
# # 		    set $mods(ViewSlices)-crop_minAxis$i 0
# # 		    set $mods(UnuCrop)-maxAxis$i [expr [set size$i]-1]
# # 		    set $mods(ViewSlices)-crop_maxAxis$i [expr [set size$i]-1]
# # 		}
# # 	    }

# # 	    configure_slice_sliders

# # 	    # update samples
# # 	    set path f.p.childsite.tnb.canvas.notebook.cs.page1.cs.stats.childsite
# # 	    $attachedPFr.$path.samples configure -text "Samples: $orig_size0, $orig_size1, $orig_size2"
# # 	    $detachedPFr.$path.samples configure -text "Samples: $orig_size0, $orig_size1, $orig_size2"
# # 	} elseif {$which == $mods(NrrdInfo-Size) && $state == "Completed"} {
# # 	    global $which-dimension

# # 	    global $which-size0 $which-size1 $which-size2
# # 	    set size0 [set $which-size0]
# # 	    set size1 [set $which-size1]
# # 	    set size2 [set $which-size2]

# # 	    configure_slice_sliders

# # 	    # Set Smoothing Paste and Extract values
# # 	    global axis
# # 	    global $mods(PasteImageFilter-Smooth)-size0
# # 	    global $mods(PasteImageFilter-Smooth)-size1
# # 	    global $mods(PasteImageFilter-Smooth)-size2
# # 	    global $mods(PasteImageFilter-Smooth)-index
# # 	    global $mods(PasteImageFilter-Smooth)-axis
# # 	    global $mods(PasteImageFilter-Smooth)-fill_value
# # #  	    set $mods(PasteImageFilter-Smooth)-size0 [expr $size0 - 1]
# # #  	    set $mods(PasteImageFilter-Smooth)-size1 [expr $size1 - 1]
# # #  	    set $mods(PasteImageFilter-Smooth)-size2 [expr $size2 - 1]
# #  	    set $mods(PasteImageFilter-Smooth)-size0 $size0
# #  	    set $mods(PasteImageFilter-Smooth)-size1 $size1
# #  	    set $mods(PasteImageFilter-Smooth)-size2 $size2
# # 	    set $mods(PasteImageFilter-Smooth)-index 0
# # 	    set $mods(PasteImageFilter-Smooth)-axis $axis
# # 	    set $mods(PasteImageFilter-Smooth)-fill_value 0

# # 	    global $mods(Extract-Smooth)-minDim0
# # 	    global $mods(Extract-Smooth)-minDim1
# # 	    global $mods(Extract-Smooth)-minDim2
# # 	    global $mods(Extract-Smooth)-maxDim0
# # 	    global $mods(Extract-Smooth)-maxDim1
# # 	    global $mods(Extract-Smooth)-maxDim2
	    
# # 	    set $mods(Extract-Smooth)-minDim0 0
# # 	    set $mods(Extract-Smooth)-minDim1 0
# # 	    set $mods(Extract-Smooth)-minDim2 0
# # # 	    set $mods(Extract-Smooth)-maxDim0 [expr $size0 - 1]
# # # 	    set $mods(Extract-Smooth)-maxDim1 [expr $size1 - 1]
# # # 	    set $mods(Extract-Smooth)-maxDim2 [expr $size2 - 1]
# # 	    set $mods(Extract-Smooth)-maxDim0 $size0
# # 	    set $mods(Extract-Smooth)-maxDim1 $size1
# # 	    set $mods(Extract-Smooth)-maxDim2 $size2

# # 	    # Initialize segs array
# # 	    set extent 0
# # 	    set scale 0
# # 	    if {$axis == 0} {
# # 		set extent $size0
# # 		set scale $size2
# # 		set $mods(Extract-Smooth)-minDim0 0
# # 		set $mods(Extract-Smooth)-maxDim0 1
# # 	    } elseif {$axis == 1} {
# # 		set extent $size1
# # 		set scale $size1
# # 		set $mods(Extract-Smooth)-minDim1 0
# # 		set $mods(Extract-Smooth)-maxDim1 1
# # 	    } else {
# # 		set extent $size2
# # 		set scale $size0
# # 		set $mods(Extract-Smooth)-minDim2 0
# # 		set $mods(Extract-Smooth)-maxDim2 1
# # 	    }

# # 	    for {set i 0} {$i < $extent} {incr i} {
# # 		set segs($i) 0
# # 	    }

# # 	    # Re-configure slice indicator
# # 	    $this change_slice_icon 0
# # 	} elseif {$which == $mods(UnuMinMax-Reader) && $state == "Completed"} {
# # 	    global $which-min0 $which-max0
# # 	    set range_min [set $which-min0]
# # 	    set range_max [set $which-max0]

# # 	    $this configure_threshold_sliders $range_min $range_max

# # 	    # update range
# # 	    set path f.p.childsite.tnb.canvas.notebook.cs.page1.cs.stats.childsite
# # 	    $attachedPFr.$path.range configure -text "Data Range: $range_min - $range_max"
# # 	    $detachedPFr.$path.range configure -text "Data Range: $range_min - $range_max"
# # 	} elseif {$which == $mods(UnuMinMax-Size) && $state == "Completed"} {
# # 	    global $which-min0 $which-max0
# # 	    set range_min [set $which-min0]
# # 	    set range_max [set $which-max0]

# # 	    $this configure_threshold_sliders $range_min $range_max
# # 	} elseif {$which == $mods(UnuMinMax-Smoothed) && $state == "Completed"} {
# # 	    global $which-min0 $which-max0
# # 	    set range_min [set $which-min0]
# # 	    set range_max [set $which-max0]

# # #	    $this configure_threshold_sliders
# # 	} elseif {$which == $mods(UnuMinMax-Thresholds) \
# # 		      && $state == "Completed"} {
# # 	    global $which-min0 $which-max0
# # 	    $this configure_threshold_sliders [set $which-min0] [set $which-max0]
# # 	} elseif {$which == $mods(Smooth-Gradient) && $state == "JustStarted"} {
# # 	    change_indicate_val 1
# # 	    change_indicator_labels "Peforming GradientAnisotropicDiffusion Smoothing..."
# # 	} elseif {$which == $mods(Smooth-Gradient) && $state == "Completed"} { 
# # 	    change_indicate_val 2
# # 	    change_indicator_labels "Done Performing GradientAnisotropicDiffusion Smoothing"
# # 	} elseif {$which == $mods(Smooth-Curvature) && $state == "JustStarted"} {
# # 	    change_indicate_val 1
# # 	    change_indicator_labels "Peforming CurvatureAnisotropicDiffusion Smoothing..."
# # 	} elseif {$which == $mods(Smooth-Curvature) && $state == "Completed"} { 
# # 	    change_indicate_val 2
# # 	    change_indicator_labels "Done Performing CurvatureAnisotropicDiffusion Smoothing"
# # 	} elseif {$which == $mods(Smooth-Blur) && $state == "JustStarted"} {
# # 	    change_indicate_val 1
# # 	    change_indicator_labels "Peforming Gaussian Blurring..."
# # 	} elseif {$which == $mods(Smooth-Blur) && $state == "Completed"} { 
# # 	    change_indicate_val 2
# # 	    change_indicator_labels "Done Performing Gaussian Blurring"
# # 	} elseif {$which == $mods(ShowField-Seg) && $state == "JustStarted"} { 
# # 	    change_indicate_val 1
# # 	} elseif {$which == $mods(ShowField-Seg) && $state == "Completed"} {
# # 	    change_indicate_val 2
# # 	    # Turn off Current Segmentation in ViewWindow 1
# # 	    after 100 \
# # 		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_1-Transparent Faces (6)\}\" 0; $mods(Viewer)-ViewWindow_1-c redraw"
# #  	} elseif {$which == $mods(ShowField-Speed) && $state == "JustStarted"} { 
# # 	    change_indicate_val 1
# # 	} elseif {$which == $mods(ShowField-Speed) && $state == "Completed"} {
# # 	    change_indicate_val 2
# # 	    # Turn off Speed Image in ViewWindow 0
# # 	    after 100 \
# # 		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (5)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
# # 	} elseif {$which == $mods(LevelSet) && $state == "JustStarted"} { 
# # 	    if {$updating_speed == 1} {
# # 		change_indicator_labels "Updating Speed Image..."
# # 	    } elseif {$segmenting == 1} {
# # 		change_indicator_labels "Performing LevelSet Segmentation..."
# # 	    } else {
# # 		change_indicator_labels "Generating Seeds..."
# # 	    }
# # 	    change_indicate_val 1
# # 	} elseif {$which == $mods(LevelSet) && $state == "Completed"} { 
# # 	    if {$updating_speed == 1} {
# # 		change_indicator_labels "Done Updating Speed Image"
# # 	    } elseif {$segmenting == 1} {
# # 		change_indicator_labels "Done Performing LevelSet Segmentation"
# # 		set has_segmented 1
# # 	    } 
# # 	    change_indicate_val 2
# # 	    set updating_speed 0
# # 	    set segmenting 0

# # 	    after 500 "set $mods(LevelSet)-max_iterations 0"
# # 	} elseif {$which == $mods(PasteImageFilter-Binary) && $state == "Completed"} { 
# # 	    if {$pasting_binary == 1} {
# # #		disableModule $mods(Image2DTo3D-Binary) 1
# # 		disableModule $mods(PasteImageFilter-Binary) 1

# # 		global slice
# # 		set segs($slice) 2
# # 		change_slice_icon 2
# # 	    }
# # 	    set pasting_binary 0
# # 	} elseif {$which == $mods(PasteImageFilter-Float) && $state == "Completed"} { 
# # 	    if {$pasting_float == 1} {
# # #		disableModule $mods(Image2DTo3D-Float) 1
# # 		disableModule $mods(PasteImageFilter-Float) 1
# # 	    }
# # 	    set pasting_float 0
# # 	} elseif {$which == $mods(VolumeVisualizer) && $state == "JustStarted"} { 
# # 	    change_indicate_val 1
# # 	} elseif {$which == $mods(VolumeVisualizer) && $state == "Completed"} { 
# # 	    if {$volren_has_autoviewed == 0} {
# # 		set volren_has_autoviewed 1
# # 		after 500 "$mods(Viewer-Vol)-ViewWindow_0-c autoview; $mods(Viewer-Vol)-ViewWindow_0-c redraw"		
# # 	    }
# # 	    change_indicate_val 2
# # 	    change_indicator_labels "Done Updating Volume Rendering"
# # 	} elseif {$which == $mods(ImageFileWriter-Binary) && $state == "JustStarted"} { 
# # 	    change_indicate_val 1
# # 	    change_indicator_labels "Writing out binary segmentation..."
# # 	} elseif {$which == $mods(ImageFileWriter-Binary) && $state == "Completed"} { 
# # 	    change_indicate_val 2
# # 	    change_indicator_labels "Done writing out binary segmentation"

# # 	    # disable writer
# # 	    disableModule $mods(ImageFileWriter-Binary) 1
# # 	} elseif {$which == $mods(ImageFileWriter-Float) && $state == "JustStarted"} { 
# # 	    change_indicate_val 1
# # 	    change_indicator_labels "Writing out float segmentation..."
# # 	} elseif {$which == $mods(ImageFileWriter-Float) && $state == "Completed"} { 
# # 	    change_indicate_val 2
# # 	    change_indicator_labels "Done writing out float segmentation"

# # 	    # disable writer
# # 	    disableModule $mods(ImageFileWriter-Float) 1
# # 	} elseif {$which == $mods(UnuMinmax-Vol) && $state == "Completed"} { 
# # 	    global $mods(UnuJhisto-Vol)-mins $mods(UnuJhisto-Vol)-maxs
# # 	    global $mods(RescaleColorMap-Vol)-min $mods(RescaleColorMap-Vol)-max
# # 	    global $mods(NrrdSetupTexture-Vol)-minf $mods(NrrdSetupTexture-Vol)-maxf
# # 	    global $mods(UnuMinmax-Vol)-min0 $mods(UnuMinmax-Vol)-max0

# # 	    # Change UnuJhisto, RescaleColorMap, and NrrdSetupTexture values
# # 	    set min [set $mods(UnuMinmax-Vol)-min0]
# # 	    set max [set $mods(UnuMinmax-Vol)-max0]

# # 	    set ww [expr abs($max-$min)]
# # 	    set wl [expr ($min+$max)/2.0]

# # 	    set minv [expr $wl-$ww/2.0]
# # 	    set maxv [expr $wl+$ww/2.0]

# # 	    set $mods(UnuJhisto-Vol)-mins "$minv nan"
# # 	    set $mods(UnuJhisto-Vol)-maxs "$maxv nan"
# # 	    set $mods(RescaleColorMap-Vol)-min $minv
# # 	    set $mods(RescaleColorMap-Vol)-max $maxv
# # 	    set $mods(NrrdSetupTexture-Vol)-minf $minv
# # 	    set $mods(NrrdSetupTexture-Vol)-maxf $maxv


# # 	    # now enable volume rendering and execute them - also
# # 	    # re-disale ImageToNrrd module
# # 	    disableModule $mods(NrrdSetupTexture-Vol) 0
	    
# # 	    $mods(NrrdSetupTexture-Vol)-c needexecute

# # 	    after 500 "disableModule $mods(ImageToNrrd-Vol) 1"
# # 	}
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
		change_indicator_labels "Segmenting..."
		change_indicate_val 0
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
	$indicatorL0 configure -text $msg
	$indicatorL1 configure -text $msg
    }

# #     ##############################
# #     ### configure_readers
# #     ##############################
# #     # Keeps the readers in sync.  Every time a different
# #     # data tab is selected (Nrrd, Dicom, Analyze) the other
# #     # readers must be disabled to avoid errors.
# #     method configure_readers { which } {
# #         global mods
# #         global $mods(ChooseNrrd-Reader)-port-index

# # 	if {$which == "Generic"} {
# # 	    set $mods(ChooseNrrd-Reader)-port-index 0
# # 	    disableModule $mods(NrrdReader) 0
# # 	    disableModule $mods(DicomReader) 1
# # 	    disableModule $mods(AnalyzeReader) 1

# # 	    if {$initialized != 0} {
# # 		$data_tab1 view "Generic"
# # 		$data_tab2 view "Generic"
# # 		set curr_data_tab "Generic"
# # 	    }
# #         } elseif {$which == "Dicom"} {
# # 	    set $mods(ChooseNrrd-Reader)-port-index 1

# # 	    disableModule $mods(NrrdReader) 1
# # 	    disableModule $mods(DicomReader) 0
# # 	    disableModule $mods(AnalyzeReader) 1

# #             if {$initialized != 0} {
# # 		$data_tab1 view "Dicom"
# # 		$data_tab2 view "Dicom"
# # 		set curr_data_tab "Dicom"
# # 	    }
# #         } elseif {$which == "Analyze"} {
# # 	    # Analyze
# # 	    set $mods(ChooseNrrd-Reader)-port-index 2
# # 	    disableModule $mods(NrrdReader) 1
# # 	    disableModule $mods(DicomReader) 1
# # 	    disableModule $mods(AnalyzeReader) 0

# # 	    if {$initialized != 0} {
# # 		$data_tab1 view "Analyze"
# # 		$data_tab2 view "Analyze"
# # 		set curr_data_tab "Analyze"
# # 	    }
# #         } elseif {$which == "all"} {
# # 	    if {[set $mods(ChooseNrrd-Reader)-port-index] == 0} {
# # 		# nrrd
# # 		disableModule $mods(NrrdReader) 0
# # 		disableModule $mods(DicomReader) 1
# # 		disableModule $mods(AnalyzeReader) 1
# # 	    } elseif {[set $mods(ChooseNrrd-Reader)-port-index] == 1} {
# # 		# dicom
# # 		disableModule $mods(NrrdReader) 1
# # 		disableModule $mods(DicomReader) 0
# # 		disableModule $mods(AnalyzeReader) 1
# # 	    } else {
# # 		# analyze
# # 		disableModule $mods(NrrdReader) 1
# # 		disableModule $mods(DicomReader) 1
# # 		disableModule $mods(AnalyzeReader) 0
# # 	    }
# # 	}
# #     }

# #     method set_curr_data_tab {which} {
# # 	if {$initialized} {
# # 	    set curr_data_tab $which
# # 	}
# #     }

# #     method open_nrrd_reader_ui {} {
# # 	global mods
# # 	$mods(NrrdReader) initialize_ui

# # 	.ui$mods(NrrdReader).f7.execute configure -state disabled

# # 	# rebind execute command to just withdraw
# # 	upvar \#0 .ui$mods(NrrdReader) data	
# # 	set data(-command) "wm withdraw .ui$mods(NrrdReader)"
# #     }

# #     method dicom_ui { } {
# # 	global mods
# # 	$mods(DicomReader) initialize_ui

# # 	if {[winfo exists .ui$mods(DicomReader)]} {
# # 	    # disable execute button 
# # 	    .ui$mods(DicomReader).buttonPanel.btnBox.execute configure -state disabled
# # 	}
# #     }

     method analyze_ui { } {
 	global mods
 	$mods(AnalyzeNrrdReader) initialize_ui
 	if {[winfo exists .ui$mods(AnalyzeNrrdReader)]} {
 	    # disable execute button 
 	    .ui$mods(AnalyzeNrrdReader).buttonPanel.btnBox.execute configure -state disabled
 	}
     }

    method slice_reader_ui { } {
	global mods
	set win [$mods(SliceReader) choose_file]
 	if {[winfo exists $win]} {
 	    # disable execute button 
 	    $win.f7.execute configure -state disabled

	    upvar \#0 $win data
	    set data(-command) "wm withdraw  $win"
 	}
     }

# #     method load_data {} {
# # 	global mods
# # 	# execute the appropriate reader

# #         global $mods(ChooseNrrd-Reader)-port-index
# #         set port [set $mods(ChooseNrrd-Reader)-port-index]
# #         set mod ""
# #         if {$port == 0} {
# # 	    # Nrrd
# #             set mod $mods(NrrdReader)
# # 	} elseif {$port == 1} {
# # 	    # Dicom
# #             set mod $mods(DicomReader)
# # 	} else {
# # 	    # Analyze
# #             set mod $mods(AnalyzeReader)
# # 	} 

# # 	# enable next button
# # 	$attachedPFr.$next_load configure -state normal -activebackground $next_color \
# # 	    -background $next_color 
# # 	$detachedPFr.$next_load configure -state normal -activebackground $next_color \
# # 	    -background $next_color 

# # 	set has_loaded 0
# # 	set 2D_fixed 0
	
# # 	$mod-c needexecute
# #     }

# #     ############################
# #     ### update_histo_graph_callback
# #     ############################
# #     # Called when the ScalarFieldStats updates the graph
# #     # so we can update ours
# #     method update_histo_graph_callback {varname varele varop} {
# # 	global mods

# #         global $mods(ScalarFieldStats)-min $mods(ScalarFieldStats)-max

# # 	global $mods(ScalarFieldStats)-args
# #         global $mods(ScalarFieldStats)-nmin
# #         global $mods(ScalarFieldStats)-nmax

# # 	set nmin [set $mods(ScalarFieldStats)-nmin]
# # 	set nmax [set $mods(ScalarFieldStats)-nmax]
# # 	set args [set $mods(ScalarFieldStats)-args]

# # 	if {$args == "?"} {
# # 	    return
# # 	}
        
# #         # for some reason the other graph will only work if I set temp 
# #         # instead of using the $i value 
# #  	set graph $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page1.cs.stats.childsite.histo.childsite.graph

# #          if { ($nmax - $nmin) > 1000 || ($nmax - $nmin) < 1e-3 } {
# #              $graph axis configure y -logscale yes
# #          } else {
# #              $graph axis configure y -logscale no
# #          }

# #          set min [set $mods(ScalarFieldStats)-min]
# #          set max [set $mods(ScalarFieldStats)-max]
# #          set xvector {}
# #          set yvector {}
# #          set yvector [concat $yvector $args]
# #          set frac [expr double(1.0/[llength $yvector])]

# #          $graph configure -barwidth $frac
# #          $graph axis configure x -min $min -max $max \
# # 	    -subdivisions 4 -loose 1 \
# # 	    -stepsize 0

# #          for {set i 0} { $i < [llength $yvector] } {incr i} {
# #              set val [expr $min + $i*$frac*($max-$min)]
# #              lappend xvector $val
# #          }
        
# #           if { [$graph element exists data] == 1 } {
# #               $graph element delete data
# #           }

# # 	$graph element create data -label {} -xdata $xvector -ydata $yvector
# # 	$graph element configure data -fg blue

# # 	# 	## other window
# #  	set graph $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page1.cs.stats.childsite.histo.childsite.graph

# #          if { ($nmax - $nmin) > 1000 || ($nmax - $nmin) < 1e-3 } {
# #              $graph axis configure y -logscale yes
# #          } else {
# #              $graph axis configure y -logscale no
# #          }

# #          set min [set $mods(ScalarFieldStats)-min]
# #          set max [set $mods(ScalarFieldStats)-max]
# #          set xvector {}
# #          set yvector {}
# #          set yvector [concat $yvector $args]
# #          set frac [expr double(1.0/[llength $yvector])]

# #          $graph configure -barwidth $frac
# #          $graph axis configure x -min $min -max $max \
# # 	    -subdivisions 4 -loose 1 \
# # 	    -stepsize 0

# #          for {set i 0} { $i < [llength $yvector] } {incr i} {
# #              set val [expr $min + $i*$frac*($max-$min)]
# #              lappend xvector $val
# #          }
        
# #           if { [$graph element exists data] == 1 } {
# #               $graph element delete data
# #           }

# # 	$graph element create data -label {} -xdata $xvector -ydata $yvector
# # 	$graph element configure data -fg blue
# #     }
    
# #     method select_region_of_interest {} {
# # 	global mods

# # 	# change ChooseNrrd port
# # 	global $mods(ChooseNrrd-Crop)-port-index
# # 	set $mods(ChooseNrrd-Crop)-port-index 1

# # 	# This causes region radiobutton to change
# # 	global smooth_region
# # 	set smooth_region "roi"
# # 	$this change_smooth_region

# # 	# turn off crop widget
# # 	global show_roi
# # 	set show_roi 0
# # 	$this toggle_show_roi

# # 	# set ViewSlices pad values
# # 	global $mods(ViewSlices)-crop_minPadAxis0
# # 	global $mods(ViewSlices)-crop_maxPadAxis0
# # 	global $mods(ViewSlices)-crop_minPadAxis1
# # 	global $mods(ViewSlices)-crop_maxPadAxis1
# # 	global $mods(ViewSlices)-crop_minPadAxis2
# # 	global $mods(ViewSlices)-crop_maxPadAxis2
	
# # 	global $mods(UnuCrop)-minAxis0 $mods(UnuCrop)-maxAxis0
# # 	global $mods(UnuCrop)-minAxis1 $mods(UnuCrop)-maxAxis1
# # 	global $mods(UnuCrop)-minAxis2 $mods(UnuCrop)-maxAxis2

	
# # 	set $mods(ViewSlices)-crop_minPadAxis0 [set $mods(UnuCrop)-minAxis0]
# # 	set $mods(ViewSlices)-crop_maxPadAxis0 \
# # 	    [expr $orig_size0 - [set $mods(UnuCrop)-maxAxis0] - 1]
# # 	set $mods(ViewSlices)-crop_minPadAxis1 [set $mods(UnuCrop)-minAxis1]
# # 	set $mods(ViewSlices)-crop_maxPadAxis1 \
# # 	    [expr $orig_size1 - [set $mods(UnuCrop)-maxAxis1] - 1]
# # 	set $mods(ViewSlices)-crop_minPadAxis2 [set $mods(UnuCrop)-minAxis2]
# # 	set $mods(ViewSlices)-crop_maxPadAxis2 \
# # 	    [expr $orig_size2 - [set $mods(UnuCrop)-maxAxis2] - 1]

# # 	# execute UnuCrop
# # 	$mods(UnuCrop)-c needexecute
# #     }

# #     method change_axis {} {
# # 	global axis mods

# # 	# update ViewSlices orientation
# # 	global $mods(ViewSlices)-slice-viewport0-axis
	
# # 	set $mods(ViewSlices)-slice-viewport0-axis $axis

# # 	# update Viewer orientation
# # # 	global $mods(Viewer)-ViewWindow_0-pos
# # # 	global $mods(Viewer)-ViewWindow_1-pos
# # # 	set $mods(Viewer)-ViewWindow_0-pos "z1_y1"
# # # 	set $mods(Viewer)-ViewWindow_1-pos "z1_y1"
# # # 	if {$axis == 0} {
# # # 	    set $mods(Viewer)-ViewWindow_0-pos "x0_y0"
# # # 	    set $mods(Viewer)-ViewWindow_1-pos "x0_y0"
# # # 	} elseif {$axis == 1} {
# # # 	    set $mods(Viewer)-ViewWindow_0-pos "y0_x0"
# # # 	    set $mods(Viewer)-ViewWindow_1-pos "y0_x0"
# # # 	} else {
# # # 	    set $mods(Viewer)-ViewWindow_0-pos "z1_y1"
# # # 	    set $mods(Viewer)-ViewWindow_1-pos "z1_y1"
# # # 	}
	

# # 	# re-configure slice slider
# # 	$this configure_slice_sliders

# # 	# Change UnuAxdelete axis
# # 	global $mods(UnuAxdelete-Feature)-axis
# # 	global $mods(UnuAxdelete-Prev)-axis
# # 	global $mods(UnuAxdelete-Smooth)-axis
# # 	set $mods(UnuAxdelete-Feature)-axis $axis
# # 	set $mods(UnuAxdelete-Prev)-axis $axis
# # 	set $mods(UnuAxdelete-Smooth)-axis $axis

# # 	# ExtractImageFilter
# # 	if {$axis == 0} {
# # 	    set $mods(ExtractSlice)-minDim0 0
# # 	    set $mods(ExtractSlice)-maxDim0 1
# # 	} elseif {$axis == 1} {
# # 	    set $mods(ExtractSlice)-minDim1 0
# # 	    set $mods(ExtractSlice)-maxDim1 1
# # 	} else {
# # 	    set $mods(ExtractSlice)-minDim2 0
# # 	    set $mods(ExtractSlice)-maxDim2 1
# # 	}

# # 	# PasteImageFilters
# # 	global $mods(PasteImageFilter-Binary)-axis
# # 	global $mods(PasteImageFilter-Float)-axis
# # 	global $mods(PasteImageFilter-Smooth)-axis
# # 	set $mods(PasteImageFilter-Binary)-axis $axis
# # 	set $mods(PasteImageFilter-Float)-axis $axis
# # 	set $mods(PasteImageFilter-Smooth)-axis $axis

# # 	# execute needed modules
# # 	$mods(ViewSlices)-c rebind $slice_frame

# # 	after 100 "$mods(Viewer)-ViewWindow_0-c autoview; $mods(Viewer)-ViewWindow_1-c autoview; $mods(Viewer)-ViewWindow_0-c Views; $mods(Viewer)-ViewWindow_1-c Views"
# #     }

     method configure_slice_sliders { } {
 	global axis

 	set max 0
 	if {$axis == 0} {
 	    set max [expr $size0 - 1]
 	} elseif {$axis == 1} {
 	    set max [expr $size1 - 1]
 	} else {
 	    set max [expr $size2 - 1]
 	}

 	# configure slice slider in 2D viewer 
 	.standalone.viewers.topbot.pane1.childsite.lmr.pane0.childsite.modes.slider.slice.s \
 	    configure -from 0 -to $max
     }

#     method change_smooth_region {} {
# 	global smooth_region mods
	
# 	# Change ChooseNrrd to send down desired region
# 	global $mods(ChooseNrrd-Crop)-port-index
# 	if {$smooth_region == "roi"} {
# 	    set $mods(ChooseNrrd-Crop)-port-index 1
# 	} else {
# 	    set $mods(ChooseNrrd-Crop)-port-index 0
# 	}
# 	set region_changed 1
#     }

    method change_filter {w} {
 	global mods
 	set which [$w get]
	
 	# change attached/detached menu
 	$filter_menu1 select $which
 	$filter_menu2 select $which
	
	set filter_type $which

	# hide/show proper ui
	pack forget $attachedPFr.f.p.childsite.smoothing.childsite.gradient
	pack forget $detachedPFr.f.p.childsite.smoothing.childsite.gradient

	pack forget $attachedPFr.f.p.childsite.smoothing.childsite.curvature
	pack forget $detachedPFr.f.p.childsite.smoothing.childsite.curvature
	
	pack forget $attachedPFr.f.p.childsite.smoothing.childsite.gaussian
	pack forget $detachedPFr.f.p.childsite.smoothing.childsite.gaussian

	if {$which == "GradientAnisotropicDiffusion"} {
	    pack $attachedPFr.f.p.childsite.smoothing.childsite.gradient \
		-side top -anchor n -pady 4 -expand yes -fill x
	    
	    pack $detachedPFr.f.p.childsite.smoothing.childsite.gradient \
		-side top -anchor n -pady 4 -expand yes -fill x
	} elseif {$which == "CurvatureAnisotropicDiffusion"} {
	    pack $attachedPFr.f.p.childsite.smoothing.childsite.curvature \
		-side top -anchor n -pady 4 -expand yes -fill x
	    
	    pack $detachedPFr.f.p.childsite.smoothing.childsite.curvature \
		-side top -anchor n -pady 4 -expand yes -fill x
	} else {
	    pack $attachedPFr.f.p.childsite.smoothing.childsite.gaussian \
		-side top -anchor n -pady 4 -expand yes -fill x

	    pack $detachedPFr.f.p.childsite.smoothing.childsite.gaussian \
		-side top -anchor n -pady 4 -expand yes -fill x
	}

	# enable/disable proper modules
	global to_smooth
	if {$to_smooth} {
	    $this enable_proper_smooth_module
	}
    }

    method change_seed_method {w} {
 	global mods
 	set which [$w get]
	
 	# change attached/detached menu
 	$seed_menu1 select $which
 	$seed_menu2 select $which
	
	set seed_type $which

	global $mods(ChooseNrrd-Seeds)-port-index
	global $mods(ChooseNrrd-Combine)-port-index

	disableModule $mods(ImageReaderFloat2D) 1
	disableModule $mods(ImageToNrrd-Prev) 1

	# change Choose module and turn on/off seeds
	if {$seed_type == "Seed Points Only"} {
	    set $mods(ChooseNrrd-Seeds)-port-index 1
	    set $mods(ChooseNrrd-Combine)-port-index 0
	} elseif {$seed_type == "Previous Segmentation and Seed Points"} {
	    set $mods(ChooseNrrd-Seeds)-port-index 0
	    set $mods(ChooseNrrd-Combine)-port-index 1
	    disableModule $mods(ImageReaderFloat2D) 0
	    disableModule $mods(ImageToNrrd-Prev) 0
	} elseif {$seed_type == "Thresholds and Seed Points"} {
	    set $mods(ChooseNrrd-Combine)-port-index 1
	    set $mods(ChooseNrrd-Seeds)-port-index 1
	} else {
	    puts "ERROR: unsupported seed method"
	    return
	}	
    }
    
    method enable_proper_smooth_module {} {
	global mods to_smooth
	
	# Set proper index for ChooseImage-Smooth
	global $mods(ChooseImage-Smooth)-port-index

	disableModule $mods(Smooth-Gradient) 1
	disableModule $mods(Smooth-Curvature) 1
	disableModule $mods(Smooth-Gaussian) 1
	if {$filter_type == "GradientAnisotropicDiffusion"} {
	    disableModule $mods(Smooth-Gradient) 0
	    set $mods(ChooseImage-Smooth)-port-index 1
	} elseif {$filter_type == "CurvatureAnisotropicDiffusion"} {
	    disableModule $mods(Smooth-Curvature) 0
	    set $mods(ChooseImage-Smooth)-port-index 2
	} elseif {$filter_type == "Gaussian"} {
	    disableModule $mods(Smooth-Gaussian) 0
	    set $mods(ChooseImage-Smooth)-port-index 3
	} else {
	    set $mods(ChooseImage-Smooth)-port-index 0
	}

	# Fix BuildSeedVolume modules
# 	if {$pos_seeds_used == 0} {
# 	    disableModule $mods(BuildSeedVolume-PosSeeds) 1
# 	}
# 	if {$neg_seeds_used == 0} {
# 	    disableModule $mods(BuildSeedVolume-NegSeeds) 1
# 	}	
    }

    method smooth_slice { } {
	global mods
	
	# execute appropriate smooth modules and show smoothed
	global $mods(ChooseImage-ShowSlice)-port-index
	if {$filter_type == "GradientAnisotropicDiffusion"} {
	    set $mods(ChooseImage-ShowSlice)-port-index 1
	    $mods(Smooth-Gradient)-c needexecute
	} elseif {$filter_type == "CurvatureAnisotropicDiffusion"} {
	    set $mods(ChooseImage-ShowSlice)-port-index 1
	    $mods(Smooth-Curvature)-c needexecute
	} elseif {$filter_type == "Gaussian"} {
	    set $mods(ChooseImage-ShowSlice)-port-index 1
	    $mods(Smooth-Gaussian)-c needexecute
	}
    }
    
#     method smooth_data {type} {
# 	global mods
    
# 	global $mods(ChooseImage-SmoothInput)-port-index
# 	if {$type == "Go" && $has_smoothed == 1} {
# 	    set $mods(ChooseImage-SmoothInput)-port-index 1
# 	    set type "Go"
# 	    set smoothing_type "Go"
# 	} else {
# 	    set $mods(ChooseImage-SmoothInput)-port-index 0
# 	    set type "Reset"
# 	    set smoothing_type "Reset"
# 	}

# 	# Reset Paste and Extract values
# 	global axis

# 	global $mods(PasteImageFilter-Smooth)-size0
# 	global $mods(PasteImageFilter-Smooth)-size1
# 	global $mods(PasteImageFilter-Smooth)-size2
# 	global $mods(PasteImageFilter-Smooth)-index
# 	global $mods(PasteImageFilter-Smooth)-axis
# 	global $mods(PasteImageFilter-Smooth)-fill_value
# # 	set $mods(PasteImageFilter-Smooth)-size0 [expr $size0 - 1]
# # 	set $mods(PasteImageFilter-Smooth)-size1 [expr $size1 - 1]
# # 	set $mods(PasteImageFilter-Smooth)-size2 [expr $size2 - 1]
# 	set $mods(PasteImageFilter-Smooth)-size0 $size0
# 	set $mods(PasteImageFilter-Smooth)-size1 $size1
# 	set $mods(PasteImageFilter-Smooth)-size2 $size2
# 	set $mods(PasteImageFilter-Smooth)-index 0
# 	set $mods(PasteImageFilter-Smooth)-axis $axis
# 	set $mods(PasteImageFilter-Smooth)-fill_value 0

# 	global $mods(Extract-Smooth)-minDim0
# 	global $mods(Extract-Smooth)-minDim1
# 	global $mods(Extract-Smooth)-minDim2
# 	global $mods(Extract-Smooth)-maxDim0
# 	global $mods(Extract-Smooth)-maxDim1
# 	global $mods(Extract-Smooth)-maxDim2
	
# 	set $mods(Extract-Smooth)-minDim0 0
# 	set $mods(Extract-Smooth)-minDim1 0
# 	set $mods(Extract-Smooth)-minDim2 0
# # 	set $mods(Extract-Smooth)-maxDim0 [expr $size0 - 1]
# # 	set $mods(Extract-Smooth)-maxDim1 [expr $size1 - 1]
# # 	set $mods(Extract-Smooth)-maxDim2 [expr $size2 - 1]
# 	set $mods(Extract-Smooth)-maxDim0 $size0
# 	set $mods(Extract-Smooth)-maxDim1 $size1
# 	set $mods(Extract-Smooth)-maxDim2 $size2

# 	if {$axis == 0} {
# 	    set $mods(Extract-Smooth)-minDim0 0
# 	    set $mods(Extract-Smooth)-maxDim0 1
# 	} elseif {$axis == 1} {
# 	    set $mods(Extract-Smooth)-minDim1 0
# 	    set $mods(Extract-Smooth)-maxDim1 1
# 	} else {
# 	    set $mods(Extract-Smooth)-minDim2 0
# 	    set $mods(Extract-Smooth)-maxDim2 1
# 	}

# 	# Make sure a filter has been selected
# 	if {[string equal $smoothing_method "None"] == 0} {

# 	    if {$type == "Reset"} {
# 		# Enable extract and paste module
# 		disableModule $mods(Extract-Smooth) 0
# 		disableModule $mods(PasteImageFilter-Smooth) 0
		
# 		# disable downstream modules so loop of smoothing
# 		# can execute
# 		disableModule $mods(ChooseImage-ToSmooth) 1
# 		disableModule $mods(ImageToNrrd-ViewSlices) 0
		
# 		# execute
# 		set smoothing 1
# 		disableModule $mods(ChooseImage-SmoothInput) 0
		
# 		after 500 "$mods(ChooseImage-SmoothInput)-c needexecute"
# 	    } else {
# 		disableModule $mods(PasteImageFilter-Smooth) 0

# 		# We have to get previously blurred data to
# 		# Choose module before enabling downstream modules
# 		disableModule $mods(ChooseImage-SmoothInput) 0
# 		disableModule $mods(Extract-Smooth) 1

# 		set smoothing 1
		
# 		# Execute choose, when it finishes, we will disable it
# 		# and then enable downstream modules and execute them
# 		after 500 "$mods(ChooseImage-SmoothInput)-c needexecute"
# 	    }
# 	}
#     }

    method update_speed_image {} {
 	global mods
	
 	# set number of iterations 0
 	global $mods(LevelSet)-max_iterations
 	set $mods(LevelSet)-max_iterations 0

	# Turn on speed image
	global $mods(ShowField-Speed)-faces-on
	set $mods(ShowField-Speed)-faces-on 1
	
	if {$reverse_changed == 1} {
	    $mods(GenStandardColorMaps-Speed)-c needexecute
	}
	
 	# execute ThresholdLevelSet filter
# 	$mods(LevelSet)-c needexecute
	$mods(ChooseImage-Smooth)-c needexecute
    }

#     method configure_threshold_sliders {r_min r_max} {
# 	# configure threshold range sliders
# 	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.params.childsite.lthresh.s \
# 	    configure -from $r_min -to $r_max
# 	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.params.childsite.lthresh.s \
# 	    configure -from $r_min -to $r_max

# 	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.params.childsite.uthresh.s \
# 	    configure -from $r_min -to $r_max
# 	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.params.childsite.uthresh.s \
# 	    configure -from $r_min -to $r_max
#     }

#     method initialize_segmentation {} {
# 	if {!$segmentation_initialized} {
# 	    global mods slice

# 	    disableModule $mods(ChooseImage-ToSmooth) 0
	    
# 	    # Set segmenting slice to 0
# 	    set slice 0
	    
# 	    # Set threshold values to reasonable defaults (and other Level Set vals)
# 	    global $mods(LevelSet)-lower_threshold
# 	    global $mods(LevelSet)-upper_threshold
# 	    global $mods(LevelSet)-curvature_scaling
# 	    global $mods(LevelSet)-propagation_scaling
# 	    global $mods(LevelSet)-edge_weight
# 	    global $mods(LevelSet)-max_iterations
# 	    global $mods(LevelSet)-max_rms_change
# 	    global $mods(LevelSet)-reverse_expansion_direction
# 	    global $mods(LevelSet)-smoothing_iterations
# 	    global $mods(LevelSet)-smoothing_time_step
# 	    global $mods(LevelSet)-smoothing_conductance
# 	    global $mods(LevelSet)-update_OutputImage
# 	    global $mods(LevelSet)-update_iters_OutputImage

# 	    set $mods(LevelSet)-update_OutputImage 1
# 	    set $mods(LevelSet)-update_iters_OutputImage 2

# 	    set range [expr $range_max - $range_min]
# 	    set $mods(LevelSet)-lower_threshold \
# 		[expr int([expr $range_min + [expr $range/3]])]
# 	    set $mods(LevelSet)-upper_threshold \
# 		[expr int([expr $range_max - [expr $range/3]])]
# 	    set $mods(LevelSet)-curvature_scaling 1.0
# 	    set $mods(LevelSet)-propagation_scaling 1.0
# 	    set $mods(LevelSet)-edge_weight 1.0
# 	    set $mods(LevelSet)-reverse_expansion_direction 0
# 	    set $mods(LevelSet)-smoothing_iterations 0
# 	    set $mods(LevelSet)-max_rms_change 0.02
	    
# 	    # set LevelSet iterations to 0
# 	    set $mods(LevelSet)-max_iterations 0
	    
# 	    # configure ExtractImageFilter bounds and enable it
# 	    global $mods(ExtractSlice)-minDim0
# 	    global $mods(ExtractSlice)-minDim1
# 	    global $mods(ExtractSlice)-minDim2
# 	    global $mods(ExtractSlice)-maxDim0
# 	    global $mods(ExtractSlice)-maxDim1
# 	    global $mods(ExtractSlice)-maxDim2
	    
# 	    set $mods(ExtractSlice)-minDim0 0
# 	    set $mods(ExtractSlice)-minDim1 0
# 	    set $mods(ExtractSlice)-minDim2 0
# # 	    set $mods(ExtractSlice)-maxDim0 [expr $size0 - 1]
# # 	    set $mods(ExtractSlice)-maxDim1 [expr $size1 - 1]
# # 	    set $mods(ExtractSlice)-maxDim2 [expr $size2 - 1]
# 	    set $mods(ExtractSlice)-maxDim0 $size0
# 	    set $mods(ExtractSlice)-maxDim1 $size1
# 	    set $mods(ExtractSlice)-maxDim2 $size2

# 	    global axis
# 	    if {$axis == 0} {
# 		set $mods(ExtractSlice)-minDim0 0
# 		set $mods(ExtractSlice)-maxDim0 1
# 	    } elseif {$axis == 1} {
# 		set $mods(ExtractSlice)-minDim1 0
# 		set $mods(ExtractSlice)-maxDim1 1
# 	    } else {
# 		set $mods(ExtractSlice)-minDim2 0
# 		set $mods(ExtractSlice)-maxDim2 1
# 	    }

# 	    # configure binary/float PasteImageFilter
# 	    global $mods(PasteImageFilter-Binary)-size0
# 	    global $mods(PasteImageFilter-Binary)-size1
# 	    global $mods(PasteImageFilter-Binary)-size2
# 	    global $mods(PasteImageFilter-Binary)-index
# 	    global $mods(PasteImageFilter-Binary)-axis
# 	    global $mods(PasteImageFilter-Binary)-fill_value
# #  	    set $mods(PasteImageFilter-Binary)-size0 [expr $size0 - 1]
# #  	    set $mods(PasteImageFilter-Binary)-size1 [expr $size1 - 1]
# #  	    set $mods(PasteImageFilter-Binary)-size2 [expr $size2 - 1]
#  	    set $mods(PasteImageFilter-Binary)-size0 $size0
#  	    set $mods(PasteImageFilter-Binary)-size1 $size1
#  	    set $mods(PasteImageFilter-Binary)-size2 $size2
# 	    set $mods(PasteImageFilter-Binary)-index 0
# 	    set $mods(PasteImageFilter-Binary)-axis $axis
# 	    set $mods(PasteImageFilter-Binary)-fill_value 0

# 	    global $mods(PasteImageFilter-Float)-size0
# 	    global $mods(PasteImageFilter-Float)-size1
# 	    global $mods(PasteImageFilter-Float)-size2
# 	    global $mods(PasteImageFilter-Float)-index
# 	    global $mods(PasteImageFilter-Float)-axis
# 	    global $mods(PasteImageFilter-Float)-fill_value
# #  	    set $mods(PasteImageFilter-Float)-size0 [expr $size0 - 1]
# #  	    set $mods(PasteImageFilter-Float)-size1 [expr $size1 - 1]
# #  	    set $mods(PasteImageFilter-Float)-size2 [expr $size2 - 1]
#  	    set $mods(PasteImageFilter-Float)-size0 $size0
#  	    set $mods(PasteImageFilter-Float)-size1 $size1
#  	    set $mods(PasteImageFilter-Float)-size2 $size2
# 	    set $mods(PasteImageFilter-Float)-index 0
# 	    set $mods(PasteImageFilter-Float)-axis $axis
# 	    set $mods(PasteImageFilter-Float)-fill_value 0

# 	    # enable segmentation modules
# 	    disableModule $mods(LevelSet) 0
# 	    disableModule $mods(BinaryThreshold-Slice) 0
# 	    disableModule $mods(ExtractSlice) 0
# 	    disableModule $mods(ImageToField-Feature) 0
# 	    disableModule $mods(NrrdToImage-Feature) 0
# 	    disableModule $mods(ExtractSlice) 0

# 	    # disable downstream modules
# 	    disableModule $mods(PasteImageFilter-Binary) 1
# 	    disableModule $mods(PasteImageFilter-Float) 1

# 	    disableModule $mods(ImageToNrrd-Vol) 1

# 	    # execute feature image stuff
# 	    $mods(SampleField-Seeds) send
# 	    $mods(SampleField-SeedsNeg) send

# 	    set segmentation_initialized 1

# 	    # turn off speed, seed, and segmentation in Viewer 0
# 	    after 100 \
# 		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (4)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (5)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (6)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
# 	}
#     }

#     method update_seed_binary_threshold {var1 var2 var3} {
# 	global mods
# 	global $mods(LevelSet)-lower_threshold
# 	global $mods(LevelSet)-upper_threshold

# 	# update thresholds of BinaryThresholdImageFilter that
# 	# feeds in when user selects thresholding seeding method
# 	global $mods(BinaryThreshold-Seeds)-lower_threshold
# 	global $mods(BinaryThreshold-Seeds)-upper_threshold
# 	set $mods(BinaryThreshold-Seeds)-lower_threshold [set $mods(LevelSet)-lower_threshold]
# 	set $mods(BinaryThreshold-Seeds)-upper_threshold [set $mods(LevelSet)-upper_threshold]
#     }

#     method change_seed_method {} {
# 	global mods seed_method 
# 	global $mods(ChooseNrrd-Seeds)-port-index
# 	global $mods(ChooseNrrd-Combine)-port-index

# 	# change Choose module and turn on/off seeds
# 	if {$seed_method == "points"} {
# 	    set $mods(ChooseNrrd-Seeds)-port-index 0
# 	} elseif {$seed_method == "prev" || $seed_method == "next" || \
# 		  $seed_method == "curr"} {
# 	    set $mods(ChooseNrrd-Combine)-port-index 0
# 	    set $mods(ChooseNrrd-Seeds)-port-index 1
# 	} elseif {$seed_method == "thresh"} {
# 	    set $mods(ChooseNrrd-Combine)-port-index 1
# 	    set $mods(ChooseNrrd-Seeds)-port-index 1
# 	} else {
# 	    puts "ERROR: unsupported seed method"
# 	    return
# 	}
#     }

     method change_number_of_seeds {type dir} {
 	global mods

 	if {$type == "+"} {
 	    # Positive Seeds
 	    global $mods(SeedPoints-PosSeeds)-num_seeds
 	    if {$dir == "+"} {
 		set $mods(SeedPoints-PosSeeds)-num_seeds \
		    [expr [set $mods(SeedPoints-PosSeeds)-num_seeds] + 1]
 	    } elseif {$dir == "-"} {
 		set $mods(SeedPoints-PosSeeds)-num_seeds \
		    [expr [set $mods(SeedPoints-PosSeeds)-num_seeds] - 1]
 	    } 
 	    if {[set $mods(SeedPoints-PosSeeds)-num_seeds] < 0} {
 		set $mods(SeedPoints-PosSeeds)-num_seeds 0
 	    }
 	    $mods(SeedPoints-PosSeeds) send
 	} else {
 	    # Negative Seeds
 	    global $mods(SeedPoints-NegSeeds)-num_seeds
  	    if {$dir == "+"} {
 		set $mods(SeedPoints-NegSeeds)-num_seeds \
		    [expr [set $mods(SeedPoints-NegSeeds)-num_seeds] + 1]
 	    } elseif {$dir == "-"} {
 		set $mods(SeedPoints-NegSeeds)-num_seeds \
		    [expr [set $mods(SeedPoints-NegSeeds)-num_seeds] - 1]
 	    } 
 	    if {[set $mods(SeedPoints-NegSeeds)-num_seeds] < 0} {
 		set $mods(SeedPoints-NegSeeds)-num_seeds 0
 	    }
 	    $mods(SeedPoints-NegSeeds) send
 	}
     }
    
    method create_seeds {} {
	global seed_method mods

	# Create initial segmentation and display it
	# by executing the appropriate set of modules

 	# Turn on seed in top viewer and segmentation and speed off
	global $mods(ShowField-Seg)-faces-on
	set $mods(ShowField-Seg)-faces-on 0
	$mods(ShowField-Seg)-c toggle_display_faces

	global $mods(ShowField-Speed)-faces-on
	set $mods(ShowField-Speed)-faces-on 0

	global $mods(ShowField-Seed)-faces-on
	set $mods(ShowField-Seed)-faces-on 1
	$mods(ShowField-Seed)-c toggle_display_faces

 	# Turn seed point widgets back on
 	global show_seeds
 	set show_seeds 1
 	$this seeds_changed 1 2 3

        global $mods(SeedPoints-PosSeeds)-send
        global $mods(SeedPoints-NegSeeds)-send
        set $mods(SeedPoints-PosSeeds)-send 1
        set $mods(SeedPoints-NegSeeds)-send 1
# 	if {$seed_type == "Seed Points Only"} {
# 	    $mods(SeedPoints-PosSeeds) send
# 	    $mods(SeedPoints-NegSeeds) send
# 	} elseif {$seed_type == "Previous Segmentation and Seed Points"} {
#  	    $mods(SeedPoints-PosSeeds) send
#  	    $mods(SeedPoints-NegSeeds) send
# 	    $this check_previous_filename
# 	    $mods(ImageReaderFloat2D)-c needexecute
#  	} elseif {$seed_type == "Thresholds and Seed Points"} {
#  	    $mods(SeedPoints-PosSeeds) send
#  	    $mods(SeedPoints-NegSeeds) send
#  	    $mods(BinaryThreshold-Seed)-c needexecute
#  	} else {
#  	    puts "ERROR: cannot create seeds"
#  	    return
#  	}
 	
	if {$seed_type == "Previous Segmentation and Seed Points"} {
	    $this check_previous_filename
	    $mods(ImageReaderFloat2D)-c needexecute
	}
	$mods(ChooseImage-Smooth)-c needexecute
    }

    method start_segmentation {} {
 	global max_iter mods
	
 	global $mods(LevelSet)-max_iterations
 	# set level set max iterations to be what 
 	# global max_iter is and then after execute
 	# set it back to 0 so user can update speed image
 	set $mods(LevelSet)-max_iterations $max_iter
	
	global $mods(ShowField-Seed)-faces-on
	set $mods(ShowField-Seed)-faces-on 0
	$mods(ShowField-Seed)-c toggle_display_faces

	global $mods(ShowField-Seg)-faces-on
	set $mods(ShowField-Seg)-faces-on 1
	$mods(ShowField-Seg)-c toggle_display_faces

 	# Turn seeds off
 	global show_seeds
 	set show_seeds 0
 	$this seeds_changed 1 2 3 
	
 	set segmenting 1

	# execute Level Set
	$mods(LevelSet)-c needexecute
     }

     method stop_segmentation {} {
 	global mods
 	$mods(LevelSet) stop_segmentation
     }

    method commit_segmentation {} {
  	global mods commit_dir base_filename

	# Set up writer filename
	global $mods(ImageFileWriter-Binary)-filename
	global $mods(SliceReader)-slice
#	upvar \#0 $mods(SliceReader)-slice s
	set s [set $mods(SliceReader)-slice]
	set $mods(ImageFileWriter-Binary)-filename \
	    [file join $commit_dir $base_filename$s.hdr]


  	# enable writing module and 3D vis
	disableModule $mods(ImageFileWriter-Binary) 0
	disableModule $mods(ImageToField-Iso) 0

  	# execute
	set committing 1
	$mods(ImageFileWriter-Binary)-c needexecute
	$mods(ImageToField-Iso)-c needexecute

	set has_committed 1
     }

#     method current_slice_changed {} {
# 	global slice mods axis
# 	# slice value changed via spinner so update

# 	# Set ChooseImage-SegInput back to not use a this
# 	# slice's previous segmentation since we are one a
# 	# new slice
# 	global $mods(ChooseImage-SegInput)-port-index
# 	set $mods(ChooseImage-SegInput)-port-index 0
	
# 	# extract modules 
# 	if {$axis == 0} {
# 	    global $mods(ExtractSlice)-minDim0
# 	    global $mods(ExtractSlice)-maxDim0
# 	    set $mods(ExtractSlice)-minDim0 $slice
# 	    set $mods(ExtractSlice)-maxDim0 [expr $slice + 1]
# 	} elseif {$axis == 1} {
# 	    global $mods(ExtractSlice)-minDim1
# 	    global $mods(ExtractSlice)-maxDim1
# 	    set $mods(ExtractSlice)-minDim1 $slice
# 	    set $mods(ExtractSlice)-maxDim1 [expr $slice + 1]
# 	} else {
# 	    global $mods(ExtractSlice)-minDim2
# 	    global $mods(ExtractSlice)-maxDim2
# 	    set $mods(ExtractSlice)-minDim2 $slice
# 	    set $mods(ExtractSlice)-maxDim2 [expr $slice + 1]
# 	}

# 	# top window should just show original data
# 	after 100 \
# 	    "uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (4)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (5)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (6)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	
# 	# update speed image

# 	$mods(ExtractSlice)-c needexecute

# 	# update slice window
# 	global $mods(ViewSlices)-slice-viewport0-slice
# 	set $mods(ViewSlices)-slice-viewport0-slice $slice
# 	$mods(ViewSlices)-c rebind .standalone.viewers.topbot.pane1.childsite.lmr.pane0.childsite.slice
# 	$mods(ViewSlices)-c redrawall

# 	# udpate indicator
# 	$this change_slice_icon $segs($slice)

# 	set has_segmented 0
#     }

#     method update_volume_rendering {} {
# 	global mods

# 	# enable ImageToField and execute to get min/max for volume rendering modules.
# 	# when the unuminmax module completes it will execute the rest of them
# 	disableModule $mods(ImageToNrrd-Vol) 0

# 	disableModule $mods(NrrdSetupTexture-Vol) 1

# 	$mods(ImageToNrrd-Vol)-c needexecute

# 	# disable update volume rendering button
# 	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.volren \
# 	    configure -background "grey75"\
# 	    -activebackground "grey75" -state disabled
# 	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.volren \
# 	    configure -background "grey75" \
# 	    -activebackground "grey75" -state disabled
#     }

#     method start_crop {} {
#         global mods show_roi
# 	global $mods(ViewSlices)-crop
	
# 	if {$show_roi == 1} {
# 	    $mods(ViewSlices)-c startcrop
# 	}
#     }

#     method update_crop_values { varname varele varop } {
# 	global mods 

# 	if {$updating_crop_widget == 1} {
# 	    return
# 	}

# 	if {[string first "crop_minAxis0" $varname] != -1} {
# 	    global $mods(UnuCrop)-minAxis0
# 	    global $mods(ViewSlices)-crop_minAxis0
# 	    global $mods(ViewSlices)-crop_minPadAxis0
# 	    set $mods(UnuCrop)-minAxis0 \
# 		[expr [set $mods(ViewSlices)-crop_minAxis0] + \
# 		     [set $mods(ViewSlices)-crop_minPadAxis0]]
# 	} elseif {[string first "crop_maxAxis0" $varname] != -1} {
# 	    global $mods(UnuCrop)-maxAxis0
# 	    global $mods(ViewSlices)-crop_maxAxis0
# 	    set $mods(UnuCrop)-maxAxis0 [set $mods(ViewSlices)-crop_maxAxis0]
# 	} elseif {[string first "crop_minAxis1" $varname] != -1} {
# 	    global $mods(UnuCrop)-minAxis1
# 	    global $mods(ViewSlices)-crop_minAxis1
# 	    global $mods(ViewSlices)-crop_minPadAxis1
# 	    set $mods(UnuCrop)-minAxis1 \
# 		[expr [set $mods(ViewSlices)-crop_minAxis1] + \
# 		     [set $mods(ViewSlices)-crop_minPadAxis1]]
# 	} elseif {[string first "crop_maxAxis1" $varname] != -1} {
# 	    global $mods(UnuCrop)-maxAxis1
# 	    global $mods(ViewSlices)-crop_maxAxis1
# 	    set $mods(UnuCrop)-maxAxis1 [set $mods(ViewSlices)-crop_maxAxis1]
# 	} elseif {[string first "crop_minAxis2" $varname] != -1} {
# 	    global $mods(UnuCrop)-minAxis2
# 	    global $mods(ViewSlices)-crop_minAxis2
# 	    global $mods(ViewSlices)-crop_minPadAxis2
# 	    set $mods(UnuCrop)-minAxis2 \
# 		[expr [set $mods(ViewSlices)-crop_minAxis2] + \
# 		     [set $mods(ViewSlices)-crop_minPadAxis2]]
# 	} elseif {[string first "crop_maxAxis2" $varname] != -1} {
# 	    global $mods(UnuCrop)-maxAxis2
# 	    global $mods(ViewSlices)-crop_maxAxis2
# 	    set $mods(UnuCrop)-maxAxis2 [set $mods(ViewSlices)-crop_maxAxis2]
# 	}
#     }

#     method update_crop_widget {type i} {
# 	global mods

# 	# get values from UnuCrop, then
# 	# set ViewSlices crop values
#         if {$type == "min"} {
#     	    global $mods(UnuCrop)-minAxis$i $mods(ViewSlices)-crop_minAxis$i
#             set min [set $mods(UnuCrop)-minAxis$i]
#             set $mods(ViewSlices)-crop_minAxis$i $min           
#         } else {
#     	    global $mods(UnuCrop)-maxAxis$i $mods(ViewSlices)-crop_maxAxis$i
#             set max [set $mods(UnuCrop)-maxAxis$i]
#             set $mods(ViewSlices)-crop_maxAxis$i $max 
#         }

# 	global $mods(ViewSlices)-crop
# 	if {[set $mods(ViewSlices)-crop] == 1} {
# 	    $mods(ViewSlices)-c updatecrop
# 	}
#     }

#     method save_binary {} {
# 	global mods
# 	global $mods(ImageFileWriter-Binary)-filename
	
# 	# enable writer, open ui
# 	disableModule $mods(ImageFileWriter-Binary) 0

# 	if {[set $mods(ImageFileWriter-Binary)-filename] != ""} {
# 	    $mods(ImageFileWriter-Binary)-c needexecute
# 	} else {
# 	    $mods(ImageFileWriter-Binary) initialize_ui

# 	    # Disable execute behavior
# 	    set m $mods(ImageFileWriter-Binary)
# 	    .ui$m.f7.execute configure -state disabled
	    
# 	    upvar \#0 .ui$m data	
# 	    set data(-command) "wm withdraw .ui$m"
# 	}
#     }

#     method open_save_binary_ui {} {
# 	global mods

# 	# enable writer, open ui
# 	disableModule $mods(ImageFileWriter-Binary) 0
# 	$mods(ImageFileWriter-Binary) initialize_ui

# 	# Disable execute behavior
# 	set m $mods(ImageFileWriter-Binary)
# 	.ui$m.f7.execute configure -state disabled
	
# 	upvar \#0 .ui$m data	
# 	set data(-command) "wm withdraw .ui$m"
#     }

#     method save_float {} {
# 	global mods
# 	global $mods(ImageFileWriter-Float)-filename

# 	# enable writer, open ui
# 	disableModule $mods(ImageFileWriter-Float) 0

# 	if {[set $mods(ImageFileWriter-Float)-filename] != ""} {
# 	    $mods(ImageFileWriter-Float)-c needexecute
# 	} else {
# 	    $mods(ImageFileWriter-Float) initialize_ui

# 	    # Disable execute behavior
# 	    set m $mods(ImageFileWriter-Float)
# 	    .ui$m.f7.execute configure -state disabled
	    
# 	    upvar \#0 .ui$m data	
# 	    set data(-command) "wm withdraw .ui$m"
# 	}
#     }

#     method open_save_float_ui {} {
# 	global mods

# 	# enable writer, open ui
# 	disableModule $mods(ImageFileWriter-Float) 0
# 	$mods(ImageFileWriter-Float) initialize_ui

# 	# Disable execute behavior
# 	set m $mods(ImageFileWriter-Float)
# 	.ui$m.f7.execute configure -state disabled
	
# 	upvar \#0 .ui$m data	
# 	set data(-command) "wm withdraw .ui$m"
#     }

     method seeds_changed {a b c} {
	 global mods show_seeds
	 
	 set val 0
	 if {$show_seeds == 1} {
	     set val 1
	 }

	 # Turn ViewWindow_0 seeds on/off depending on val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint0 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint1 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint2 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint3 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint4 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint5 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint6 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint7 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint8 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint9 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint10 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint11 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint12 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint13 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint14 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint15 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint16 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint17 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint18 (3)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint19 (3)} $val
	 
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint0 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint1 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint2 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint3 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint4 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint5 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint6 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint7 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint8 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint9 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint10 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint11 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint12 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint13 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint14 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint15 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint16 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint17 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint18 (4)} $val
	 setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint19 (4)} $val
	 
	 $mods(Viewer)-ViewWindow_0-c redraw
	 
	 #	for {set i 0} {$i < [set $mods(SeedPoints-PosSeeds)-num_seeds]} {incr i} {
	 #	    setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint (5)} $val
	 #	    upvar \#0 {$mods(Viewer)-ViewWindow_1-SeedPoint$i (5)} point
	 #	    puts [set point]
	 #	}
	
     }

#     method change_slice_icon {state} {
# 	set icon ""
# 	if {$state == 0} {
# 	    global no_seg_icon
# 	    set icon $no_seg_icon
# 	} elseif {$state == 1} {
# 	    global old_seg_icon
# 	    set icon $old_seg_icon
# 	} else {
# 	    global updated_seg_icon
# 	    set icon $updated_seg_icon
# 	}

# 	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.slice.status configure -image $icon
# 	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.slice.status configure -image $icon
	

#     }

#     method toggle_show_roi {} {
# 	global mods show_roi
# 	if {$show_roi == 0} {
# 	    $mods(ViewSlices)-c stopcrop
# 	} else {
# 	    set updating_crop_widget 1

# 	    $this start_crop

# 	    # Make sure widget is over volume
# 	    global $mods(ViewSlices)-dim0 $mods(ViewSlices)-dim1
# 	    global $mods(ViewSlices)-dim2
# 	    upvar \#0 $mods(ViewSlices)-dim0 dim0
# 	    upvar \#0 $mods(ViewSlices)-dim1 dim1
# 	    upvar \#0 $mods(ViewSlices)-dim2 dim2

# 	    global $mods(ViewSlices)-crop_minAxis0 
# 	    global $mods(ViewSlices)-crop_maxAxis0
# 	    global $mods(ViewSlices)-crop_minAxis1 
# 	    global $mods(ViewSlices)-crop_maxAxis1
# 	    global $mods(ViewSlices)-crop_minAxis2 
# 	    global $mods(ViewSlices)-crop_maxAxis2
	    
# 	    set $mods(ViewSlices)-crop_minAxis0 0
# 	    set $mods(ViewSlices)-crop_minAxis1 0
# 	    set $mods(ViewSlices)-crop_minAxis2 0

# 	    set $mods(ViewSlices)-crop_maxAxis0 $dim0
# 	    set $mods(ViewSlices)-crop_maxAxis1 $dim1
# 	    set $mods(ViewSlices)-crop_maxAxis2 $dim2

# 	    $mods(ViewSlices)-c updatecrop
# 	    set updating_crop_widget 0
# 	} 
#     }

#     method toggle_volume_render_object {} {
# 	global mods vol_foreground
# 	global $mods(EditColorMap2D)-on-0
# 	global $mods(EditColorMap2D)-on-1
# 	if {$vol_foreground == 1} {
# 	    set $mods(EditColorMap2D)-on-0 1
# 	    set $mods(EditColorMap2D)-on-1 0
# 	} else {
# 	    set $mods(EditColorMap2D)-on-0 0
# 	    set $mods(EditColorMap2D)-on-1 1
# 	}
# 	$mods(EditColorMap2D)-c toggle 0
# 	$mods(EditColorMap2D)-c toggle 1
#     }

#     method update_ViewSlices_input {} {
# 	global mods
	
# 	# execute ChooseNrrd-2D to send the new 
# 	# input to ViewSlices
# 	$mods(ChooseNrrd-2D)-c needexecute

# 	# reconfigure windows width/level?
#     }

    method go_lowres {} {
	global mods
	global $mods(AnalyzeNrrdReader)-num-files
	global $mods(AnalyzeNrrdReader)-filenames0

	# check for a valid file
	if {[set $mods(AnalyzeNrrdReader)-num-files] > 0 && \
		[file exists [set $mods(AnalyzeNrrdReader)-filenames0]]} {
	    $mods(AnalyzeNrrdReader)-c needexecute
	} else {
	    tk_messageBox -type ok -icon info -parent .standalone \
		-message "Invalid filename specified.\nPlease select a valid filename\nand click the Go button." 
	    return    
	}
    }

    method go_highres {} {
	global mods

	
	# check for a valid file
	global $mods(SliceReader)-filename
	if {![file exists [set $mods(SliceReader)-filename]]} {
	    tk_messageBox -type ok -icon info -parent .standalone \
		-message "Invalid filename specified.\nPlease select a valid filename\nand click the Go button." 
	    return    
	}

	# Turn off segmentation
	global $mods(ShowField-Seg)-faces-on
	set $mods(ShowField-Seg)-faces-on 0

	# Turn on current seed if has_loaded
# 	if {$has_loaded == 1} {
# 	    global $mods(ShowField-Seed)-faces-on
# 	    set $mods(ShowField-Seed)-faces-on 1
# 	}

	# Prepare previous reader filename
	global commit_dir base_filename
	global $mods(SliceReader)-slice
#	upvar \#0 $mods(SliceReader)-slice slice
	
	set slice [set $mods(SliceReader)-slice]
	global $mods(ImageReaderFloat2D)-filename
	set $mods(ImageReaderFloat2D)-filename \
	    [file join $commit_dir $base_filename[expr $slice - 1].hdr]

	# Prepare ChangeFieldBounds center
	global axis
	if {$axis == 0} {
	    global $mods(ChangeFieldBounds)-outputcenterx
	    set $mods(ChangeFieldBounds)-outputcenterx $slice
	} elseif {$axis == 1} {
	    global $mods(ChangeFieldBounds)-outputcentery
	    set $mods(ChangeFieldBounds)-outputcentery $slice
	} else {
	    global $mods(ChangeFieldBounds)-outputcenterz
	    set $mods(ChangeFieldBounds)-outputcenterz $slice
	}


	# enable/disable using previous slice option
	set prev_avail 0
	if {$slice > 0 && [info exists commits([expr $slice - 1])] && \
		$commits([expr $slice - 1]) == 1} {
	    $attachedPFr.f.p.childsite.seeds.childsite.method enable \
		"Previous Segmentation and Seed Points"
	    $detachedPFr.f.p.childsite.seeds.childsite.method enable \
		"Previous Segmentation and Seed Points"
	    set prev_avail 1
	} else {
	    $attachedPFr.f.p.childsite.seeds.childsite.method disable \
		"Previous Segmentation and Seed Points"
	    $detachedPFr.f.p.childsite.seeds.childsite.method disable \
		"Previous Segmentation and Seed Points"
	}

	

	$mods(SliceReader)-c needexecute

	# if previous slice option available and being used,
	# and there is a previous slice commited, execute
	# ImageReaderFloat2D
	if {$prev_avail == 1 && \
		$seed_type == "Previous Segmentation and Seed Points" && \
		$slice > 0 && $commits([expr $slice - 1]) == 1} {
	    $mods(ImageReaderFloat2D)-c needexecute
	}

	set has_loaded 1
    }

    method next_highres {} {
	global mods
	global $mods(SliceReader)-slice
#	upvar \#0 $mods(SliceReader)-slice slice
#	set slice [expr $slice + 1]
	set $mods(SliceReader)-slice [expr [set $mods(SliceReader)-slice] + 1]

	$this go_highres
    }

    method SliceReader_spacing_changed {var1 var2 var3} {
	global mods axis spacing

	if {$axis == 0} {
	    global $mods(SliceReader)-spacing_0
	    set spacing [set $mods(SliceReader)-spacing_0]
	} elseif {$axis == 1} {
	    global $mods(SliceReader)-spacing_1
	    set spacing [set $mods(SliceReader)-spacing_1]
	} else {
	    global $mods(SliceReader)-spacing_2
	    set spacing [set $mods(SliceReader)-spacing_2]
	}	
    }

    method SliceReader_size_changed {var1 var2 var3} {
	# Slice Reader read in new file so adjust the slice
	# slider and status canvas width
	global mods axis spacing
	set max 0
	
	if {$axis == 0} {
	    global $mods(SliceReader)-size_0
	    upvar \#0 $mods(SliceReader)-size_0 size
	    set max $size
	    set size0 $size
	} elseif {$axis == 1} {
	    global $mods(SliceReader)-size_1
	    upvar \#0 $mods(SliceReader)-size_1 size
	    set max $size
	    set size1 $size
	} else {
	    global $mods(SliceReader)-size_2
	    upvar \#0 $mods(SliceReader)-size_2 size
	    set max $size
	    set size2 $size
	}

	if {$max > 0} {
	    $attachedPFr.f.p.childsite.volumes.childsite.slice.sl configure \
		-to [expr $max - 1]
	    $detachedPFr.f.p.childsite.volumes.childsite.slice.sl configure \
		-to [expr $max - 1]
	    
	    set status_width [expr [expr $process_width - 75]/$max]
	}

	# Clear canvas
	$status_canvas1 create rectangle \
	    0 0 [expr $process_width - 75] 10 \
	    -fill "white" -outline "black" -tags completed
	$status_canvas2 create rectangle \
	    0 0 [expr $process_width - 75] 10 \
	    -fill "white" -outline "black" -tags completed

	# Set commits array
	for {set i 0} {$i < $max} {incr i} {
	    set commits($i) 0
	}
    }
    
    method to_smooth_changed {} {
	global mods to_smooth
	
	# If smoothing, enable appropriate modules, else disable them
	global $mods(ChooseImage-Smooth)-port-index
	
	disableModule $mods(ChooseImage-Smooth) 0
	if {$to_smooth == 0} {
	    set $mods(ChooseImage-Smooth)-port-index 0
	    disableModule $mods(Smooth-Gradient) 1
	    disableModule $mods(Smooth-Curvature) 1
	    disableModule $mods(Smooth-Gaussian) 1
	} else {
	    $this enable_proper_smooth_module
	}
    }

    method update_seed_binary_threshold {var1 var2 var3} {
	global mods
	global $mods(LevelSet)-lower_threshold
	global $mods(LevelSet)-upper_threshold

	# update thresholds of BinaryThresholdImageFilter that
	# feeds in when user selects thresholding seeding method
	global $mods(BinaryThreshold-Seed)-lower_threshold
	global $mods(BinaryThreshold-Seed)-upper_threshold
	set $mods(BinaryThreshold-Seed)-lower_threshold \
	    [set $mods(LevelSet)-lower_threshold]
	set $mods(BinaryThreshold-Seed)-upper_threshold \
	    [set $mods(LevelSet)-upper_threshold]
    }
    

    method toggle_show_speed {} {
	global mods
	global $mods(ShowField-Speed)-faces-on

	if {$reverse_changed == 1} {
	    $mods(GenStandardColorMaps-Speed)-c needexecute
	}
	
	$mods(ShowField-Speed)-c toggle_display_faces
    }

    method toggle_show_seed {} {
	global mods
	global $mods(ShowField-Seed)-faces-on
	
	if {[set $mods(ShowField-Seed)-faces-on] == 0} {
#	    disableModule $mods(ImageToNrrd-CurSeed) 1
	    $mods(ShowField-Seed)-c toggle_display_faces
	} else {
#	    disableModule $mods(ImageToNrrd-CurSeed) 0
	    
#	    $mods(ImageToField-Seed)-c needexecute
	    $mods(ShowField-Seed)-c toggle_display_faces
	}
    }

    method toggle_show_segmentation {} {
	global mods
	global $mods(ShowField-Seg)-faces-on
	
	if {[set $mods(ShowField-Seg)-faces-on] == 0} {
#	    disableModule $mods(ImageToField-Seg) 1
	    $mods(ShowField-Seg)-c toggle_display_faces
	} else {
#	    disableModule $mods(ImageToField-Seg) 0
	    
#	    $mods(ImageToField-Seg)-c needexecute
	    $mods(ShowField-Seg)-c toggle_display_faces
	}
    }

    method check_previous_filename {} {
  	global mods commit_dir base_filename

	# Set up writer filename
	global $mods(ImageFileWriter-Binary)-filename
	global $mods(SliceReader)-slice
	upvar \#0 $mods(SliceReader)-slice s
	set $mods(ImageFileWriter-Binary)-filename \
	    [file join $commit_dir $base_filename$s.hdr]
    }

    method reverse_expansion_direction_changed {} {
	global mods
	
	global $mods(LevelSet)-reverse_expansion_direction
	global $mods(GenStandardColorMaps-Speed)-reverse

	upvar \#0 $mods(LevelSet)-reverse_expansion_direction rev_dir
	upvar \#0 $mods(GenStandardColorMaps-Speed)-reverse reverse

	if {$rev_dir == 0} {
	    set reverse 0
	} else {
	    set reverse 1
	}

	# Only execute GenStandarColorMaps if speed image
	# is turned on, otherwise, it will need to be executed
	# when the speed image is either turned on via the checkbox
	# or by selecting the Go button in the speed image frame
	global $mods(ShowField-Speed)-faces-on
	
	if {[set $mods(ShowField-Speed)-faces-on] == 1} {
	    $mods(GenStandardColorMaps-Speed)-c needexecute
	} else {
	    set reverse_changed 1
	}
    }

    method update_spacing {var} {
	global mods spacing
	global $mods(BuildTransform)-scale_z
	
	set $mods(BuildTransform)-scale_z [expr log10($spacing)]
	if {$initialized == 1 && $has_committed == 1} {
	    $mods(BuildTransform)-c needexecute
	}
    }
    
    method make_entry {w text v {wi -1}} {
	frame $w
	label $w.l -text "$text" 
	pack $w.l -side left
	global $v
 	if {$wi == -1} {
 	    entry $w.e -textvariable $v -width 5
 	} else {
 	    entry $w.e -textvariable $v -width $wi
 	}
	pack $w.e -side right
    }
    
    method do_nothing {} {
	# do nothing because the user shouldn't
	# be able to use middle mouse in ortho windows
    }


    # Application placing and size
    variable notebook_width
    variable notebook_height

    variable eviewer2
    variable has_loaded
    variable 2D_fixed


    variable size0
    variable size1
    variable size2
    variable range_min
    variable range_max
    variable slice_frame

    variable filter_menu1
    variable filter_menu2
    variable filter_type

    variable seed_menu1
    variable seed_menu2
    variable seed_type

    variable next_load
    variable next_smooth

    variable execute_active_color
    variable has_smoothed

#    variable has_segmented
    variable segmentation_initialized
#    variable updating_speed
    variable pasting_binary
    variable pasting_float
    
    variable volren_has_autoviewed

    variable segs
    
    variable region_changed

    variable updating_crop_widget

    variable smoothing
    variable smoothing_method
    variable smoothing_type

    variable segmenting_type
    variable segmenting 

    variable has_committed

    variable pos_seeds_used
    variable neg_seeds_used

    variable committing
    variable status_canvas1
    variable status_canvas2
    variable status_width
    variable commits

    variable has_autoviewed

    variable reverse_changed
}

LevelSetSegmenterApp app

setProgressText "Displaying LevelSetSegmenter GUI..."

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
    $mods(Viewer-Vol)-ViewWindow_0-c autoview
}

# Bugs/ToDo
##############
# Reverse Expansion doesn't seem to do anything to speed image.

# Status indicator doesn't keep spinning at first
#  when the speed image and stuff is still being updated.

# If positive seed points are set to 0, the entire seed is on,
#  regardless of the number of negative points (only for Seed
#  Points Only case).

# Clean up vis toggle window (hide/showable?)

# Arg passing and filenames


