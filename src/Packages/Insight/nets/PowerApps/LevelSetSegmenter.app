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

# Create a Teem->DataIO->NrrdReader Module
set m1 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 10 10]

# Create a Teem->DataIO->DicomNrrdReader Module
set m2 [addModuleAtPosition "Teem" "DataIO" "DicomNrrdReader" 183 10]

# Create a Teem->DataIO->AnalyzeNrrdReader Module
set m3 [addModuleAtPosition "Teem" "DataIO" "AnalyzeNrrdReader" 355 10]

# Create a Teem->NrrdData->ChooseNrrd Module
set m4 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 88]
set Notes($m4) {Determines which
file type is read.}
set Notes($m4-Position) {def}
set Notes($m4-Color) {white}

# Create a Teem->UnuAtoM->UnuCrop Module
set m5 [addModuleAtPosition "Teem" "UnuAtoM" "UnuCrop" 28 393]

# Create a Teem->NrrdData->ChooseNrrd Module
set m6 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 456]
set Notes($m6) {Use cropped vs. original
volume for smoothing.}
set Notes($m6-Position) {def}
set Notes($m6-Color) {white}

# Create a Teem->Converters->NrrdToField Module
set m7 [addModuleAtPosition "Teem" "Converters" "NrrdToField" 82 167]

# Create a SCIRun->FieldsOther->ScalarFieldStats Module
set m8 [addModuleAtPosition "SCIRun" "FieldsOther" "ScalarFieldStats" 82 226]

# Create a Teem->NrrdData->NrrdInfo Module
set m9 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 254 228]

# Create a Teem->UnuAtoM->UnuMinmax Module
set m10 [addModuleAtPosition "Teem" "UnuAtoM" "UnuMinmax" 279 166]

# Create a Insight->Converters->NrrdToImage Module
set m11 [addModuleAtPosition "Insight" "Converters" "NrrdToImage" 454 560]

# Create a Insight->Filters->GradientAnisotropicDiffusionImageFilter Module
set m12 [addModuleAtPosition "Insight" "Filters" "GradientAnisotropicDiffusionImageFilter" 473 961]

# Create a Insight->Filters->CurvatureAnisotropicDiffusionImageFilter Module
set m13 [addModuleAtPosition "Insight" "Filters" "CurvatureAnisotropicDiffusionImageFilter" 769 962]

# Create a Insight->Filters->DiscreteGaussianImageFilter Module
set m14 [addModuleAtPosition "Insight" "Filters" "DiscreteGaussianImageFilter" 1074 961]

# Create a Insight->Filters->ThresholdSegmentationLevelSetImageFilter Module
set m15 [addModuleAtPosition "Insight" "Filters" "ThresholdSegmentationLevelSetImageFilter" 338 2328]

# Create a Insight->Filters->ExtractImageFilter Module
set m16 [addModuleAtPosition "Insight" "Filters" "ExtractImageFilter" 454 1246]
set Notes($m16) {Pull out
current slice.}
set Notes($m16-Position) {def}
set Notes($m16-Color) {white}

# Create a Insight->DataIO->ChooseImage Module
set m17 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 473 1042]

# Create a Insight->Filters->BinaryThresholdImageFilter Module
set m18 [addModuleAtPosition "Insight" "Filters" "BinaryThresholdImageFilter" 60 2413]

# Create a SCIRun->FieldsCreate->SeedPoints Module
set m19 [addModuleAtPosition "SCIRun" "FieldsCreate" "SeedPoints" 709 1584]

# Create a Insight->Filters->BinaryThresholdImageFilter Module
set m20 [addModuleAtPosition "Insight" "Filters" "BinaryThresholdImageFilter" 399 1527]
set Notes($m20) {Thresholding
seed.}
set Notes($m20-Position) {def}
set Notes($m20-Color) {white}

# Create a Insight->Converters->ImageToField Module
set m21 [addModuleAtPosition "Insight" "Converters" "ImageToField" 709 1526]
set Notes($m21) {Positive
seed points.}
set Notes($m21-Position) {def}
set Notes($m21-Color) {white}

# Create a Insight->Filters->ExtractImageFilter Module
set m22 [addModuleAtPosition "Insight" "Filters" "ExtractImageFilter" 129 1521]
set Notes($m22) {Prev, next, or
current seed.}
set Notes($m22-Position) {def}
set Notes($m22-Color) {white}

# Create a Insight->DataIO->ImageFileWriter Module
set m23 [addModuleAtPosition "Insight" "DataIO" "ImageFileWriter" 99 2823]
set Notes($m23) {Writer for 
binary volume.}
set Notes($m23-Position) {def}
set Notes($m23-Color) {white}

# Create a SCIRun->Visualization->RescaleColorMap Module
set m24 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 424 2614]

# Create a Insight->Converters->ImageToNrrd Module
set m25 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 129 1578]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m26 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 900 2611]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m27 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 666 2608]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m28 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1161 2604]

# Create a Insight->Filters->PasteImageFilter Module
set m29 [addModuleAtPosition "Insight" "Filters" "PasteImageFilter" 60 2745]
set Notes($m29) {Execute from
pressing Commit.}
set Notes($m29-Position) {def}
set Notes($m29-Color) {white}

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m30 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 175 3109]

# Create a Insight->Converters->ImageToNrrd Module
set m31 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 1262 1206]

# Create a SCIRun->Visualization->RescaleColorMap Module
set m32 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 175 3171]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m33 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1280 1412]

# Create a Teem->UnuAtoM->UnuMinmax Module
set m34 [addModuleAtPosition "Teem" "UnuAtoM" "UnuMinmax" 547 3026]

# Create a SCIRun->Render->ViewSlices Module
set m35 [addModuleAtPosition "SCIRun" "Render" "ViewSlices" 1244 1474]
set Notes($m35) {Viewer 3}
set Notes($m35-Position) {def}
set Notes($m35-Color) {#ff02ff}

# Create a Insight->Converters->BuildSeedVolume Module
set m36 [addModuleAtPosition "Insight" "Converters" "BuildSeedVolume" 727 1651]

# Create a SCIRun->Render->Viewer Module
set m37 [addModuleAtPosition "SCIRun" "Render" "Viewer" 1069 3149]

# Create a SCIRun->Visualization->ShowField Module
set m38 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 406 2673]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m39 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 424 2553]

# Create a Insight->Converters->ImageToField Module
set m40 [addModuleAtPosition "Insight" "Converters" "ImageToField" 406 2491]
set Notes($m40) {Current
slice.}
set Notes($m40-Position) {def}
set Notes($m40-Color) {white}

# Create a Insight->Converters->ImageToField Module
set m41 [addModuleAtPosition "Insight" "Converters" "ImageToField" 882 2491]
set Notes($m41) {Segmentation
result slice.}
set Notes($m41-Position) {def}
set Notes($m41-Color) {white}

# Create a SCIRun->Visualization->ShowField Module
set m42 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 882 2672]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m43 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 900 2551]

# Create a Insight->Converters->ImageToField Module
set m44 [addModuleAtPosition "Insight" "Converters" "ImageToField" 1143 2482]
set Notes($m44) {Speed
image slice}
set Notes($m44-Position) {def}
set Notes($m44-Color) {white}

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m45 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1161 2542]

# Create a SCIRun->Visualization->ShowField Module
set m46 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1143 2669]

# Create a Insight->Converters->ImageToNrrd Module
set m47 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 399 1589]

# Create a SCIRun->Visualization->GenTitle Module
set m48 [addModuleAtPosition "SCIRun" "Visualization" "GenTitle" 910 3070]

# Create a SCIRun->Visualization->GenTitle Module
set m49 [addModuleAtPosition "SCIRun" "Visualization" "GenTitle" 933 3010]

# Create a Insight->Filters->PasteImageFilter Module
set m50 [addModuleAtPosition "Insight" "Filters" "PasteImageFilter" 338 2746]
set Notes($m50) {Execute from
pressing Commit.}
set Notes($m50-Position) {def}
set Notes($m50-Color) {white}

# Create a Insight->Converters->ImageToNrrd Module
set m51 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 338 2936]
set Notes($m51) {Volume rendering
modules}
set Notes($m51-Position) {e}
set Notes($m51-Color) {#fffeff}

# Create a SCIRun->Render->Viewer Module
set m52 [addModuleAtPosition "SCIRun" "Render" "Viewer" 501 3609]

# Create a Teem->UnuNtoZ->UnuQuantize Module
set m53 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuQuantize" 387 3109]

# Create a SCIRun->Visualization->NrrdTextureBuilder Module
set m54 [addModuleAtPosition "SCIRun" "Visualization" "NrrdTextureBuilder" 369 3171]

# Create a SCIRun->Visualization->EditColorMap2D Module
set m55 [addModuleAtPosition "SCIRun" "Visualization" "EditColorMap2D" 537 3479]

# Create a SCIRun->Visualization->VolumeVisualizer Module
set m56 [addModuleAtPosition "SCIRun" "Visualization" "VolumeVisualizer" 501 3544]

# Create a Teem->NrrdData->NrrdSetupTexture Module
set m57 [addModuleAtPosition "Teem" "NrrdData" "NrrdSetupTexture" 369 3025]

# Create a SCIRun->Visualization->GenTitle Module
set m58 [addModuleAtPosition "SCIRun" "Visualization" "GenTitle" 709 3481]

# Create a Teem->UnuAtoM->UnuHeq Module
set m59 [addModuleAtPosition "Teem" "UnuAtoM" "UnuHeq" 555 3293]

# Create a Teem->UnuAtoM->UnuGamma Module
set m60 [addModuleAtPosition "Teem" "UnuAtoM" "UnuGamma" 555 3355]

# Create a Teem->UnuNtoZ->UnuQuantize Module
set m61 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuQuantize" 555 3417]

# Create a Teem->UnuAtoM->UnuJhisto Module
set m62 [addModuleAtPosition "Teem" "UnuAtoM" "UnuJhisto" 573 3109]

# Create a Teem->UnuAtoM->Unu2op Module
set m63 [addModuleAtPosition "Teem" "UnuAtoM" "Unu2op" 555 3171]

# Create a Teem->UnuAtoM->Unu1op Module
set m64 [addModuleAtPosition "Teem" "UnuAtoM" "Unu1op" 555 3232]

# Create a Teem->NrrdData->ChooseNrrd Module
set m65 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 1244 1265]
set Notes($m65) {Toggle for ViewSlices
to view original or smoothed.}
set Notes($m65-Position) {def}
set Notes($m65-Color) {white}

# Create a Teem->NrrdData->NrrdInfo Module
set m66 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 75 597]

# Create a Teem->UnuAtoM->UnuMinmax Module
set m67 [addModuleAtPosition "Teem" "UnuAtoM" "UnuMinmax" 94 535]

# Create a Teem->UnuAtoM->UnuMinmax Module
set m68 [addModuleAtPosition "Teem" "UnuAtoM" "UnuMinmax" 1287 1345]

# Create a Insight->Converters->ImageToNrrd Module
set m69 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 454 1310]

# Create a Teem->UnuAtoM->UnuAxdelete Module
set m70 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxdelete" 454 1368]

# Create a Insight->Converters->NrrdToImage Module
set m71 [addModuleAtPosition "Insight" "Converters" "NrrdToImage" 454 1428]

# Create a Teem->UnuAtoM->UnuAxdelete Module
set m72 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxdelete" 129 1637]

# Create a Insight->Converters->FloatToUChar Module
set m73 [addModuleAtPosition "Insight" "Converters" "FloatToUChar" 60 2681]

# Create a Insight->DataIO->ImageFileWriter Module
set m74 [addModuleAtPosition "Insight" "DataIO" "ImageFileWriter" 382 2823]
set Notes($m74) {Writer for
float volume.}
set Notes($m74-Position) {def}
set Notes($m74-Color) {white}

# Create a Teem->Converters->NrrdToField Module
set m75 [addModuleAtPosition "Teem" "Converters" "NrrdToField" 648 2487]
set Notes($m75) {Current
seed slice}
set Notes($m75-Position) {def}
set Notes($m75-Color) {white}

# Create a SCIRun->Visualization->ShowField Module
set m76 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 648 2667]

# Create a SCIRun->Visualization->GenStandardColorMaps Module
set m77 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 666 2549]

# Create a Insight->Converters->ImageToField Module
set m78 [addModuleAtPosition "Insight" "Converters" "ImageToField" 963 1524]
set Notes($m78) {Negative 
seed points.}
set Notes($m78-Position) {def}
set Notes($m78-Color) {white}

# Create a SCIRun->FieldsCreate->SeedPoints Module
set m79 [addModuleAtPosition "SCIRun" "FieldsCreate" "SeedPoints" 963 1581]

# Create a Insight->Converters->BuildSeedVolume Module
set m80 [addModuleAtPosition "Insight" "Converters" "BuildSeedVolume" 981 1647]

# Create a Teem->NrrdData->ChooseNrrd Module
set m81 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 381 1719]
set Notes($m81) {Change when
seeding method
changes.}
set Notes($m81-Position) {def}
set Notes($m81-Color) {white}

# Create a Teem->UnuAtoM->Unu2op Module
set m82 [addModuleAtPosition "Teem" "UnuAtoM" "Unu2op" 841 1798]
set Notes($m82) {Add positive seeds
to selected seed
method output.}
set Notes($m82-Position) {def}
set Notes($m82-Color) {white}

# Create a Teem->NrrdData->ChooseNrrd Module
set m83 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 823 1865]
set Notes($m83) {Change if using
seeds only or
combination.}
set Notes($m83-Position) {def}
set Notes($m83-Color) {white}

# Create a Teem->UnuAtoM->Unu2op Module
set m84 [addModuleAtPosition "Teem" "UnuAtoM" "Unu2op" 823 1925]
set Notes($m84) {Subtract
negative seeds.}
set Notes($m84-Position) {def}
set Notes($m84-Color) {white}

# Create a Insight->Converters->NrrdToImage Module
set m85 [addModuleAtPosition "Insight" "Converters" "NrrdToImage" 801 2004]

# Create a Insight->DataIO->ChooseImage Module
set m86 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 473 639]
set Notes($m86) {Change depending on
if Go or Reset was pressed.}
set Notes($m86-Position) {def}
set Notes($m86-Color) {white}

# Create a Insight->Filters->PasteImageFilter Module
set m87 [addModuleAtPosition "Insight" "Filters" "PasteImageFilter" 473 1103]
set Notes($m87) {Every time Paste finishes,
increment Extract min/maxs
and execute ExtractImageFilter.}
set Notes($m87-Position) {def}
set Notes($m87-Color) {white}

# Create a Insight->Filters->ExtractImageFilter Module
set m88 [addModuleAtPosition "Insight" "Filters" "ExtractImageFilter" 473 699]
set Notes($m88) {With each slice 
iteration, change
axis min/max.}
set Notes($m88-Position) {def}
set Notes($m88-Color) {white}

# Create a Insight->Converters->ImageToNrrd Module
set m89 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 473 762]

# Create a Teem->UnuAtoM->UnuAxdelete Module
set m90 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxdelete" 473 820]

# Create a Insight->Converters->NrrdToImage Module
set m91 [addModuleAtPosition "Insight" "Converters" "NrrdToImage" 473 881]

# Create a Insight->DataIO->ChooseImage Module
set m92 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 454 1180]

# Create a Insight->DataIO->ChooseImage Module
set m93 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 286 674]

# Create a Insight->DataIO->ChooseImage Module
set m94 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 338 2130]

# Create a Insight->DataIO->ChooseImage Module
set m95 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 338 2191]

# Create a Insight->DataIO->ChooseImage Module
set m96 [addModuleAtPosition "Insight" "DataIO" "ChooseImage" 338 2255]

# Create a Insight->Converters->ImageToNrrd Module
set m97 [addModuleAtPosition "Insight" "Converters" "ImageToNrrd" 660 2338]

# Create the Connections between Modules
set c1 [addConnection $m36 0 $m83 0]
set c2 [addConnection $m36 0 $m82 0]
set c3 [addConnection $m73 0 $m29 0]
set c4 [addConnection $m21 0 $m19 0]
set c5 [addConnection $m40 0 $m38 0]
set c6 [addConnection $m41 0 $m42 0]
set c7 [addConnection $m44 0 $m46 0]
set c8 [addConnection $m78 0 $m79 0]
set c9 [addConnection $m25 0 $m72 0]
set c10 [addConnection $m51 0 $m57 0]
set c11 [addConnection $m51 0 $m34 0]
set c12 [addConnection $m69 0 $m70 0]
set c13 [addConnection $m89 0 $m90 0]
set c14 [addConnection $m11 0 $m86 0]
set c15 [addConnection $m11 0 $m92 0]
set c16 [addConnection $m71 0 $m21 0]
set c17 [addConnection $m71 0 $m40 0]
set c18 [addConnection $m71 0 $m78 0]
set c19 [addConnection $m71 0 $m20 0]
set c20 [addConnection $m85 0 $m94 0]
set c21 [addConnection $m91 0 $m13 0]
set c22 [addConnection $m91 0 $m14 0]
set c23 [addConnection $m91 0 $m12 0]
set c24 [addConnection $m17 0 $m87 0]
set c25 [addConnection $m86 0 $m93 0]
set c26 [addConnection $m92 0 $m16 0]
set c27 [addConnection $m93 0 $m88 0]
set c28 [addConnection $m94 0 $m95 0]
set c29 [addConnection $m95 0 $m96 0]
set c30 [addConnection $m96 0 $m97 0]
set c31 [addConnection $m96 0 $m15 0]
set c32 [addConnection $m18 0 $m73 0]
set c33 [addConnection $m18 0 $m41 0]
set c34 [addConnection $m20 0 $m47 0]
set c35 [addConnection $m16 0 $m69 0]
set c36 [addConnection $m22 0 $m25 0]
set c37 [addConnection $m88 0 $m89 0]
set c38 [addConnection $m12 0 $m17 0]
set c39 [addConnection $m29 0 $m23 0]
set c40 [addConnection $m29 0 $m22 0]
set c41 [addConnection $m50 0 $m51 0]
set c42 [addConnection $m50 0 $m74 0]
set c43 [addConnection $m87 0 $m31 0]
set c44 [addConnection $m15 0 $m18 0]
set c45 [addConnection $m15 0 $m50 0]
set c46 [addConnection $m15 1 $m44 0]
set c47 [addConnection $m19 1 $m36 0]
set c48 [addConnection $m79 1 $m80 0]
set c49 [addConnection $m30 0 $m32 0]
set c50 [addConnection $m39 0 $m24 0]
set c51 [addConnection $m43 0 $m26 0]
set c52 [addConnection $m45 0 $m28 0]
set c53 [addConnection $m77 0 $m27 0]
set c54 [addConnection $m48 0 $m37 0]
set c55 [addConnection $m54 0 $m56 0]
set c56 [addConnection $m56 0 $m52 0]
set c57 [addConnection $m7 0 $m8 0]
set c58 [addConnection $m75 0 $m76 0]
set c59 [addConnection $m1 0 $m4 0]
set c60 [addConnection $m4 0 $m6 0]
set c61 [addConnection $m4 0 $m9 0]
set c62 [addConnection $m4 0 $m5 0]
set c63 [addConnection $m4 0 $m10 0]
set c64 [addConnection $m6 0 $m11 0]
set c65 [addConnection $m6 0 $m65 0]
set c66 [addConnection $m6 0 $m66 0]
set c67 [addConnection $m6 0 $m67 0]
set c68 [addConnection $m65 0 $m35 0]
set c69 [addConnection $m65 0 $m68 0]
set c70 [addConnection $m83 0 $m84 0]
set c71 [addConnection $m57 0 $m54 0]
set c72 [addConnection $m57 1 $m53 0]
set c73 [addConnection $m64 0 $m59 0]
set c74 [addConnection $m63 0 $m64 0]
set c75 [addConnection $m84 0 $m85 0]
set c76 [addConnection $m70 0 $m71 0]
set c77 [addConnection $m72 0 $m81 0]
set c78 [addConnection $m90 0 $m91 0]
set c79 [addConnection $m60 0 $m61 0]
set c80 [addConnection $m59 0 $m60 0]
set c81 [addConnection $m80 0 $m84 1]
set c82 [addConnection $m40 0 $m24 1]
set c83 [addConnection $m41 0 $m26 1]
set c84 [addConnection $m44 0 $m28 1]
set c85 [addConnection $m31 0 $m65 1]
set c86 [addConnection $m47 0 $m81 1]
set c87 [addConnection $m51 0 $m62 1]
set c88 [addConnection $m71 0 $m36 1]
set c89 [addConnection $m71 0 $m80 1]
set c90 [addConnection $m71 0 $m15 1]
set c91 [addConnection $m18 0 $m94 1]
set c92 [addConnection $m13 0 $m17 1]
set c93 [addConnection $m87 0 $m86 1]
set c94 [addConnection $m87 0 $m92 1]
set c95 [addConnection $m49 0 $m37 1]
set c96 [addConnection $m58 0 $m52 1]
set c97 [addConnection $m24 0 $m38 1]
set c98 [addConnection $m26 0 $m42 1]
set c99 [addConnection $m27 0 $m76 1]
set c100 [addConnection $m28 0 $m46 1]
set c101 [addConnection $m32 0 $m56 1]
set c102 [addConnection $m75 0 $m27 1]
set c103 [addConnection $m2 0 $m4 1]
set c104 [addConnection $m81 0 $m82 1]
set c105 [addConnection $m82 0 $m83 1]
set c106 [addConnection $m5 0 $m6 1]
set c107 [addConnection $m62 0 $m63 1]
set c108 [addConnection $m53 0 $m54 1]
set c109 [addConnection $m61 0 $m55 1]
set c110 [addConnection $m97 0 $m75 2]
set c111 [addConnection $m14 0 $m17 2]
set c112 [addConnection $m55 0 $m56 2]
set c113 [addConnection $m33 0 $m35 2]
set c114 [addConnection $m38 0 $m37 2]
set c115 [addConnection $m3 0 $m4 2]
set c116 [addConnection $m4 0 $m7 2]
set c117 [addConnection $m57 1 $m62 2]
set c118 [addConnection $m42 0 $m37 3]
set c119 [addConnection $m46 0 $m37 4]
set c120 [addConnection $m76 0 $m37 5]
set c121 [addConnection $m19 0 $m37 6]
set c122 [addConnection $m79 0 $m37 7]


# Set GUI variables
# Determine if load file was passed in, 
# and if it is an analyze file, otherwise it
# defaults to use the nrrd reader (generic tab)
global load_file
set load_file ""
global load_file_type
set load_file_type "Generic"
if {[netedit getenv LEVELSETSEGMENTER_LOAD_FILE] == ""} {
    set load_file  "/usr/sci/data/Medical/ucsd/king_filt/king_filt-full.nhdr"
    set load_file_type "Generic"
} else {
    # Determine which reader to use by looking at file extension
    set index [string last "." [netedit getenv LEVELSETSEGMENTER_LOAD_FILE]]
    if {$index > 0} {
	set ext [string range [netedit getenv LEVELSETSEGMENTER_LOAD_FILE] $index end]

	if {[string equal $ext ".hdr"] == 1} {
	    # Analyze file
	    set load_file [netedit getenv LEVELSETSEGMENTER_LOAD_FILE]
	    set load_file_type "Analyze"
	} else {
	    # Some other file, hopefully something the generic
	    # tab (NrrdReader) can handle -- DICOM can't be read
	    # in via command line
	    set load_file [netedit getenv LEVELSETSEGMENTER_LOAD_FILE]
	    set load_file_type "Generic"
	}
    } else {
	set load_file  "/usr/sci/data/Medical/ucsd/king_filt/king_filt-full.nhdr"
	set load_file_type "Generic"
    }
}

# NrrdReader_0
if {[string equal $load_file_type "Generic"] == 1} {
    setGlobal $m1-filename $load_file
}


# AnalyzeNrrdReader_0
if {[string equal $load_file_type "Analyze"] == 1} {
    setGlobal $m3-file $load_file
    setGlobal $m3-num-files {1}
    setGlobal $m3-filenames0 $load_file
}

# UnuCrop_0
setGlobal $m5-uis {3}

# ScalarFieldStats_0
setGlobal $m8-setdata 1
trace variable $m8-args w "app update_histo_graph_callback"

# GradientAnisotropicDiffusion_0
setGlobal $m12-time_step 0.0625
setGlobal $m12-iterations 5
setGlobal $m12-conductance 0.5

# CurvatureAnisotropicDiffusion_0
setGlobal $m13-time_step 0.0625
setGlobal $m13-iterations 5
setGlobal $m13-conductance 0.5

# ThresholdSegmentationLevelSetImageFilter_0
setGlobal $m15-isovalue {0.5}
setGlobal $m15-curvature_scaling {1.0}
setGlobal $m15-propagation_scaling {1.0}
setGlobal $m15-edge_weight {1.0}
setGlobal $m15-reverse_expansion_direction 0
setGlobal $m15-smoothing_iterations 0
setGlobal $m15-smoothing_conductance {0.5}
setGlobal $m15-smoothing_time_step {0.1}
setGlobal $m15-maximum_iterations {0}

# ExtractImageFilter_0
setGlobal $m16-uis {3}
setGlobal $m16-num-dims {3}

# BinaryThresholdImageFilter_0
setGlobal $m18-lower_threshold 0.0
setGlobal $m18-upper_threshold 100.0
setGlobal $m18-inside_value {1}
setGlobal $m18-outside_value {0}

# SeedPoints_0
setGlobal $m19-num_seeds 0
setGlobal $m19-probe_scale 10
setGlobal $m19-widget 1
setGlobal $m19-red 0.5
setGlobal $m19-green 0.0
setGlobal $m19-blue 0.0
setGlobal $m19-auto_execute 0

# BinaryThresholdImageFilter_1
setGlobal $m20-inside_value {0}
setGlobal $m20-outside_value {1}

# ImageToField_0
setGlobal $m21-copy {1}

# ExtractImageFilter_1
setGlobal $m22-uis {3}

# ImageFileWriter_0
if {[netedit getenv LEVELSETSEGMENTER_SAVE_BINARY_FILE] == ""} {
    setGlobal $m23-filename "/tmp/binary.mhd"
} else {
    setGlobal $m23-filename "[netedit getenv LEVELSETSEGMENTER_SAVE_BINARY_FILE]"
}

# PasteImageFilter_0
setGlobal $m29-uis 3

# RescaleColorMap_0
setGlobal $m32-isFixed 1
setGlobal $m32-min 0
setGlobal $m32-max 1

# GenStandardColorMaps_0
setGlobal $m33-width {552}
setGlobal $m33-height {40}
setGlobal $m33-mapName {Gray}

# UnuMinmax_0
setGlobal $m34-min0 0
setGlobal $m34-max0 0

# ViewSlices_0
setGlobal $m35-crop_minPadAxis0 0
setGlobal $m35-crop_maxPadAxis0 0
setGlobal $m35-crop_minPadAxis1 0
setGlobal $m35-crop_maxPadAxis1 0
setGlobal $m35-crop_minPadAxis2 0
setGlobal $m35-crop_maxPadAxis2 0
setGlobal $m35-show_text {1}
setGlobal $m35-anatomical_coordinates {0}

# BuildSeedVolume_0
setGlobal $m36-inside_value {0}
setGlobal $m36-outside_value {1}

# Viewer_0
setGlobal $m37-ViewWindow_0-raxes 0
setGlobal $m37-ViewWindow_0-ortho-view 1
setGlobal $m37-ViewWindow_0-pos "z1_y1"
setGlobal $m37-ViewWindow_1-raxes 0
setGlobal $m37-ViewWindow_1-ortho-view 1
setGlobal $m37-ViewWindow_1-pos "z1_y1"
setGlobal {$m37-ViewWindow_1-Transparent Faces (2)} 0
setGlobal {$m37-ViewWindow_1-Transparent Faces (4)} 0
setGlobal {$m37-ViewWindow_0-Title (2)} 0
setGlobal {$m37-ViewWindow_1-Title (1)} 0

# ShowField_0
setGlobal $m38-faces-on 1
setGlobal $m38-faces-usetexture 1
setGlobal $m38-nodes-on 0
setGlobal $m38-edges-on 0

# GenStandardColorMaps_1
setGlobal $m39-width {552}
setGlobal $m39-height {40}
setGlobal $m39-mapName {Gray}

# ImageToField_0
setGlobal $m40-copy 1

# ImageToField_1
setGlobal $m41-copy 1

# ShowField_1
setGlobal $m42-faces-on 1
setGlobal $m42-faces-usetexture 1
setGlobal $m42-use-transparency 1
setGlobal $m42-nodes-on 0
setGlobal $m42-edges-on 0

# GenStandardColorMaps_2
setGlobal $m43-positionList {{0 0} {273 00} {277 40} {552 40}}
setGlobal $m43-nodeList {3 4 5 6}
setGlobal $m43-width {552}
setGlobal $m43-height {40}
setGlobal $m43-mapName {BP Seismic}
setGlobal $m43-resolution {2}
setGlobal $m43-realres {2}
setGlobal $m43-reverse {1}

# ImageToField_2
setGlobal $m44-copy 1

# GenStandardColorMaps_3
setGlobal $m45-width {552}
setGlobal $m45-height {40}
setGlobal $m45-mapName {BP Seismic}
setGlobal $m45-resolution {2}
setGlobal $m45-realres {2}

# ShowField_2
setGlobal $m46-faces-on 1
setGlobal $m46-use-transparency 1
setGlobal $m46-faces-usetexture 0
setGlobal $m46-nodes-on 0
setGlobal $m46-edges-on 0

# GenTitle_0
setGlobal $m48-bbox 0
setGlobal $m48-showValue 0
setGlobal $m48-format "Current Seeds and Segmentation"
setGlobal $m48-location "Top Center"

# GenTitle_1
setGlobal $m49-bbox 0
setGlobal $m49-showValue 0
setGlobal $m49-format "Speed Image"
setGlobal $m49-location "Top Center"

# PasteImageFilter_1
setGlobal $m50-uis {3}

# UnuQuantize_0
setGlobal $m53-nbits {8}

# EditColorMap2D_0
setGlobal $m55-on-0 {1}
setGlobal $m55-on-1 {0}
setGlobal $m55-panx {-0.02734375}
setGlobal $m55-pany {-0.01171875}
setGlobal $m55-scale_factor {1.0}
setGlobal $m55-faux {1}
setGlobal $m55-histo {0.5}
setGlobal $m55-selected_widget {0}
setGlobal $m55-selected_object {9}
setGlobal $m55-name-0 {Generic}
setGlobal $m55-0-color-r {0.8}
setGlobal $m55-0-color-g {0.0}
setGlobal $m55-0-color-b {0.0}
setGlobal $m55-0-color-a {1.0}
setGlobal $m55-state-0 {r 0 0.00390622 0.015625 0.537109 0.613281 0.25}
setGlobal $m55-shadeType-0 {0}
setGlobal $m55-name-1 {Rectangle}
setGlobal $m55-1-color-r {0.0}
setGlobal $m55-1-color-g {0.0}
setGlobal $m55-1-color-b {0.8}
setGlobal $m55-1-color-a {1.0}
setGlobal $m55-state-1 {r 0 0.50304 0.0117188 0.459851 0.61407 0.737198}
setGlobal $m55-shadeType-1 {0}
setGlobal $m55-marker {end}

# VolumeVisualizer_0
setGlobal $m56-alpha_scale {0.0}
setGlobal $m56-shading {1}
setGlobal $m56-specular {0.388}
setGlobal $m56-shine {24}

# NrrdSetupTexture_0
setGlobal $m57-useinputmin 0
setGlobal $m57-useinputmax 0
setGlobal $m57-minf 0
setGlobal $m57-maxf 0

# GenTitle_2
setGlobal $m58-bbox {0}
setGlobal $m58-showValue {0}
setGlobal $m58-format "Volume Rendering"
setGlobal $m58-location "Top Center"

# UnuHeq_0
setGlobal $m59-bins {3000}
setGlobal $m59-sbins {1}

# UnuGamma_0
setGlobal $m60-gamma {0.5}

# UnuQuantize_1
setGlobal $m61-nbits {8}

# UnuJhisto_0
setGlobal $m62-bins {512 256}
setGlobal $m62-mins {0 nan}
setGlobal $m62-maxs {6 nan}
setGlobal $m62-type {nrrdTypeFloat}

# Unu2op_0
setGlobal $m63-operator {+}

# Unu1op_0
setGlobal $m64-operator {log}

# UnuAxdelete_0
setGlobal $m70-axis 2

# UnuAxdelete_1
setGlobal $m72-axis 2

# ImageFileWriter_1
if {[netedit getenv LEVELSETSEGMENTER_SAVE_FLOAT_FILE] == ""} {
    setGlobal $m74-filename "/tmp/float.mhd"
} else {
    setGlobal $m74-filename "[netedit getenv LEVELSETSEGMENTER_SAVE_FLOAT_FILE]"
}

# NrrdToField_0
setGlobal $m75-copy 1

# ShowField_3
setGlobal $m76-faces-on 1
setGlobal $m76-faces-usetexture 1
setGlobal $m76-use-transparency 1
setGlobal $m76-nodes-on 0
setGlobal $m76-edges-on 0

# GenStandardColorMaps_4
setGlobal $m77-positionList {{0 0} {273 0} {277 40} {552 40}}
setGlobal $m77-nodeList {3 4 5 6}
setGlobal $m77-width {552}
setGlobal $m77-height {40}
setGlobal $m77-mapName {BP Seismic}
setGlobal $m77-resolution {2}
setGlobal $m77-realres {2}
setGlobal $m77-reverse {1}

# ImageToField_3
setGlobal $m78-copy {1}

# SeedPoints_1
setGlobal $m79-num_seeds 0
setGlobal $m79-probe_scale 10
setGlobal $m79-widget 1
setGlobal $m79-red 0.0
setGlobal $m79-green 0.0
setGlobal $m79-blue 0.5
setGlobal $m79-auto_execute 0

# BuildSeedVolume_1
setGlobal $m80-inside_value {1}
setGlobal $m80-outside_value {0}

# ChooseNrrd_0
setGlobal $m81-port-index {1}

# Unu2op_1
setGlobal $m82-operator {min}

# ChooseNrrd_1
setGlobal $m83-port-index {1}

# Unu2op_2
setGlobal $m84-operator {max}

# ExtractImageFilter_0
setGlobal $m88-uis 3

# UnuAxdelete_2
setGlobal $m90-axis 2


::netedit scheduleok


# global array indexed by module name to keep track of modules
global mods
set mods(Viewer) $m37
set mods(ViewSlices) $m35

# readers
set mods(NrrdReader) $m1
set mods(DicomReader) $m2
set mods(AnalyzeReader) $m3
set mods(ChooseNrrd-Reader) $m4
set mods(NrrdInfo-Reader) $m9
set mods(UnuMinMax-Reader) $m10
set mods(ScalarFieldStats) $m8
set mods(NrrdToImage-Reader) $m11

# Region of Interest
set mods(UnuCrop) $m5
set mods(ChooseNrrd-Crop) $m6

set mods(NrrdInfo-Size) $m66
set mods(UnuMinMax-Size) $m67
set mods(UnuMinMax-Smoothed) $m68

# Smoothing
set mods(Smooth-Gradient) $m12
set mods(Smooth-Curvature) $m13
set mods(Smooth-Blur) $m14
set mods(ChooseImage-Smooth) $m17
set mods(ChooseImage-ToSmooth) $m92
set mods(PasteImageFilter-Smooth) $m87
set mods(Extract-Smooth) $m88
set mods(UnuAxdelete-Smooth) $m90
set mods(ChooseImage-SmoothInput) $m86
set mods(ChooseImage-Hack) $m93


# Segmenting
set mods(LevelSet) $m15
set mods(ExtractSlice) $m16
set mods(BinaryThreshold-Slice) $m18
set mods(ImageToField-Seg) $m41
set mods(GenStandard-Seg) $m43
set mods(ChooseImage-SegInput) $m94
set mods(ChooseImage-Hack2) $m95
set mods(ChooseImage-Hack3) $m96

# Seeds
set mods(BinaryThreshold-Seeds) $m20 
set mods(SampleField-Seeds) $m19
set mods(SampleField-SeedsNeg) $m79
set mods(BuildSeedVolume) $m36
    # NrrdToField
set mods(ImageToField-Seeds) $m75 
    # NrrdToField
set mods(ImageToField-SeedsNeg) $m78
set mods(BuildSeedVolume-Neg) $m80
set mods(Unu2op-Max) $m82
set mods(Unu2op-Min) $m84
set mods(ChooseNrrd-Combine) $m81
set mods(ChooseNrrd-Seeds) $m83
set mods(GenTitle-Seg) $m48

# 2D vis
set mods(ChooseNrrd-2D) $m65

# volume rendering
set mods(Viewer-VolRen) $m52
set mods(EditColorMap2D) $m55

# Prev/Next slice
set mods(ImageToNrrd-Prev) $m25
set mods(UnuAxdelete-Prev) $m72
set mods(Extract-Prev) $m22 

# Feature Image
set mods(ImageToField-Feature) $m69 
set mods(UnuAxdelete-Feature) $m70
set mods(NrrdToImage-Feature) $m71
set mods(ShowField-Feature) $m38

set mods(ShowField-Seg) $m42
set mods(ShowField-Speed) $m46
set mods(ShowField-Seed) $m76

set mods(GenTitle-Speed) $m49
set mods(ImageToNrrd-ViewSlices) $m31


# Binary/Float Segmented Volume
set mods(PasteImageFilter-Binary) $m29
set mods(PasteImageFilter-Float) $m50

# binary readers/writers
set mods(ImageFileWriter-Binary) $m23
set mods(ImageFileWriter-Float) $m74

# volume rendering
set mods(ImageToNrrd-Vol) $m51
set mods(VolumeVisualizer) $m56
set mods(UnuMinmax-Vol) $m34
set mods(RescaleColorMap-Vol) $m32
set mods(UnuJhisto-Vol) $m62
set mods(NrrdSetupTexture-Vol) $m57
set mods(GenTitle-Vol) $m58



global axis
set axis 2
setGlobal $m70-axis $axis
setGlobal $m72-axis $axis
setGlobal $m90-axis $axis

global smooth_region
set smooth_region vol

global current_slice
set current_slice 0

global seed_method
set seed_method "thresh"

global max_iter
set max_iter 100

global slice
set slice 0

global show_roi
set show_roi 0

global image_dir no_seg_icon
global old_seg_icon updated_seg_icon

set image_dir [netedit getenv SCIRUN_SRCDIR]/Packages/Insight/Dataflow/GUI
set no_seg_icon [image create photo -file ${image_dir}/no-seg.ppm]
set old_seg_icon [image create photo -file ${image_dir}/old-seg.ppm]
set updated_seg_icon [image create photo -file ${image_dir}/updated-seg.ppm]

global vol_foreground
set vol_foreground 1

global show_seeds
set show_seeds 1

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

	set viewer_width 700
	set viewer_height 750

	set process_width 350
	set process_height $viewer_height

        set initialized 0

        set i_width [expr $process_width - 50]
        set i_height 20
        set stripes 10

        set error_module ""

        set indicatorID 0

	set curr_proc_tab "Load"
	set proc_tab1 ""
	set proc_tab2 ""
	global load_file_type
	set curr_data_tab $load_file_type
	set data_tab1 ""
	set data_tab2 ""
	set eviewer2 ""
	set eviewer3 ""
	set has_loaded 0
	set 2D_fixed 0

	set orig_size0 0
	set orig_size1 0
	set orig_size2 0

	set size0 0
	set size1 0
	set size2 0

	set range_min 0
	set range_max 0
	set slice_frame ""

	set filter_menu1 ""
	set filter_menu2 ""
#	set filter_enabled 0

	set next_load ""
	set next_smooth ""

	set execute_color "#008b45"
	set execute_active_color "#31a065"

	set has_smoothed 0

	set has_segmented 0
	set segmentation_initialized 0

	set updating_speed 0
	set segmenting 0
	
	set pasting_binary 0
	set pasting_float 0

	set current_slice 0

	set volren_has_autoviewed 0

	set region_changed 0

	set updating_crop_widget 0

	set smoothing 0
	set smoothing_method "None"
	set smoothing_type "Reset"

	set segmenting_type "Reset"

	set has_committed 0


	### Define Tooltips
	##########################
	# General
	global tips

	# Data Acquisition Tab
	# FIX ME: DEFINE TOOLTIPS

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
	# disable non-default readers
	disableModule $mods(DicomReader) 1
	disableModule $mods(AnalyzeReader) 1

	# Disable smoothers
	disableModule $mods(ExtractSlice) 1
	disableModule $mods(Smooth-Gradient) 1
	disableModule $mods(Smooth-Curvature) 1
	disableModule $mods(Smooth-Blur) 1
	disableModule $mods(PasteImageFilter-Smooth) 1
	disableModule $mods(Extract-Smooth) 1
	
	# Disable prev/next segmentation as seeds
	disableModule $mods(ImageToNrrd-Prev) 1
	disableModule $mods(Extract-Prev) 1

	# Level Set
	disableModule $mods(LevelSet) 1

	# Feature image into Level Set
	disableModule $mods(ImageToField-Feature) 1

	disableModule $mods(NrrdToImage-Feature) 1

	disableModule $mods(BinaryThreshold-Slice) 1

	disableModule $mods(ImageFileWriter-Binary) 1

	disableModule $mods(PasteImageFilter-Binary)  1

	# Seed Points
	disableModule $mods(ImageToField-Seeds) 1

	disableModule $mods(ImageToNrrd-ViewSlices) 1
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
	$this build_viewers $mods(Viewer) $mods(Viewer-VolRen) $mods(ViewSlices)

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
	$mods(GenTitle-Speed)-c needexecute
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
	    
            iwidgets::tabnotebook $process.tnb \
                -width [expr $process_width - 50] \
                -height [expr $process_height - 140] \
                -tabpos n -equaltabs 0
	    pack $process.tnb -side top -anchor n 

	    set proc_tab$case $process.tnb

	    ############# Load ###########
            set step_tab [$process.tnb add -label "Load" -command "$this change_processing_tab Load"]

	    iwidgets::labeledframe $step_tab.load \
		-labeltext "Data File Format and Orientation" \
		-labelpos nw
	    pack $step_tab.load -side top -anchor nw -pady 3 \
		-expand yes -fill x

	    set load [$step_tab.load childsite]

	    # Build data tabs
	    iwidgets::tabnotebook $load.tnb \
		-width [expr $process_width - 110] -height 75 \
		-tabpos n -equaltabs false
	    pack $load.tnb -side top -anchor n \
		-padx 0 -pady 3

	    set data_tab$case $load.tnb
	
	    # Nrrd
	    set page [$load.tnb add -label "Generic" \
			  -command "$this set_curr_data_tab Generic; $this configure_readers Generic"]       
	    
	    global $mods(NrrdReader)-filename
	    frame $page.file
	    pack $page.file -side top -anchor nw -padx 3 -pady 0 -fill x
	    
	    label $page.file.l -text ".vol/.vff/.nrrd file:" 
	    entry $page.file.e -textvariable $mods(NrrdReader)-filename \
		-width [expr $process_width - 120]
	    pack $page.file.l $page.file.e -side left -padx 3 -pady 0 -anchor nw \
		-fill x 
	    
	    bind $page.file.e <Return> "$this load_data"
	    
	    button $page.load -text "Browse" \
		-command "$this open_nrrd_reader_ui" \
		-width 12
	    pack $page.load -side top -anchor n -padx 3 -pady 1
	    
	    
	    ### Dicom
	    set page [$load.tnb add -label "Dicom" \
			  -command "$this set_curr_data_tab Dicom; $this configure_readers Dicom"]
	    
	    button $page.load -text "Dicom Loader" \
		-command "$this dicom_ui"
	    
	    pack $page.load -side top -anchor n \
		-padx 3 -pady 10 -ipadx 2 -ipady 2
	    
	    ### Analyze
	    set page [$load.tnb add -label "Analyze" \
			  -command "$this set_curr_data_tab Analyze; $this configure_readers Analyze"]
	    
	    button $page.load -text "Analyze Loader" \
		-command "$this analyze_ui"
	    
	    pack $page.load -side top -anchor n \
		-padx 3 -pady 10 -ipadx 2 -ipady 2

	    global load_file_type
	    $load.tnb view $load_file_type

	    # Viewing slices axis
	    frame $load.axis
	    pack $load.axis -side top -anchor nw -expand yes -fill x -pady 2

	    set a $load.axis
	    label $a.label -text "View Slices Along:"
	    
	    global axis
	    radiobutton $a.x -text "X Axis" \
		-variable axis -value 0 \
		-command "$this change_axis"

	    radiobutton $a.y -text "Y Axis" \
		-variable axis -value 1 \
		-command "$this change_axis"

	    radiobutton $a.z -text "Z Axis" \
		-variable axis -value 2 \
		-command "$this change_axis"

	    pack $a.label $a.x $a.y $a.z -side left -expand yes -fill x

	    button $load.load -text "Load" \
		-command "$this load_data" \
		-background $execute_color \
		-activebackground $execute_active_color \
		-width 10
	    pack $load.load -side top -anchor n -padx 3 -pady 4 \
		-ipadx 2



	    # Histogram

	    iwidgets::labeledframe $step_tab.stats \
		-labeltext "Volume Statistics" \
		-labelpos nw
	    pack $step_tab.stats -side top -anchor nw -expand yes -fill x
	    set stats [$step_tab.stats childsite]

	    label $stats.samples -text "Samples: $orig_size0, $orig_size1, $orig_size2"
	    label $stats.range -text "Data Range: $range_min - $range_max"

	    pack $stats.samples $stats.range -side top -anchor nw \
		-expand no

	    iwidgets::labeledframe $stats.histo \
		-labelpos nw -labeltext "Histogram"
	    pack $stats.histo -side top -fill x -anchor n -expand 1
	    
	    set histo [$stats.histo childsite]
	    
	    global $mods(ScalarFieldStats)-min
	    global $mods(ScalarFieldStats)-max
	    global $mods(ScalarFieldStats)-nbuckets
	    
	    blt::barchart $histo.graph -title "" \
		-height 220 \
		-width [expr $process_width - 50] -plotbackground gray80
	    pack $histo.graph

	    pack $stats.histo -side top -anchor nw \
		-fill x -expand yes
	    
	    # Next button
	    button $step_tab.next -text "Next" \
                -command "$this change_processing_tab Smooth" -width 8 \
                -state disabled
	    pack $step_tab.next -side top -anchor ne \
		-padx 3 -pady 3

	    set next_load "f.p.childsite.tnb.canvas.notebook.cs.page1.cs.next"


	    ############# Smooth ###########
            set step_tab [$process.tnb add -label "Smooth" -command "$this change_processing_tab Smooth"]

	    iwidgets::labeledframe $step_tab.roi \
		-labeltext "Region of Interest" \
		-labelpos nw
	    pack $step_tab.roi -side top -anchor nw -expand no -fill x -pady 3
	    
	    set roi [$step_tab.roi childsite]
	    global $mods(UnuCrop)-digits_only

	    global show_roi
	    checkbutton $roi.t -text "Show Crop Widget" \
		-variable show_roi \
		-command "$this toggle_show_roi"
	    pack $roi.t -side top -anchor nw

	    foreach l {{0 X} {1 Y} {2 Z}} {
		set i [lindex $l 0]
		set label [lindex $l 1]
		global $mods(UnuCrop)-minAxis$i
		global $mods(UnuCrop)-maxAxis$i

		set $mods(UnuCrop)-minAxis$i 0
		if {[set $mods(UnuCrop)-digits_only] == 1} {
		    set $mods(UnuCrop)-maxAxis$i 0
		} else {
		    set $mods(UnuCrop)-maxAxis$i M
		}

		frame $roi.$i
		pack $roi.$i -side top -anchor nw -expand yes -fill x \
		    -padx 2 -pady 2

		label $roi.$i.minl -text "Min Axis $label:"
		entry $roi.$i.minv -textvariable $mods(UnuCrop)-minAxis$i \
		    -width 6
		label $roi.$i.maxl -text "Max Axis $label:"
		entry $roi.$i.maxv -textvariable $mods(UnuCrop)-maxAxis$i \
		    -width 6
		grid configure $roi.$i.minl -row $i -column 0 -sticky "w"
		grid configure $roi.$i.minv -row $i -column 1 -sticky "e"
		grid configure $roi.$i.maxl -row $i -column 2 -sticky "w"
		grid configure $roi.$i.maxv -row $i -column 3 -sticky "e"

		bind $roi.$i.minv <ButtonPress-1> "$this start_crop"
		bind $roi.$i.maxv <ButtonPress-1> "$this start_crop"
		bind $roi.$i.minv <Return> "$this update_crop_widget min $i"
		bind $roi.$i.maxv <Return> "$this update_crop_widget max $i"

		global $mods(ViewSlices)-crop_minAxis$i $mods(ViewSlices)-crop_maxAxis$i
		trace variable $mods(ViewSlices)-crop_minAxis$i w "$this update_crop_values"
		trace variable $mods(ViewSlices)-crop_maxAxis$i w "$this update_crop_values"
	    }

	    button $roi.button -text "Crop" \
		-background $execute_color \
		-activebackground $execute_active_color \
		-command "$this select_region_of_interest" \
		-width 10
	    pack $roi.button -side top -anchor n -padx 3 -pady 4 -ipadx 2

	    # Smoothing Filters
	    iwidgets::labeledframe $step_tab.smooth \
		-labeltext "Smoothing" \
		-labelpos nw
	    pack $step_tab.smooth -side top -anchor nw -expand yes -fill x -pady 3
	    
	    set smooth [$step_tab.smooth childsite]

	    global smooth_region
	    radiobutton $smooth.roi -text "Smooth Region of Interest" \
		-variable smooth_region \
		-value roi \
		-command "$this change_smooth_region"

	    radiobutton $smooth.vol -text "Smooth Entire Volume" \
		-variable smooth_region \
		-value vol \
		-command "$this change_smooth_region"
	    pack $smooth.roi $smooth.vol -side top -anchor nw

	    iwidgets::optionmenu $smooth.filter -labeltext "Filter:" \
		-labelpos w -command "$this change_filter $smooth.filter"
	    pack $smooth.filter -side top -anchor nw 

	    set filter_menu$case $smooth.filter

	    $smooth.filter insert end "GradientAnisotropicDiffusion" \
		"CurvatureAnisotropicDiffusion" \
		"Gaussian" "None"
	    $smooth.filter select "None"

	    # pack ui for GradientAnisotropic first
	    # Gradient
	    frame $smooth.gradient
	    pack $smooth.gradient -side top -anchor n -pady 4 

	    global $mods(Smooth-Gradient)-time_step
	    global $mods(Smooth-Gradient)-iterations
	    global $mods(Smooth-Gradient)-conductance_parameter
	    make_entry $smooth.gradient.time "Time Step" $mods(Smooth-Gradient)-time_step 8
	    make_entry $smooth.gradient.iter "Iterations" $mods(Smooth-Gradient)-iterations 4 
	    make_entry $smooth.gradient.cond "Conductance" $mods(Smooth-Gradient)-conductance_parameter 4
	    
	    pack $smooth.gradient.time $smooth.gradient.iter $smooth.gradient.cond -side left

	    # Curvature
	    frame $smooth.curvature
	    global $mods(Smooth-Curvature)-time_step
	    global $mods(Smooth-Curvature)-iterations
	    global $mods(Smooth-Curvature)-conductance_parameter
	    make_entry $smooth.curvature.time "Time Step" $mods(Smooth-Curvature)-time_step 8
	    make_entry $smooth.curvature.iter "Iterations" $mods(Smooth-Curvature)-iterations 4 
	    make_entry $smooth.curvature.cond "Conductance" $mods(Smooth-Curvature)-conductance_parameter 4
	    
	    pack $smooth.curvature.time $smooth.curvature.iter $smooth.curvature.cond -side left

	    # Blur
	    frame $smooth.blur
	    global $mods(Smooth-Blur)-variance
	    global $mods(Smooth-Blur)-maximum_error
	    global $mods(Smooth-Blur)-maximum_kernel_width
	    make_entry $smooth.blur.var "Variance" $mods(Smooth-Blur)-variance 
	    make_entry $smooth.blur.err "Max Error" $mods(Smooth-Blur)-maximum_error 
	    make_entry $smooth.blur.kern "Kernel Width" $mods(Smooth-Blur)-maximum_kernel_width 
	    
	    pack $smooth.blur.var $smooth.blur.err $smooth.blur.kern \
		-side left	 

	    # Smooth button
	    frame $smooth.buttons
	    pack $smooth.buttons -side bottom -anchor n -padx 3 \
		-pady 4

	    button $smooth.buttons.reset -text "Reset" \
		-background $execute_color \
		-activebackground $execute_active_color \
		-command "$this smooth_data Reset" -width 10

	    button $smooth.buttons.go -text "Go" \
		-background $execute_color \
		-activebackground $execute_active_color \
		-command "$this smooth_data Go" -width 10

	    pack $smooth.buttons.reset $smooth.buttons.go  \
		-side left -anchor nw -padx 3 \
		-pady 4 -ipadx 2   

	    # ViewSlices toggle
	    frame $smooth.toggle
	    pack $smooth.toggle -side bottom -anchor n -pady 3

	    global $mods(ChooseNrrd-2D)-port-index
	    radiobutton $smooth.toggle.orig -text "Show Original" \
		-variable $mods(ChooseNrrd-2D)-port-index -value 0 \
		-command "$this update_ViewSlices_input"

	    radiobutton $smooth.toggle.smooth -text "Show Smoothed" \
		-variable $mods(ChooseNrrd-2D)-port-index -value 1 \
		-command "$this update_ViewSlices_input"

	    pack $smooth.toggle.orig $smooth.toggle.smooth -side left \
		-padx 4



	    # Next button
	    button $step_tab.next -text "Next" \
                -command "$this change_processing_tab Segment" -width 8 \
		-activebackground $next_color \
		-background $next_color 
	    pack $step_tab.next -side top -anchor ne \
		-padx 3 -pady 3

	    set next_smooth "f.p.childsite.tnb.canvas.notebook.cs.page2.cs.next"
	    

	    ############# Segment ###########
            set step_tab [$process.tnb add -label "Segment" -command "$this change_processing_tab Segment"]
	    
	    # Current Slice
	    frame $step_tab.slice
	    pack $step_tab.slice

	    global no_seg_icon
	    button $step_tab.slice.status -relief flat -image $no_seg_icon
	    pack $step_tab.slice.status -side left
	    
	    global slice
	    iwidgets::spinint $step_tab.slice.sp -labeltext "Segmenting Slice:" -width 6 \
		-range {0 100} -step 1 -textvariable slice
	    button $step_tab.slice.up -text "Change Slice" \
		-command "$this current_slice_changed"

	    pack $step_tab.slice.sp $step_tab.slice.up -side left

	    # Level Set Parameters
	    iwidgets::labeledframe $step_tab.params \
		-labeltext "Tune Speed Image" \
		-labelpos nw
	    pack $step_tab.params -side top -anchor nw -expand no -fill x \
		-pady 3

	    set params [$step_tab.params childsite]

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

	    trace variable $mods(LevelSet)-lower_threshold w "$this update_seed_binary_threshold"
	    trace variable $mods(LevelSet)-upper_threshold w "$this update_seed_binary_threshold"

	    # thresholds
	    frame $params.lthresh
	    pack $params.lthresh -side top -anchor nw -expand yes -fill x
	    label $params.lthresh.l -text "Lower Threshold"
	    scale $params.lthresh.s -variable $mods(LevelSet)-lower_threshold \
		-from 0 -to 255 -width 15 \
		-showvalue false -length 150 \
		-orient horizontal
	    entry $params.lthresh.e -textvariable $mods(LevelSet)-lower_threshold \
		-width 6
	    pack $params.lthresh.l $params.lthresh.s $params.lthresh.e -side left -pady 2

	    frame $params.uthresh
	    pack $params.uthresh -side top -anchor nw -expand yes -fill x
	    label $params.uthresh.l -text "Upper Threshold"
	    scale $params.uthresh.s -variable $mods(LevelSet)-upper_threshold \
		-from 0 -to 255 -width 15 \
		-showvalue false -length 150 \
		-orient horizontal
	    entry $params.uthresh.e -textvariable $mods(LevelSet)-upper_threshold \
		-width 6
	    pack $params.uthresh.l $params.uthresh.s $params.uthresh.e -side left -pady 2

	    # Equation Term Weights
	    iwidgets::labeledframe $params.terms \
		-labeltext "Equation Term Weights" \
		-labelpos nw
	    pack $params.terms -side top -anchor n -padx 3
	    
	    set terms [$params.terms childsite]
	    frame $terms.scaling1
	    pack $terms.scaling1 -side top -anchor nw \
		-padx 3 -pady 1
	    make_entry $terms.scaling1.curv "Curvature" $mods(LevelSet)-curvature_scaling
	    make_entry $terms.scaling1.prop "Propagation" $mods(LevelSet)-propagation_scaling

	    pack $terms.scaling1.curv $terms.scaling1.prop \
		-side left -anchor ne \
		-padx 3 -pady 1

	    make_entry $terms.edge "Edge Weight (Laplacian)" $mods(LevelSet)-edge_weight
	    checkbutton $terms.exp -text "Reverse Expansion Direction" \
		-variable $mods(LevelSet)-reverse_expansion_direction

	    pack $terms.edge $terms.exp -side top -anchor nw \
		-padx 3 -pady 1


	    button $params.button -text "Update Speed Image" \
		-background $execute_color \
		-activebackground $execute_active_color \
		-command "$this update_speed_image" \
		-width 20
	    pack $params.button -side top -anchor n -padx 3 -pady 3 -ipadx 2


	    # Seeding parameters
	    iwidgets::labeledframe $step_tab.seeds \
		-labeltext "Initial Segmentation" \
		-labelpos nw
	    pack $step_tab.seeds -side top -anchor nw -expand no -fill x \
		-pady 3

	    set seeds [$step_tab.seeds childsite]
	    

	    global seed_method
	    global $mods(SampleField-Seeds)-num_seeds
	    global $mods(SampleField-SeedsNeg)-num_seeds
	    trace variable $mods(SampleField-Seeds)-num_seeds w "$this seeds_changed"
	    trace variable $mods(SampleField-SeedsNeg)-num_seeds w "$this seeds_changed"

	    frame $seeds.options
	    frame $seeds.options.a
	    frame $seeds.options.b
	    pack $seeds.options -side top -anchor n

	    pack $seeds.options.a $seeds.options.b -side left -anchor nw

	    # Previous
	    radiobutton $seeds.options.a.prev \
		-text "Use Previous Segmentation" \
		-variable seed_method -value "prev" \
		-command "$this change_seed_method"

	    # Current
	    radiobutton $seeds.options.a.curr \
		-text "Use Current Segmentation" \
		-variable seed_method -value "curr" \
		-command "$this change_seed_method"

	    # Current
	    radiobutton $seeds.options.a.next \
		-text "Use Next Segmentation" \
		-variable seed_method -value "next" \
		-command "$this change_seed_method"

	    # Thresholds
	    radiobutton $seeds.options.b.thresh \
		-text "Use Thresholds" \
		-variable seed_method -value "thresh" \
		-command "$this change_seed_method"

	    # Seeds
	    radiobutton $seeds.options.b.point -text "Use Seed Points Only" \
		-variable seed_method -value "points" \
		-command "$this change_seed_method"

	    pack $seeds.options.a.prev $seeds.options.a.curr \
		$seeds.options.a.next -side top -anchor nw

	    pack $seeds.options.b.thresh $seeds.options.b.point \
		-side top -anchor nw


	    frame $seeds.points -relief groove -borderwidth 2

	    frame $seeds.points.pos
	    set f $seeds.points.pos
	    label $f.l -text "Positive Seed Points: " \
		-foreground "#990000"
	    button $f.decr -text "-" -command "$this change_number_of_seeds + -"
	    entry $f.e -textvariable $mods(SampleField-Seeds)-num_seeds \
		-width 4 -foreground "#990000"
	    button $f.incr -text "+" -command "$this change_number_of_seeds + +"
	    bind $f.e <Return> "$this change_number_of_seeds + ="

	    pack $f.l $f.decr $f.e $f.incr -side left -anchor nw -expand yes -fill x

	    frame $seeds.points.neg
	    set f $seeds.points.neg
	    label $f.l -text "Negative Seed Points: " \
		-foreground "blue"
	    button $f.decr -text "-" -command "$this change_number_of_seeds - -"
	    entry $f.e -textvariable $mods(SampleField-SeedsNeg)-num_seeds \
		-width 4 -foreground "blue"
	    button $f.incr -text "+" -command "$this change_number_of_seeds - +"
	    bind $f.e <Return> "$this change_number_of_seeds - ="

	    pack $f.l $f.decr $f.e $f.incr -side left -anchor nw -expand yes -fill x

	    global show_seeds
	    checkbutton $seeds.points.toggle -text "Show Seeds" \
		-variable show_seeds -command "$this seeds_changed 1 2 3"

	    pack $seeds.points.pos $seeds.points.neg \
		-side top -anchor ne
	    pack $seeds.points.toggle -side top -anchor n

	    button $seeds.generate -text "Update Initial Segmentation" \
		-background $execute_color \
		-activebackground $execute_active_color \
		-command "$this create_seeds"


	    pack $seeds.points $seeds.generate -side top -anchor n -pady 3 -ipadx 2

	    # Segment frame

	    iwidgets::labeledframe $step_tab.segment \
		-labeltext "Segment" \
		-labelpos nw
	    pack $step_tab.segment -side top -anchor nw \
		-expand no -fill x
	    set segment [$step_tab.segment childsite]

	    global max_iter
	    global $mods(LevelSet)-max_rms_change

	    frame $segment.params
	    make_entry $segment.params.iter "Maximum Iterations:" max_iter 5
	    make_entry $segment.params.rms "Maximum RMS:" $mods(LevelSet)-max_rms_change 5
	    pack $segment.params.iter $segment.params.rms \
		-side left -anchor nw
	    frame $segment.buttons
	    button $segment.buttons.reset -text "Reset" \
		-background $execute_color \
		-activebackground $execute_active_color \
		-command "$this start_segmentation Reset"
	    button $segment.buttons.go -text "Go" \
		-background $execute_color \
		-activebackground $execute_active_color \
		-command "$this start_segmentation Go"
	    button $segment.buttons.stop -text "Stop" \
		-background "#990000" \
		-activebackground "#CC0000" \
		-command "$this stop_segmentation"
	    button $segment.buttons.commit -text "Commit" \
		-activebackground $next_color \
		-background $next_color \
		-command "$this commit_segmentation"
	    pack $segment.buttons.reset $segment.buttons.go \
		$segment.buttons.stop $segment.buttons.commit \
		-side left -anchor n -padx 4 -pady 3 -expand yes \
		-ipadx 2

	    pack $segment.params -side top -anchor nw 
	    pack $segment.buttons -side top -anchor n -expand yes -fill x
	    

	    button $step_tab.volren -text "Update Volume Rendering" \
		-state disabled \
		-command "$this update_volume_rendering"
	    pack $step_tab.volren -side top -anchor n -pady 3

# 	    # Radiobuttons for what is volume rendered
 	    frame $step_tab.whichvol 
 	    pack $step_tab.whichvol -side top -anchor n -pady 3
 	    global vol_foreground
 	    radiobutton $step_tab.whichvol.a -text "Show Segmentation" \
 		-variable vol_foreground -value 1 \
 		-command "$this toggle_volume_render_object"
 	    radiobutton $step_tab.whichvol.b -text "Show Background" \
 		-variable vol_foreground -value 0 \
 		-command "$this toggle_volume_render_object"
 	    pack $step_tab.whichvol.a $step_tab.whichvol.b -side left \
 		-anchor nw -pady 3

	    frame $step_tab.savebin 
	    frame $step_tab.savefl
	    pack $step_tab.savebin $step_tab.savefl -side top -anchor n \
		-pady 2
	    
	    global $mods(ImageFileWriter-Binary)-filename
	    button $step_tab.savebin.btn -text "Save Binary" \
		-command "$this save_binary" -state disabled
	    label $step_tab.savebin.l -text "File:"
	    entry $step_tab.savebin.e \
		-textvariable $mods(ImageFileWriter-Binary)-filename
	    button $step_tab.savebin.browse -text "Browse" \
		-command "$this open_save_binary_ui"
	    pack $step_tab.savebin.btn $step_tab.savebin.l \
		$step_tab.savebin.e $step_tab.savebin.browse -side left

	    global $mods(ImageFileWriter-Float)-filename
	    button $step_tab.savefl.btn -text " Save Float " \
		-command "$this save_float" -state disabled
	    label $step_tab.savefl.l -text "File:"
	    entry $step_tab.savefl.e \
		-textvariable $mods(ImageFileWriter-Float)-filename
	    button $step_tab.savefl.browse -text "Browse" \
		-command "$this open_save_float_ui"
	    pack $step_tab.savefl.btn $step_tab.savefl.l \
		$step_tab.savefl.e $step_tab.savefl.browse -side left
	    
	    
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
	    
            $process.tnb view "Load"

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

	$w.topbot fraction 50 50

	# top
	iwidgets::panedwindow $top.lmr -orient vertical -thickness 0 \
	    -sashheight 5000 -sashwidth 6 -sashindent 0 -sashborderwidth 2 \
	    -sashcursor sb_h_double_arrow

	$top.lmr add left -margin 3 -minimum 0
	$top.lmr add right -margin 3 -minimum 0

	set topl [$top.lmr childsite left]
	set topr [$top.lmr childsite right]

	pack $top.lmr -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0

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

	# embed viewer1 on top right and left
	global mods

 	set eviewer [$mods(Viewer) ui_embedded]
 	$eviewer setWindow $topl $viewer_width \
 	    [expr $viewer_height/2] \

 	pack $topl -side top -anchor n \
 	    -expand 1 -fill both -padx 4 -pady 0

 	set eviewer2 [$mods(Viewer) ui_embedded]
 	$eviewer2 setWindow $topr [expr $viewer_width/3] \
 	    [expr $viewer_height/2] \

 	pack $topr -side top -anchor n \
 	    -expand 1 -fill both -padx 4 -pady 0

	# embed volume rendering viewer in bottom right
 	set eviewer3 [$mods(Viewer-VolRen) ui_embedded]
 	$eviewer3 setWindow $botr $viewer_width \
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

	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane1.childsite \
	    <ButtonPress-2> "$this do_nothing"
	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane1.childsite \
	    <ButtonRelease-2> "$this do_nothing"
	bind .standalone.viewers.topbot.pane0.childsite.lmr.pane1.childsite \
	    <Button2-Motion> "$this do_nothing"

    }

    method create_2d_frame { window axis } {
	global mods 

	# Modes for $axis
	frame $window.modes
	pack $window.modes -side bottom -padx 0 -pady 0 -expand 0 -fill x
	
	frame $window.modes.slider
	pack $window.modes.slider \
	    -side top -pady 0 -anchor n -expand yes -fill x
	
	# Initialize with slice scale visible
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
	    -command "$this hide_control_panel $window.modes"
	pack $window.modes.expand -side bottom -fill both
    }

    method show_control_panel { w } {
	pack forget $w.expand
	pack $w.slider -side top -pady 0 -anchor nw -expand yes -fill x
	pack $w.expand -side bottom -fill both

	$w.expand configure -command "$this hide_control_panel $w" \
	    -cursor based_arrow_down
    }

    method hide_control_panel { w } {
	pack forget $w.slider
	pack $w.expand -side bottom -fill both

	$w.expand configure -command "$this show_control_panel $w" \
	    -cursor based_arrow_up
    }

    method change_processing_tab {which} {
	if {$initialized} {
	    if {$which == "Segment" && !$loading} {
		if {$has_smoothed == 0} {
		    # $this smooth_data
		}
		# get speed image
		$this initialize_segmentation
	    }
	    
	    $proc_tab1 view $which
	    $proc_tab2 view $which
	    set curr_proc_tab $which
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
	
	if {$which == $mods(PasteImageFilter-Smooth) \
		&& $state == "Completed"} {

	    global axis

	    # determine if this was the last run
	    set last 0
	    if {$axis == 0} {
		set last [expr $size0 - 1]
	    } elseif {$axis == 1} {
		set last [expr $size1 - 1]
	    } else {
		set last [expr $size2 - 1]
	    }
	    
	    global $mods(PasteImageFilter-Smooth)-index
	    if {[set $mods(PasteImageFilter-Smooth)-index] == $last} {
		# set 2D viewer to use smoothed data as input
		global $mods(ChooseNrrd-2D)-port-index
		set $mods(ChooseNrrd-2D)-port-index 1
		
		set which [$filter_menu1 get]
		set has_smoothed 1
		set region_changed 0
		set smoothing 0

 		# enable modules downstream of smoothing and execute
 		# them
###  		disableModule $mods(ImageToNrrd-ViewSlices) 0
		after 500 "$mods(ImageToNrrd-ViewSlices)-c needexecute"
	    } else {
		# increment Extract values and Paste index
		if {$axis == 0} {
		    global $mods(Extract-Smooth)-minDim0
		    global $mods(Extract-Smooth)-maxDim0

		    set prev [expr [set $mods(Extract-Smooth)-minDim0] + 1]
		    set $mods(Extract-Smooth)-minDim0 $prev
		    set $mods(Extract-Smooth)-maxDim0 [expr $prev + 1]
		} elseif {$axis == 1} {
		    global $mods(Extract-Smooth)-minDim1
		    global $mods(Extract-Smooth)-maxDim1

		    set prev [expr [set $mods(Extract-Smooth)-minDim1] + 1]
		    set $mods(Extract-Smooth)-minDim1 $prev
		    set $mods(Extract-Smooth)-maxDim1 [expr $prev + 1]
		} else {
		    global $mods(Extract-Smooth)-minDim2
		    global $mods(Extract-Smooth)-maxDim2

		    set prev [expr [set $mods(Extract-Smooth)-minDim2] + 1]
		    set $mods(Extract-Smooth)-minDim2 $prev
		    set $mods(Extract-Smooth)-maxDim2 [expr $prev + 1]
		}
		
		global $mods(PasteImageFilter-Smooth)-index
		set $mods(PasteImageFilter-Smooth)-index \
		    [expr [set $mods(PasteImageFilter-Smooth)-index] + 1]

		# Execute Extract Again
		after 500 "$mods(Extract-Smooth)-c needexecute"
	    }
	} elseif {$which == $mods(ChooseImage-Hack) && \
		      $state == "Completed"} {	    
	    if {$smoothing_type == "Reset"} {
		disableModule $mods(ChooseImage-SmoothInput) 1
	    } else {
		# Data has gotten to Choose module so disable it
		# and then enable Extract and execute it
		disableModule $mods(ChooseImage-SmoothInput) 1
		disableModule $mods(Extract-Smooth) 0
		
		$mods(Extract-Smooth)-c needexecute
	    }
	} elseif {$which == $mods(ChooseImage-Hack2) && \
		      $state == "Completed"} {	    
	    if {$segmenting_type == "Reset"} {
		disableModule $mods(ChooseImage-SegInput) 1
	    } else {
		# Data has gotten to Choose module so disable it
		# and then enable next Choose and execute it
		disableModule $mods(ChooseImage-SegInput) 1
		disableModule $mods(ChooseImage-Hack3) 0
		
		$mods(ChooseImage-Hack3)-c needexecute
	    }
	} elseif {$which == $mods(ChooseNrrd-Reader) && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Loading Volume..."
	} elseif {$which == $mods(ChooseNrrd-Reader) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ScalarFieldStats) && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Building Histogram..."
	} elseif {$which == $mods(ScalarFieldStats) && $state == "Completed"} {
	    change_indicate_val 2
	    set has_loaded 1
	    change_indicator_labels "Done Loading Volume"
	} elseif {$which == $mods(NrrdInfo-Reader) && $state == "Completed"} {
	    global $which-dimension

	    if {[set $which-dimension] != 3} {
		tk_messageBox -message "Data must be 3 dimensional scalar data." -type ok -icon info -parent .standalone
		return
	    }

	    global $which-size0 $which-size1 $which-size2
	    set size0 [set $which-size0]
	    set size1 [set $which-size1]
	    set size2 [set $which-size2]

	    set orig_size0 $size0
	    set orig_size1 $size1
	    set orig_size2 $size2

	    # Fix initial crop values
	    if {!$loading} {
		foreach i {0 1 2} {
		    global $mods(UnuCrop)-minAxis$i $mods(UnuCrop)-maxAxis$i
		    global $mods(ViewSlices)-crop_minAxis$i 
		    global $mods(ViewSlices)-crop_maxAxis$i
		    set $mods(UnuCrop)-minAxis$i 0
		    set $mods(ViewSlices)-crop_minAxis$i 0
		    set $mods(UnuCrop)-maxAxis$i [expr [set size$i]-1]
		    set $mods(ViewSlices)-crop_maxAxis$i [expr [set size$i]-1]
		}
	    }

	    configure_slice_sliders

	    # update samples
	    set path f.p.childsite.tnb.canvas.notebook.cs.page1.cs.stats.childsite
	    $attachedPFr.$path.samples configure -text "Samples: $orig_size0, $orig_size1, $orig_size2"
	    $detachedPFr.$path.samples configure -text "Samples: $orig_size0, $orig_size1, $orig_size2"
	} elseif {$which == $mods(NrrdInfo-Size) && $state == "Completed"} {
	    global $which-dimension

	    global $which-size0 $which-size1 $which-size2
	    set size0 [set $which-size0]
	    set size1 [set $which-size1]
	    set size2 [set $which-size2]

	    configure_slice_sliders

	    # Set Smoothing Paste and Extract values
	    global axis
	    global $mods(PasteImageFilter-Smooth)-size0
	    global $mods(PasteImageFilter-Smooth)-size1
	    global $mods(PasteImageFilter-Smooth)-size2
	    global $mods(PasteImageFilter-Smooth)-index
	    global $mods(PasteImageFilter-Smooth)-axis
	    global $mods(PasteImageFilter-Smooth)-fill_value
#  	    set $mods(PasteImageFilter-Smooth)-size0 [expr $size0 - 1]
#  	    set $mods(PasteImageFilter-Smooth)-size1 [expr $size1 - 1]
#  	    set $mods(PasteImageFilter-Smooth)-size2 [expr $size2 - 1]
 	    set $mods(PasteImageFilter-Smooth)-size0 $size0
 	    set $mods(PasteImageFilter-Smooth)-size1 $size1
 	    set $mods(PasteImageFilter-Smooth)-size2 $size2
	    set $mods(PasteImageFilter-Smooth)-index 0
	    set $mods(PasteImageFilter-Smooth)-axis $axis
	    set $mods(PasteImageFilter-Smooth)-fill_value 0

	    global $mods(Extract-Smooth)-minDim0
	    global $mods(Extract-Smooth)-minDim1
	    global $mods(Extract-Smooth)-minDim2
	    global $mods(Extract-Smooth)-maxDim0
	    global $mods(Extract-Smooth)-maxDim1
	    global $mods(Extract-Smooth)-maxDim2
	    
	    set $mods(Extract-Smooth)-minDim0 0
	    set $mods(Extract-Smooth)-minDim1 0
	    set $mods(Extract-Smooth)-minDim2 0
# 	    set $mods(Extract-Smooth)-maxDim0 [expr $size0 - 1]
# 	    set $mods(Extract-Smooth)-maxDim1 [expr $size1 - 1]
# 	    set $mods(Extract-Smooth)-maxDim2 [expr $size2 - 1]
	    set $mods(Extract-Smooth)-maxDim0 $size0
	    set $mods(Extract-Smooth)-maxDim1 $size1
	    set $mods(Extract-Smooth)-maxDim2 $size2

	    # Initialize segs array
	    set extent 0
	    set scale 0
	    if {$axis == 0} {
		set extent $size0
		set scale $size2
		set $mods(Extract-Smooth)-minDim0 0
		set $mods(Extract-Smooth)-maxDim0 1
	    } elseif {$axis == 1} {
		set extent $size1
		set scale $size1
		set $mods(Extract-Smooth)-minDim1 0
		set $mods(Extract-Smooth)-maxDim1 1
	    } else {
		set extent $size2
		set scale $size0
		set $mods(Extract-Smooth)-minDim2 0
		set $mods(Extract-Smooth)-maxDim2 1
	    }

	    for {set i 0} {$i < $extent} {incr i} {
		set segs($i) 0
	    }

	    # Re-configure slice indicator
	    $this change_slice_icon 0
	} elseif {$which == $mods(UnuMinMax-Reader) && $state == "Completed"} {
	    global $which-min0 $which-max0
	    set range_min [set $which-min0]
	    set range_max [set $which-max0]

	    $this configure_threshold_sliders

	    # update range
	    set path f.p.childsite.tnb.canvas.notebook.cs.page1.cs.stats.childsite
	    $attachedPFr.$path.range configure -text "Data Range: $range_min - $range_max"
	    $detachedPFr.$path.range configure -text "Data Range: $range_min - $range_max"
	} elseif {$which == $mods(UnuMinMax-Size) && $state == "Completed"} {
	    global $which-min0 $which-max0
	    set range_min [set $which-min0]
	    set range_max [set $which-max0]

	    $this configure_threshold_sliders
	} elseif {$which == $mods(UnuMinMax-Smoothed) && $state == "Completed"} {
	    global $which-min0 $which-max0
	    set range_min [set $which-min0]
	    set range_max [set $which-max0]

	    $this configure_threshold_sliders
	} elseif {$which == $mods(ViewSlices) && $state == "Completed"} {
            if {$2D_fixed == 0} {
		# setup correct axis
		global $mods(ViewSlices)-slice-viewport0-axis axis

		set $mods(ViewSlices)-slice-viewport0-axis $axis

		# fix window width and level
		upvar \#0 $mods(ViewSlices)-min val_min $mods(ViewSlices)-max val_max
                set ww [expr abs($val_max-$val_min)]
                set wl [expr ($val_min+$val_max)/2.0]

		setGlobal $mods(ViewSlices)-clut_ww $ww
		setGlobal $mods(ViewSlices)-clut_wl $wl

                $mods(ViewSlices)-c rebind $slice_frame

		$mods(ViewSlices)-c setclut

                set 2D_fixed 1
	    } 
	} elseif {$which == $mods(Smooth-Gradient) && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Peforming GradientAnisotropicDiffusion Smoothing..."
	} elseif {$which == $mods(Smooth-Gradient) && $state == "Completed"} { 
	    change_indicate_val 2
	    change_indicator_labels "Done Performing GradientAnisotropicDiffusion Smoothing"
	} elseif {$which == $mods(Smooth-Curvature) && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Peforming CurvatureAnisotropicDiffusion Smoothing..."
	} elseif {$which == $mods(Smooth-Curvature) && $state == "Completed"} { 
	    change_indicate_val 2
	    change_indicator_labels "Done Performing CurvatureAnisotropicDiffusion Smoothing"
	} elseif {$which == $mods(Smooth-Blur) && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Peforming Gaussian Blurring..."
	} elseif {$which == $mods(Smooth-Blur) && $state == "Completed"} { 
	    change_indicate_val 2
	    change_indicator_labels "Done Performing Gaussian Blurring"
	} elseif {$which == $mods(ShowField-Feature) && $state == "JustStarted"} { 
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Feature) && $state == "Completed"} { 
	    change_indicate_val 2
	    # Setup views of 2 View Windows
	    global axis
	    global $mods(Viewer)-ViewWindow_0-pos
	    global $mods(Viewer)-ViewWindow_1-pos
	    set $mods(Viewer)-ViewWindow_0-pos "z1_y1"
	    set $mods(Viewer)-ViewWindow_1-pos "z1_y1"

# 	    if {$axis == 0} {
# 		set $mods(Viewer)-ViewWindow_0-pos "x0_y0"
# 		set $mods(Viewer)-ViewWindow_1-pos "x0_y0"
# 	    } elseif {$axis == 1} {
# 		set $mods(Viewer)-ViewWindow_0-pos "y0_x0"
# 		set $mods(Viewer)-ViewWindow_1-pos "y0_x0"
# 	    } else {
# 		set $mods(Viewer)-ViewWindow_0-pos "z1_y1"
# 		set $mods(Viewer)-ViewWindow_1-pos "z1_y1"
# 	    }

	    after 100 "$mods(Viewer)-ViewWindow_0-c autoview; $mods(Viewer)-ViewWindow_1-c autoview; $mods(Viewer)-ViewWindow_0-c Views; $mods(Viewer)-ViewWindow_1-c Views"
	} elseif {$which == $mods(ShowField-Seg) && $state == "JustStarted"} { 
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Seg) && $state == "Completed"} {
	    change_indicate_val 2
	    # Turn off Current Segmentation in ViewWindow 1
	    after 100 \
		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_1-Transparent Faces (6)\}\" 0; $mods(Viewer)-ViewWindow_1-c redraw"
 	} elseif {$which == $mods(ShowField-Speed) && $state == "JustStarted"} { 
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Speed) && $state == "Completed"} {
	    change_indicate_val 2
	    # Turn off Speed Image in ViewWindow 0
	    after 100 \
		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (5)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	} elseif {$which == $mods(LevelSet) && $state == "JustStarted"} { 
	    if {$updating_speed == 1} {
		change_indicator_labels "Updating Speed Image..."
	    } elseif {$segmenting == 1} {
		change_indicator_labels "Performing LevelSet Segmentation..."
	    } else {
		change_indicator_labels "Generating Seeds..."
	    }
	    change_indicate_val 1
	} elseif {$which == $mods(LevelSet) && $state == "Completed"} { 
	    if {$updating_speed == 1} {
		change_indicator_labels "Done Updating Speed Image"
	    } elseif {$segmenting == 1} {
		change_indicator_labels "Done Performing LevelSet Segmentation"
		set has_segmented 1
	    } 
	    change_indicate_val 2
	    set updating_speed 0
	    set segmenting 0

	    after 500 "set $mods(LevelSet)-max_iterations 0"
	} elseif {$which == $mods(PasteImageFilter-Binary) && $state == "Completed"} { 
	    if {$pasting_binary == 1} {
#		disableModule $mods(Image2DTo3D-Binary) 1
		disableModule $mods(PasteImageFilter-Binary) 1

		global slice
		set segs($slice) 2
		change_slice_icon 2
	    }
	    set pasting_binary 0
	} elseif {$which == $mods(PasteImageFilter-Float) && $state == "Completed"} { 
	    if {$pasting_float == 1} {
#		disableModule $mods(Image2DTo3D-Float) 1
		disableModule $mods(PasteImageFilter-Float) 1
	    }
	    set pasting_float 0
	} elseif {$which == $mods(VolumeVisualizer) && $state == "JustStarted"} { 
	    change_indicate_val 1
	} elseif {$which == $mods(VolumeVisualizer) && $state == "Completed"} { 
	    if {$volren_has_autoviewed == 0} {
		set volren_has_autoviewed 1
		after 500 "$mods(Viewer-VolRen)-ViewWindow_0-c autoview; $mods(Viewer-VolRen)-ViewWindow_0-c redraw"		
	    }
	    change_indicate_val 2
	    change_indicator_labels "Done Updating Volume Rendering"
	} elseif {$which == $mods(ImageFileWriter-Binary) && $state == "JustStarted"} { 
	    change_indicate_val 1
	    change_indicator_labels "Writing out binary segmentation..."
	} elseif {$which == $mods(ImageFileWriter-Binary) && $state == "Completed"} { 
	    change_indicate_val 2
	    change_indicator_labels "Done writing out binary segmentation"

	    # disable writer
	    disableModule $mods(ImageFileWriter-Binary) 1
	} elseif {$which == $mods(ImageFileWriter-Float) && $state == "JustStarted"} { 
	    change_indicate_val 1
	    change_indicator_labels "Writing out float segmentation..."
	} elseif {$which == $mods(ImageFileWriter-Float) && $state == "Completed"} { 
	    change_indicate_val 2
	    change_indicator_labels "Done writing out float segmentation"

	    # disable writer
	    disableModule $mods(ImageFileWriter-Float) 1
	} elseif {$which == $mods(UnuMinmax-Vol) && $state == "Completed"} { 
	    global $mods(UnuJhisto-Vol)-mins $mods(UnuJhisto-Vol)-maxs
	    global $mods(RescaleColorMap-Vol)-min $mods(RescaleColorMap-Vol)-max
	    global $mods(NrrdSetupTexture-Vol)-minf $mods(NrrdSetupTexture-Vol)-maxf
	    global $mods(UnuMinmax-Vol)-min0 $mods(UnuMinmax-Vol)-max0

	    # Change UnuJhisto, RescaleColorMap, and NrrdSetupTexture values
	    set min [set $mods(UnuMinmax-Vol)-min0]
	    set max [set $mods(UnuMinmax-Vol)-max0]

	    set ww [expr abs($max-$min)]
	    set wl [expr ($min+$max)/2.0]

	    set minv [expr $wl-$ww/2.0]
	    set maxv [expr $wl+$ww/2.0]

	    set $mods(UnuJhisto-Vol)-mins "$minv nan"
	    set $mods(UnuJhisto-Vol)-maxs "$maxv nan"
	    set $mods(RescaleColorMap-Vol)-min $minv
	    set $mods(RescaleColorMap-Vol)-max $maxv
	    set $mods(NrrdSetupTexture-Vol)-minf $minv
	    set $mods(NrrdSetupTexture-Vol)-maxf $maxv


	    # now enable volume rendering and execute them - also
	    # re-disale ImageToNrrd module
	    disableModule $mods(NrrdSetupTexture-Vol) 0
	    
	    $mods(NrrdSetupTexture-Vol)-c needexecute

	    after 500 "disableModule $mods(ImageToNrrd-Vol) 1"
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

    ##############################
    ### configure_readers
    ##############################
    # Keeps the readers in sync.  Every time a different
    # data tab is selected (Nrrd, Dicom, Analyze) the other
    # readers must be disabled to avoid errors.
    method configure_readers { which } {
        global mods
        global $mods(ChooseNrrd-Reader)-port-index

	if {$which == "Generic"} {
	    set $mods(ChooseNrrd-Reader)-port-index 0
	    disableModule $mods(NrrdReader) 0
	    disableModule $mods(DicomReader) 1
	    disableModule $mods(AnalyzeReader) 1

	    if {$initialized != 0} {
		$data_tab1 view "Generic"
		$data_tab2 view "Generic"
		set curr_data_tab "Generic"
	    }
        } elseif {$which == "Dicom"} {
	    set $mods(ChooseNrrd-Reader)-port-index 1

	    disableModule $mods(NrrdReader) 1
	    disableModule $mods(DicomReader) 0
	    disableModule $mods(AnalyzeReader) 1

            if {$initialized != 0} {
		$data_tab1 view "Dicom"
		$data_tab2 view "Dicom"
		set curr_data_tab "Dicom"
	    }
        } elseif {$which == "Analyze"} {
	    # Analyze
	    set $mods(ChooseNrrd-Reader)-port-index 2
	    disableModule $mods(NrrdReader) 1
	    disableModule $mods(DicomReader) 1
	    disableModule $mods(AnalyzeReader) 0

	    if {$initialized != 0} {
		$data_tab1 view "Analyze"
		$data_tab2 view "Analyze"
		set curr_data_tab "Analyze"
	    }
        } elseif {$which == "all"} {
	    if {[set $mods(ChooseNrrd-Reader)-port-index] == 0} {
		# nrrd
		disableModule $mods(NrrdReader) 0
		disableModule $mods(DicomReader) 1
		disableModule $mods(AnalyzeReader) 1
	    } elseif {[set $mods(ChooseNrrd-Reader)-port-index] == 1} {
		# dicom
		disableModule $mods(NrrdReader) 1
		disableModule $mods(DicomReader) 0
		disableModule $mods(AnalyzeReader) 1
	    } else {
		# analyze
		disableModule $mods(NrrdReader) 1
		disableModule $mods(DicomReader) 1
		disableModule $mods(AnalyzeReader) 0
	    }
	}
    }

    method set_curr_data_tab {which} {
	if {$initialized} {
	    set curr_data_tab $which
	}
    }

    method open_nrrd_reader_ui {} {
	global mods
	$mods(NrrdReader) initialize_ui

	.ui$mods(NrrdReader).f7.execute configure -state disabled

	# rebind execute command to just withdraw
	upvar \#0 .ui$mods(NrrdReader) data	
	set data(-command) "wm withdraw .ui$mods(NrrdReader)"
    }

    method dicom_ui { } {
	global mods
	$mods(DicomReader) initialize_ui

	if {[winfo exists .ui$mods(DicomReader)]} {
	    # disable execute button 
	    .ui$mods(DicomReader).buttonPanel.btnBox.execute configure -state disabled
	}
    }

    method analyze_ui { } {
	global mods
	$mods(AnalyzeReader) initialize_ui
	if {[winfo exists .ui$mods(AnalyzeReader)]} {
	    # disable execute button 
	    .ui$mods(AnalyzeReader).buttonPanel.btnBox.execute configure -state disabled
	}
    }

    method load_data {} {
	global mods
	# execute the appropriate reader

        global $mods(ChooseNrrd-Reader)-port-index
        set port [set $mods(ChooseNrrd-Reader)-port-index]
        set mod ""
        if {$port == 0} {
	    # Nrrd
            set mod $mods(NrrdReader)
	} elseif {$port == 1} {
	    # Dicom
            set mod $mods(DicomReader)
	} else {
	    # Analyze
            set mod $mods(AnalyzeReader)
	} 

	# enable next button
	$attachedPFr.$next_load configure -state normal -activebackground $next_color \
	    -background $next_color 
	$detachedPFr.$next_load configure -state normal -activebackground $next_color \
	    -background $next_color 

	set has_loaded 0
	set 2D_fixed 0
	
	$mod-c needexecute
    }

    ############################
    ### update_histo_graph_callback
    ############################
    # Called when the ScalarFieldStats updates the graph
    # so we can update ours
    method update_histo_graph_callback {varname varele varop} {
	global mods

        global $mods(ScalarFieldStats)-min $mods(ScalarFieldStats)-max

	global $mods(ScalarFieldStats)-args
        global $mods(ScalarFieldStats)-nmin
        global $mods(ScalarFieldStats)-nmax

	set nmin [set $mods(ScalarFieldStats)-nmin]
	set nmax [set $mods(ScalarFieldStats)-nmax]
	set args [set $mods(ScalarFieldStats)-args]

	if {$args == "?"} {
	    return
	}
        
        # for some reason the other graph will only work if I set temp 
        # instead of using the $i value 
 	set graph $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page1.cs.stats.childsite.histo.childsite.graph

         if { ($nmax - $nmin) > 1000 || ($nmax - $nmin) < 1e-3 } {
             $graph axis configure y -logscale yes
         } else {
             $graph axis configure y -logscale no
         }

         set min [set $mods(ScalarFieldStats)-min]
         set max [set $mods(ScalarFieldStats)-max]
         set xvector {}
         set yvector {}
         set yvector [concat $yvector $args]
         set frac [expr double(1.0/[llength $yvector])]

         $graph configure -barwidth $frac
         $graph axis configure x -min $min -max $max \
	    -subdivisions 4 -loose 1 \
	    -stepsize 0

         for {set i 0} { $i < [llength $yvector] } {incr i} {
             set val [expr $min + $i*$frac*($max-$min)]
             lappend xvector $val
         }
        
          if { [$graph element exists data] == 1 } {
              $graph element delete data
          }

	$graph element create data -label {} -xdata $xvector -ydata $yvector
	$graph element configure data -fg blue

	# 	## other window
 	set graph $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page1.cs.stats.childsite.histo.childsite.graph

         if { ($nmax - $nmin) > 1000 || ($nmax - $nmin) < 1e-3 } {
             $graph axis configure y -logscale yes
         } else {
             $graph axis configure y -logscale no
         }

         set min [set $mods(ScalarFieldStats)-min]
         set max [set $mods(ScalarFieldStats)-max]
         set xvector {}
         set yvector {}
         set yvector [concat $yvector $args]
         set frac [expr double(1.0/[llength $yvector])]

         $graph configure -barwidth $frac
         $graph axis configure x -min $min -max $max \
	    -subdivisions 4 -loose 1 \
	    -stepsize 0

         for {set i 0} { $i < [llength $yvector] } {incr i} {
             set val [expr $min + $i*$frac*($max-$min)]
             lappend xvector $val
         }
        
          if { [$graph element exists data] == 1 } {
              $graph element delete data
          }

	$graph element create data -label {} -xdata $xvector -ydata $yvector
	$graph element configure data -fg blue
    }
    
    method select_region_of_interest {} {
	global mods

	# change ChooseNrrd port
	global $mods(ChooseNrrd-Crop)-port-index
	set $mods(ChooseNrrd-Crop)-port-index 1

	# This causes region radiobutton to change
	global smooth_region
	set smooth_region "roi"
	$this change_smooth_region

	# turn off crop widget
	global show_roi
	set show_roi 0
	$this toggle_show_roi

	# set ViewSlices pad values
	global $mods(ViewSlices)-crop_minPadAxis0
	global $mods(ViewSlices)-crop_maxPadAxis0
	global $mods(ViewSlices)-crop_minPadAxis1
	global $mods(ViewSlices)-crop_maxPadAxis1
	global $mods(ViewSlices)-crop_minPadAxis2
	global $mods(ViewSlices)-crop_maxPadAxis2
	
	global $mods(UnuCrop)-minAxis0 $mods(UnuCrop)-maxAxis0
	global $mods(UnuCrop)-minAxis1 $mods(UnuCrop)-maxAxis1
	global $mods(UnuCrop)-minAxis2 $mods(UnuCrop)-maxAxis2

	
	set $mods(ViewSlices)-crop_minPadAxis0 [set $mods(UnuCrop)-minAxis0]
	set $mods(ViewSlices)-crop_maxPadAxis0 \
	    [expr $orig_size0 - [set $mods(UnuCrop)-maxAxis0] - 1]
	set $mods(ViewSlices)-crop_minPadAxis1 [set $mods(UnuCrop)-minAxis1]
	set $mods(ViewSlices)-crop_maxPadAxis1 \
	    [expr $orig_size1 - [set $mods(UnuCrop)-maxAxis1] - 1]
	set $mods(ViewSlices)-crop_minPadAxis2 [set $mods(UnuCrop)-minAxis2]
	set $mods(ViewSlices)-crop_maxPadAxis2 \
	    [expr $orig_size2 - [set $mods(UnuCrop)-maxAxis2] - 1]

	# execute UnuCrop
	$mods(UnuCrop)-c needexecute
    }

    method change_axis {} {
	global axis mods

	# update ViewSlices orientation
	global $mods(ViewSlices)-slice-viewport0-axis
	
	set $mods(ViewSlices)-slice-viewport0-axis $axis

	# update Viewer orientation
# 	global $mods(Viewer)-ViewWindow_0-pos
# 	global $mods(Viewer)-ViewWindow_1-pos
# 	set $mods(Viewer)-ViewWindow_0-pos "z1_y1"
# 	set $mods(Viewer)-ViewWindow_1-pos "z1_y1"
# 	if {$axis == 0} {
# 	    set $mods(Viewer)-ViewWindow_0-pos "x0_y0"
# 	    set $mods(Viewer)-ViewWindow_1-pos "x0_y0"
# 	} elseif {$axis == 1} {
# 	    set $mods(Viewer)-ViewWindow_0-pos "y0_x0"
# 	    set $mods(Viewer)-ViewWindow_1-pos "y0_x0"
# 	} else {
# 	    set $mods(Viewer)-ViewWindow_0-pos "z1_y1"
# 	    set $mods(Viewer)-ViewWindow_1-pos "z1_y1"
# 	}
	

	# re-configure slice slider
	$this configure_slice_sliders

	# Change UnuAxdelete axis
	global $mods(UnuAxdelete-Feature)-axis
	global $mods(UnuAxdelete-Prev)-axis
	global $mods(UnuAxdelete-Smooth)-axis
	set $mods(UnuAxdelete-Feature)-axis $axis
	set $mods(UnuAxdelete-Prev)-axis $axis
	set $mods(UnuAxdelete-Smooth)-axis $axis

	# ExtractImageFilter
	if {$axis == 0} {
	    set $mods(ExtractSlice)-minDim0 0
	    set $mods(ExtractSlice)-maxDim0 1
	} elseif {$axis == 1} {
	    set $mods(ExtractSlice)-minDim1 0
	    set $mods(ExtractSlice)-maxDim1 1
	} else {
	    set $mods(ExtractSlice)-minDim2 0
	    set $mods(ExtractSlice)-maxDim2 1
	}

	# PasteImageFilters
	global $mods(PasteImageFilter-Binary)-axis
	global $mods(PasteImageFilter-Float)-axis
	global $mods(PasteImageFilter-Smooth)-axis
	set $mods(PasteImageFilter-Binary)-axis $axis
	set $mods(PasteImageFilter-Float)-axis $axis
	set $mods(PasteImageFilter-Smooth)-axis $axis

	# execute needed modules
	$mods(ViewSlices)-c rebind $slice_frame

	after 100 "$mods(Viewer)-ViewWindow_0-c autoview; $mods(Viewer)-ViewWindow_1-c autoview; $mods(Viewer)-ViewWindow_0-c Views; $mods(Viewer)-ViewWindow_1-c Views"
    }

    method configure_slice_sliders {} {
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

	# configure current slice spinint on Segment tab
	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.slice.sp configure \
	    -range "0 $max"

	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.slice.sp configure \
	    -range "0 $max"
    }

    method change_smooth_region {} {
	global smooth_region mods
	
	# Change ChooseNrrd to send down desired region
	global $mods(ChooseNrrd-Crop)-port-index
	if {$smooth_region == "roi"} {
	    set $mods(ChooseNrrd-Crop)-port-index 1
	} else {
	    set $mods(ChooseNrrd-Crop)-port-index 0
	}
	set region_changed 1
    }

    method change_filter {w} {
	global mods
	set which [$w get]

	# change attached/detached menu
	$filter_menu1 select $which
	$filter_menu2 select $which
	
	# enable/disable appropriate modules
	# change ChooseImage port
	# pack/forget appropriate ui

	global $mods(ChooseImage-Smooth)-port-index
	global $mods(ChooseImage-ToSmooth)-port-index
	set $mods(ChooseImage-ToSmooth)-port-index 1
	if {$which == "GradientAnisotropicDiffusion"} {
	    set smoothing_method "GradientAnisotropicDiffusion"
	    disableModule $mods(Smooth-Gradient) 0
	    disableModule $mods(Smooth-Curvature) 1
	    disableModule $mods(Smooth-Blur) 1

	    set $mods(ChooseImage-Smooth)-port-index 0

	    pack forget $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.curvature
	    pack forget $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.curvature

	    pack forget $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.blur
	    pack forget $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.blur

	    pack $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.gradient -side top -anchor n -pady 4

	    pack $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.gradient -side top -anchor n -pady 4
	} elseif {$which == "CurvatureAnisotropicDiffusion"} {
	    set smoothing_method "CurvatureAnisotropicDiffusion"
	    disableModule $mods(Smooth-Gradient) 1
	    disableModule $mods(Smooth-Curvature) 0
	    disableModule $mods(Smooth-Blur) 1

	    set $mods(ChooseImage-Smooth)-port-index 1

	    pack forget $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.gradient
	    pack forget $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.gradient

	    pack forget $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.blur
	    pack forget $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.blur

	    pack $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.curvature -side top -anchor n -pady 4

	    pack $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.curvature -side top -anchor n -pady 4
	} elseif {$which == "Gaussian"} {
	    set smoothing_method "Gaussian"
	    disableModule $mods(Smooth-Gradient) 1
	    disableModule $mods(Smooth-Curvature) 1
	    disableModule $mods(Smooth-Blur) 0

	    set $mods(ChooseImage-Smooth)-port-index 2

	    pack forget $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.gradient
	    pack forget $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.gradient

	    pack forget $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.curvature
	    pack forget $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.curvature

	    pack $attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.blur -side top -anchor n -pady 4

	    pack $detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page2.cs.smooth.childsite.blur -side top -anchor n -pady 4
	} else {
	    set smoothing_method "None"
	    # no smoothing
	    disableModule $mods(Smooth-Gradient) 1
	    disableModule $mods(Smooth-Curvature) 1
	    disableModule $mods(Smooth-Blur) 1

	    # Doesn't matter
	    set $mods(ChooseImage-Smooth)-port-index 0

	    # Change ToSmooth
	    set $mods(ChooseImage-ToSmooth)-port-index 0

	}

#	set filter_enabled 1
    }

    method smooth_data {type} {
	global mods

	global $mods(ChooseImage-SmoothInput)-port-index
	if {$type == "Go" && $has_smoothed == 1} {
	    set $mods(ChooseImage-SmoothInput)-port-index 1
	    set type "Go"
	    set smoothing_type "Go"
	} else {
	    set $mods(ChooseImage-SmoothInput)-port-index 0
	    set type "Reset"
	    set smoothing_type "Reset"
	}

	# Reset Paste and Extract values
	global axis

	global $mods(PasteImageFilter-Smooth)-size0
	global $mods(PasteImageFilter-Smooth)-size1
	global $mods(PasteImageFilter-Smooth)-size2
	global $mods(PasteImageFilter-Smooth)-index
	global $mods(PasteImageFilter-Smooth)-axis
	global $mods(PasteImageFilter-Smooth)-fill_value
# 	set $mods(PasteImageFilter-Smooth)-size0 [expr $size0 - 1]
# 	set $mods(PasteImageFilter-Smooth)-size1 [expr $size1 - 1]
# 	set $mods(PasteImageFilter-Smooth)-size2 [expr $size2 - 1]
	set $mods(PasteImageFilter-Smooth)-size0 $size0
	set $mods(PasteImageFilter-Smooth)-size1 $size1
	set $mods(PasteImageFilter-Smooth)-size2 $size2
	set $mods(PasteImageFilter-Smooth)-index 0
	set $mods(PasteImageFilter-Smooth)-axis $axis
	set $mods(PasteImageFilter-Smooth)-fill_value 0

	global $mods(Extract-Smooth)-minDim0
	global $mods(Extract-Smooth)-minDim1
	global $mods(Extract-Smooth)-minDim2
	global $mods(Extract-Smooth)-maxDim0
	global $mods(Extract-Smooth)-maxDim1
	global $mods(Extract-Smooth)-maxDim2
	
	set $mods(Extract-Smooth)-minDim0 0
	set $mods(Extract-Smooth)-minDim1 0
	set $mods(Extract-Smooth)-minDim2 0
# 	set $mods(Extract-Smooth)-maxDim0 [expr $size0 - 1]
# 	set $mods(Extract-Smooth)-maxDim1 [expr $size1 - 1]
# 	set $mods(Extract-Smooth)-maxDim2 [expr $size2 - 1]
	set $mods(Extract-Smooth)-maxDim0 $size0
	set $mods(Extract-Smooth)-maxDim1 $size1
	set $mods(Extract-Smooth)-maxDim2 $size2

	if {$axis == 0} {
	    set $mods(Extract-Smooth)-minDim0 0
	    set $mods(Extract-Smooth)-maxDim0 1
	} elseif {$axis == 1} {
	    set $mods(Extract-Smooth)-minDim1 0
	    set $mods(Extract-Smooth)-maxDim1 1
	} else {
	    set $mods(Extract-Smooth)-minDim2 0
	    set $mods(Extract-Smooth)-maxDim2 1
	}

	# Make sure a filter has been selected
	if {[string equal $smoothing_method "None"] == 0} {

	    if {$type == "Reset"} {
		# Enable extract and paste module
		disableModule $mods(Extract-Smooth) 0
		disableModule $mods(PasteImageFilter-Smooth) 0
		
		# disable downstream modules so loop of smoothing
		# can execute
		disableModule $mods(ChooseImage-ToSmooth) 1
		disableModule $mods(ImageToNrrd-ViewSlices) 0
		
		# execute
		set smoothing 1
		disableModule $mods(ChooseImage-SmoothInput) 0
		
		after 500 "$mods(ChooseImage-SmoothInput)-c needexecute"
	    } else {
		disableModule $mods(PasteImageFilter-Smooth) 0

		# We have to get previously blurred data to
		# Choose module before enabling downstream modules
		disableModule $mods(ChooseImage-SmoothInput) 0
		disableModule $mods(Extract-Smooth) 1

		set smoothing 1
		
		# Execute choose, when it finishes, we will disable it
		# and then enable downstream modules and execute them
		after 500 "$mods(ChooseImage-SmoothInput)-c needexecute"
	    }
	}
    }

    method update_speed_image {} {
	global mods
	
	# set number of iterations 0
	global $mods(LevelSet)-max_iterations
	set $mods(LevelSet)-max_iterations 0

	# execute ThresholdLevelSet filter
	set updating_speed 1
	$mods(LevelSet)-c needexecute
    }

    method configure_threshold_sliders {} {
	# configure threshold range sliders
	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.params.childsite.lthresh.s \
	    configure -from $range_min -to $range_max
	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.params.childsite.lthresh.s \
	    configure -from $range_min -to $range_max

	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.params.childsite.uthresh.s \
	    configure -from $range_min -to $range_max
	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.params.childsite.uthresh.s \
	    configure -from $range_min -to $range_max
    }

    method initialize_segmentation {} {
	if {!$segmentation_initialized} {
	    global mods slice

	    disableModule $mods(ChooseImage-ToSmooth) 0
	    
	    # Set segmenting slice to 0
	    set slice 0
	    
	    # Set threshold values to reasonable defaults (and other Level Set vals)
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
	    global $mods(LevelSet)-update_OutputImage
	    global $mods(LevelSet)-update_iters_OutputImage

	    set $mods(LevelSet)-update_OutputImage 1
	    set $mods(LevelSet)-update_iters_OutputImage 2

	    set range [expr $range_max - $range_min]
	    set $mods(LevelSet)-lower_threshold \
		[expr int([expr $range_min + [expr $range/3]])]
	    set $mods(LevelSet)-upper_threshold \
		[expr int([expr $range_max - [expr $range/3]])]
	    set $mods(LevelSet)-curvature_scaling 1.0
	    set $mods(LevelSet)-propagation_scaling 1.0
	    set $mods(LevelSet)-edge_weight 1.0
	    set $mods(LevelSet)-reverse_expansion_direction 0
	    set $mods(LevelSet)-smoothing_iterations 0
	    set $mods(LevelSet)-max_rms_change 0.02
	    
	    # set LevelSet iterations to 0
	    set $mods(LevelSet)-max_iterations 0
	    
	    # configure ExtractImageFilter bounds and enable it
	    global $mods(ExtractSlice)-minDim0
	    global $mods(ExtractSlice)-minDim1
	    global $mods(ExtractSlice)-minDim2
	    global $mods(ExtractSlice)-maxDim0
	    global $mods(ExtractSlice)-maxDim1
	    global $mods(ExtractSlice)-maxDim2
	    
	    set $mods(ExtractSlice)-minDim0 0
	    set $mods(ExtractSlice)-minDim1 0
	    set $mods(ExtractSlice)-minDim2 0
# 	    set $mods(ExtractSlice)-maxDim0 [expr $size0 - 1]
# 	    set $mods(ExtractSlice)-maxDim1 [expr $size1 - 1]
# 	    set $mods(ExtractSlice)-maxDim2 [expr $size2 - 1]
	    set $mods(ExtractSlice)-maxDim0 $size0
	    set $mods(ExtractSlice)-maxDim1 $size1
	    set $mods(ExtractSlice)-maxDim2 $size2

	    global axis
	    if {$axis == 0} {
		set $mods(ExtractSlice)-minDim0 0
		set $mods(ExtractSlice)-maxDim0 1
	    } elseif {$axis == 1} {
		set $mods(ExtractSlice)-minDim1 0
		set $mods(ExtractSlice)-maxDim1 1
	    } else {
		set $mods(ExtractSlice)-minDim2 0
		set $mods(ExtractSlice)-maxDim2 1
	    }

	    # configure binary/float PasteImageFilter
	    global $mods(PasteImageFilter-Binary)-size0
	    global $mods(PasteImageFilter-Binary)-size1
	    global $mods(PasteImageFilter-Binary)-size2
	    global $mods(PasteImageFilter-Binary)-index
	    global $mods(PasteImageFilter-Binary)-axis
	    global $mods(PasteImageFilter-Binary)-fill_value
#  	    set $mods(PasteImageFilter-Binary)-size0 [expr $size0 - 1]
#  	    set $mods(PasteImageFilter-Binary)-size1 [expr $size1 - 1]
#  	    set $mods(PasteImageFilter-Binary)-size2 [expr $size2 - 1]
 	    set $mods(PasteImageFilter-Binary)-size0 $size0
 	    set $mods(PasteImageFilter-Binary)-size1 $size1
 	    set $mods(PasteImageFilter-Binary)-size2 $size2
	    set $mods(PasteImageFilter-Binary)-index 0
	    set $mods(PasteImageFilter-Binary)-axis $axis
	    set $mods(PasteImageFilter-Binary)-fill_value 0

	    global $mods(PasteImageFilter-Float)-size0
	    global $mods(PasteImageFilter-Float)-size1
	    global $mods(PasteImageFilter-Float)-size2
	    global $mods(PasteImageFilter-Float)-index
	    global $mods(PasteImageFilter-Float)-axis
	    global $mods(PasteImageFilter-Float)-fill_value
#  	    set $mods(PasteImageFilter-Float)-size0 [expr $size0 - 1]
#  	    set $mods(PasteImageFilter-Float)-size1 [expr $size1 - 1]
#  	    set $mods(PasteImageFilter-Float)-size2 [expr $size2 - 1]
 	    set $mods(PasteImageFilter-Float)-size0 $size0
 	    set $mods(PasteImageFilter-Float)-size1 $size1
 	    set $mods(PasteImageFilter-Float)-size2 $size2
	    set $mods(PasteImageFilter-Float)-index 0
	    set $mods(PasteImageFilter-Float)-axis $axis
	    set $mods(PasteImageFilter-Float)-fill_value 0

	    # enable segmentation modules
	    disableModule $mods(LevelSet) 0
	    disableModule $mods(BinaryThreshold-Slice) 0
	    disableModule $mods(ExtractSlice) 0
	    disableModule $mods(ImageToField-Feature) 0
	    disableModule $mods(NrrdToImage-Feature) 0
	    disableModule $mods(ExtractSlice) 0

	    # disable downstream modules
	    disableModule $mods(PasteImageFilter-Binary) 1
	    disableModule $mods(PasteImageFilter-Float) 1

	    disableModule $mods(ImageToNrrd-Vol) 1

	    # execute feature image stuff
	    $mods(SampleField-Seeds) send
	    $mods(SampleField-SeedsNeg) send

	    set segmentation_initialized 1

	    # turn off speed, seed, and segmentation in Viewer 0
	    after 100 \
		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (4)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (5)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (6)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	}
    }

    method update_seed_binary_threshold {var1 var2 var3} {
	global mods
	global $mods(LevelSet)-lower_threshold
	global $mods(LevelSet)-upper_threshold

	# update thresholds of BinaryThresholdImageFilter that
	# feeds in when user selects thresholding seeding method
	global $mods(BinaryThreshold-Seeds)-lower_threshold
	global $mods(BinaryThreshold-Seeds)-upper_threshold
	set $mods(BinaryThreshold-Seeds)-lower_threshold [set $mods(LevelSet)-lower_threshold]
	set $mods(BinaryThreshold-Seeds)-upper_threshold [set $mods(LevelSet)-upper_threshold]
    }

    method change_seed_method {} {
	global mods seed_method 
	global $mods(ChooseNrrd-Seeds)-port-index
	global $mods(ChooseNrrd-Combine)-port-index

	# change Choose module and turn on/off seeds
	if {$seed_method == "points"} {
	    set $mods(ChooseNrrd-Seeds)-port-index 0
	} elseif {$seed_method == "prev" || $seed_method == "next" || \
		  $seed_method == "curr"} {
	    set $mods(ChooseNrrd-Combine)-port-index 0
	    set $mods(ChooseNrrd-Seeds)-port-index 1
	} elseif {$seed_method == "thresh"} {
	    set $mods(ChooseNrrd-Combine)-port-index 1
	    set $mods(ChooseNrrd-Seeds)-port-index 1
	} else {
	    puts "ERROR: unsupported seed method"
	    return
	}
    }

    method change_number_of_seeds {type dir} {
	global mods

	if {$type == "+"} {
	    # Positive Seeds
	    global $mods(SampleField-Seeds)-num_seeds
	    if {$dir == "+"} {
		set $mods(SampleField-Seeds)-num_seeds [expr [set $mods(SampleField-Seeds)-num_seeds] + 1]
	    } elseif {$dir == "-"} {
		set $mods(SampleField-Seeds)-num_seeds [expr [set $mods(SampleField-Seeds)-num_seeds] - 1]
	    } 
	    if {[set $mods(SampleField-Seeds)-num_seeds] < 0} {
		set $mods(SampleField-Seeds)-num_seeds 0
	    }
	    $mods(SampleField-Seeds) send
	} else {
	    # Negative Seeds
	    global $mods(SampleField-SeedsNeg)-num_seeds
	    if {$dir == "+"} {
		set $mods(SampleField-SeedsNeg)-num_seeds [expr [set $mods(SampleField-SeedsNeg)-num_seeds] + 1]
	    } elseif {$dir == "-"} {
		set $mods(SampleField-SeedsNeg)-num_seeds [expr [set $mods(SampleField-SeedsNeg)-num_seeds] - 1]
	    } 
	    if {[set $mods(SampleField-SeedsNeg)-num_seeds] < 0} {
		set $mods(SampleField-SeedsNeg)-num_seeds 0
	    }
	    $mods(SampleField-SeedsNeg) send
	}
    }

    method create_seeds {} {
	global seed_method mods
	# Create initial segmentation and display it
	# by executing the appropriate set of modules
	disableModule $mods(Extract-Prev) 1
	disableModule $mods(ImageToNrrd-Prev) 1
	disableModule $mods(UnuAxdelete-Prev) 1
	disableModule $mods(BinaryThreshold-Seeds) 1

	disableModule $mods(ImageToField-Seeds) 0
	disableModule $mods(ImageToField-SeedsNeg) 0

	disableModule $mods(PasteImageFilter-Binary) 1
	disableModule $mods(PasteImageFilter-Float) 1

	# Turn on seed in top viewer and segmentation off
	after 100 \
	    "uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (4)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (6)\}\" 1; $mods(Viewer)-ViewWindow_0-c redraw"

	# Turn seed point widgets back on
	global show_seeds
	set show_seeds 1
	$this seeds_changed 1 2 3

	# Enable Choose module and disable Level Set
	disableModule $mods(ChooseImage-SegInput) 0
	disableModule $mods(LevelSet) 1

	if {$seed_method == "points"} {
	    $mods(SampleField-Seeds) send
	    $mods(SampleField-SeedsNeg) send
	} elseif {$seed_method == "prev"} {
	    global $mods(Extract-Prev)-minDim0
	    global $mods(Extract-Prev)-minDim1
	    global $mods(Extract-Prev)-minDim2
	    global $mods(Extract-Prev)-maxDim0
	    global $mods(Extract-Prev)-maxDim1
	    global $mods(Extract-Prev)-maxDim2
	    
	    set $mods(Extract-Prev)-minDim0 0
	    set $mods(Extract-Prev)-minDim1 0
	    set $mods(Extract-Prev)-minDim2 0
	    set $mods(Extract-Prev)-maxDim0 $size0
	    set $mods(Extract-Prev)-maxDim1 $size1
	    set $mods(Extract-Prev)-maxDim2 $size2
	    global axis
	    global slice
	    if {$slice == 0} {
		# Use last slice
		if {$axis == 0} {
		    set $mods(Extract-Prev)-minDim0 [expr $size0-1]
		    set $mods(Extract-Prev)-maxDim0 $size0
		} elseif {$axis == 1} {
		    set $mods(Extract-Prev)-minDim1 [expr $size1-1]
		    set $mods(Extract-Prev)-maxDim1 $size1
		} else {
		    set $mods(Extract-Prev)-minDim2 [expr $size2-1]
		    set $mods(Extract-Prev)-maxDim2 $size2
		}
	    } else {
		set prev [expr $slice - 1]
		if {$axis == 0} {
		    set $mods(Extract-Prev)-minDim0 $prev
		    set $mods(Extract-Prev)-maxDim0 $slice
		} elseif {$axis == 1} {
		    set $mods(Extract-Prev)-minDim1 $prev
		    set $mods(Extract-Prev)-maxDim1 $slice
		} else {
		    set $mods(Extract-Prev)-minDim2 $prev
		    set $mods(Extract-Prev)-maxDim2 $slice
		}
	    }

	    disableModule $mods(Extract-Prev) 0
	    disableModule $mods(ImageToNrrd-Prev) 0
	    disableModule $mods(UnuAxdelete-Prev) 0
	    $mods(SampleField-Seeds) send
	    $mods(SampleField-SeedsNeg) send
	    $mods(Extract-Prev)-c needexecute
	}  elseif {$seed_method == "next"} {
	    global $mods(Extract-Prev)-minDim0
	    global $mods(Extract-Prev)-minDim1
	    global $mods(Extract-Prev)-minDim2
	    global $mods(Extract-Prev)-maxDim0
	    global $mods(Extract-Prev)-maxDim1
	    global $mods(Extract-Prev)-maxDim2
	    
	    set $mods(Extract-Prev)-minDim0 0
	    set $mods(Extract-Prev)-minDim1 0
	    set $mods(Extract-Prev)-minDim2 0
	    set $mods(Extract-Prev)-maxDim0 $size0
	    set $mods(Extract-Prev)-maxDim1 $size1
	    set $mods(Extract-Prev)-maxDim2 $size2
	    global axis
	    global slice
	    set next [expr $slice + 1]
	    set next_1 [expr $next + 1]
	    if {$axis == 0} {
		# Determine if it is the last slice, if so use first
		if {$slice == [expr $size0 - 1]} {
		    set $mods(Extract-Prev)-minDim0 0
		    set $mods(Extract-Prev)-maxDim0 1
		} else {
		    set $mods(Extract-Prev)-minDim0 $next
		    set $mods(Extract-Prev)-maxDim0 $next_1
		}
	    } elseif {$axis == 1} {
		if {$slice == [expr $size1 - 1]} {
		    set $mods(Extract-Prev)-minDim1 0
		    set $mods(Extract-Prev)-maxDim1 1
		} else {
		    set $mods(Extract-Prev)-minDim1 $next
		    set $mods(Extract-Prev)-maxDim1 $next_1
		}
	    } else {
		if {$slice == [expr $size2 - 1]} {
		    set $mods(Extract-Prev)-minDim2 0
		    set $mods(Extract-Prev)-maxDim2 1
		} else {
		    set $mods(Extract-Prev)-minDim2 $next
		    set $mods(Extract-Prev)-maxDim2 $next_1
		}
	    }

	    disableModule $mods(Extract-Prev) 0
	    disableModule $mods(ImageToNrrd-Prev) 0
	    disableModule $mods(UnuAxdelete-Prev) 0
	    $mods(SampleField-Seeds) send
	    $mods(SampleField-SeedsNeg) send
	    $mods(Extract-Prev)-c needexecute
	}  elseif {$seed_method == "curr"} {
	    global $mods(Extract-Prev)-minDim0
	    global $mods(Extract-Prev)-minDim1
	    global $mods(Extract-Prev)-minDim2
	    global $mods(Extract-Prev)-maxDim0
	    global $mods(Extract-Prev)-maxDim1
	    global $mods(Extract-Prev)-maxDim2
	    
	    set $mods(Extract-Prev)-minDim0 0
	    set $mods(Extract-Prev)-minDim1 0
	    set $mods(Extract-Prev)-minDim2 0
	    set $mods(Extract-Prev)-maxDim0 $size0
	    set $mods(Extract-Prev)-maxDim1 $size1
	    set $mods(Extract-Prev)-maxDim2 $size2
	    global axis
	    global slice
	    set next [expr $slice + 1]
	    if {$axis == 0} {
		set $mods(Extract-Prev)-minDim0 $slice
		set $mods(Extract-Prev)-maxDim0 $next
	    } elseif {$axis == 1} {
		set $mods(Extract-Prev)-minDim1 $slice
		set $mods(Extract-Prev)-maxDim1 $next
	    } else {
		set $mods(Extract-Prev)-minDim2 $slice
		set $mods(Extract-Prev)-maxDim2 $next
	    }

	    disableModule $mods(Extract-Prev) 0
	    disableModule $mods(ImageToNrrd-Prev) 0
	    disableModule $mods(UnuAxdelete-Prev) 0
	    $mods(SampleField-Seeds) send
	    $mods(SampleField-SeedsNeg) send
	    $mods(Extract-Prev)-c needexecute
	} elseif {$seed_method == "thresh"} {
	    disableModule $mods(BinaryThreshold-Seeds) 0
	    $mods(SampleField-Seeds) send
	    $mods(SampleField-SeedsNeg) send
	    $mods(BinaryThreshold-Seeds)-c needexecute
	} else {
	    puts "ERROR: cannot create seeds"
	    return
	}
    }

    method start_segmentation {type} {
	global max_iter mods


	global $mods(ChooseImage-SegInput)-port-index
	if {$type == "Go" && $has_segmented == 1} {
	    set $mods(ChooseImage-SegInput)-port-index 1
	    set type "Go"
	    set segmenting_type "Go"
	} else {
	    set $mods(ChooseImage-SegInput)-port-index 0
	    set type "Reset"
	    set segmenting_type "Reset"
	}

	# enable LevelSet
	disableModule $mods(LevelSet) 0

	global $mods(LevelSet)-max_iterations
	# set level set max iterations to be what 
	# global max_iter is and then after execute
	# set it back to 0 so user can update speed image
	set $mods(LevelSet)-max_iterations $max_iter

	# Turn seed off and segmentation on
	after 100 \
	    "uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (4)\}\" 1; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (5)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (6)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"

	# Turn seeds off
	global show_seeds
	set show_seeds 0
	$this seeds_changed 1 2 3 
	
	set segmenting 1
	if {$type == "Reset"} {
	    disableModule $mods(ChooseImage-Hack3) 0
	    disableModule $mods(ChooseImage-SegInput) 0

	    # execute Level Set
#	    $mods(LevelSet)-c needexecute
	    $mods(ChooseImage-SegInput)-c needexecute	   
	} else {
	    disableModule $mods(ChooseImage-SegInput) 0
	    disableModule $mods(ChooseImage-Hack3) 1
	    
	    after 500 "$mods(ChooseImage-SegInput)-c needexecute"
	}

	global slice
	if {$segs($slice) == 2} {
	    $this change_slice_icon 1
	}
    }

    method stop_segmentation {} {
	global mods
	$mods(LevelSet) stop_segmentation
    }

    method commit_segmentation {} {
	global slice mods axis
	
	# Add current segmentation into paste image filters 
	if {$axis == 0} {
	    global $mods(PasteImageFilter-Binary)-index
	    global $mods(PasteImageFilter-Float)-index
	    set $mods(PasteImageFilter-Binary)-index $slice
	    set $mods(PasteImageFilter-Float)-index $slice	    
	} elseif {$axis == 1} {
	    global $mods(PasteImageFilter-Binary)-index
	    global $mods(PasteImageFilter-Float)-index
	    set $mods(PasteImageFilter-Binary)-index $slice
	    set $mods(PasteImageFilter-Float)-index $slice
	} else {
	    global $mods(PasteImageFilter-Binary)-index
	    global $mods(PasteImageFilter-Float)-index
	    set $mods(PasteImageFilter-Binary)-index $slice
	    set $mods(PasteImageFilter-Float)-index $slice
	}

	# enable pasting modules
	disableModule $mods(PasteImageFilter-Binary) 0
	disableModule $mods(PasteImageFilter-Float) 0

	# re-disable volume rendering module and Extract
	disableModule $mods(ImageToNrrd-Vol) 1
	disableModule $mods(Extract-Prev) 1

	disableModule $mods(ImageFileWriter-Binary) 1
	disableModule $mods(ImageFileWriter-Float) 1

	# execute
	$mods(PasteImageFilter-Binary)-c needexecute
	$mods(PasteImageFilter-Float)-c needexecute

	set pasting_binary 1
	set pasting_float 1

	# enable volume rendering button
	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.volren \
	    configure -background $execute_color \
	    -activebackground $execute_active_color -state normal
	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.volren \
	    configure -background $execute_color \
	    -activebackground $execute_active_color -state normal

	set has_committed 1
	# enable saving buttons
	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.savefl.btn configure -state normal
	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.savefl.btn configure -state normal

	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.savebin.btn configure -state normal
	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.savebin.btn configure -state normal
    }

    method current_slice_changed {} {
	global slice mods axis
	# slice value changed via spinner so update

	# Set ChooseImage-SegInput back to not use a this
	# slice's previous segmentation since we are one a
	# new slice
	global $mods(ChooseImage-SegInput)-port-index
	set $mods(ChooseImage-SegInput)-port-index 0
	
	# extract modules 
	if {$axis == 0} {
	    global $mods(ExtractSlice)-minDim0
	    global $mods(ExtractSlice)-maxDim0
	    set $mods(ExtractSlice)-minDim0 $slice
	    set $mods(ExtractSlice)-maxDim0 [expr $slice + 1]
	} elseif {$axis == 1} {
	    global $mods(ExtractSlice)-minDim1
	    global $mods(ExtractSlice)-maxDim1
	    set $mods(ExtractSlice)-minDim1 $slice
	    set $mods(ExtractSlice)-maxDim1 [expr $slice + 1]
	} else {
	    global $mods(ExtractSlice)-minDim2
	    global $mods(ExtractSlice)-maxDim2
	    set $mods(ExtractSlice)-minDim2 $slice
	    set $mods(ExtractSlice)-maxDim2 [expr $slice + 1]
	}

	# top window should just show original data
	after 100 \
	    "uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (4)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (5)\}\" 0; uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Transparent Faces (6)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	
	# update speed image

	$mods(ExtractSlice)-c needexecute

	# update slice window
	global $mods(ViewSlices)-slice-viewport0-slice
	set $mods(ViewSlices)-slice-viewport0-slice $slice
	$mods(ViewSlices)-c rebind .standalone.viewers.topbot.pane1.childsite.lmr.pane0.childsite.slice
	$mods(ViewSlices)-c redrawall

	# udpate indicator
	$this change_slice_icon $segs($slice)

	set has_segmented 0
    }

    method update_volume_rendering {} {
	global mods

	# enable ImageToField and execute to get min/max for volume rendering modules.
	# when the unuminmax module completes it will execute the rest of them
	disableModule $mods(ImageToNrrd-Vol) 0

	disableModule $mods(NrrdSetupTexture-Vol) 1

	$mods(ImageToNrrd-Vol)-c needexecute

	# disable update volume rendering button
	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.volren \
	    configure -background "grey75"\
	    -activebackground "grey75" -state disabled
	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.volren \
	    configure -background "grey75" \
	    -activebackground "grey75" -state disabled
    }

    method start_crop {} {
        global mods show_roi
	global $mods(ViewSlices)-crop
	
	if {$show_roi == 1} {
	    $mods(ViewSlices)-c startcrop
	}
    }

    method update_crop_values { varname varele varop } {
	global mods 

	if {$updating_crop_widget == 1} {
	    return
	}

	if {[string first "crop_minAxis0" $varname] != -1} {
	    global $mods(UnuCrop)-minAxis0
	    global $mods(ViewSlices)-crop_minAxis0
	    global $mods(ViewSlices)-crop_minPadAxis0
	    set $mods(UnuCrop)-minAxis0 \
		[expr [set $mods(ViewSlices)-crop_minAxis0] + \
		     [set $mods(ViewSlices)-crop_minPadAxis0]]
	} elseif {[string first "crop_maxAxis0" $varname] != -1} {
	    global $mods(UnuCrop)-maxAxis0
	    global $mods(ViewSlices)-crop_maxAxis0
	    set $mods(UnuCrop)-maxAxis0 [set $mods(ViewSlices)-crop_maxAxis0]
	} elseif {[string first "crop_minAxis1" $varname] != -1} {
	    global $mods(UnuCrop)-minAxis1
	    global $mods(ViewSlices)-crop_minAxis1
	    global $mods(ViewSlices)-crop_minPadAxis1
	    set $mods(UnuCrop)-minAxis1 \
		[expr [set $mods(ViewSlices)-crop_minAxis1] + \
		     [set $mods(ViewSlices)-crop_minPadAxis1]]
	} elseif {[string first "crop_maxAxis1" $varname] != -1} {
	    global $mods(UnuCrop)-maxAxis1
	    global $mods(ViewSlices)-crop_maxAxis1
	    set $mods(UnuCrop)-maxAxis1 [set $mods(ViewSlices)-crop_maxAxis1]
	} elseif {[string first "crop_minAxis2" $varname] != -1} {
	    global $mods(UnuCrop)-minAxis2
	    global $mods(ViewSlices)-crop_minAxis2
	    global $mods(ViewSlices)-crop_minPadAxis2
	    set $mods(UnuCrop)-minAxis2 \
		[expr [set $mods(ViewSlices)-crop_minAxis2] + \
		     [set $mods(ViewSlices)-crop_minPadAxis2]]
	} elseif {[string first "crop_maxAxis2" $varname] != -1} {
	    global $mods(UnuCrop)-maxAxis2
	    global $mods(ViewSlices)-crop_maxAxis2
	    set $mods(UnuCrop)-maxAxis2 [set $mods(ViewSlices)-crop_maxAxis2]
	}
    }

    method update_crop_widget {type i} {
	global mods

	# get values from UnuCrop, then
	# set ViewSlices crop values
        if {$type == "min"} {
    	    global $mods(UnuCrop)-minAxis$i $mods(ViewSlices)-crop_minAxis$i
            set min [set $mods(UnuCrop)-minAxis$i]
            set $mods(ViewSlices)-crop_minAxis$i $min           
        } else {
    	    global $mods(UnuCrop)-maxAxis$i $mods(ViewSlices)-crop_maxAxis$i
            set max [set $mods(UnuCrop)-maxAxis$i]
            set $mods(ViewSlices)-crop_maxAxis$i $max 
        }

	global $mods(ViewSlices)-crop
	if {[set $mods(ViewSlices)-crop] == 1} {
	    $mods(ViewSlices)-c updatecrop
	}
    }

    method save_binary {} {
	global mods
	global $mods(ImageFileWriter-Binary)-filename
	
	# enable writer, open ui
	disableModule $mods(ImageFileWriter-Binary) 0

	if {[set $mods(ImageFileWriter-Binary)-filename] != ""} {
	    $mods(ImageFileWriter-Binary)-c needexecute
	} else {
	    $mods(ImageFileWriter-Binary) initialize_ui

	    # Disable execute behavior
	    set m $mods(ImageFileWriter-Binary)
	    .ui$m.f7.execute configure -state disabled
	    
	    upvar \#0 .ui$m data	
	    set data(-command) "wm withdraw .ui$m"
	}
    }

    method open_save_binary_ui {} {
	global mods

	# enable writer, open ui
	disableModule $mods(ImageFileWriter-Binary) 0
	$mods(ImageFileWriter-Binary) initialize_ui

	# Disable execute behavior
	set m $mods(ImageFileWriter-Binary)
	.ui$m.f7.execute configure -state disabled
	
	upvar \#0 .ui$m data	
	set data(-command) "wm withdraw .ui$m"
    }

    method save_float {} {
	global mods
	global $mods(ImageFileWriter-Float)-filename

	# enable writer, open ui
	disableModule $mods(ImageFileWriter-Float) 0

	if {[set $mods(ImageFileWriter-Float)-filename] != ""} {
	    $mods(ImageFileWriter-Float)-c needexecute
	} else {
	    $mods(ImageFileWriter-Float) initialize_ui

	    # Disable execute behavior
	    set m $mods(ImageFileWriter-Float)
	    .ui$m.f7.execute configure -state disabled
	    
	    upvar \#0 .ui$m data	
	    set data(-command) "wm withdraw .ui$m"
	}
    }

    method open_save_float_ui {} {
	global mods

	# enable writer, open ui
	disableModule $mods(ImageFileWriter-Float) 0
	$mods(ImageFileWriter-Float) initialize_ui

	# Disable execute behavior
	set m $mods(ImageFileWriter-Float)
	.ui$m.f7.execute configure -state disabled
	
	upvar \#0 .ui$m data	
	set data(-command) "wm withdraw .ui$m"
    }

    method seeds_changed {a b c} {
	global mods show_seeds
	global $mods(SampleField-Seeds)-num_seeds

	global $mods(SampleField-Seeds)-num_seeds

	set val 0
	if {$show_seeds == 1} {
	    set val 1
	}

	# Turn ViewWindow_0 seeds on/off depending on val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint0 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint1 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint2 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint3 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint4 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint5 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint6 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint7 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint8 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint9 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint10 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint11 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint12 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint13 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint14 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint15 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint16 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint17 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint18 (7)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint19 (7)} $val

	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint0 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint1 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint2 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint3 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint4 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint5 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint6 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint7 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint8 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint9 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint10 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint11 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint12 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint13 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint14 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint15 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint16 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint17 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint18 (8)} $val
	setGlobal {$mods(Viewer)-ViewWindow_0-SeedPoint19 (8)} $val
	

	# Assume no more than 20 seeds - HACK
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint0 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint1 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint2 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint3 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint4 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint5 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint6 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint7 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint8 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint9 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint10 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint11 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint12 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint13 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint14 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint15 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint16 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint17 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint18 (7)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint19 (7)} 0

	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint0 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint1 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint2 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint3 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint4 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint5 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint6 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint7 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint8 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint9 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint10 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint11 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint12 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint13 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint14 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint15 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint16 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint17 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint18 (8)} 0
	setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint19 (8)} 0

	$mods(Viewer)-ViewWindow_0-c redraw
	$mods(Viewer)-ViewWindow_1-c redraw


#	for {set i 0} {$i < [set $mods(SampleField-Seeds)-num_seeds]} {incr i} {
#	    setGlobal {$mods(Viewer)-ViewWindow_1-SeedPoint (5)} $val
#	    upvar \#0 {$mods(Viewer)-ViewWindow_1-SeedPoint$i (5)} point
#	    puts [set point]
#	}
	
    }

    method change_slice_icon {state} {
	set icon ""
	if {$state == 0} {
	    global no_seg_icon
	    set icon $no_seg_icon
	} elseif {$state == 1} {
	    global old_seg_icon
	    set icon $old_seg_icon
	} else {
	    global updated_seg_icon
	    set icon $updated_seg_icon
	}

	$attachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.slice.status configure -image $icon
	$detachedPFr.f.p.childsite.tnb.canvas.notebook.cs.page3.cs.slice.status configure -image $icon
	

    }

    method toggle_show_roi {} {
	global mods show_roi
	if {$show_roi == 0} {
	    $mods(ViewSlices)-c stopcrop
	} else {
	    set updating_crop_widget 1

	    $this start_crop

	    # Make sure widget is over volume
	    global $mods(ViewSlices)-dim0 $mods(ViewSlices)-dim1
	    global $mods(ViewSlices)-dim2
	    upvar \#0 $mods(ViewSlices)-dim0 dim0
	    upvar \#0 $mods(ViewSlices)-dim1 dim1
	    upvar \#0 $mods(ViewSlices)-dim2 dim2

	    global $mods(ViewSlices)-crop_minAxis0 
	    global $mods(ViewSlices)-crop_maxAxis0
	    global $mods(ViewSlices)-crop_minAxis1 
	    global $mods(ViewSlices)-crop_maxAxis1
	    global $mods(ViewSlices)-crop_minAxis2 
	    global $mods(ViewSlices)-crop_maxAxis2
	    
	    set $mods(ViewSlices)-crop_minAxis0 0
	    set $mods(ViewSlices)-crop_minAxis1 0
	    set $mods(ViewSlices)-crop_minAxis2 0

	    set $mods(ViewSlices)-crop_maxAxis0 $dim0
	    set $mods(ViewSlices)-crop_maxAxis1 $dim1
	    set $mods(ViewSlices)-crop_maxAxis2 $dim2

	    $mods(ViewSlices)-c updatecrop
	    set updating_crop_widget 0
	} 
    }

    method toggle_volume_render_object {} {
	global mods vol_foreground
	global $mods(EditColorMap2D)-on-0
	global $mods(EditColorMap2D)-on-1
	if {$vol_foreground == 1} {
	    set $mods(EditColorMap2D)-on-0 1
	    set $mods(EditColorMap2D)-on-1 0
	} else {
	    set $mods(EditColorMap2D)-on-0 0
	    set $mods(EditColorMap2D)-on-1 1
	}
	$mods(EditColorMap2D)-c toggle 0
	$mods(EditColorMap2D)-c toggle 1
    }

    method update_ViewSlices_input {} {
	global mods
	
	# execute ChooseNrrd-2D to send the new 
	# input to ViewSlices
	$mods(ChooseNrrd-2D)-c needexecute

	# reconfigure windows width/level?
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

    variable curr_proc_tab
    variable proc_tab1
    variable proc_tab2
    variable curr_data_tab
    variable data_tab1
    variable data_tab2
    variable eviewer2
    variable eviewer3
    variable has_loaded
    variable 2D_fixed

    variable orig_size0
    variable orig_size1
    variable orig_size2

    variable size0
    variable size1
    variable size2
    variable range_min
    variable range_max
    variable slice_frame

    variable filter_menu1
    variable filter_menu2
#    variable filter_enabled

    variable next_load
    variable next_smooth

    variable execute_active_color
    variable has_smoothed

    variable has_segmented
    variable segmentation_initialized
    variable updating_speed
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
    $mods(Viewer-VolRen)-ViewWindow_0-c autoview
}
