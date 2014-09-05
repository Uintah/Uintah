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

setProgressText "Loading BioTensor Modules, Please Wait..."

#######################################################################
# Check environment variables.  Ask user for input if not set:
# Attempt to get environment variables:
set DATADIR [netedit getenv SCIRUN_DATA]
set DATASET brain-dt
#######################################################################

############# NET ##############
::netedit dontschedule

set m0 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 14 9]
set m1 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSlice" 603 769]
set m2 [addModuleAtPosition "Teem" "Tend" "TendEpireg" 14 780]
set m3 [addModuleAtPosition "Teem" "Tend" "TendEstim" 14 1318]
set m4 [addModuleAtPosition "Teem" "Tend" "TendBmat" 231 781]
set m5 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSlice" 992 1116]
set m6 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 603 1183]
set m7 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 992 1249]
set m8 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 769 1412]
set m9 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 992 1412]
set m10 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 1121 1528]
set m11 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 14 1528]
set m12 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 14 1594]
set m13 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 114 1978]
#set m14 [addModuleAtPosition "Teem" "Tend" "TendEval" 1450 23]
#set m15 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 1449 86]
set m16 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 545 1595]
set m17 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 14 1743]
set m18 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 46 1820]
set m19 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 189 1528]
set m20 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 364 1528]
set m21 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 189 1594]
set m22 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 364 1594]
set m23 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 14 1672]
set m24 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 196 1681]
set m25 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 218 1822]
set m26 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 218 1756]
set m27 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 473 1681]
set m28 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 506 2005]
set m29 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 737 1657]
set m30 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 737 1758]
set m31 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 470 2250]
#set m32 [addModuleAtPosition "SCIRun" "FieldsCreate" "ClipField" 1451 533]
set m33 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 488 2127]
set m34 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1050 1681]
#set m35 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldMeasures" 1451 213]
#set m36 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldMeasures" 1450 149]
#set m37 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldMeasures" 1452 464]
#set m38 [addModuleAtPosition "SCIRun" "FieldsData" "ManageFieldData" 1451 656]
set m39 [addModuleAtPosition "SCIRun" "FieldsData" "ManageFieldData" 1492 2014]
set m40 [addModuleAtPosition "SCIRun" "FieldsData" "ManageFieldData" 1492 2138]
#set m41 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1452 339]
#set m42 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1269 86]
#set m43 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1268 26]
set m44 [addModuleAtPosition "SCIRun" "Render" "Viewer" 88 2990]
set m45 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 37 112]
set m46 [addModuleAtPosition "SCIRun" "FieldsGeometry" "ChangeFieldBounds" 603 1245]
set m47 [addModuleAtPosition "Teem" "UnuAtoM" "UnuJoin" 14 1055]
set m48 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 32 916]
set m49 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 231 1158]
set m50 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 231 703]
#set m51 [addModuleAtPosition "Teem" "Unu" "UnuJoin" 1269 149]
#set m52 [addModuleAtPosition "Teem" "Tend" "TendEstim" 1269 212]
#set m53 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 1269 275]
set m54 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 1102 2071]
#set m55 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1269 463]
#set m56 [addModuleAtPosition "SCIRun" "FieldsGeometry" "ChangeFieldBounds" 1269 400]
set m57 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuProject" 603 1110]
set m58 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 787 1248]
set m59 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 787 1345]
set m60 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuProject" 992 1180]
set m61 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1010 1348]
set m62 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 774 769]
set m63 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 249 1095]
#set m64 [addModuleAtPosition "Teem" "Tend" "TendEvec" 1451 718]
#set m65 [addModuleAtPosition "Teem" "Unu" "UnuCrop" 1448 594]
#set m66 [addModuleAtPosition "Teem" "NrrdData" "EditTupleAxis" 1269 336]
set m67 [addModuleAtPosition "SCIRun" "FieldsCreate" "SamplePlane" 488 1864]
set m68 [addModuleAtPosition "SCIRun" "FieldsCreate" "SamplePlane" 674 1863]
set m69 [addModuleAtPosition "SCIRun" "FieldsCreate" "SamplePlane" 862 1863]
#set m70 [addModuleAtPosition "SCIRun" "FieldsGeometry" "Unstructure" 1263 591]
#set m71 [addModuleAtPosition "SCIRun" "FieldsGeometry" "Unstructure" 1270 529]
#set m72 [addModuleAtPosition "SCIRun" "FieldsGeometry" "Unstructure" 1267 652]
set m73 [addModuleAtPosition "SCIRun" "FieldsGeometry" "QuadToTri" 524 1941]
set m74 [addModuleAtPosition "SCIRun" "FieldsGeometry" "QuadToTri" 710 1941]
set m75 [addModuleAtPosition "SCIRun" "FieldsGeometry" "QuadToTri" 898 1940]
set m76 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 470 2186]
set m77 [addModuleAtPosition "SCIRun" "FieldsCreate" "IsoClip" 506 2065]
set m78 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 692 2006]
set m79 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 880 2005]
set m80 [addModuleAtPosition "SCIRun" "FieldsCreate" "IsoClip" 692 2065]
set m81 [addModuleAtPosition "SCIRun" "FieldsCreate" "IsoClip" 880 2066]
set m82 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 674 2127]
set m83 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 862 2127]
set m84 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 656 2188]
set m85 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 844 2189]
set m86 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 656 2251]
set m87 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 844 2251]
#set m88 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherPoints" 1451 400]
set m89 [addModuleAtPosition "SCIRun" "FieldsCreate" "ClipByFunction" 1492 2264]
set m90 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherPoints" 1157 2210]
set m91 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1050 2449]
set m92 [addModuleAtPosition "SCIRun" "FieldsCreate" "SampleField" 1138 2134]
set m93 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1120 2326]
set m94 [addModuleAtPosition "SCIRun" "FieldsCreate" "Probe" 1298 2211]
set m95 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1050 2913]
set m96 [addModuleAtPosition "Teem" "DataIO" "AnalyzeNrrdReader" 446 8]
set m97 [addModuleAtPosition "Teem" "DataIO" "DicomNrrdReader" 232 9]
set m98 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 32 992]
set m99 [addModuleAtPosition "Teem" "DataIO" "DicomNrrdReader" 204 913]
set m100 [addModuleAtPosition "Teem" "DataIO" "AnalyzeNrrdReader" 377 913]
#set m101 [addModuleAtPosition "Teem" "Unu" "UnuResample" 1450 275]
set m102 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1102 2388]
set m103 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuResample" 14 1117]
set m104 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 14 1252]
set m105 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 14 848]
set m106 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuResample" 14 1188]
#set m107 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 1267 713]
set m108 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1068 1744]
set m109 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1068 1806]
set m110 [addModuleAtPosition "Teem" "DataIO" "FieldToNrrd" 1102 2513]
set m111 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 1086 2851]
set m112 [addModuleAtPosition "Teem" "Tend" "TendEvalClamp" 14 1450]
set m113 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 410 689]
set m114 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 14 1386]
set m115 [addModuleAtPosition "Teem" "Tend" "TendAnscale" 1086 2724]
set m116 [addModuleAtPosition "Teem" "Tend" "TendNorm" 1102 2579]
set m117 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 1102 2646]
set m118 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 1086 2790]
set m119 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherPoints" 1492 2075]
set m120 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1438 2325]
set m121 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1367 2526]
set m122 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1367 1681]
set m123 [addModuleAtPosition "Teem" "Tend" "TendFiber" 1420 2392]
set m124 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1402 2463]
set m125 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1367 2593]
set m126 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1385 1744]
set m127 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1385 1806]
set m128 [addModuleAtPosition "Teem" "Tend" "TendEvecRGB" 545 1528]
#set m129 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldInfo" 1451 780]
set m130 [addModuleAtPosition "SCIRun" "FieldsCreate" "Probe" 1420 1888]
set m131 [addModuleAtPosition "SCIRun" "FieldsCreate" "SampleField" 1456 1948]
set m132 [addModuleAtPosition "SCIRun" "Visualization" "ChooseColorMap" 218 1894]
set m133 [addModuleAtPosition "SCIRun" "Visualization" "ChooseColorMap" 540 1761]
set m134 [addModuleAtPosition "SCIRun" "Visualization" "ChooseColorMap" 1068 1889]
set m135 [addModuleAtPosition "SCIRun" "Visualization" "ChooseColorMap" 1274 1958]
set m136 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 204 992]
set m137 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxinfo"  269 1266]
set m138 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSave" 198 1450]
set m139 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 199 1371]
set m156 [addModuleAtPosition "SCIRun" "FieldsGeometry" "Unstructure" 1492 2206]
# set m157 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSlice" 558 912]
# set m158 [addModuleAtPosition "Teem" "UnuAtoM" "UnuCrop" 428 773]
# set m159 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 428 839]
# set m160 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 606 701]

set m157 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 14 704]
set m158 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxinsert" 67 193]
set m159 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuReshape" 67 251]
set m160 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuPermute" 67 309]
set m161 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxinfo" 67 371]
set m162 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxinfo" 67 431]
set m163 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxinfo" 67 489]
set m164 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 255 196]
set m165 [addModuleAtPosition "Teem" "UnuAtoM" "UnuCrop" 95 625]
set m166 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuSlice" 267 625]
set m167 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 49 546]


set $m23-use-first-valid {0}
set $m24-use-first-valid {0} 
set $m27-use-first-valid {0}
set $m33-use-first-valid {0}
set $m45-use-first-valid {0}
set $m49-use-first-valid {0}
set $m82-use-first-valid {0}
set $m83-use-first-valid {0}
set $m93-use-first-valid {0}
set $m98-use-first-valid {0}
set $m104-use-first-valid {0}
set $m105-use-first-valid {0}
set $m113-use-first-valid {0}
set $m114-use-first-valid {0}
set $m117-use-first-valid {0}
set $m118-use-first-valid {0}
set $m120-use-first-valid {0}
set $m122-use-first-valid {0}
set $m132-use-first-valid {0}
set $m133-use-first-valid {0}
set $m134-use-first-valid {0}
set $m135-use-first-valid {0}
set $m136-use-first-valid {0}
set $m139-use-first-valid {0}
set $m157-use-first-valid {0}
set $m167-use-first-valid {0}


setProgressText "Loading BioTensor Connections, Please Wait..."
# Create the Connections between Modules
set c0 [addConnection $m119 0 $m40 0]
set c1 [addConnection $m94 1 $m93 0]
set c2 [addConnection $m130 1 $m120 0]
set c3 [addConnection $m67 0 $m73 0]
set c4 [addConnection $m67 0 $m33 0]
set c5 [addConnection $m68 0 $m74 0]
set c6 [addConnection $m68 0 $m82 0]
set c7 [addConnection $m69 0 $m75 0]
set c8 [addConnection $m69 0 $m83 0]
set c9 [addConnection $m18 0 $m13 0]
set c10 [addConnection $m28 0 $m77 0]
set c11 [addConnection $m76 0 $m31 0]
set c12 [addConnection $m78 0 $m80 0]
set c13 [addConnection $m79 0 $m81 0]
set c14 [addConnection $m84 0 $m86 0]
set c15 [addConnection $m85 0 $m87 0]
set c16 [addConnection $m91 0 $m95 0]
set c17 [addConnection $m102 0 $m110 0]
set c18 [addConnection $m121 0 $m125 0]
set c19 [addConnection $m39 0 $m119 0]
#set c20 [addConnection $m40 0 $m89 0]
set c20 [addConnection $m40 0 $m156 0]
set c21 [addConnection $m46 0 $m8 0]
set c22 [addConnection $m23 0 $m28 0]
set c23 [addConnection $m23 0 $m78 0]
set c24 [addConnection $m23 0 $m79 0]
set c25 [addConnection $m23 0 $m39 0]
set c26 [addConnection $m23 0 $m17 0]
set c27 [addConnection $m24 0 $m18 0]
set c28 [addConnection $m27 0 $m67 0]
set c29 [addConnection $m27 0 $m68 0]
set c30 [addConnection $m27 0 $m69 0]
set c31 [addConnection $m27 0 $m76 0]
set c32 [addConnection $m27 0 $m84 0]
set c33 [addConnection $m27 0 $m85 0]
set c34 [addConnection $m33 0 $m90 0]
set c35 [addConnection $m34 0 $m91 0]
set c36 [addConnection $m122 0 $m121 0]
set c37 [addConnection $m26 0 $m25 0]
set c38 [addConnection $m29 0 $m30 0]
set c39 [addConnection $m58 0 $m59 0]
set c40 [addConnection $m58 0 $m61 0]
set c41 [addConnection $m108 0 $m109 0]
set c42 [addConnection $m126 0 $m127 0]
set c43 [addConnection $m25 0 $m132 0]
set c44 [addConnection $m30 0 $m133 0]
set c45 [addConnection $m109 0 $m134 0]
set c46 [addConnection $m127 0 $m135 0]
set c47 [addConnection $m8 0 $m44 0]
set c48 [addConnection $m110 2 $m116 0]
#set c49 [addConnection $m0 0 $m45 0]
set c50 [addConnection $m0 0 $m113 0]
set c51 [addConnection $m48 0 $m98 0]
set c52 [addConnection $m50 0 $m4 0]
set c53 [addConnection $m6 0 $m46 0]
set c54 [addConnection $m7 0 $m9 0]
set c55 [addConnection $m10 0 $m94 0]
set c56 [addConnection $m10 0 $m130 0]
set c57 [addConnection $m10 0 $m92 0]
set c58 [addConnection $m10 0 $m131 0]
set c59 [addConnection $m12 0 $m23 0]
set c60 [addConnection $m12 0 $m24 0]
set c61 [addConnection $m12 0 $m27 0]
set c62 [addConnection $m12 0 $m34 0]
set c63 [addConnection $m12 0 $m122 0]
set c64 [addConnection $m54 0 $m124 0]
set c65 [addConnection $m54 0 $m102 0]
set c66 [addConnection $m45 0 $m164 0]
set c67 [addConnection $m157 0 $m2 0]
set c68 [addConnection $m157 0 $m1 0]
#set c66 [addConnection $m159 0 $m62 0]
#set c67 [addConnection $m159 0 $m2 0]
#set c68 [addConnection $m159 0 $m1 0]

set c69 [addConnection $m98 0 $m47 0]
set c70 [addConnection $m104 0 $m3 0]
set c71 [addConnection $m105 0 $m5 0]
set c72 [addConnection $m114 0 $m112 0]
set c73 [addConnection $m117 0 $m115 0]
set c74 [addConnection $m118 0 $m111 2]
set c75 [addConnection $m115 0 $m118 0]
set c76 [addConnection $m11 0 $m12 2]
set c77 [addConnection $m19 0 $m21 2]
set c78 [addConnection $m20 0 $m22 2]
set c79 [addConnection $m4 0 $m49 0]
set c80 [addConnection $m2 0 $m105 0]
set c81 [addConnection $m3 0 $m114 0]
set c82 [addConnection $m112 0 $m10 2]
set c83 [addConnection $m112 0 $m54 2]
set c84 [addConnection $m112 0 $m11 0]
set c85 [addConnection $m112 0 $m19 0]
set c86 [addConnection $m112 0 $m20 0]
set c87 [addConnection $m112 0 $m128 0]
set c88 [addConnection $m112 0 $m123 0]
set c89 [addConnection $m128 0 $m16 2]
set c90 [addConnection $m116 0 $m117 0]
set c91 [addConnection $m47 0 $m136 0]
set c92 [addConnection $m57 0 $m6 2]
set c93 [addConnection $m60 0 $m7 2]
set c94 [addConnection $m103 0 $m106 0]
set c95 [addConnection $m106 0 $m104 0]
set c96 [addConnection $m1 0 $m57 0]
set c97 [addConnection $m5 0 $m60 0]
set c98 [addConnection $m77 0 $m33 1]
set c99 [addConnection $m80 0 $m82 1]
set c100 [addConnection $m81 0 $m83 1]
set c101 [addConnection $m92 0 $m93 1]
set c102 [addConnection $m131 0 $m120 1]
set c103 [addConnection $m39 1 $m40 1]
set c104 [addConnection $m73 0 $m28 1]
set c105 [addConnection $m74 0 $m78 1]
set c106 [addConnection $m75 0 $m79 1]
set c107 [addConnection $m24 0 $m25 1]
set c108 [addConnection $m27 0 $m30 1]
set c109 [addConnection $m33 0 $m76 1]
set c110 [addConnection $m34 0 $m109 1]
set c111 [addConnection $m82 0 $m90 1]
set c112 [addConnection $m82 0 $m84 1]
set c113 [addConnection $m83 0 $m85 1]
set c114 [addConnection $m93 0 $m91 1]
set c115 [addConnection $m93 0 $m102 1]
set c116 [addConnection $m120 0 $m123 1]
set c117 [addConnection $m122 0 $m127 1]
set c118 [addConnection $m132 0 $m13 1]
set c119 [addConnection $m133 0 $m31 1]
set c120 [addConnection $m133 0 $m86 1]
set c121 [addConnection $m133 0 $m87 1]
set c122 [addConnection $m134 0 $m95 1]
set c123 [addConnection $m135 0 $m125 1]
set c124 [addConnection $m17 0 $m18 1]
set c125 [addConnection $m59 0 $m8 1]
set c126 [addConnection $m61 0 $m9 1]
set c127 [addConnection $m9 0 $m44 1]
#set c128 [addConnection $m97 0 $m45 1]
set c129 [addConnection $m97 0 $m113 1]
set c130 [addConnection $m99 0 $m98 1]
set c131 [addConnection $m110 2 $m117 1]
set c132 [addConnection $m50 0 $m2 1]
set c133 [addConnection $m63 0 $m49 1]
set c134 [addConnection $m6 0 $m59 1]
set c135 [addConnection $m7 0 $m61 1]
set c136 [addConnection $m21 0 $m23 1]
set c137 [addConnection $m21 0 $m24 1]
set c138 [addConnection $m21 0 $m27 1]
set c139 [addConnection $m21 0 $m34 1]
set c140 [addConnection $m21 0 $m122 1]
set c141 [addConnection $m157 0 $m105 1]
#set c141 [addConnection $m159 0 $m105 1]

set c142 [addConnection $m49 0 $m3 1]
set c143 [addConnection $m105 0 $m47 1]
set c144 [addConnection $m113 0 $m114 1]
set c144 [addConnection $m113 0 $m137 0]
set c145 [addConnection $m117 0 $m118 1]
set c146 [addConnection $m123 0 $m124 1]
set c147 [addConnection $m123 0 $m121 1]
set c148 [addConnection $m136 0 $m104 1]
set c149 [addConnection $m90 0 $m93 2]
set c150 [addConnection $m90 0 $m120 2]
set c151 [addConnection $m124 0 $m125 2]
set c152 [addConnection $m83 0 $m90 2]
set c153 [addConnection $m13 0 $m44 2]
#set c154 [addConnection $m96 0 $m45 2]
set c155 [addConnection $m96 0 $m113 2]
set c156 [addConnection $m100 0 $m98 2]
set c157 [addConnection $m22 0 $m23 2]
set c158 [addConnection $m22 0 $m24 2]
set c159 [addConnection $m22 0 $m27 2]
set c160 [addConnection $m22 0 $m34 2]
set c161 [addConnection $m22 0 $m122 2]
set c162 [addConnection $m111 0 $m95 2]
set c163 [addConnection $m89 0 $m93 3]
set c164 [addConnection $m89 0 $m120 3]
set c165 [addConnection $m31 0 $m44 3]
set c166 [addConnection $m16 0 $m23 3]
set c167 [addConnection $m16 0 $m24 3]
set c168 [addConnection $m16 0 $m27 3]
set c169 [addConnection $m16 0 $m34 3]
set c170 [addConnection $m16 0 $m122 3]
set c171 [addConnection $m86 0 $m44 4]
set c172 [addConnection $m87 0 $m44 5]
set c173 [addConnection $m92 1 $m44 6]
set c174 [addConnection $m94 0 $m44 7]
set c175 [addConnection $m95 0 $m44 8]
set c176 [addConnection $m125 0 $m44 9]
set c177 [addConnection $m130 0 $m44 10]
set c178 [addConnection $m131 1 $m44 11]
set c179 [addConnection $m136 0 $m103 0]
set c180 [addConnection $m45 0 $m136 1]
set c181 [addConnection $m102 0 $m111 3]
set c182 [addConnection $m137 0 $m114 1]
set c182 [addConnection $m114 0 $m139 0]
set c183 [addConnection $m139 0 $m138 0]
set c184 [addConnection $m156 0 $m89 0]
#set c185 [addConnection $m45 0 $m157 0]
#set c186 [addConnection $m45 0 $m158 0]
# set c187 [addConnection $m157 0 $m98 3]
# set c188 [addConnection $m45 0 $m159 0]
# set c189 [addConnection $m158 0 $m159 1]
# set c190 [addConnection $m45 0 $m160 0]
set c185 [addConnection $m0 0 $m45 0]
set c186 [addConnection $m97 0 $m45 1]
set c187 [addConnection $m96 0 $m45 2]
set c189 [addConnection $m45 0 $m158 0]
set c189 [addConnection $m157 0 $m62 0]
set c189 [addConnection $m158 0 $m159 0]
set c189 [addConnection $m159 0 $m160 0]
set c189 [addConnection $m160 0 $m161 0]
set c189 [addConnection $m161 0 $m162 0]
set c189 [addConnection $m162 0 $m163 0]
set c189 [addConnection $m167 0 $m165 0]
set c189 [addConnection $m167 0 $m166 0]
set c192 [addConnection $m45 0 $m157 0]
set c190 [addConnection $m165 0 $m157 1] 
set c191 [addConnection $m166 0 $m98 3]
set c192 [addConnection $m45 0 $m167 0]
set c192 [addConnection $m163 0 $m167 1]




setProgressText "Loading BioTensor Settings, Please Wait..."

set $m0-notes {}
#set $m0-add {0}
if {[file exists $DATADIR/$DATASET/demo-DWI.nrrd]} {
    set $m0-filename $DATADIR/$DATASET/demo-DWI.nrrd
} else {
    set $m0-filename {}
} 
set $m1-notes {}
set $m1-axis {3}
set $m1-position {0}
set $m2-notes {}
set $m2-gradient_list {}
set $m2-reference {-1}
set $m2-blur_x {1}
set $m2-blur_y {1}
set $m2-use-default-threshold {1}
set $m2-threshold {100}
set $m2-cc_analysis {1}
set $m2-fitting {0.7}
set $m2-kernel {cubicCR}
set $m2-sigma {1}
set $m2-extent {1}
set $m3-notes {}
set $m3-knownB0 {0}
set $m3-use-default-threshold {1}
set $m3-threshold {100}
set $m3-soft {0}
set $m3-scale {1}
set $m4-notes {}
set $m4-gradient_list {  -0.41782350500888   0.82380935278446   0.38309485630448
   0.50198668225440   0.56816446503362   0.65207247412560
   0.14374011122662  -0.42965895097966  -0.89147740648186
   0.69798942421263   0.04821234467449  -0.71448326328075
  -0.08966694118359  -0.82868717571573   0.55248294495221
  -0.22401802662322  -0.96424889055732  -0.14156270980317
   0.95269761674347   0.19440683125031   0.23360915010872
   0.61723322128312  -0.16621570162639   0.76902242560104
  -0.91787981980257   0.35358981414646   0.18019678057914
  -0.57743422830171   0.74041863058356  -0.34402029514314
   0.04765815481080   0.27630610772691  -0.95988730333974
  -0.73488583063772  -0.61688185895817   0.2817793250333



}
set $m5-notes {}
set $m5-axis {3}
set $m5-position {0}
set $m6-notes {}
set $m6-build-eigens {0}
set $m7-notes {}
set $m7-build-eigens {0}
set $m8-notes {}
set $m8-nodes-on {0}
set $m8-nodes-transparency {0}
set $m8-nodes-as-disks {0}
set $m8-edges-on {0}
set $m8-edges-transparency {0}
set $m8-faces-on {1}
set $m8-use-normals {0}
set $m8-use-transparency {0}
set $m8-vectors-on {0}
set $m8-normalize-vectors {}
set $m8-has_vector_data {0}
set $m8-bidirectional {0}
set $m8-vector-usedefcolor {0}
set $m8-tensors-on {0}
set $m8-has_tensor_data {0}
set $m8-scalars-on {0}
set $m8-scalars-transparency {0}
set $m8-has_scalar_data {1}
set $m8-text-on {0}
set $m8-text-use-default-color {1}
set $m8-text-color-r {1.0}
set $m8-text-color-g {1.0}
set $m8-text-color-b {1.0}
set $m8-text-backface-cull {0}
set $m8-text-fontsize {1}
set $m8-text-precision {2}
set $m8-text-render_locations {0}
set $m8-text-show-data {1}
set $m8-text-show-nodes {0}
set $m8-text-show-edges {0}
set $m8-text-show-faces {0}
set $m8-text-show-cells {0}
set $m8-def-color-r {0.5}
set $m8-def-color-g {0.5}
set $m8-def-color-b {0.5}
set $m8-def-color-a {0.5}
set $m8-node_display_type {Points}
set $m8-edge_display_type {Lines}
set $m8-data_display_type {Arrows}
set $m8-tensor_display_type {Boxes}
set $m8-scalar_display_type {Points}
set $m8-active_tab {Faces}
set $m8-node_scale {0.0300}
set $m8-edge_scale {0.0150}
set $m8-vectors_scale {0.300}
set $m8-tensors_scale {0.30}
set $m8-scalars_scale {0.30}
set $m8-show_progress {}
set $m8-interactive_mode {Interactive}
set $m8-field-name {}
set $m8-field-name-update {1}
set $m8-node-resolution {6}
set $m8-edge-resolution {6}
set $m8-data-resolution {6}
set $m9-notes {}
set $m9-nodes-on {0}
set $m9-nodes-transparency {0}
set $m9-nodes-as-disks {0}
set $m9-edges-on {0}
set $m9-edges-transparency {0}
set $m9-faces-on {1}
set $m9-use-normals {0}
set $m9-use-transparency {0}
set $m9-vectors-on {0}
set $m9-normalize-vectors {}
set $m9-has_vector_data {0}
set $m9-bidirectional {0}
set $m9-vector-usedefcolor {0}
set $m9-tensors-on {0}
set $m9-has_tensor_data {0}
set $m9-scalars-on {0}
set $m9-scalars-transparency {0}
set $m9-has_scalar_data {1}
set $m9-text-on {0}
set $m9-text-use-default-color {1}
set $m9-text-color-r {1.0}
set $m9-text-color-g {1.0}
set $m9-text-color-b {1.0}
set $m9-text-backface-cull {0}
set $m9-text-fontsize {1}
set $m9-text-precision {2}
set $m9-text-render_locations {0}
set $m9-text-show-data {1}
set $m9-text-show-nodes {0}
set $m9-text-show-edges {0}
set $m9-text-show-faces {0}
set $m9-text-show-cells {0}
set $m9-def-color-r {0.5}
set $m9-def-color-g {0.5}
set $m9-def-color-b {0.5}
set $m9-def-color-a {0.5}
set $m9-node_display_type {Points}
set $m9-edge_display_type {Lines}
set $m9-data_display_type {Arrows}
set $m9-tensor_display_type {Boxes}
set $m9-scalar_display_type {Points}
set $m9-active_tab {Faces}
set $m9-node_scale {0.0300}
set $m9-edge_scale {0.0150}
set $m9-vectors_scale {0.300}
set $m9-tensors_scale {0.30}
set $m9-scalars_scale {0.30}
set $m9-show_progress {}
set $m9-interactive_mode {Interactive}
set $m9-field-name {}
set $m9-field-name-update {1}
set $m9-node-resolution {6}
set $m9-edge-resolution {6}
set $m9-data-resolution {6}
set $m10-notes {}
set $m10-build-eigens {0}
set $m11-notes {}
set $m11-aniso_metric {tenAniso_FA}
set $m11-threshold {0.5}
set $m12-notes {}
set $m12-build-eigens {0}
set $m13-notes {}
set $m13-nodes-on {0}
set $m13-nodes-transparency {0}
set $m13-nodes-as-disks {0}
set $m13-edges-on {0}
set $m13-edges-transparency {0}
set $m13-faces-on {0}
set $m13-use-normals {1}
set $m13-use-transparency {0}
set $m13-vectors-on {0}
set $m13-normalize-vectors {}
set $m13-has_vector_data {0}
set $m13-bidirectional {0}
set $m13-vector-usedefcolor {0}
set $m13-tensors-on {0}
set $m13-has_tensor_data {0}
set $m13-scalars-on {0}
set $m13-scalars-transparency {0}
set $m13-has_scalar_data {1}
set $m13-text-on {0}
set $m13-text-use-default-color {1}
set $m13-text-color-r {1.0}
set $m13-text-color-g {1.0}
set $m13-text-color-b {1.0}
set $m13-text-backface-cull {0}
set $m13-text-fontsize {1}
set $m13-text-precision {2}
set $m13-text-render_locations {0}
set $m13-text-show-data {1}
set $m13-text-show-nodes {0}
set $m13-text-show-edges {0}
set $m13-text-show-faces {0}
set $m13-text-show-cells {0}
set $m13-def-color-r {0.5}
set $m13-def-color-g {0.5}
set $m13-def-color-b {0.5}
set $m13-def-color-a {0.5}
set $m13-node_display_type {Points}
set $m13-edge_display_type {Lines}
set $m13-data_display_type {Arrows}
set $m13-tensor_display_type {Boxes}
set $m13-scalar_display_type {Points}
set $m13-active_tab {Faces}
set $m13-node_scale {3.000000e-02}
set $m13-edge_scale {1.500000e-02}
set $m13-vectors_scale {0.30}
set $m13-tensors_scale {0.30}
set $m13-scalars_scale {3.000000e-01}
set $m13-show_progress {}
set $m13-interactive_mode {Interactive}
set $m13-field-name {}
set $m13-field-name-update {1}
set $m13-node-resolution {6}
set $m13-edge-resolution {6}
set $m13-data-resolution {6}
# set $m14-notes {}
# set $m14-threshold {}
# set $m15-notes {}
# set $m15-axis {}
# set $m15-position {}
set $m16-notes {}
set $m16-build-eigens {0}
set $m17-notes {}
set $m17-isoval {0.5000}
set $m17-isoval-min {0}
set $m17-isoval-max {0.99999994039535522}
set $m17-isoval-typed {0}
set $m17-isoval-quantity {1}
set $m17-quantity-range {colormap}
set $m17-quantity-min {0}
set $m17-quantity-max {100}
set $m17-isoval-list {0.0 1.0 2.0 3.0}
set $m17-extract-from-new-field {1}
set $m17-algorithm {1}
set $m17-build_trisurf {1}
set $m17-np {1}
set $m17-active-isoval-selection-tab {0}
set $m17-active_tab {NOISE}
set $m17-update_type {on release}
set $m17-color-r {0.40}
set $m17-color-g {0.78}
set $m17-color-b {0.73}
set $m18-notes {}
set $m18-interpolation_basis {linear}
set $m18-map_source_to_single_dest {0}
set $m18-exhaustive_search {0}
set $m18-exhaustive_search_max_dist {-1}
set $m18-np {1}
set $m19-notes {}
set $m19-aniso_metric {tenAniso_Cl1}
set $m19-threshold {0.5}
set $m20-notes {}
set $m20-aniso_metric {tenAniso_Cp1}
set $m20-threshold {0.5}
set $m21-notes {}
set $m21-build-eigens {0}
set $m22-notes {}
set $m22-build-eigens {0}
set $m23-notes {}
set $m23-port-selected-index {0}
set $m24-notes {}
set $m24-port-selected-index {3}
set $m25-notes {}
set $m25-isFixed {0}
set $m25-min {0}
set $m25-max {0.99999970197677612}
set $m25-makeSymmetric {0}
set $m26-notes {}
set $m26-tcl_status {Calling GenStandardColorMaps!}
set $m26-positionList {}
set $m26-nodeList {}
set $m26-width {1}
set $m26-height {1}
set $m26-mapType {2}
set $m26-minRes {12}
set $m26-resolution {256}
set $m26-realres {256}
set $m26-gamma {0}
set $m27-notes {}
set $m27-port-selected-index {3}
set $m28-notes {}
set $m28-interpolation_basis {linear}
set $m28-map_source_to_single_dest {0}
set $m28-exhaustive_search {0}
set $m28-exhaustive_search_max_dist {-1}
set $m28-np {1}
set $m29-notes {}
set $m29-tcl_status {Calling GenStandardColorMaps!}
set $m29-positionList {}
set $m29-nodeList {}
set $m29-width {398}
set $m29-height {40}
set $m29-mapType {17}
set $m29-minRes {19}
set $m29-resolution {256}
set $m29-realres {256}
set $m29-gamma {0}
set $m30-notes {}
set $m30-isFixed {0}
set $m30-min {0}
set $m30-max {0.99999994039535522}
set $m30-makeSymmetric {0}
set $m31-notes {}
set $m31-nodes-on {0}
set $m31-nodes-transparency {0}
set $m31-nodes-as-disks {0}
set $m31-edges-on {0}
set $m31-edges-transparency {0}
set $m31-faces-on {1}
set $m31-use-normals {0}
set $m31-use-transparency {0}
set $m31-vectors-on {0}
set $m31-normalize-vectors {}
set $m31-has_vector_data {0}
set $m31-bidirectional {0}
set $m31-vector-usedefcolor {0}
set $m31-tensors-on {0}
set $m31-has_tensor_data {0}
set $m31-scalars-on {0}
set $m31-scalars-transparency {0}
set $m31-has_scalar_data {1}
set $m31-text-on {0}
set $m31-text-use-default-color {1}
set $m31-text-color-r {1.0}
set $m31-text-color-g {1.0}
set $m31-text-color-b {1.0}
set $m31-text-backface-cull {0}
set $m31-text-fontsize {1}
set $m31-text-precision {2}
set $m31-text-render_locations {0}
set $m31-text-show-data {1}
set $m31-text-show-nodes {0}
set $m31-text-show-edges {0}
set $m31-text-show-faces {0}
set $m31-text-show-cells {0}
set $m31-def-color-r {0.5}
set $m31-def-color-g {0.5}
set $m31-def-color-b {0.5}
set $m31-def-color-a {0.5}
set $m31-node_display_type {Points}
set $m31-edge_display_type {Lines}
set $m31-data_display_type {Arrows}
set $m31-tensor_display_type {Boxes}
set $m31-scalar_display_type {Points}
set $m31-active_tab {Faces}
set $m31-node_scale {0.0300}
set $m31-edge_scale {0.0150}
set $m31-vectors_scale {0.30}
set $m31-tensors_scale {0.30}
set $m31-scalars_scale {0.300}
set $m31-show_progress {}
set $m31-interactive_mode {Interactive}
set $m31-field-name {}
set $m31-field-name-update {1}
set $m31-node-resolution {6}
set $m31-edge-resolution {6}
set $m31-data-resolution {6}
# set $m32-notes {}
# set $m32-clip-location {cell}
# set $m32-clipmode {replace}
# set $m32-autoexecute {0}
# set $m32-autoinvert {0}
# set $m32-execmode {0}
set $m33-notes {}
set $m33-port-selected-index {0}
set $m34-notes {}
set $m34-port-selected-index {3}
# set $m35-notes {}
# set $m35-simplexString {Node}
# set $m35-xFlag {1}
# set $m35-yFlag {1}
# set $m35-zFlag {1}
# set $m35-idxFlag {0}
# set $m35-sizeFlag {0}
# set $m35-numNbrsFlag {0}
# set $m36-notes {}
# set $m36-simplexString {Node}
# set $m36-xFlag {1}
# set $m36-yFlag {1}
# set $m36-zFlag {1}
# set $m36-idxFlag {0}
# set $m36-sizeFlag {0}
# set $m36-numNbrsFlag {0}
# set $m37-notes {}
# set $m37-simplexString {Node}
# set $m37-xFlag {1}
# set $m37-yFlag {1}
# set $m37-zFlag {1}
# set $m37-idxFlag {0}
# set $m37-sizeFlag {0}
# set $m37-numNbrsFlag {0}
# set $m38-notes {}
set $m39-notes {}
set $m40-notes {}
# set $m41-notes {}
# set $m41-isoval {0}
# set $m41-isoval-min {0}
# set $m41-isoval-max {99}
# set $m41-isoval-typed {0}
# set $m41-isoval-quantity {1}
# set $m41-quantity-range {colormap}
# set $m41-quantity-min {0}
# set $m41-quantity-max {100}
# set $m41-isoval-list {0.0 1.0 2.0 3.0}
# set $m41-extract-from-new-field {1}
# set $m41-algorithm {0}
# set $m41-build_trisurf {1}
# set $m41-np {1}
# set $m41-active-isoval-selection-tab {0}
# set $m41-active_tab {MC}
# set $m41-update_type {on release}
# set $m41-color-r {0.4}
# set $m41-color-g {0.2}
# set $m41-color-b {0.9}
# set $m42-notes {}
# set $m42-isoval {0}
# set $m42-isoval-min {0}
# set $m42-isoval-max {99}
# set $m42-isoval-typed {0}
# set $m42-isoval-quantity {1}
# set $m42-quantity-range {colormap}
# set $m42-quantity-min {0}
# set $m42-quantity-max {100}
# set $m42-isoval-list {0.0 1.0 2.0 3.0}
# set $m42-extract-from-new-field {1}
# set $m42-algorithm {0}
# set $m42-build_trisurf {1}
# set $m42-np {1}
# set $m42-active-isoval-selection-tab {0}
# set $m42-active_tab {MC}
# set $m42-update_type {on release}
# set $m42-color-r {0.4}
# set $m42-color-g {0.2}
# set $m42-color-b {0.9}
# set $m43-notes {}
# set $m43-isoval {0}
# set $m43-isoval-min {0}
# set $m43-isoval-max {99}
# set $m43-isoval-typed {0}
# set $m43-isoval-quantity {1}
# set $m43-quantity-range {colormap}
# set $m43-quantity-min {0}
# set $m43-quantity-max {100}
# set $m43-isoval-list {0.0 1.0 2.0 3.0}
# set $m43-extract-from-new-field {1}
# set $m43-algorithm {0}
# set $m43-build_trisurf {1}
# set $m43-np {1}
# set $m43-active-isoval-selection-tab {0}
# set $m43-active_tab {MC}
# set $m43-update_type {on release}
# set $m43-color-r {0.4}
# set $m43-color-g {0.2}
# set $m43-color-b {0.9}


 set $m44-notes {}
# set $m44-ViewWindow_0-pos {z0_y0}
# set $m44-ViewWindow_0-caxes {0}
# set $m44-ViewWindow_0-raxes {1}
# set $m44-ViewWindow_0-iaxes {}
# set $m44-ViewWindow_0-have_collab_vis {0}
# set $m44-ViewWindow_0-view-eyep-x {-7.6630625798236407}
# set $m44-ViewWindow_0-view-eyep-y {75.028175882540012}
# set $m44-ViewWindow_0-view-eyep-z {1015.1705499788859}
# set $m44-ViewWindow_0-view-lookat-x {-3.0535172036584237}
# set $m44-ViewWindow_0-view-lookat-y {89.966866425184293}
# set $m44-ViewWindow_0-view-lookat-z {35.278357023344782}
# set $m44-ViewWindow_0-view-up-x {-0.0064389303591822541}
# set $m44-ViewWindow_0-view-up-y {-0.99986262175693541}
# set $m44-ViewWindow_0-view-up-z {-0.015273434099017805}
# set $m44-ViewWindow_0-view-fov {20}
# set $m44-ViewWindow_0-view-eyep_offset-x {}
# set $m44-ViewWindow_0-view-eyep_offset-y {}
# set $m44-ViewWindow_0-view-eyep_offset-z {}
# set $m44-ViewWindow_0-lightColors {{1.0 1.0 1.0} {1.0 1.0 1.0} {1.0 1.0 1.0} {1.0 1.0 1.0}}
# set $m44-ViewWindow_0-lightVectors {{ 0 0 1 } { 0 0 1 } { 0 0 1 } { 0 0 1 }}
# set $m44-ViewWindow_0-bgcolor-r {0}
# set $m44-ViewWindow_0-bgcolor-g {0}
# set $m44-ViewWindow_0-bgcolor-b {0}
# set $m44-ViewWindow_0-shading {}
# set $m44-ViewWindow_0-do_stereo {0}
# set $m44-ViewWindow_0-ambient-scale {1.0}
# set $m44-ViewWindow_0-diffuse-scale {1.0}
# set $m44-ViewWindow_0-specular-scale {0.4}
# set $m44-ViewWindow_0-emission-scale {1.0}
# set $m44-ViewWindow_0-shininess-scale {1.0}
# set $m44-ViewWindow_0-polygon-offset-factor {1.0}
# set $m44-ViewWindow_0-polygon-offset-units {0.0}
# set $m44-ViewWindow_0-point-size {1.0}
# set $m44-ViewWindow_0-line-width {1.0}
# set $m44-ViewWindow_0-sbase {0.40}
# set $m44-ViewWindow_0-sr {1}
# set $m44-ViewWindow_0-do_bawgl {0}
# set $m44-ViewWindow_0-drawimg {}
# set $m44-ViewWindow_0-saveprefix {}
# set $m44-ViewWindow_0-resx {}
# set $m44-ViewWindow_0-resy {}
# set $m44-ViewWindow_0-aspect {}
# set $m44-ViewWindow_0-aspect_ratio {}
# set $m44-ViewWindow_0-ortho-view {1}
# set $m44-ViewWindow_0-unused {1}
# set $m44-ViewWindow_0-unused {1}

set $m44-ViewWindow_0-view-eyep-x {-95.50000095367432}
set $m44-ViewWindow_0-view-eyep-y {110.5000021118158}
set $m44-ViewWindow_0-view-eyep-z {-792.366673105479}
set $m44-ViewWindow_0-view-lookat-x {-95.50000095367432}
set $m44-ViewWindow_0-view-lookat-y {110.5000021118158}
set $m44-ViewWindow_0-view-lookat-z {0}
set $m44-ViewWindow_0-view-up-x {0}
set $m44-ViewWindow_0-view-up-y {-1}
set $m44-ViewWindow_0-view-up-z {0}
set $m44-ViewWindow_0-view-fov {20}
set $m44-ViewWindow_0-specular-scale {0.4}
set $m44-ViewWindow_0-global-light {1}
set $m44-ViewWindow_0-global-fog {0}
set $m44-ViewWindow_0-global-debug {0}
set $m44-ViewWindow_0-global-clip {0}
set $m44-ViewWindow_0-global-cull {1}
set $m44-ViewWindow_0-global-dl {0}
set $m44-ViewWindow_0-global-type {Gouraud}


set $m45-notes {}
set $m45-port-selected-index {0}
set $m46-notes {}
set $m46-outputcenterx {-95.5}
set $m46-outputcentery {110.5}
set $m46-outputcenterz {0}
set $m46-outputsizex {171}
set $m46-outputsizey {221}
set $m46-outputsizez {0}
set $m46-useoutputcenter {1}
set $m46-useoutputsize {0}
set $m46-box-scale {-1}
set $m46-box-center-x {}
set $m46-box-center-y {}
set $m46-box-center-z {}
set $m46-box-right-x {}
set $m46-box-right-y {}
set $m46-box-right-z {}
set $m46-box-down-x {}
set $m46-box-down-y {}
set $m46-box-down-z {}
set $m46-box-in-x {}
set $m46-box-in-y {}
set $m46-box-in-z {}
set $m46-resetting {0}
set $m47-notes {}
set $m47-join-axis {0}
set $m47-incr-dim {0}
set $m47-dim {4}
set $m48-notes {}
#set $m48-add {1}
if {[file exists $DATADIR/$DATASET/demo-DWI.nrrd]} {
    set $m48-filename $DATADIR/$DATASET/demo-B0.nrrd
} else {
    set $m48-filename {}    
} 
set $m49-notes {}
set $m49-port-selected-index {0}
set $m50-notes {}
#set $m50-add {0}
if {[file exists $DATADIR/$DATASET/demo-DWI.nrrd]} {
    set $m50-filename $DATADIR/$DATASET/demo-gradients.txt
} else {
    set $m50-filename {}
} 
# set $m51-notes {}
# set $m51-join-axis {0}
# set $m51-incr-dim {0}
# set $m51-dim {4}
# set $m52-notes {}
# set $m52-use-default-threshold {1}
# set $m52-threshold {100}
# set $m52-soft {0}
# set $m52-scale {1}
# set $m53-notes {}
# set $m53-aniso_metric {tenAniso_FA}
# set $m53-threshold {100}
set $m54-notes {}
set $m54-build-eigens {1}
# set $m55-notes {}
# set $m55-isoval {0.5000}
# set $m55-isoval-min {0}
# set $m55-isoval-max {1}
# set $m55-isoval-typed {0}
# set $m55-isoval-quantity {1}
# set $m55-quantity-range {colormap}
# set $m55-quantity-min {0}
# set $m55-quantity-max {100}
# set $m55-isoval-list {0.0 1.0 2.0 3.0}
# set $m55-extract-from-new-field {1}
# set $m55-algorithm {1}
# set $m55-build_trisurf {0}
# set $m55-np {1}
# set $m55-active-isoval-selection-tab {0}
# set $m55-active_tab {NOISE}
# set $m55-update_type {on release}
# set $m55-color-r {0.40}
# set $m55-color-g {0.78}
# set $m55-color-b {0.73}
# set $m56-notes {}
# set $m56-outputcenterx {-95.5}
# set $m56-outputcentery {110.5}
# set $m56-outputcenterz {51}
# set $m56-outputsizex {171}
# set $m56-outputsizey {221}
# set $m56-outputsizez {102}
# set $m56-useoutputcenter {1}
# set $m56-useoutputsize {0}
# set $m56-box-scale {-1}
# set $m56-box-center-x {}
# set $m56-box-center-y {}
# set $m56-box-center-z {}
# set $m56-box-right-x {}
# set $m56-box-right-y {}
# set $m56-box-right-z {}
# set $m56-box-down-x {}
# set $m56-box-down-y {}
# set $m56-box-down-z {}
# set $m56-box-in-x {}
# set $m56-box-in-y {}
# set $m56-box-in-z {}
# set $m56-resetting {0}
set $m57-notes {}
set $m57-axis {0}
set $m57-measure {12}
set $m58-notes {}
set $m58-tcl_status {Calling GenStandardColorMaps!}
set $m58-positionList {}
set $m58-nodeList {}
set $m58-width {398}
set $m58-height {40}
set $m58-mapType {7}
set $m58-minRes {13}
set $m58-resolution {256}
set $m58-realres {256}
set $m58-gamma {0}
set $m59-notes {}
set $m59-isFixed {0}
set $m59-min {3.0945742130279541}
set $m59-max {130.40342712402344}
set $m59-makeSymmetric {0}
set $m60-notes {}
set $m60-axis {0}
set $m60-measure {12}
set $m61-notes {}
set $m61-isFixed {0}
set $m61-min {2.8124227523803711}
set $m61-max {105.93144989013672}
set $m61-makeSymmetric {0}
set $m62-notes {}
set $m63-notes {}
#set $m63-add {1}
set $m63-filename {}
# set $m64-notes {}
# set $m64-threshold {0.5}
# set $m65-notes {}
# set $m65-num-axes {4}
# set $m65-minAxis0 {0}
# set $m65-maxAxis0 {0}
# set $m65-absmaxAxis0 {2}
# set $m65-minAxis1 {0}
# set $m65-maxAxis1 {0}
# set $m65-absmaxAxis1 {0}
# set $m65-minAxis2 {0}
# set $m65-maxAxis2 {0}
# set $m65-absmaxAxis2 {0}
# set $m65-minAxis3 {0}
# set $m65-maxAxis3 {0}
# set $m65-absmaxAxis3 {0}
# set $m65-minAxis0 {0}
# set $m65-maxAxis0 {0}
# set $m65-absmaxAxis0 {2}
# set $m66-notes {}
# set $m66-input-label {Unknown:Vector}
# set $m66-output-label {unknown:Vector}
set $m67-notes {}
set $m67-sizex {128}
set $m67-sizey {128}
set $m67-axis {0}
set $m67-padpercent {0}
set $m67-pos {0}
set $m67-data-at {Nodes}
set $m67-update_type {on release}
set $m68-notes {}
set $m68-sizex {128}
set $m68-sizey {128}
set $m68-axis {1}
set $m68-padpercent {0}
set $m68-pos {0}
set $m68-data-at {Nodes}
set $m68-update_type {on release}
set $m69-notes {}
set $m69-sizex {128}
set $m69-sizey {128}
set $m69-axis {2}
set $m69-padpercent {0}
set $m69-pos {0}
set $m69-data-at {Nodes}
set $m69-update_type {on release}
# set $m70-notes {}
# set $m71-notes {}
# set $m72-notes {}
set $m73-notes {}
set $m74-notes {}
set $m75-notes {}
set $m76-notes {}
set $m76-interpolation_basis {linear}
set $m76-map_source_to_single_dest {0}
set $m76-exhaustive_search {0}
set $m76-exhaustive_search_max_dist {-1}
set $m76-np {1}
set $m77-notes {}
set $m77-isoval {0.5}
set $m77-lte {0}
set $m78-notes {}
set $m78-interpolation_basis {linear}
set $m78-map_source_to_single_dest {0}
set $m78-exhaustive_search {0}
set $m78-exhaustive_search_max_dist {-1}
set $m78-np {1}
set $m79-notes {}
set $m79-interpolation_basis {linear}
set $m79-map_source_to_single_dest {0}
set $m79-exhaustive_search {0}
set $m79-exhaustive_search_max_dist {-1}
set $m79-np {1}
set $m80-notes {}
set $m80-isoval {0.5}
set $m80-lte {0}
set $m81-notes {}
set $m81-isoval {0.5}
set $m81-lte {0}
set $m82-notes {}
set $m82-port-selected-index {0}
set $m83-notes {}
set $m83-port-selected-index {0}
set $m84-notes {}
set $m84-interpolation_basis {linear}
set $m84-map_source_to_single_dest {0}
set $m84-exhaustive_search {0}
set $m84-exhaustive_search_max_dist {-1}
set $m84-np {1}
set $m85-notes {}
set $m85-interpolation_basis {linear}
set $m85-map_source_to_single_dest {0}
set $m85-exhaustive_search {0}
set $m85-exhaustive_search_max_dist {-1}
set $m85-np {1}
set $m86-notes {}
set $m86-nodes-on {0}
set $m86-nodes-transparency {0}
set $m86-nodes-as-disks {0}
set $m86-edges-on {0}
set $m86-edges-transparency {0}
set $m86-faces-on {1}
set $m86-use-normals {0}
set $m86-use-transparency {0}
set $m86-vectors-on {0}
set $m86-normalize-vectors {}
set $m86-has_vector_data {0}
set $m86-bidirectional {0}
set $m86-vector-usedefcolor {0}
set $m86-tensors-on {0}
set $m86-has_tensor_data {0}
set $m86-scalars-on {0}
set $m86-scalars-transparency {0}
set $m86-has_scalar_data {1}
set $m86-text-on {0}
set $m86-text-use-default-color {1}
set $m86-text-color-r {1.0}
set $m86-text-color-g {1.0}
set $m86-text-color-b {1.0}
set $m86-text-backface-cull {0}
set $m86-text-fontsize {1}
set $m86-text-precision {2}
set $m86-text-render_locations {0}
set $m86-text-show-data {1}
set $m86-text-show-nodes {0}
set $m86-text-show-edges {0}
set $m86-text-show-faces {0}
set $m86-text-show-cells {0}
set $m86-def-color-r {0.5}
set $m86-def-color-g {0.5}
set $m86-def-color-b {0.5}
set $m86-def-color-a {0.5}
set $m86-node_display_type {Points}
set $m86-edge_display_type {Lines}
set $m86-data_display_type {Arrows}
set $m86-tensor_display_type {Boxes}
set $m86-scalar_display_type {Points}
set $m86-active_tab {Faces}
set $m86-node_scale {0.0300}
set $m86-edge_scale {0.0150}
set $m86-vectors_scale {0.30}
set $m86-tensors_scale {0.30}
set $m86-scalars_scale {0.300}
set $m86-show_progress {}
set $m86-interactive_mode {Interactive}
set $m86-field-name {}
set $m86-field-name-update {1}
set $m86-node-resolution {6}
set $m86-edge-resolution {6}
set $m86-data-resolution {6}
set $m87-notes {}
set $m87-nodes-on {0}
set $m87-nodes-transparency {0}
set $m87-nodes-as-disks {0}
set $m87-edges-on {0}
set $m87-edges-transparency {0}
set $m87-faces-on {1}
set $m87-use-normals {0}
set $m87-use-transparency {0}
set $m87-vectors-on {0}
set $m87-normalize-vectors {}
set $m87-has_vector_data {0}
set $m87-bidirectional {0}
set $m87-vector-usedefcolor {0}
set $m87-tensors-on {0}
set $m87-has_tensor_data {0}
set $m87-scalars-on {0}
set $m87-scalars-transparency {0}
set $m87-has_scalar_data {1}
set $m87-text-on {0}
set $m87-text-use-default-color {1}
set $m87-text-color-r {1.0}
set $m87-text-color-g {1.0}
set $m87-text-color-b {1.0}
set $m87-text-backface-cull {0}
set $m87-text-fontsize {1}
set $m87-text-precision {2}
set $m87-text-render_locations {0}
set $m87-text-show-data {1}
set $m87-text-show-nodes {0}
set $m87-text-show-edges {0}
set $m87-text-show-faces {0}
set $m87-text-show-cells {0}
set $m87-def-color-r {0.5}
set $m87-def-color-g {0.5}
set $m87-def-color-b {0.5}
set $m87-def-color-a {0.5}
set $m87-node_display_type {Points}
set $m87-edge_display_type {Lines}
set $m87-data_display_type {Arrows}
set $m87-tensor_display_type {Boxes}
set $m87-scalar_display_type {Points}
set $m87-active_tab {Faces}
set $m87-node_scale {0.0300}
set $m87-edge_scale {0.0150}
set $m87-vectors_scale {0.30}
set $m87-tensors_scale {0.30}
set $m87-scalars_scale {0.300}
set $m87-show_progress {}
set $m87-interactive_mode {Interactive}
set $m87-field-name {}
set $m87-field-name-update {1}
set $m87-node-resolution {6}
set $m87-edge-resolution {6}
set $m87-data-resolution {6}
#set $m88-notes {}
set $m89-notes {}
set $m89-clipmode {allnodes}
set $m89-clipfunction {v > 0.5}
set $m90-notes {}
set $m90-force-pointcloud {1}
set $m91-notes {}
set $m91-interpolation_basis {linear}
set $m91-map_source_to_single_dest {0}
set $m91-exhaustive_search {0}
set $m91-exhaustive_search_max_dist {-1}
set $m91-np {1}
set $m92-notes {}
set $m92-maxseeds {15}
set $m92-numseeds {10}
set $m92-rngseed {1}
set $m92-rnginc {1}
set $m92-clamp {0}
set $m92-autoexecute {1}
set $m92-type {}
set $m92-dist {uniuni}
set $m92-whichtab {Widget}
set $m93-notes {}
set $m93-port-selected-index {2}
set $m94-notes {}
set $m94-show-value {0}
set $m94-show-node {0}
set $m94-show-edge {0}
set $m94-show-face {0}
set $m94-show-cell {0}
set $m95-notes {}
set $m95-nodes-on {0}
set $m95-nodes-transparency {0}
set $m95-nodes-as-disks {0}
set $m95-edges-on {0}
set $m95-edges-transparency {0}
set $m95-faces-on {0}
set $m95-use-normals {0}
set $m95-use-transparency {0}
set $m95-vectors-on {0}
set $m95-normalize-vectors {}
set $m95-has_vector_data {0}
set $m95-bidirectional {0}
set $m95-vector-usedefcolor {0}
set $m95-tensors-on {0}
set $m95-has_tensor_data {1}
set $m95-scalars-on {0}
set $m95-scalars-transparency {0}
set $m95-has_scalar_data {0}
set $m95-text-on {0}
set $m95-text-use-default-color {1}
set $m95-text-color-r {1.0}
set $m95-text-color-g {1.0}
set $m95-text-color-b {1.0}
set $m95-text-backface-cull {0}
set $m95-text-fontsize {1}
set $m95-text-precision {2}
set $m95-text-render_locations {0}
set $m95-text-show-data {1}
set $m95-text-show-nodes {0}
set $m95-text-show-edges {0}
set $m95-text-show-faces {0}
set $m95-text-show-cells {0}
set $m95-def-color-r {0.5}
set $m95-def-color-g {0.5}
set $m95-def-color-b {0.5}
set $m95-def-color-a {0.5}
set $m95-node_display_type {Spheres}
set $m95-edge_display_type {Lines}
set $m95-data_display_type {Arrows}
set $m95-tensor_display_type {Boxes}
set $m95-scalar_display_type {Points}
set $m95-active_tab {Tensors}
set $m95-node_scale {1.00}
set $m95-edge_scale {0.0150}
set $m95-vectors_scale {0.30}
set $m95-tensors_scale {1.0}
set $m95-scalars_scale {0.300}
set $m95-show_progress {}
set $m95-interactive_mode {Interactive}
set $m95-field-name {}
set $m95-field-name-update {1}
set $m95-node-resolution {6}
set $m95-edge-resolution {6}
set $m95-data-resolution {6}
set $m96-notes {}
set $m96-file {}
set $m96-file-del {}
set $m96-messages {}
set $m97-notes {}
set $m97-dir {.}
set $m97-series-uid {}
set $m97-series-files {}
set $m97-messages {}
set $m97-suid-sel {}
set $m97-series-del {}
set $m98-notes {}
set $m98-port-selected-index {0}
set $m99-notes {}
set $m99-dir {.}
set $m99-series-uid {}
set $m99-series-files {}
set $m99-messages {}
set $m99-suid-sel {}
set $m99-series-del {}
set $m100-notes {}
set $m100-file {}
set $m100-file-del {}
set $m100-messages {}
# set $m101-notes {}
# set $m101-filtertype {gaussian}
# set $m101-dim {4}
# set $m101-sigma {1}
# set $m101-extent {4}
set $m102-notes {}
set $m102-interpolation_basis {linear}
set $m102-map_source_to_single_dest {0}
set $m102-exhaustive_search {0}
set $m102-exhaustive_search_max_dist {-1}
set $m102-np {1}
set $m103-notes {}
set $m103-filtertype {gaussian}
set $m103-dim {4}
set $m103-sigma {1}
set $m103-extent {3}
set $m103-resampAxis0 {x1}
set $m103-resampAxis1 {x1}
set $m103-resampAxis2 {x1}
set $m103-resampAxis3 {=}
set $m104-notes {}
set $m104-port-selected-index {1}
set $m105-notes {}
set $m105-port-selected-index {0}
set $m106-notes {}
set $m106-filtertype {gaussian}
set $m106-dim {4}
set $m106-sigma {1}
set $m106-extent {3}
set $m106-resampAxis0 {x1}
set $m106-resampAxis1 {=}
set $m106-resampAxis2 {=}
set $m106-resampAxis3 {x1}
#set $m107-notes {}
set $m108-notes {}
set $m108-tcl_status {Calling GenStandardColorMaps!}
set $m108-positionList {}
set $m108-nodeList {}
set $m108-width {1}
set $m108-height {1}
set $m108-mapType {2}
set $m108-minRes {12}
set $m108-resolution {256}
set $m108-realres {256}
set $m108-gamma {0}
set $m109-notes {}
set $m109-isFixed {0}
set $m109-min {0}
set $m109-max {0.99999970197677612}
set $m109-makeSymmetric {0}
set $m110-notes {}
set $m110-label {unknown}
set $m111-notes {}
set $m111-build-eigens {1}
set $m112-notes {}
set $m112-min {0.0001}
set $m112-max {NaN}
set $m113-notes {}
set $m113-port-selected-index {0}
set $m114-notes {}
set $m114-port-selected-index {0}
set $m115-notes {}
set $m115-scale {1.0}
set $m116-notes {}
set $m116-major-weight {1.0}
set $m116-medium-weight {1.0}
set $m116-minor-weight {1.0}
set $m116-amount {1.0}
set $m116-target {1.0}
set $m117-notes {}
set $m117-port-selected-index {0}
set $m118-notes {}
set $m118-port-selected-index {1}
set $m119-notes {}
set $m119-force-pointcloud {1}
set $m120-notes {}
set $m120-port-selected-index {1}
set $m121-notes {}
set $m121-interpolation_basis {linear}
set $m121-map_source_to_single_dest {0}
set $m121-exhaustive_search {0}
set $m121-exhaustive_search_max_dist {-1}
set $m121-np {1}
set $m122-notes {}
set $m122-port-selected-index {3}
set $m123-notes {}
set $m123-fibertype {tensorline}
set $m123-puncture {0.0}
set $m123-neighborhood {2.0}
set $m123-stepsize {0.005}
set $m123-integration {Euler}
set $m123-use-aniso {1}
set $m123-aniso-metric {tenAniso_Cl2}
set $m123-aniso-thresh {0.14}
set $m123-use-length {1}
set $m123-length {1}
set $m123-use-steps {0}
set $m123-steps {200}
set $m123-use-conf {1}
set $m123-conf-thresh {0.5}
set $m123-kernel {tent}
set $m124-notes {}
set $m124-interpolation_basis {linear}
set $m124-map_source_to_single_dest {0}
set $m124-exhaustive_search {0}
set $m124-exhaustive_search_max_dist {-1}
set $m124-np {1}
set $m125-notes {}
set $m125-nodes-on {0}
set $m125-nodes-transparency {0}
set $m125-nodes-as-disks {0}
set $m125-edges-on {0}
set $m125-edges-transparency {0}
set $m125-faces-on {0}
set $m125-use-normals {0}
set $m125-use-transparency {0}
set $m125-vectors-on {0}
set $m125-normalize-vectors {}
set $m125-has_vector_data {0}
set $m125-bidirectional {0}
set $m125-vector-usedefcolor {0}
set $m125-tensors-on {0}
set $m125-has_tensor_data {1}
set $m125-tensor-usedefcolor {0}
set $m125-scalars-on {0}
set $m125-scalars-transparency {0}
set $m125-has_scalar_data {0}
set $m125-text-on {0}
set $m125-text-use-default-color {1}
set $m125-text-color-r {1.0}
set $m125-text-color-g {1.0}
set $m125-text-color-b {1.0}
set $m125-text-backface-cull {0}
set $m125-text-fontsize {1}
set $m125-text-precision {2}
set $m125-text-render_locations {0}
set $m125-text-show-data {1}
set $m125-text-show-nodes {0}
set $m125-text-show-edges {0}
set $m125-text-show-faces {0}
set $m125-text-show-cells {0}
set $m125-def-color-r {0.5}
set $m125-def-color-g {0.5}
set $m125-def-color-b {0.5}
set $m125-def-color-a {0.5}
set $m125-node_display_type {Spheres}
set $m125-edge_display_type {Cylinders}
set $m125-data_display_type {Arrows}
set $m125-tensor_display_type {Boxes}
set $m125-scalar_display_type {Points}
set $m125-active_tab {Edges}
set $m125-node_scale {0.240}
set $m125-edge_scale {0.1250}
set $m125-vectors_scale {0.30}
set $m125-tensors_scale {3.000000e-01}
set $m125-scalars_scale {0.30}
set $m125-show_progress {}
set $m125-interactive_mode {Interactive}
set $m125-field-name {}
set $m125-field-name-update {1}
set $m125-node-resolution {10}
set $m125-edge-resolution {10}
set $m125-data-resolution {6}
set $m126-notes {}
set $m126-tcl_status {Calling GenStandardColorMaps!}
set $m126-positionList {}
set $m126-nodeList {}
set $m126-width {1}
set $m126-height {1}
set $m126-mapType {2}
set $m126-minRes {12}
set $m126-resolution {256}
set $m126-realres {256}
set $m126-gamma {0}
set $m127-notes {}
set $m127-isFixed {0}
set $m127-min {0}
set $m127-max {1}
set $m127-makeSymmetric {0}
set $m128-notes {}
set $m128-evec {0}
set $m128-aniso_metric {tenAniso_FA}
set $m128-background {0.0}
set $m128-gray {0.5}
set $m128-gamma {1.6}
#set $m129-notes {}
set $m130-notes {}
set $m130-show-value {0}
set $m130-show-node {0}
set $m130-show-edge {0}
set $m130-show-face {0}
set $m130-show-cell {0}

set $m131-maxseeds {75}
set $m131-endpoint0x {62.222220637490381}
set $m131-endpoint0y {49.102145897390322}
set $m131-endpoint0z {32.328802538672456}
set $m131-endpoint1x {66.779997917931084}
set $m131-endpoint1y {145.80466651910558}
set $m131-endpoint1z {37.932737971668956}
set $m131-widgetscale {4.0164163130830941}
set $m131-endpoints {1}
set $m131-done_bld_icon {0}

set $m132-notes {}
set $m132-port-selected-index {1}
set $m133-notes {}
set $m133-port-selected-index {1}
set $m134-notes {}
set $m134-port-selected-index {1}
set $m135-notes {}
set $m135-port-selected-index {1}
set $m136-port-selected-index {1}

set $m137-kind {nrrdKind3DMaskedSymTensor}

set $m138-filename {/tmp/tensors.nrrd}

::netedit scheduleok


# global array indexed by module name to keep track of modules
global mods

set mods(NrrdReader1) $m0
set mods(DicomToNrrd1) $m97
set mods(AnalyzeToNrrd1) $m96
set mods(ChooseNrrd1) $m45
set mods(NrrdInfo1) $m62

set mods(UnuAxinsert) $m158
set mods(UnuCrop-DWI) $m165
set mods(UnuSlice-B0) $m166
set mods(ChooseNrrd-B0) $m157
set mods(NrrdInfo-full) $m164
set mods(UnuAxinfo-X) $m161
set mods(UnuAxinfo-Y) $m162
set mods(UnuAxinfo-Z) $m163
set mods(UnuReshape) $m159
set mods(UnuPermute) $m160
set mods(ChooseNrrd-preprocess) $m167


### Original Data Stuff
set mods(UnuSlice1) $m1
set mods(UnuProject1) $m57
set mods(ShowField-Orig) $m8
set mods(ChooseNrrd-ToProcess) $m113

### Registered
set mods(ShowField-Reg) $m9
set mods(UnuSlice2) $m5
set mods(UnuJoin) $m47
set mods(TendEpireg) $m2
set mods(ChooseNrrd-BMatrix) $m49
set mods(ChooseNrrd-ToReg) $m105
set mods(ChooseNrrd-KnownB0) $m136
set mods(ChooseNrrd-ToSmooth) $m104
set mods(ChangeFieldBounds-Variance) $m46

set mods(NrrdReader-Gradient) $m50

set mods(NrrdReader-BMatrix) $m63

### T2 Reference Image
set mods(NrrdReader-T2) $m48
set mods(DicomToNrrd-T2) $m99
set mods(AnalyzeToNrrd-T2) $m100
set mods(ChooseNrrd-T2) $m98


set mods(GenStandardColorMaps1)  $m58
set mods(RescaleColorMap2) $m61

### Build DT
set mods(TendEstim) $m3
set mods(UnuResample-XY) $m103
set mods(UnuResample-Z) $m106
set mods(ChooseNrrd-DT) $m114
set mods(UnuSave-Tensors) $m138
set mods(ChooseNrrd-Save) $m139

### Planes
set mods(ChooseField-ColorPlanes) $m27
set mods(GenStandardColorMaps-ColorPlanes) $m29
set mods(RescaleColorMap-ColorPlanes) $m30

### Isosurface
set mods(DirectInterpolate-Isosurface) $m18
set mods(ShowField-Isosurface) $m13
set mods(ChooseField-Isoval) $m23
set mods(ChooseField-Isosurface) $m24
set mods(GenStandardColorMaps-Isosurface) $m26
set mods(RescaleColorMap-Isosurface) $m25
set mods(Isosurface) $m17
set mods(ChooseColorMap-Isosurface) $m132

# Planes
set mods(SamplePlane-X) $m67
set mods(SamplePlane-Y) $m68
set mods(SamplePlane-Z) $m69

set mods(QuadToTri-X) $m73
set mods(QuadToTri-Y) $m74
set mods(QuadToTri-Z) $m75

set mods(IsoClip-X) $m77
set mods(IsoClip-Y) $m80
set mods(IsoClip-Z) $m81

set mods(ChooseField-X) $m33
set mods(ChooseField-Y) $m82
set mods(ChooseField-Z) $m83

set mods(ShowField-X) $m31
set mods(ShowField-Y) $m86
set mods(ShowField-Z) $m87

set mods(ChooseColorMap-Planes) $m133

### Glyphs
set mods(NrrdToField-GlyphSeeds) $m10
set mods(Probe-GlyphSeeds) $m94
set mods(SampleField-GlyphSeeds) $m92
set mods(DirectInterpolate-GlyphSeeds) $m102
set mods(ClipByFunction-Seeds) $m89
set mods(ShowField-Glyphs) $m95
set mods(ChooseField-GlyphSeeds) $m93
set mods(ChooseField-Glyphs) $m34
set mods(GenStandardColorMaps-Glyphs) $m108
set mods(RescaleColorMap-Glyphs) $m109
set mods(DirectInterpolate-Glyphs) $m91
set mods(TendNorm-Glyphs) $m116
set mods(TendAnscale-Glyphs) $m115
set mods(ChooseNrrd-Norm) $m117
set mods(ChooseNrrd-Exag) $m118
set mods(ChooseColorMap-Glyphs) $m134

### Fibers
set mods(ChooseField-FiberSeeds) $m120
set mods(GenStandardColorMaps-Fibers) $m126
set mods(RescaleColorMap-Fibers) $m127
set mods(Probe-FiberSeeds) $m130
set mods(SampleField-FiberSeeds) $m131
set mods(DirectInterpolate-FiberSeeds) $m124
set mods(ShowField-Fibers) $m125
set mods(ChooseField-Fibers) $m122
set mods(DirectInterpolate-Fibers) $m121
set mods(TendFiber) $m123
set mods(ChooseColorMap-Fibers) $m135
set mods(GatherPoints) $m90

### Anisotropy modules
set mods(TendAnvol-0) $m11
set mods(TendAnvol-1) $m19
set mods(TendAnvol-2) $m20
set mods(TendEvecRGB) $m128
set mods(NrrdToField-0) $m12
set mods(NrrdToField-1) $m21
set mods(NrrdToField-2) $m22
set mods(NrrdToField-3) $m16


### Viewer
set mods(Viewer) $m44

# Tooltips
global tips

global data_mode
set data_mode "DWIknownB0"
global fast_axis
set fast_axis "volumes"
global B0_first
set B0_first 1
global channels
set channels 6

### planes variables that must be globals (all checkbuttons)
global show_planes
set show_planes 1
global show_plane_x
set show_plane_x 1
global show_plane_y
set show_plane_y 1
global show_plane_z
set show_plane_z 1
global plane_x
set plane_x 0
global plane_y
set plane_y 0
global plane_z
set plane_z 0
global exec_planes
set exec_planes(GatherPoints) 0
set exec_planes(ChooseField-ColorPlanes) 0
set exec_planes(GenStandardColorMaps-ColorPlanes) 0
set exec_planes(ShowField-X) 0
set exec_planes(ShowField-Y) 0
set exec_planes(ShowField-Z) 0
set exec_planes(update-X) 0
set exec_planes(update-Y) 0
set exec_planes(update-Z) 0


### registration globals
global ref_image
set ref_image 1
global ref_image_state
set ref_image_state 0

global clip_to_isosurface
set clip_to_isosurface 0

global clip_to_isosurface_color
set clip_to_isosurface_color ""
global clip_to_isosurface_color-r
set clip_to_isosurface_color-r 0.5
global clip_to_isosurface_color-g
set clip_to_isosurface_color-g 0.5
global clip_to_isosurface_color-b
set clip_to_isosurface_color-b 0.5

global bmatrix
set bmatrix "compute"

### DT Globals
global xy_radius
set xy_radius 1.0
global z_radius 
set z_radius 1.0
global save_tensors
set save_tensors 0

### isosurface variables
global clip_by_planes
set clip_by_planes 0

global do_registration 
set do_registration 1

global do_smoothing
set do_smoothing 0

global isosurface_color
set isosurface_color ""
global isosurface_color-r
set isosurface_color-r 0.5
global isosurface_color-g
set isosurface_color-g 0.5
global isosurface_color-b
set isosurface_color-b 0.5

global exec_iso
set exec_iso(Isosurface) 0
set exec_iso(IsoClip-X) 0
set exec_iso(IsoClip-Y) 0
set exec_iso(IsoClip-Z) 0
set exec_iso(ChooseField-Isoval) 0
set exec_iso(ShowField-Isosurface) 0
set exec_iso(GenStandardColorMaps-Isosurface) 0
set exec_iso(global-clip) "off"

# glyphs
global glyph_display_type
set glyph_display_type boxes

global glyph_color
set glyph_color ""
global glyph_color-r
set glyph_color-r 0.5
global glyph_color-g
set glyph_color-g 0.5
global glyph_color-b
set glyph_color-b 0.5

global scale_glyph
set scale_glyph 1

global glyph_scale_val
set glyph_scale_val 0.5

global exag_glyph
set exag_glyph 0

global glyph_rake
set glyph_rake 1

global glyph_point
set glyph_point 1

global exec_glyphs
set exec_glyphs(ChooseNrrd-Norm) 0
set exec_glyphs(TendNorm-Glyphs) 0
set exec_glyphs(ShowField-Glyphs) 0 
set exec_glyphs(TendAnscale-Glyphs) 0 
set exec_glyphs(ChooseField-Glyphs) 0
set exec_glyphs(ChooseField-GlyphSeeds) 0
set exec_glyphs(GenStandardColorMaps-Glyphs) 0


# fibers
global fiber_color
set fiber_color ""
global fiber_color-r
set fiber_color-r 0.5
global fiber_color-g
set fiber_color-g 0.5
global fiber_color-b
set fiber_color-b 0.5

global fibers_stepsize
set fibers_stepsize 0.5

global fibers_length
set fibers_length 100

global fibers_steps
set fibers_steps 200

global fiber_rake
set fiber_rake 1

global fiber_point
set fiber_point 1

global exec_fibers
set exec_fibers(TendFiber) 0
set exec_fibers(ChooseField-FiberSeeds) 0
set exec_fibers(ChooseField-Fibers) 0
# rerender_edges
set exec_fibers(ShowField-Fibers) 0
set exec_fibers(GenStandardColorMaps-Fibers) 0


setProgressText "Loading BioTensor Application, Please Wait..."

                                                                               
#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

set auto_index(::PowerAppBase) "source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/PowerAppBase.app"

class BioTensorApp {
    inherit ::PowerAppBase
        
    constructor {} {
	toplevel .standalone
	wm title .standalone "BioTensor"	 
	set win .standalone

	# Set window sizes
	set viewer_width 436
	set viewer_height 566
	
	set notebook_width 281
	set notebook_height [expr $viewer_height - 50]
	
	set process_width 267
	set process_height $viewer_height
	
	set vis_width [expr $notebook_width + 30]
	set vis_height $viewer_height

	# set state variables
	set data_completed 0
	set reg_completed 0
	set dt_completed 0
	set vis_activated 0

	set c_procedure_tab "Load Data"
	set c_data_tab "Generic"
	set c_left_tab "Vis Options"
	set c_vis_tab "Variance"

	set last_B0_port 0

        set proc_tab1 ""
        set proc_tab2 ""

        set vis_frame_tab1 ""
        set vis_frame_tab2 ""

        set data_tab1 ""
        set data_tab2 ""

        set vis_tab1 ""
        set vis_tab2 ""
     
        set reg_tab1 ""
        set reg_tab2 ""

        set dt_tab1 ""
        set dt_tab2 ""

	set volumes 0
        set size_x 0
	set spacing_x 0
	set min_x 0

        set size_y 0
	set spacing_y 0
	set min_y 0

        set size_z 0
	set spacing_z 0
	set min_z 0

	set average_spacing 0

        set ref_image1 ""
        set ref_image2 ""

        set reg_thresh1 ""
        set reg_thresh2 ""

	# Vis tabs
        set variance_tab1 ""
        set variance_tab2 ""

        set planes_tab1 ""
        set planes_tab2 ""

        set isosurface_tab1 ""
        set isosurface_tab2 ""

        set glyphs_tab1 ""
        set glyphs_tab2 ""

        set fibers_tab1 ""
        set fibers_tab2 ""

	# Load Data internal tabs
        set nrrd_tab1 ""
        set nrrd_tab2 ""
        set dicom_tab1 ""
        set dicom_tab2 ""
        set analyze_tab1 ""
        set analyze_tab2 ""

	set data_next_button1 ""
	set data_next_button2 ""
	set data_ex_button1 ""
	set data_ex_button2 ""

        # planes
        set last_x 2
        set last_y 4
        set last_z 6
        set plane_inc "-0.1"
        set plane_type "Principle Eigenvector"

	#isosurfaces
        set iso_type "Principle Eigenvector"

        # glyphs
        set clip_x "<"
        set clip_y "<"
        set clip_z "<"
        set glyph_type "Principle Eigenvector"

	# fibers
        set fiber_type "Principle Eigenvector"

        # colormaps
        set colormap_width 80
        set colormap_height 15
        set colormap_res 64

        set has_autoviewed 0
	
	### Define Tooltips
	##########################
	global tips

 	# Process Tabs
 	set tips(LoadDataTab) \
 	    "Load DWI images,\na T2 Reference image\n or Tensors depending\non mode."
 	set tips(RegistrationTab) \
 	    "EPI Registration paramters."
 	set tips(BuildTensorsTab) \
 	    "Parameters for\nBuilding Tensors."

 	# Data Tab
 	set tips(DataExecute) \
 	    "Click to load data"
 	set tips(DataNext) \
 	    "Click to proceed to\nthe Registration step\nonly after completing\nthe Load Data step.\nIf Tensors were loaded\ndirectly, proceed to\nvisualization."
	set tips(NrrdTab) \
	    "Load Data in\nNrrd format."
	set tips(NrrdFile1) \
	    "Specify a .nrrd or\n.nhdr file."
	set tips(NrrdFile2) \
	    "Browse to a .nrrd\nor .nhdr file."
	set tips(DicomTab) \
	    "Load Data in\nDicom format."
	set tips(DicomFiles) \
	    "Load a series of\nDicom images using\nour Dicom Loader."
	set tips(AnalyzeTab) \
	    "Load Data in\nAnalyze format."
	set tips(AnalyzeFiles) \
	    "Load Analyze files\nusing our Analyze\nLoader."
	set tips(LoadModeDWIknownB0) \
	    "Load a set of Diffusion\nWeighted Images. These\ncan be registered and\nused to build diffusion\ntensors. B0 is in a\nseparate file."
	set tips(LoadModeB0DWI) \
	    "Load a set of Diffusion\nWeighted Images. These\ncan be registered and\nused to build diffusion\ntensors. B0 is first\nimage in 4D volume."
	set tips(LoadModeDWI) \
	    "Load a set of Diffusion\nWeighted Images\nwithout any B.  These\ncan be registered and\nused to build diffusion\ntensors."
	set tips(LoadModeTensor) \
	    "Load Tensors directly.\nThe Registration and\nBuild Tensors steps\nwill be skipped. "

 	# Registration Tab
 	set tips(RegToggle) \
 	    "Perform EPI Registration\nor skip step entirely and\nbuild tensors."
 	set tips(RegImpRefImg) \
 	    "Select to register all\nimages to each other."
 	set tips(RegChooseRefImg) \
 	    "Select a reference\nimage to register\nall images to."
 	set tips(RegRefImgSlider) \
 	    "Select a reference image\nby adjusting the slider."
 	set tips(RegBlurX) \
 	    "Gaussian Smoothing\nin the X direction.\n(units=samples)"
 	set tips(RegBlurY) \
 	    "Gaussian Smoothing\nin the Y direction\n(units=samples)"
 	set tips(RegFitting) \
 	    "Select the percentage of\nslices for parameter estimation."
 	set tips(RegExecute) \
 	    "Click to apply\nRegistration\nchanges."
 	set tips(RegNext) \
 	    "Click to proceed to\nBuilding Tensors."
	
 	# Build Tensors Tab
 	set tips(DTToggleSmoothing) \
 	    "Perform Smoothing."
 	set tips(DTSmoothXY) \
 	    "DWI Smoothing in the\nX and Y directions.\n(units=samples)"
 	set tips(DTSmoothZ) \
 	    "DWI Smoothing in the\nZ direction.\n(units=samples)"
 	set tips(DTBMatrixCompute) \
 	    "Select to compute the\nB-Matrix using the\nGradients file specified\nin the Registration step."
 	set tips(DTBMatrixLoad) \
 	    "Select to Load a\nB-Matrix file\ndirectly. Specify\nthe file using the\nentry or browse\nbutton."
 	set tips(DTExecute) \
 	    "Click to apply changes\nfor Building Tensors\nand start visualization."

 	# Variance Tab
 	set tips(VarToggleOrig) \
 	    "Turn visibility of\noriginal variance\nslice on or off."
 	set tips(VarToggleReg) \
 	    "Turn visibility of\nregistered variance\nslice on or off."

 	# Planes Tab
 	set tips(PlanesToggle) \
 	    "Turn visibility of all\nplanes on or off."
 	set tips(PlanesXToggle) \
 	    "Turn visibility of\nX plane on or off.\nThis will also effect\nresults when seeding\nby planes."
 	set tips(PlanesYToggle) \
 	    "Turn visibility of\nY plane on or off.\nThis will also effect\nresults when seeding\nby planes."
 	set tips(PlanesZToggle) \
 	    "Turn visibility of\nZ plane on or off.\nThis will also effect\nresults when seeding\nby planes."
 	set tips(PlanesXSlider) \
 	    "Select a position of\nthe plane in X.\nThis applies to the\nvisible planes,\nclipping planes,\nand grid selections."
 	set tips(PlanesYSlider) \
 	    "Select a position of\nthe plane in Y.\nThis applies to the\nvisible planes,\nclipping planes,\nand grid selections."
 	set tips(PlanesZSlider) \
 	    "Select a position of\nthe plane in Z.\nThis applies to the\nvisible planes,\nclipping planes,\nand grid selections."
 	set tips(PlanesColorMap) \
 	    "Select a colormap to\ncolor the planes. This\nwill not apply when\nPrinciple Eigenvector\nis selected."
 	set tips(PlanesClipToIso) \
 	    "Select to clip the planes\nto the Isosurface."

	# Isosurface Tab
	set tips(IsoToggle) \
	    "Turn visibility of\nisosurface on or off."
	set tips(IsoColorMap) \
 	    "Select a colormap to\ncolor the isosurface.\nThis will not apply when\nPrinciple Eigenvector\nis selected."
	set tips(ToggleClipPlanes) \
	    "Turn clipping by\nplanes on or off.\nThis will clip all\nobjects in the Viewer."
	set tips(FlipX) \
	    "Flip about X Plane where\nclipping occurs. This is\nthe plane specified by\nthe slider on the planes\ntab. This option is only\navailable when Clip by\nPlanes is set to on."
	set tips(FlipY) \
	    "Flip about Y Plane where\nclipping occurs. This is\nthe plane specified by\nthe slider on the planes\ntab. This option is only\navailable when Clip by\nPlanes is set to on."
	set tips(FlipZ) \
	    "Flip about Z Plane where\nclipping occurs. This is\nthe plane specified by\nthe slider on the planes\ntab. This option is only\navailable when Clip by\nPlanes is set to on."



	# Glyphs Tab
	set tips(GlyphsToggle) \
	    "Turn visibility of\nGlyphs on or off."
	set tips(GlyphsRes) \
	    "Change resolution\nof glyphs."
	set tips(GlyphsNormalize) \
	    "Normalize all glyphs."
	set tips(GlyphsScale) \
	    "Scale all glyphs."
	set tips(GlyphsShape) \
	    "Toggle shape exaggeration.\nValues < 1.0 will\nmake it more isotropic.\nValues > 1.0 will make\nit more anisotropic."
	set tips(GlyphsSeedPoint) \
	    "Seed the Glyphs\nat a Point using\nthe Probe widget\n(sphere)."
	set tips(GlyphsSeedLine) \
	    "Seed the Glyphs\nalong a line\nusing the Rake\nwidget."
	set tips(GlyphsSeedPlanes) \
	    "Seed the Glyphs\non the planes."
	set tips(GlyphsSeedGrid) \
	    "Seed the Glyphs\nin the grid defined\nby the planes."
	set tips(GlyphsColorMap) \
 	    "Select a colormap\nto color the Glyphs.\nThis will not apply when\nPrinciple Eigenvector\nis selected."
	set tips(GlyphsTogglePoint) \
	    "Turn the visibility\nof the Probe widget\non or off."
	set tips(GlyphsToggleRake) \
	    "Turn the visibility\nof the Rake widget\non or off."
	set tips(GlyphsBoxes) \
	    "View glyphs\nas boxes."
	set tips(GlyphsEllipsoids) \
	    "View glyphs\n as ellipsoids."
	set tips(GlyphsSQ) \
	    "View glyphs as\nsuper quadrics."


	# Fibers Tab
	set tips(FibersToggle) \
	    "Turn visibility of\nFibers on or off."
	set tips(FibersAlgEigen) \
	    "Select the Major\nEigenvector algorithm."
	set tips(FibersAlgTL) \
	    "Select the TensorLines\nalgorithm."
	set tips(FibersSeedPoint) \
	    "Seed the Fibers\nat a Point\nusing the Probe\nwidget (sphere)."
	set tips(FibersSeedLine) \
	    "Seed the Fibers\nalong a line\nusing the Rake\nwidget."
	set tips(FibersSeedPlanes) \
	    "Seed the Fibers\non the planes."
	set tips(FibersSeedGrid) \
	    "Seed the Fibers\nin the grid defined\nby the planes."
	set tips(FibersTogglePoint) \
	    "Turn the visibility\nof the Probe widget\non or off."
	set tips(FibersToggleLine) \
	    "Turn the visibility\nof the Rake widget\non or off."
	set tips(FibersColorMap) \
 	    "Select a colormap\nto color the fibers.\nThis will not apply when\nPrinciple Eigenvector\nis selected."
	set tips(FibersColorMap) \
 	    "Select a colormap\nto color the Fibers.\nThis will not apply when\nPrinciple Eigenvector\nis selected."


        initialize_blocks
    }
    

    destructor {
	destroy $this
    }


    method appname {} {
	return "BioTensor"
    }


    #############################
    ### initialize_blocks
    #############################
    # Disable modules for any steps beyond first one.
    method initialize_blocks {} { 
	global mods
	
        # Blocking Data Section (Analyze and Dicom modules)
        disableModule $mods(DicomToNrrd1) 1
        disableModule $mods(AnalyzeToNrrd1) 1
	
        disableModule $mods(DicomToNrrd-T2) 1
        disableModule $mods(AnalyzeToNrrd-T2) 1

	disableModule $mods(UnuAxinsert) 1
	disableModule $mods(UnuReshape) 1
	disableModule $mods(UnuAxinfo-Z) 1
	disableModule $mods(UnuCrop-DWI) 1
	disableModule $mods(UnuSlice-B0) 1
	disableModule $mods(ChooseNrrd-preprocess) 1
	disableModule $mods(ChooseNrrd-B0) 1
	#disableModule $mods(UnuCrop-DWI) 1
	#disableModule $mods(ChooseNrrd-B0) 1


	# Blocking Registration
        disableModule $mods(TendEpireg) 1
	disableModule $mods(UnuJoin) 1
        disableModule $mods(ChooseNrrd-ToReg) 1
        disableModule $mods(RescaleColorMap2) 1

	# DT Smoothing is intially turned off
	disableModule $mods(UnuResample-XY) 1
	disableModule $mods(UnuResample-Z) 1
  
        # Building Diffusion Tensors
        disableModule $mods(NrrdReader-BMatrix) 1
        disableModule $mods(TendEstim) 1
	disableModule $mods(ChooseNrrd-DT) 1
	disableModule $mods(UnuSave-Tensors) 1
	disableModule $mods(ChooseNrrd-Save) 1

	# Disable execute buttons so that only the set button
	# can be used for NrrdReaders
	#.ui$mods(NrrdReader1).f7.execute configure -state disabled

	# Disable planar and linear anisotropy modules (1,2)
	disableModule $mods(TendAnvol-1) 1
	disableModule $mods(NrrdToField-1) 1
	disableModule $mods(TendAnvol-2) 1
	disableModule $mods(NrrdToField-2) 1
    }

    
    ############################
    ### build_app
    ############################
    # Build the processing and visualization frames and pack along with viewer
    method build_app {} {
	global mods
	
	# Embed the Viewer
	set viewer_border 3
	set eviewer [$mods(Viewer) ui_embedded]

	#frame $win.viewer -relief sunken -borderwidth $viewer_border
	$eviewer setWindow $win.viewer $viewer_width $viewer_height
	# $eviewer setWindow $win.viewer.v $viewer_width $viewer_height
	#$eviewer setWindow $win.viewer.v [expr $viewer_width - 2*$viewer_border] [expr $viewer_height - 2*$viewer_border]
	
	
	### Processing Part
	#########################
	### Create Detached Processing Part
	toplevel $win.detachedP
	frame $win.detachedP.f -relief flat
	pack $win.detachedP.f -side left -anchor n -fill both -expand 1
	
	wm title $win.detachedP "Processing Window"
	wm protocol $win.detachedP WM_DELETE_WINDOW \
	    { app hide_processing_window }
	
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

	### create detached width and heigh
	append geomP $process_width x $process_height
	wm geometry $detachedPFr $geomP


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
	
	init_Vframe $detachedVFr.f 1
	init_Vframe $attachedVFr.f 2


	### pack 3 frames in proper order so that viewer
	# is the last to be packed and will be the one to resize
	pack $attachedVFr -side right -anchor n 

	pack $attachedPFr -side left -anchor n

	pack $win.viewer -side left -anchor n -fill both -expand 1


	set total_width [expr $process_width + $viewer_width + $vis_width]

	set total_height $viewer_height

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]
	set pos_y [expr [expr $screen_height / 2] - [expr $total_height / 2]]

	append geom $total_width x $total_height + $pos_x + $pos_y
	wm geometry .standalone $geom
	update	

	initialize_clip_info

        set initialized 1

	global PowerAppSession
	if {[info exists PowerAppSession] && [set PowerAppSession] != ""} { 
	    set saveFile $PowerAppSession
	    wm title .standalone "BioTensor - [getFileName $saveFile]"
	    $this load_session_data
	} 
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
                -width [expr $process_width - 40] \
                -height [expr $process_height - 120] \
                -tabpos n -equaltabs 0
	    pack $process.tnb -side top -anchor n 

	    
            set step_tab [$process.tnb add -label "Load Data" -command "$this change_processing_tab Data"]

	    #Tooltip $step_tab $tips(LoadDataTab)
	    
	    set proc_tab$case $process.tnb

	    # Radiobuttons for by-passing any processing and loading
	    # the tensors directly
            global data_mode

            radiobutton $step_tab.mode1 -text "Just Diffusion Weighted Images (DWI)" \
                -variable data_mode -value "DWI" \
                -command "$this toggle_data_mode"

	    Tooltip $step_tab.mode1 $tips(LoadModeDWI)

            radiobutton $step_tab.mode2 -text "DWI with separate B0 Reference Data" \
                -variable data_mode -value "DWIknownB0" \
                -command "$this toggle_data_mode"

	    Tooltip $step_tab.mode2 $tips(LoadModeDWIknownB0)


            radiobutton $step_tab.mode3 -text "B0 Reference Data and DWI (single file)" \
                -variable data_mode -value "B0DWI" \
                -command "$this toggle_data_mode"

	    Tooltip $step_tab.mode3 $tips(LoadModeB0DWI)

	    frame $step_tab.mode3quest -relief groove -borderwidth 2

	    global channels
	    iwidgets::entryfield $step_tab.mode3quest.ch -labeltext "Number of channels:" \
		-labelpos w -textvariable channels -width 5 -foreground grey64
	    global fast_axis
	    radiobutton $step_tab.mode3quest.vol -text "Volumes fastest changing axis" \
		-variable fast_axis -value "volumes" -foreground grey64
	    Tooltip $step_tab.mode3quest.vol "The volumes are changing the\nfastest in your data.\nThis means that all of the\nB0 slices are either the first z\slices or last z slices,\nbut not interspersed among\nthe DWI slices."

	    radiobutton $step_tab.mode3quest.z -text "Z fastest changing axis" \
		-variable fast_axis -value "z" -foreground grey64
	    Tooltip $step_tab.mode3quest.z "The z slices are changing the\nfastest in your data.\nThis means that the B0\nslices are interspersed among the\nDWI slices."

	    global B0_first
	    checkbutton $step_tab.mode3quest.b0 -text "B0 slices before DWI slices" \
		-variable B0_first -foreground grey64
	    Tooltip $step_tab.mode3quest.b0 "Regardless of the fastest changing\naxis, the B0 slices are either before\nor after the DWI slices. Check this\nwhen they are before."

	    pack $step_tab.mode3quest.ch \
		-side top -anchor nw -padx 2 -pady 3

	    pack $step_tab.mode3quest.vol $step_tab.mode3quest.z \
		-side top -anchor nw -padx 2 -pady 0

	    pack $step_tab.mode3quest.b0 \
		-side top -anchor nw -padx 2 -pady 3

            radiobutton $step_tab.mode4 -text "Tensor Volumes" \
                -variable data_mode -value "tensor" \
                -command "$this toggle_data_mode"

	    Tooltip $step_tab.mode4 $tips(LoadModeTensor)
	    pack $step_tab.mode1 $step_tab.mode2 $step_tab.mode3 $step_tab.mode4 -side top -anchor nw -padx 3 -pady 3 -expand 0
            pack $step_tab.mode3quest -after $step_tab.mode3 -side top -anchor n \
		-padx 12 -pady 3 -expand 0

	    global $mods(TendEstim)-knownB0
	    if { $data_mode == "DWIknownB0" || $data_mode == "B0DWI"} {
	       set $mods(TendEstim)-knownB0 1
	    } else {
		set $mods(TendEstim)-knownB0 0
	    }

	    ### Data Acquisition
            iwidgets::tabnotebook $step_tab.tnb \
		-width [expr $process_width - 55 ] -height 180 \
		-tabpos n 
            pack $step_tab.tnb -side top -anchor n \
		-padx 3 -pady 8

	    set data_tab$case $step_tab.tnb
	    
	    
            ### Nrrd
            set page [$step_tab.tnb add -label "Generic" -command {app configure_readers Generic}]

	    #Tooltip $page $tips(NrrdTab)

	    set nrrd_tab$case $page
	    
            global $mods(NrrdReader1)-filename
            label $page.dwil -text "DWI Volume:"
            pack $page.dwil -side top -anchor nw -padx 3 -pady 3

            iwidgets::entryfield $page.file -labeltext ".vol/.vff/.nrrd file:" -labelpos w \
                -textvariable $mods(NrrdReader1)-filename \
                -command "$this execute_Data"
	    Tooltip $page.file $tips(NrrdFile1)
            pack $page.file -side top -padx 3 -pady 3 -anchor n \
	        -fill x 
	    
            button $page.load -text "Browse" \
                -command "$this load_nrrd_dwi" \
                -width 12
	    Tooltip $page.load $tips(NrrdFile2)
            pack $page.load -side top -anchor n -padx 3 -pady 3
	    

            global $mods(NrrdReader-T2)-filename
            label $page.t2l -text "T2 Reference Image:"
            pack $page.t2l -side top -anchor nw -padx 3 -pady 3

            iwidgets::entryfield $page.file2 -labeltext ".vol/.vff/.nrrd file:" -labelpos w \
                -textvariable $mods(NrrdReader-T2)-filename 
	    Tooltip $page.file2 $tips(NrrdFile1)
            pack $page.file2 -side top -padx 3 -pady 3 -anchor n \
	        -fill x 
	    
            button $page.load2 -text "Browse" \
                -command "$this load_nrrd_t2" \
                -width 12
	    Tooltip $page.load2 $tips(NrrdFile2)
            pack $page.load2 -side top -anchor n -padx 3 -pady 3
	    
	    
            ### Dicom
            set page [$step_tab.tnb add -label "Dicom" -command {app configure_readers Dicom}]

	    #Tooltip $page $tips(DicomTab)

	    set dicom_tab$case $page
	    
            label $page.dwil -text "DWI Volume:"
            pack $page.dwil -side top -anchor nw -padx 3 -pady 3

	    button $page.load -text "Dicom Loader" \
		-command "$this dicom_ui $mods(DicomToNrrd1)"
	    Tooltip $page.load $tips(DicomFiles)

	    pack $page.load -side top -anchor n \
		-padx 3 -pady 5 -ipadx 2 -ipady 2
	    
            label $page.space -text "    "
            pack $page.space -side top -anchor n -padx 3 -pady 0

            label $page.t2l -text "T2 Reference Image:"
            pack $page.t2l -side top -anchor nw -padx 3 -pady 3

	    button $page.load2 -text "Dicom Loader" \
		-command "$this dicom_ui $mods(DicomToNrrd-T2)"
	    Tooltip $page.load2 $tips(DicomFiles)

	    pack $page.load2 -side top -anchor n \
		-padx 3 -pady 5 -ipadx 2 -ipady 2
	    

            ### Analyze
            set page [$step_tab.tnb add -label "Analyze" -command {app configure_readers Analyze}]

	    #Tooltip $page $tips(AnalyzeTab)

	    set analyze_tab$case $page
	    
            label $page.dwil -text "DWI Volume:"
            pack $page.dwil -side top -anchor nw -padx 3 -pady 3

	    button $page.load -text "Analyze Loader" \
		-command "$this analyze_ui $mods(AnalyzeToNrrd1)"
	    Tooltip $page.load $tips(AnalyzeFiles)

	    pack $page.load -side top -anchor n \
		-padx 3 -pady 5 -ipadx 2 -ipady 2

            label $page.space -text "    "
            pack $page.space -side top -anchor n -padx 3 -pady 0

            label $page.t2l -text "T2 Reference Image:"
            pack $page.t2l -side top -anchor nw -padx 3 -pady 3
	    
	    button $page.load2 -text "Analyze Loader" \
		-command "$this analyze_ui $mods(AnalyzeToNrrd-T2)"
	    Tooltip $page.load2 $tips(AnalyzeFiles)

	    pack $page.load2 -side top -anchor n \
		-padx 3 -pady 5 -ipadx 2 -ipady 2
            

	    # Set default view to be Nrrd
            $step_tab.tnb view "Generic"
	    
	    
	    # Execute and Next buttons
            frame $step_tab.last
            pack $step_tab.last -side bottom -anchor ne \
		-padx 3 -pady 3
	    
            button $step_tab.last.ex -text "Execute" \
		-background $execute_color \
		-activebackground $execute_color \
		-width 8 \
		-command "$this execute_Data"
	    Tooltip $step_tab.last.ex $tips(DataExecute)

	    button $step_tab.last.ne -text "Next" \
                -command "$this change_processing_tab Registration" -width 8 \
                -activebackground $next_color \
                -background grey75 -state disabled 
	    Tooltip $step_tab.last.ne $tips(DataNext)

            pack $step_tab.last.ne $step_tab.last.ex -side right -anchor ne \
		-padx 2 -pady 0

	    set data_next_button$case $step_tab.last.ne
	    set data_ex_button$case $step_tab.last.ex
	    
	    ### Registration
            set step_tab [$process.tnb add -label "Registration" -command "$this change_processing_tab Registration"]          

	    #Tooltip $step_tab $tips(RegistrationTab)
	    
	    set reg_tab$case $step_tab
	    
	    # Checkbutton to skip Registration entirely
            global do_registration
            checkbutton $step_tab.doreg -text "Perform Global EPI Registration" \
                -variable do_registration -state disabled \
                -command "$this toggle_do_registration"

	    Tooltip $step_tab.doreg $tips(RegToggle)

            pack $step_tab.doreg -side top -anchor nw -padx 7 -pady 0
	    
	    # Gradient File
	    iwidgets::labeledframe $step_tab.gradients \
                -labeltext "Gradients" \
                -labelpos nw -foreground grey64
            pack $step_tab.gradients -side top -anchor n \
		-fill x -padx 3 -pady 0
	    
	    set gradients [$step_tab.gradients childsite]
	    
            iwidgets::entryfield $gradients.file -labeltext "Gradients File:" \
                -labelpos w \
                -textvariable $mods(NrrdReader-Gradient)-filename \
	        -state disabled -foreground grey64
            pack $gradients.file -side top -padx 3 -pady 1 -anchor n \
	        -fill x 
	    
            button $gradients.load -text "Browse" \
                -command "$this load_gradient" \
                -width 12 -state disabled
            pack $gradients.load -side top -anchor n -padx 3 -pady 1
	    
	    
            # Reference Image
            global $mods(TendEpireg)-reference
            global ref_image_state ref_image

            iwidgets::labeledframe $step_tab.refimg \
		-labeltext "Reference Image" \
		-labelpos nw -foreground grey64
            pack $step_tab.refimg -side top -anchor n -padx 3 -pady 0
	    
            set refimg [$step_tab.refimg childsite]
	    
	    set ref_image$case $refimg
	    
	    radiobutton $refimg.est -text "Implicit Reference: estimate distortion\nparameters from all images" \
		-state disabled \
		-variable ref_image_state -value 0 \
		-command "$this toggle_reference_image_state"
	    Tooltip $refimg.est $tips(RegImpRefImg)
	    
            pack $refimg.est -side top -anchor nw -padx 2 -pady 0
	    
            frame $refimg.s
            pack $refimg.s -side top -anchor nw -padx 2 -pady 0
	    
            radiobutton $refimg.s.choose -text "Choose Reference Image:" \
		-state disabled \
		-variable ref_image_state -value 1  \
		-command "$this toggle_reference_image_state"
	    Tooltip $refimg.s.choose $tips(RegChooseRefImg)
	    
            label $refimg.s.label -textvariable ref_image -state disabled
            pack $refimg.s.choose $refimg.s.label -side left -anchor n
	    
            scale $refimg.s.ref -label "" \
		-state disabled \
		-variable ref_image \
		-from 1 -to 7 \
		-showvalue false \
		-length 80  -width 15 \
		-sliderlength 15 \
		-command "$this configure_reference_image" \
		-orient horizontal

	    Tooltip $refimg.s.ref $tips(RegRefImgSlider)
            pack $refimg.s.ref -side top -anchor ne -padx 0 -pady 0
	    
	    
	    
            ### Segmentation
            iwidgets::labeledframe $step_tab.seg \
		-labeltext "Segmentatation" \
		-labelpos nw -foreground grey64
            pack $step_tab.seg -side top -anchor n -padx 3 -pady 0
	    
            set seg [$step_tab.seg childsite] 
	    
            # Gaussian Smoothing
            iwidgets::labeledframe $seg.blur \
		-labeltext "Gaussian Smoothing" \
		-labelpos nw -foreground grey64
            pack $seg.blur -side top -anchor n -padx 0 -pady 0 \
		-fill x -expand 1
	    
            set blur [$seg.blur childsite]
	    
            global $mods(TendEpireg)-blur_x
            global $mods(TendEpireg)-blur_y
	    
            label $blur.labelx -text "X:" -state disabled
            scale $blur.entryx -label "" \
		-state disabled \
		-foreground grey64 \
		-variable $mods(TendEpireg)-blur_x \
		-from 0.0 -to 5.0 \
		-resolution 0.01 \
		-showvalue true \
		-sliderlength 15 -width 15 -length 50 \
		-orient horizontal
	    Tooltip $blur.entryx $tips(RegBlurX)
            label $blur.labely -text "Y:" -state disabled
            scale $blur.entryy -label "" \
		-state disabled \
		-foreground grey64 \
		-variable $mods(TendEpireg)-blur_y \
		-from 0.0 -to 5.0 \
		-resolution 0.01 \
		-showvalue true \
		-sliderlength 15 -width 15 -length 50 \
		-orient horizontal
	    Tooltip $blur.entryy $tips(RegBlurY)
            pack $blur.labelx $blur.entryx \
                $blur.labely $blur.entryy \
                -side left -anchor w -padx 2 -pady 0 \
                -fill x -expand 1
	    
	    
            iwidgets::labeledframe $seg.thresh \
		-labeltext "Background/DWI Threshold" \
		-labelpos nw -foreground grey64
            pack $seg.thresh -side top -anchor n -padx 0 -pady 0 \
		-fill x -expand 1
	    
            set thresh [$seg.thresh childsite]	    
	    set reg_thresh$case $thresh
	    
	    global $mods(TendEpireg)-threshold
            global $mods(TendEpireg)-use-default-threshold
            radiobutton $thresh.auto -text "Automatically Determine Threshold" \
		-state disabled \
		-variable $mods(TendEpireg)-use-default-threshold -value 1 \
		-command "$this toggle_registration_threshold"
            pack $thresh.auto -side top -anchor nw -padx 3 -pady 0
            frame $thresh.choose
            pack $thresh.choose -side top -anchor nw -padx 0 -pady 0 -fill x
	    
            radiobutton $thresh.choose.button -text "Specify Threshold:" \
		-state disabled \
		-variable $mods(TendEpireg)-use-default-threshold -value 0 \
		-command "$this toggle_registration_threshold"
            entry $thresh.choose.entry -width 12 \
		-textvariable $mods(TendEpireg)-threshold \
		-state disabled -foreground grey64
            pack $thresh.choose.button  -side left -anchor n -padx 2 -pady 1
            pack  $thresh.choose.entry -side left -anchor n -padx 2 -pady 1 -fill x
	    
            checkbutton $seg.cc -variable $mods(TendEpireg)-cc_analysis \
		-text "Use Connected Components"\
		-state disabled
            pack $seg.cc -side top -anchor nw -padx 6 -pady 0
	    
            # Fitting
            frame $step_tab.fit
            pack $step_tab.fit -side top -anchor n -padx 10 -pady 0
	    
            global $mods(TendEpireg)-fitting
            label $step_tab.fit.l -text "Fitting: " -state disabled
            label $step_tab.fit.f -text "70" -state disabled
            label $step_tab.fit.p -text "%" -state disabled
            scale $step_tab.fit.s -label "" \
		-state disabled \
		-variable $mods(TendEpireg)-fitting \
		-from .25 -to 1.0 \
		-resolution 0.01 \
		-length 150  -width 15 \
		-sliderlength 15 \
		-showvalue false \
		-orient horizontal \
		-command "$this configure_fitting_label "
	    Tooltip $step_tab.fit.s $tips(RegFitting)
	    
            pack $step_tab.fit.l $step_tab.fit.f $step_tab.fit.p \
		$step_tab.fit.s  -side left \
		-anchor nw -padx 0 -pady 0
            
	    
	    iwidgets::optionmenu $step_tab.rf -labeltext "Resampling Filter" \
                -labelpos w -width 160  \
                -state disabled \
                -command "$this set_resampling_filter $step_tab.rf"
	    
            pack $step_tab.rf -side top \
		-anchor nw -padx 8 -pady 0
	    
            $step_tab.rf insert end Linear Catmull-Rom "Windowed Sinc"

	    $step_tab.rf select "Catmull-Rom"
	    
	    
	    # Execute and Next buttons
            frame $step_tab.last
            pack $step_tab.last -side bottom -anchor ne  \
		-padx 3 -pady 3
            button $step_tab.last.ex -text "Execute" -state disabled -width 8 \
		-command "$this execute_Registration"
	    Tooltip $step_tab.last.ex $tips(RegExecute)

	    button $step_tab.last.ne -text "Next" -state disabled -width 8 \
		-command "$this change_processing_tab \"Build Tensors\"" 
	    Tooltip $step_tab.last.ne $tips(RegNext)

            pack $step_tab.last.ne $step_tab.last.ex -side right \
		-anchor ne -padx 2 -pady 0
	    
	    ### Build DT
            set step_tab [$process.tnb add -label "Build Tensors" -command "$this change_processing_tab \"Build Tensors\""]

	    Tooltip $step_tab $tips(BuildTensorsTab)
	    
	    set dt_tab$case $step_tab

            iwidgets::labeledframe $step_tab.blur \
		-labeltext "DWI Smoothing" \
		-labelpos nw -foreground grey64
            pack $step_tab.blur -side top -anchor nw -padx 3 -pady 0 \
		-fill x 
	    
            set blur [$step_tab.blur childsite]
	    
	    
            global do_smoothing
            checkbutton $blur.smooth -text "Do Smoothing" \
		-state disabled \
		-variable do_smoothing \
		-command "$this toggle_do_smoothing"
	    Tooltip $blur.smooth $tips(DTToggleSmoothing)
            pack $blur.smooth -side top -anchor nw -padx 3 -pady 3
	    
            frame $blur.rad1
            pack $blur.rad1 -side top -anchor n -padx 3 -pady 0
	    
            global xy_radius z_radius
            label $blur.rad1.l -text "Radius in X and Y:" -state disabled
            scale $blur.rad1.s -from 0.0 -to 5.0 \
		-resolution 0.01 \
		-state disabled \
		-variable xy_radius \
		-orient horizontal \
		-length 70  -width 15 \
		-sliderlength 15 \
		-showvalue false \
		-command "$this change_xy_smooth"
	    Tooltip $blur.rad1.s $tips(DTSmoothXY)
            label $blur.rad1.v -textvariable xy_radius -state disabled
            pack $blur.rad1.l $blur.rad1.s $blur.rad1.v -side left -anchor nw \
		-padx 1 -pady 0
	    
            frame $blur.rad2
            pack $blur.rad2 -side top -anchor n -padx 3 -pady 0
	    
            label $blur.rad2.l -text "Radius in Z:          " -state disabled
            scale $blur.rad2.s -from 0.0 -to 5.0 \
		-resolution 0.01 \
		-state disabled \
		-variable z_radius \
		-orient horizontal \
		-length 70  -width 15 \
		-sliderlength 15 \
		-showvalue false \
		-command "$this change_z_smooth"
	    Tooltip $blur.rad2.s $tips(DTSmoothZ)
            label $blur.rad2.v -textvariable z_radius -state disabled
            pack $blur.rad2.l $blur.rad2.s $blur.rad2.v -side left -anchor nw \
		-padx 1 -pady 0
	    

	    # Masking Threshold
            iwidgets::labeledframe $step_tab.thresh \
		-labeltext "Masking Threshold" \
		-labelpos nw -foreground grey64
            pack $step_tab.thresh -side top \
		-fill x -anchor nw -padx 3 -pady 0
	    
            set thresh [$step_tab.thresh childsite]
	    
	    
            global $mods(TendEstim)-use-default-threshold
            radiobutton $thresh.def -text "Automatically Determine Threshold" \
		-state disabled \
		-variable $mods(TendEstim)-use-default-threshold \
		-value 1 \
		-command "$this toggle_dt_threshold"
            pack $thresh.def -side top -anchor nw -padx 2 -pady 0
	    
            frame $thresh.choose
            pack $thresh.choose -side top -anchor nw -padx 0 -pady 0 -fill x
	    
            radiobutton $thresh.choose.button -text "Specify Threshold:" \
		-state disabled \
		-variable $mods(TendEstim)-use-default-threshold -value 0 \
		-command "$this toggle_dt_threshold" 
            pack $thresh.choose.button -side left \
		-anchor n -padx 2 -pady 3
            entry $thresh.choose.entry -width 17 \
		-textvariable $mods(TendEstim)-threshold \
		-state disabled -foreground grey64
            pack $thresh.choose.entry -side left \
		-anchor n -padx 2 -pady 3 -fill x
	    
            $thresh.def select

	    # B-Matrix
            iwidgets::labeledframe $step_tab.bm \
		-labeltext "B-Matrix" \
		-labelpos nw -foreground grey64
            pack $step_tab.bm -side top \
		-fill x -padx 3 -pady 0 -anchor n
	    
            set bm [$step_tab.bm childsite]
	    
	    global bmatrix
            radiobutton $bm.computeb -text "Compute B-Matrix Using\nGradients Provided" \
		-state disabled \
		-variable bmatrix \
		-value "compute" \
		-command "$this toggle_b_matrix"
	    Tooltip $bm.computeb $tips(DTBMatrixCompute)
            pack $bm.computeb  -side top -anchor nw -padx 2 -pady 0
	    
            frame $bm.load
            pack $bm.load -side top -anchor nw -padx 0 -pady 0 -fill x
	    
            radiobutton $bm.load.b -text "Load B-Matrix" \
		-state disabled \
		-variable bmatrix \
		-value "load" \
		-command "$this toggle_b_matrix"
	    Tooltip $bm.load.b $tips(DTBMatrixLoad)
	    
            pack $bm.load.b -side left -anchor nw \
		-padx 2 -pady 0

            entry $bm.load.e -width 20 \
		-textvariable $mods(NrrdReader-BMatrix)-filename \
		-state disabled -foreground grey64
            pack $bm.load.e -side left -anchor nw \
		-padx 2 -pady 0 -fill x
	    
            button $bm.browse -text "Browse" \
		-command "$this load_bmatrix" \
		-state disabled -width 12
	    
            pack $bm.browse -side top -anchor ne -padx 35 -pady 5

	    # Saving Tensor
	    iwidgets::labeledframe $step_tab.save \
		-labeltext "Save Tensors" \
		-labelpos nw -foreground grey64
	    pack $step_tab.save -side top \
		-fill x -padx 3 -pady 0 -anchor nw
	    set save [$step_tab.save childsite]
	    
	    global save_tensors
	    checkbutton $save.do_save -text "Save Computed Tensors" \
		-command "$this toggle_save_tensors" \
		-variable save_tensors \
		-foreground grey64

	    pack $save.do_save -side top -anchor nw -padx 3 -pady 3 

	    Tooltip $save.do_save "Save computed tensors as a nrrd.\nSpecify the filename by typing it\nin the Tensor File entry or using the\nBrowse button to navigate to a file."

	    frame $save.file
	    pack $save.file -side top -anchor nw

	    label $save.file.l -text "Tensor File:" -foreground grey64

	    entry $save.file.e -width 20 \
		-textvariable $mods(UnuSave-Tensors)-filename \
		-foreground grey64

	    pack $save.file.l $save.file.e -side left

	    Tooltip $save.file.e "Save computed tensors as a nrrd.\nSpecify the filename by typing it\nin the Tensor File entry or using the\nBrowse button to navigate to a file."

	    bind $save.file.e <Return> "$this execute_save"

	    button $save.browse -text "Browse" \
		-command "$mods(UnuSave-Tensors) create_filebox" \
		-foreground grey64 -width 15

	    pack $save.browse \
		-side top -anchor n -padx 3 -pady 3
	    
	    Tooltip $save.browse "Save computed tensors as a nrrd.\nSpecify the filename by typing it\nin the Tensor File entry or using the\nBrowse button to navigate to a file."
        
	    # Execute and Next
            frame $step_tab.last
            pack $step_tab.last -side bottom -anchor ne \
		-padx 3 -pady 3
	    
            button $step_tab.last.ex -text "Execute" \
		-width 16 -state disabled \
		-command "$this execute_DT"
	    Tooltip $step_tab.last.ex $tips(DTExecute)

            pack $step_tab.last.ex -side right -anchor ne \
		-padx 2 -pady 0
	    
	    
            ### Indicator
	    frame $process.indicator -relief sunken -borderwidth 2
            pack $process.indicator -side bottom -anchor s -padx 3 -pady 5
	    
	    canvas $process.indicator.canvas -bg "white" -width $i_width \
	        -height $i_height 
	    pack $process.indicator.canvas -side top -anchor n -padx 3 -pady 3
	    
            bind $process.indicator <Button> {app display_module_error} 
	    
            label $process.indicatorL -text "Press Execute to Load Data..."
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


	    Tooltip $process.indicatorL $tips(IndicatorLabel)
	    
            construct_indicator $process.indicator.canvas
	    
            $process.tnb view "Load Data"
	    
	    ### Attach/Detach button
            frame $m.d 
	    pack $m.d -side left -anchor e
            for {set i 0} {$i<32} {incr i} {
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
    
    

    #############################
    ### init_Vframe
    #############################
    # Initialize the visualization frame on the right. For this app
    # that includes the Vis Options and Viewer Options tabs.  For the
    # Viewer Options call the create_viewer_tab method in the base class.
    # For the Vis Options tab, build Variance, Planes, Isosurface,
    # Glyphs and Fibers tabs.  Also set variables pointing to these tabs.
    method init_Vframe { m case} {
	global mods
	global tips

	if { [winfo exists $m] } {
	    ### Visualization Frame
	    iwidgets::labeledframe $m.vis \
		-labelpos n -labeltext "Visualization" 
	    pack $m.vis -side right -anchor n 
	    
	    set vis [$m.vis childsite]
	    
	    ### Tabs
	    iwidgets::tabnotebook $vis.tnb -width $notebook_width \
		-height [expr $vis_height - 25] -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	    set vis_frame_tab$case $vis.tnb
	    
	    set page [$vis.tnb add -label "Vis Options" -command "$this change_vis_frame \"Vis Options\""]
	    
	    ### Vis Options Tab
	    # Add tabs for each visualization
            # Variance, Planes, Isosurface, Glyphs, Fibers
	    iwidgets::tabnotebook $page.vis_tabs \
                -width $notebook_width \
                -height $notebook_height \
                -tabpos n -equaltabs 0
	    
            pack $page.vis_tabs -padx 4 -pady 4
	    
	    set vis_tab$case $page.vis_tabs
	    
            ### Variance
            set vis_tab [$page.vis_tabs add -label "Variance" -command "$this change_vis_tab Variance"]
	    set variance_tab$case $vis_tab
	    build_variance_tab $vis_tab
	    
	    ### Planes
            set vis_tab [$page.vis_tabs add -label "Planes" -command "$this change_vis_tab Planes"]
	    set planes_tab$case $vis_tab
	    build_planes_tab $vis_tab
	    
	    
	    ### Isosurface
            set vis_tab [$page.vis_tabs add -label "Isosurface" -command "$this change_vis_tab Isosurface"]
	    set isosurface_tab$case $vis_tab
	    build_isosurface_tab $vis_tab

	    
	    ### Glyphs
            set vis_tab [$page.vis_tabs add -label "Glyphs" -command "$this change_vis_tab Glyphs"]
	    set glyphs_tab$case $vis_tab
	    build_glyphs_tab $vis_tab

	    
	    ### Fibers
            set vis_tab [$page.vis_tabs add -label "Fibers" -command "$this change_vis_tab Fibers"]
	    set fibers_tab$case $vis_tab
	    build_fibers_tab $vis_tab
	    
            $page.vis_tabs view "Variance"
	    

	    ### Renderer Options Tab
	    create_viewer_tab $vis
	    
	    $vis.tnb view "Vis Options"
	    
	    
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

	    # unpack everything and repack in the proper order
	    # (viewer last) so that viewer is the one to resize
	    pack forget $win.viewer

	    if { $IsVAttached } {
		pack forget $attachedVFr
		pack $attachedVFr -side right -anchor n 
	    }

	    pack $attachedPFr -side left -anchor n

	    pack $win.viewer -side left -anchor n -fill both -expand 1

	    set new_width [expr $c_width + $process_width]
            append geom $new_width x $c_height + [expr $x - $process_width] + $y
	    wm geometry $win $geom
	    set IsPAttached 1
	}
    }


    ##########################
    ### switch_V_frames
    ##########################
    # This method is called when the user wants to attach or detach
    # the visualization frame.
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

	    # unpack everything and repack in proper order
	    # (viewer last) so that viewer is the one to resize
	    pack forget $win.viewer

	    pack $attachedVFr -anchor n -side right 

	    if { $IsPAttached } {
		pack forget $attachedPFr
		pack $attachedPFr -side left -anchor n 
	    }

	    pack $win.viewer -side left -anchor n -fill both -expand 1

	    set new_width [expr $c_width + $vis_width]
            append geom $new_width x $c_height
	    wm geometry $win $geom
	    set IsVAttached 1
	}
    }


    #########################
    ### save_session
    #########################
    # This implements saving a BioTensor session
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
	    wm title .standalone "BioTensor - [getFileName $saveFile]" 
	    
	    set fileid [open $saveFile w]
	    
	    # Save out data information 
	    puts $fileid "# BioTensor Session\n"
	    puts $fileid "set app_version 1.0"
	    
	    save_module_variables $fileid
	    save_class_variables $fileid 
	    
	    save_global_variables $fileid
	    save_disabled_modules $fileid
	    save_disabled_connections $fileid
	    
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

    #########################
    ### save_disabled_connections
    #########################
    # Save out the call to disable all modules connections
    # that are currently disabled
    method save_disabled_connections { fileid } {
	global mods Disabled
	
	puts $fileid "\n# Disabled Module Connections\n"
	
	# Check the connections between the ChooseField-X, ChooseField-Y,
	# or ChooseField-Z and the GatherPoints module

	set name "$mods(ChooseField-X)_p0_to_$mods(GatherPoints)_p0"
	if {[info exists Disabled($name)] && $Disabled($name)} {
	    puts $fileid "disableConnection \"\$mods(ChooseField-X) 0 \$mods(GatherPoints) 0\""
	}

	set name "$mods(ChooseField-Y)_p0_to_$mods(GatherPoints)_p1"
	if {[info exists Disabled($name)] && $Disabled($name)} {
	    puts $fileid "disableConnection \"\$mods(ChooseField-Y) 0 \$mods(GatherPoints) 1\""
	}

	set name "$mods(ChooseField-Z)_p0_to_$mods(GatherPoints)_p2"
	if {[info exists Disabled($name)] && $Disabled($name)} {
	    puts $fileid "disableConnection \"\$mods(ChooseField-Z) 0 \$mods(GatherPoints) 2\""
	}


	
    }


    #########################
    ### save_global_variables
    #########################
    # Save out any globals used spefically in this app
    method save_global_variables { fileid } {
	global mods

	puts $fileid "\n# Global Variables\n"
	
	# Save out my globals by hand because otherwise they conflict with
	# the module variables
	global data_mode
	puts $fileid "set data_mode \{$data_mode\}"
	global fast_axis
	puts $fileid "set fast_axis \{$fast_axis\}"
	global B0_first
	puts $fileid "set B0_first \{$B0_first\}"
	global channels
	puts $fileid "set channels \{$channels\}"
	
	
	### planes variables that must be globals (all checkbuttons)
	global show_planes
	puts $fileid "set show_planes \{$show_planes\}"
	global show_plane_x
	puts $fileid "set show_plane_x \{$show_plane_x\}"
	global show_plane_y
	puts $fileid "set show_plane_y \{$show_plane_y\}"
	global show_plane_z
	puts $fileid "set show_plane_z \{$show_plane_z\}"
	global plane_x
	puts $fileid "set plane_x \{$plane_x\}"
	global plane_y
	puts $fileid "set plane_y \{$plane_y\}"
	global plane_z
	puts $fileid "set plane_z \{$plane_z\}"
	
	### registration globals
	global ref_image
	puts $fileid "set ref_image \{$ref_image\}"
	global ref_image_state
	puts $fileid "set ref_image_state \{$ref_image_state\}"
	
	global clip_to_isosurface
	puts $fileid "set clip_to_isosurface \{$clip_to_isosurface\}"
	
	global bmatrix
	puts $fileid "set bmatrix \{$bmatrix\}"
	
	### DT Globals
	global xy_radius
	puts $fileid "set xy_radius \{$xy_radius\}"
	global z_radius 
	puts $fileid "set z_radius \{$z_radius\}"
	
	### isosurface variables
	global clip_by_planes
	puts $fileid "set clip_by_planes \{$clip_by_planes\}"
	
	global do_registration 
	puts $fileid "set do_registration \{$do_registration\}"
	
	global do_smoothing
	puts $fileid "set do_smoothing \{$do_smoothing\}"
	
	# glyphs
	global glyph_display_type
	puts $fileid "set glyph_display_type \{$glyph_display_type\}"
	
	global scale_glyph
	puts $fileid "set scale_glyph \{$scale_glyph\}"
	
	global exag_glyph
	puts $fileid "set exag_glyph \{$exag_glyph\}"

	global glyph_scale_val
	puts $fileid "set glyph_scale_val \{$glyph_scale_val\}"
	
	# fibers
	global fibers_stepsize
	puts $fileid "set fibers_stepsize \{$fibers_stepsize\}"
	
	global fibers_length
	puts $fileid "set fibers_length \{$fibers_length\}"
	
	global fibers_steps
	puts $fileid "set fibers_steps \{$fibers_steps\}"

	global plane
	if {[info exists plane(-X)]} {
	    puts $fileid "set plane(-X) \{$plane(-X)\}"
	}
	if {[info exists plane(+X)]} {
	    puts $fileid "set plane(+X) \{$plane(+X)\}"
	}
	if {[info exists plane(-Y)]} {
	    puts $fileid "set plane(-Y) \{$plane(-Y)\}"
	}
	if {[info exists plane(+Y)]} {
	    puts $fileid "set plane(+Y) \{$plane(+Y)\}"
	}
	if {[info exists plane(-Z)]} {
	    puts $fileid "set plane(-Z) \{$plane(-Z)\}"
	}
	if {[info exists plane(+Z)]} {
	    puts $fileid "set plane(+Z) \{$plane(+Z)\}"
	}

	# colors
	global clip_to_isosurface_color
	set color clip_to_isosurface_color
	global $color-r
	global $color-g
	global $color-b

	puts $fileid "global clip_to_isosurface_color"
	puts $fileid "set color clip_to_isosurface_color"
	puts $fileid "global \$color-r"
	puts $fileid "set \$color-r [set $color-r]"
	puts $fileid "global \$color-g"
	puts $fileid "set \$color-g [set $color-g]"
	puts $fileid "global \$color-b"
	puts $fileid "set \$color-b [set $color-b]"
	puts $fileid "setColor \$planes_tab1.color.childsite.select.colorFrame.col clip_to_isosurface_color load"
	puts $fileid "setColor \$planes_tab2.color.childsite.select.colorFrame.col clip_to_isosurface_color load"

	global isosurface_color
	set color isosurface_color
	global $color-r
	global $color-g
	global $color-b

	puts $fileid "global isosurface_color"
	puts $fileid "set color isosurface_color"
	puts $fileid "global \$color-r"
	puts $fileid "set \$color-r [set $color-r]"
	puts $fileid "global \$color-g"
	puts $fileid "set \$color-g [set $color-g]"
	puts $fileid "global \$color-b"
	puts $fileid "set \$color-b [set $color-b]"
	puts $fileid "setColor \$isosurface_tab1.isocolor.childsite.select.colorFrame.col isosurface_color load"
	puts $fileid "setColor \$isosurface_tab2.isocolor.childsite.select.colorFrame.col isosurface_color load"

	global glyph_color
	set color glyph_color
	global $color-r
	global $color-g
	global $color-b

	puts $fileid "global glyph_color"
	puts $fileid "set color glyph_color"
	puts $fileid "global \$color-r"
	puts $fileid "set \$color-r [set $color-r]"
	puts $fileid "global \$color-g"
	puts $fileid "set \$color-g [set $color-g]"
	puts $fileid "global \$color-b"
	puts $fileid "set \$color-b [set $color-b]"
	puts $fileid "setColor \$glyphs_tab1.rep.childsite.select.colorFrame.col glyph_color load"
	puts $fileid "setColor \$glyphs_tab2.rep.childsite.select.colorFrame.col glyph_color load"


	global fiber_color
	set color fiber_color
	global $color-r
	global $color-g
	global $color-b

	puts $fileid "global fiber_color"
	puts $fileid "set color fiber_color"
	puts $fileid "global \$color-r"
	puts $fileid "set \$color-r [set $color-r]"
	puts $fileid "global \$color-g"
	puts $fileid "set \$color-g [set $color-g]"
	puts $fileid "global \$color-b"
	puts $fileid "set \$color-b [set $color-b]"
	puts $fileid "setColor \$fibers_tab1.rep.childsite.f1.colorFrame.col fiber_color load"
	puts $fileid "setColor \$fibers_tab2.rep.childsite.f1.colorFrame.col fiber_color load"


	# save clipping planes
	global mods
        global $mods(Viewer)-ViewWindow_0-global-clip
        set $mods(Viewer)-ViewWindow_0-global-clip [set $mods(Viewer)-ViewWindow_0-global-clip]
        global $mods(Viewer)-ViewWindow_0-clip
        set clip $mods(Viewer)-ViewWindow_0-clip

	puts $fileid "global mods"
	puts $fileid "global \$mods(Viewer)-ViewWindow_0-global-clip"
	puts $fileid "set \$mods(Viewer)-ViewWindow_0-global-clip [set $mods(Viewer)-ViewWindow_0-global-clip]"
	puts $fileid "global \$mods(Viewer)-ViewWindow_0-clip"
	puts $fileid "set clip \$mods(Viewer)-ViewWindow_0-clip"

	global $clip-normal-x
	global $clip-normal-y
	global $clip-normal-z
	global $clip-normal-d
	global $clip-visible

	puts $fileid "global \$clip-normal-x"
	puts $fileid "global \$clip-normal-y"
	puts $fileid "global \$clip-normal-z"
	puts $fileid "global \$clip-normal-d"
	puts $fileid "global \$clip-visible"

	for {set i 1} {$i <= 6} {incr i 1} {
	    set mod $i
	    

	    global $clip-normal-x-$mod
	    global $clip-normal-y-$mod
	    global $clip-normal-z-$mod
	    global $clip-normal-d-$mod
	    global $clip-visible-$mod

	    puts $fileid "global \$clip-normal-x-$mod"
	    puts $fileid "global \$clip-normal-y-$mod"
	    puts $fileid "global \$clip-normal-z-$mod"
	    puts $fileid "global \$clip-normal-d-$mod"
	    puts $fileid "global \$clip-visible-$mod"

	    puts $fileid "set \$clip-visible-$mod [set $clip-visible-$mod]"
	    puts $fileid "set \$clip-normal-d-$mod [set $clip-normal-d-$mod]"
	    puts $fileid "set \$clip-normal-x-$mod [set $clip-normal-x-$mod]"
	    puts $fileid "set \$clip-normal-y-$mod [set $clip-normal-y-$mod]"
	    puts $fileid "set \$clip-normal-z-$mod [set $clip-normal-z-$mod]"
        }

    }
    

    
    ###########################
    ### load_session
    ###########################
    # Load a saved session of BioTensor.  After sourcing the file,
    # reset some of the state (attached, indicate) and configure
    # the tabs and guis. This method also sets the loading to be 
    # true so that when executing, the progress labels don't get
    # all messed up.
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
	# configure title
	wm title .standalone "BioTensor - [getFileName $saveFile]" 
	
	# Reset application 
	reset_app
	
	foreach g [info globals] {
	    global $g
	}

	source $saveFile
	

	if {$c_data_tab == "Analyze"} {
	    # This is a hack for the AnalyzeToNrrd module.
	    # The problem is that the filenames$i variables
	    # are created as data is selected and loaded.  For
	    # some reason when loading the setting for an app
	    # the filenames$i variables aren't seen.  This
	    # needs to be looked into more for 1.20.2
	    global mods
	    global data_mode

	    if {$data_mode == "DWIknownB0"} {
		# Check AnalyzeToNrrd-T2
		global $mods(AnalyzeToNrrd-T2)-num-files
		global $mods(AnalyzeToNrrd-T2)-file
		set num [set $mods(AnalyzeToNrrd-T2)-num-files]
		for {set i 0} {$i < $num} {incr i} {
		    if {[info exists $mods(AnalyzeToNrrd-T2)-filenames$i]} {
			set temp [set $mods(AnalyzeToNrrd-T2)-filenames$i]
			unset $mods(AnalyzeToNrrd-T2)-filenames$i
			set $mods(AnalyzeToNrrd-T2)-file $temp
			
			global $mods(AnalyzeToNrrd-T2)-filenames$i
			set $mods(AnalyzeToNrrd-T2)-filenames$i [set $mods(AnalyzeToNrrd-T2)-file]
			
			# Call the c++ function that adds this data to its data 
			# structure.
			#			    $mods(AnalyzeToNrrd-T2)-c add_data [set $mods(AnalyzeToNrrd-T2)-file]
		    }
		}
	    }
	    # Check AnalyzeToNrrd1
	    global $mods(AnalyzeToNrrd1)-num-files
	    global $mods(AnalyzeToNrrd1)-file
	    set num [set $mods(AnalyzeToNrrd1)-num-files]
	    for {set i 0} {$i < $num} {incr i} {
		if {[info exists $mods(AnalyzeToNrrd1)-filenames$i]} {
		    set temp [set $mods(AnalyzeToNrrd1)-filenames$i]
		    unset $mods(AnalyzeToNrrd1)-filenames$i
		    set $mods(AnalyzeToNrrd1)-file $temp
		    
		    global $mods(AnalyzeToNrrd1)-filenames$i
		    set $mods(AnalyzeToNrrd1)-filenames$i [set $mods(AnalyzeToNrrd1)-file]
		    
		    # Call the c++ function that adds this data to its data 
		    # structure.
		    #			$mods(AnalyzeToNrrd1)-c add_data [set $mods(AnalyzeToNrrd1)-file]
		}
	    }
	} elseif {$c_data_tab == "Dicom"} {
	    # This is a hack for the DicomToNrrd module.
	    # The problem is that the entry-files$i variables
	    # are created as data is selected and loaded.  For
	    # some reason when loading the setting for an app
	    # the entry-files$i variables aren't seen.  This
	    # needs to be looked into more for 1.20.2
	    global mods
	    global data_mode
	    if {$data_mode == "DWIknownB0"} {
		# Check DicomToNrrd-T2
		global $mods(DicomToNrrd-T2)-num-entries
		set num [set $mods(DicomToNrrd-T2)-num-entries]
		for {set i 0} {$i < $num} {incr i} {
		    if {[info exists $mods(DicomToNrrd-T2)-entry-files$i]} {
			set temp1 [set $mods(DicomToNrrd-T2)-entry-files$i]
			unset $mods(DicomToNrrd-T2)-entry-files$i

			set temp2 [set $mods(DicomToNrrd-T2)-entry-dir$i]
			unset $mods(DicomToNrrd-T2)-entry-dir$i

			set temp3 [set $mods(DicomToNrrd-T2)-entry-suid$i]
			unset $mods(DicomToNrrd-T2)-entry-suid$i
			
			global $mods(DicomToNrrd-T2)-entry-files$i
			set $mods(DicomToNrrd-T2)-entry-files$i $temp1

			global $mods(DicomToNrrd-T2)-entry-dir$i
			set $mods(DicomToNrrd-T2)-entry-dir$i $temp2

			global $mods(DicomToNrrd-T2)-entry-suid$i
			set $mods(DicomToNrrd-T2)-entry-suid$i $temp3
			
			# Call the c++ function that adds this data to its data 
			# structure.
			# 			    $mods(DicomToNrrd-T2)-c add_data \
			    # 				[set $mods(DicomToNrrd-T2)-entry-dir$i] \
			    # 				[set $mods(DicomToNrrd-T2)-entry-suid$i] \
			    # 				[set $mods(DicomToNrrd-T2)-entry-files$i] 
		    }
		}
	    }
	    # Check DicomToNrrd1
	    global $mods(DicomToNrrd1)-num-entries
	    set num [set $mods(DicomToNrrd1)-num-entries]
	    for {set i 0} {$i < $num} {incr i} {
		if {[info exists $mods(DicomToNrrd1)-entry-files$i]} {
		    set temp1 [set $mods(DicomToNrrd1)-entry-files$i]
		    unset $mods(DicomToNrrd1)-entry-files$i
		    
		    set temp2 [set $mods(DicomToNrrd1)-entry-dir$i]
		    unset $mods(DicomToNrrd1)-entry-dir$i
		    
		    set temp3 [set $mods(DicomToNrrd1)-entry-suid$i]
		    unset $mods(DicomToNrrd1)-entry-suid$i
		    
		    global $mods(DicomToNrrd1)-entry-files$i
		    set $mods(DicomToNrrd1)-entry-files$i $temp1

		    global $mods(DicomToNrrd1)-entry-dir$i
		    set $mods(DicomToNrrd1)-entry-dir$i $temp2
		    
		    global $mods(DicomToNrrd1)-entry-suid$i
		    set $mods(DicomToNrrd1)-entry-suid$i $temp3
		    
		    # Call the c++ function that adds this data to its data 
		    # structure.
		    # 			$mods(DicomToNrrd1)-c add_data \
			# 			    [set $mods(DicomToNrrd1)-entry-dir$i] \
			# 			    [set $mods(DicomToNrrd1)-entry-suid$i] \
			# 			    [set $mods(DicomToNrrd1)-entry-files$i]
		}
	    }
	}

	if {$data_mode == "B0DWI"} {
	    global $mods(UnuPermute)-dim

	    if {[info exists $mods(UnuPermute)-axis0]} {
		set temp0 [set $mods(UnuPermute)-axis0]
		unset $mods(UnuPermute)-axis0

		global $mods(UnuPermute)-axis0
		set $mods(UnuPermute)-axis0 $temp0
	    }

	    if {[info exists $mods(UnuPermute)-axis1]} {
		set temp1 [set $mods(UnuPermute)-axis1]
		unset $mods(UnuPermute)-axis1
		global $mods(UnuPermute)-axis1
		set $mods(UnuPermute)-axis1 $temp1
	    }

	    if {[info exists $mods(UnuPermute)-axis2]} {
		set temp2 [set $mods(UnuPermute)-axis2]
		unset $mods(UnuPermute)-axis2
		global $mods(UnuPermute)-axis2
		set $mods(UnuPermute)-axis2 $temp2
	    }

	    if {[set $mods(UnuPermute)-dim] == 4} {
		set temp3 3
		if {[info exists $mods(UnuPermute)-axis3]} {
		    set temp3 [set $mods(UnuPermute)-axis3]
		    unset $mods(UnuPermute)-axis3
		    global $mods(UnuPermute)-axis3
		    set $mods(UnuPermute)-axis3 $temp3
		}
	    }
	}

	# set a few variables that need to be reset
	set indicate 0
	set cycle 0
	set IsPAttached 1
	set IsVAttached 1
	set executing_modules 0

	# configure each vis tab
	configure_variance_tabs
	configure_planes_tabs
	sync_planes_tabs
	configure_isosurface_tabs
	sync_isosurface_tabs
	configure_glyphs_tabs
	sync_glyphs_tabs
	configure_fibers_tabs
	sync_fibers_tabs
	change_glyph_scale
	
	# bring tabs forward
	$proc_tab1 view $c_procedure_tab
	$proc_tab2 view $c_procedure_tab

	if {$c_data_tab == "Nrrd"} {
	    set c_data_tab "Generic"
	}
	$data_tab1 view $c_data_tab
	$data_tab2 view $c_data_tab

	$vis_frame_tab1 view $c_left_tab
	$vis_frame_tab2 view $c_left_tab

	$vis_tab1 view $c_vis_tab
	$vis_tab2 view $c_vis_tab
	
	# activate proper step tabs
	configure_data_tab
	configure_registration_tab
	configure_dt_tab

	# because of the order of disables,
	# this module needs to be re-enabled and 
	# depending on the data_mode, the UnuCrop
	# needs to be disabled
	disableModule $mods(ChooseNrrd-B0) 0
	global data_mode
	if {$data_mode != "B0DWI"} {
	    disableModule $mods(UnuCrop-DWI) 1
	} else {
	    disableModule $mods(ChooseNrrd-preprocess) 0
	    disableModule $mods(UnuSlice-B0) 0
	}

	global do_registration
	if {!$do_registration || [string equal $data_mode "tensor"] == 1 } {
	    disableModule $mods(TendEpireg) 1
	}

	$indicatorL0 configure -text "Press Execute to run to save point..."
	$indicatorL1 configure -text "Press Execute to run to save point..."
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



    ############################
    ### show_help
    ############################
    # Show the help menu
    method show_help {} {
	set splashImageFile [file join [netedit getenv SCIRUN_SRCDIR] Packages Teem Dataflow GUI splash-tensor.ppm]
	showProgress 1 none 1

	global tutorial_link
	set tutorial_link "http://software.sci.utah.edu/doc/User/Tutorials/BioTensor"
	set help_font "-Adobe-Helvetica-normal-R-Normal-*-12-120-75-*"

	if {![winfo exists .splash.frame.m1]} {
	    label .splash.frame.m1 -text "Please refer to the online BioTensor Tutorial" \
		-font $help_font
	    
	    entry .splash.frame.m2 -relief flat -textvariable tutorial_link \
		-state disabled -width 45 -font $help_font
	    pack .splash.frame.m1 .splash.frame.m2 -before .splash.frame.ok -anchor n \
		-pady 2	   
	} else {
	    SciRaise .splash
	}
	update idletasks
    }
    

    ##########################
    ### show_about
    ##########################
    # Show about box
    method show_about {} {
	tk_messageBox -message "BioTensor is a program used to process and visualize diffusion tensor images. It can read diffusion weighted images (DWIs), perform correction for a common class of distortions in echo-planar imaging, estimate tensors from DWIs, and visualize the diffusion tensor field." -type ok -icon info -parent .standalone
    }
    
    

     ###########################
     ### indicate_dynamic_compile 
     ###########################
     # Changes the label on the progress bar to dynamic compile
     # message or changes it back
     method indicate_dynamic_compile { which mode } {
 	if {$mode == "start"} {
 	    change_indicate_val 1
 	    change_indicator_labels "Dynamically Compiling [$which name]..."
         } else {
 	    change_indicate_val 2

 	    if {$dt_completed} {
 		change_indicator_labels "Visualization..."
 	    } elseif {$c_procedure_tab == "Build Tensors"} {
 		change_indicator_labels "Building Diffusion Tensors..."
 	    } elseif {$c_procedure_tab == "Registration"} {
 		change_indicator_labels "Registration..."
 	    } else {
 		change_indicator_labels "Loading Data..."
 	    }
 	}
     }
    
    
    ##########################
    ### update_progress
    ##########################
    # This is called when any module calls update_state.
    # We only care about "JustStarted" and "Completed" calls.
    method update_progress { which state } {
	global mods
	global $mods(ShowField-Isosurface)-faces-on
	global $mods(ShowField-Glyphs)-tensors-on
	global $mods(ShowField-Fibers)-edges-on
	global show_plane_x show_plane_y show_plane_z

	if {$which == $mods(ChooseNrrd-B0) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(ChooseNrrd-B0) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ChooseNrrd1) && $state == "JustStarted"} {
	    change_indicator_labels "Loading Data..."
	    change_indicate_val 1
	} elseif {$which == $mods(ChooseNrrd1) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Orig) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Orig) && $state == "Completed"} {
	    change_indicate_val 2
	    
	    configure_variance_tabs
	    
	    # Bring images into view
	    global $mods(ShowField-Orig)-faces-on
	    if {[set $mods(ShowField-Orig)-faces-on] == 1 && !$loading} {
		after 100 "$mods(Viewer)-ViewWindow_0-c autoview; global $mods(Viewer)-ViewWindow_0-pos; set $mods(Viewer)-ViewWindow_0-pos \"z0_y0\"; $mods(Viewer)-ViewWindow_0-c Views;"
		set has_autoviewed 1
	    }
	} elseif {$which == $mods(TendEpireg) && $state == "JustStarted"} {
	    if {$data_completed} {
		change_indicator_labels "Registration..."
	    }
	    change_indicate_val 1
	} elseif {$which == $mods(TendEpireg) && $state == "Completed"} {
	    change_indicate_val 2

	    # activate next button
	    $reg_tab1.last.ne configure -state normal \
		-foreground black -background $next_color
	    $reg_tab2.last.ne configure -state normal \
		-foreground black -background $next_color
	    
	    activate_dt
	} elseif {$which == $mods(ShowField-Reg) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Reg) && $state == "Completed"} {
	    change_indicate_val 2
	    
	    configure_variance_tabs
	    
	    # Bring images into view
	    global $mods(ShowField-Reg)-faces-on
	    if {[set $mods(ShowField-Reg)-faces-on] == 1 && !$loading} {
		after 100 "$mods(Viewer)-ViewWindow_0-c autoview; global $mods(Viewer)-ViewWindow_0-pos; set $mods(Viewer)-ViewWindow_0-pos \"z0_y0\"; $mods(Viewer)-ViewWindow_0-c Views"
		set has_autoviewed 1
	    }
        } elseif {$which == $mods(TendEstim) && $state == "JustStarted"} {
	    if {$reg_completed} {
		change_indicator_labels "Building Diffusion Tensors..."
	    }
	    change_indicate_val 1
	} elseif {$which == $mods(TendEstim) && $state == "Completed"} {
	    change_indicate_val 2
	    if {!$loading} {
		activate_vis
	    }
	} elseif {$which == $mods(NrrdInfo-full) && $state == "Completed"} {
	    if {!$loading} {
		global data_mode
		if {$data_mode == "B0DWI"} {
		    global $mods(NrrdInfo-full)-dimension
		    
		    set v 0
		    set x 0
		    set y 0
		    set z 0
		    global $mods(NrrdInfo-full)-size0
		    global $mods(NrrdInfo-full)-size1
		    global $mods(NrrdInfo-full)-size2
		    global $mods(ChooseNrrd-preprocess)-port-selected-index
		    if {[set $mods(NrrdInfo-full)-dimension] == 4} {
			global $mods(NrrdInfo-full)-size3
			# just slice and crop - no preprocessing
			set v [set $mods(NrrdInfo-full)-size0]
			set x [set $mods(NrrdInfo-full)-size1]
			set y [set $mods(NrrdInfo-full)-size2]
			set z [set $mods(NrrdInfo-full)-size3]
			
			set $mods(ChooseNrrd-preprocess)-port-selected-index 0
			
			disableModule $mods(UnuAxinsert) 1
			disableModule $mods(UnuReshape) 1
			disableModule $mods(ChooseNrrd-preprocess) 0
			disableModule $mods(UnuAxinfo-Z) 1
		    } elseif {[set $mods(NrrdInfo-full)-dimension] == 3} {
			# Pre-process before crop and slice
			global channels
			set v [expr $channels + 1]
			set x [set $mods(NrrdInfo-full)-size0]
			set y [set $mods(NrrdInfo-full)-size1]
			set z [expr [set $mods(NrrdInfo-full)-size2] / $v]
			
			set $mods(ChooseNrrd-preprocess)-port-selected-index 1
			
			# set unu axinsert
			global $mods(UnuAxinsert)-axis
			set $mods(UnuAxinsert)-axis 2
			
			# set unu reshape
			global fast_axis
			global $mods(UnuReshape)-sz
			if {$fast_axis == "volumes"} {
			    set $mods(UnuReshape)-sz "$x $y $z $v"
			} else {
			    set $mods(UnuReshape)-sz "$x $y $v $z"
			}
			
			# set unu permute
			global $mods(UnuPermute)-dim
			global $mods(UnuPermute)-axis0
			global $mods(UnuPermute)-axis1
			global $mods(UnuPermute)-axis2
			global $mods(UnuPermute)-axis3
			set $mods(UnuPermute)-dim 4
			if {$fast_axis == "volumes"} {
			    set $mods(UnuPermute)-axis0 3
			    set $mods(UnuPermute)-axis1 0
			    set $mods(UnuPermute)-axis2 1
			    set $mods(UnuPermute)-axis3 2
			} else {
			    set $mods(UnuPermute)-axis0 2
			    set $mods(UnuPermute)-axis1 0
			    set $mods(UnuPermute)-axis2 1
			    set $mods(UnuPermute)-axis3 3
			}
			
			# set unu axinfos
			global $mods(UnuAxinfo-X)-axis $mods(UnuAxinfo-X)-spacing
			global $mods(UnuAxinfo-X)-min $mods(UnuAxinfo-X)-max
			global $mods(UnuAxinfo-Y)-axis $mods(UnuAxinfo-Y)-spacing
			global $mods(UnuAxinfo-Y)-min $mods(UnuAxinfo-Y)-max
			global $mods(UnuAxinfo-Z)-axis $mods(UnuAxinfo-Z)-spacing
			global $mods(UnuAxinfo-Z)-min $mods(UnuAxinfo-Z)-max
			global $mods(NrrdInfo-full)-spacing0
			global $mods(NrrdInfo-full)-spacing1
			global $mods(NrrdInfo-full)-spacing2
			global $mods(NrrdInfo-full)-min0
			global $mods(NrrdInfo-full)-min1
			global $mods(NrrdInfo-full)-min2
			
			set $mods(UnuAxinfo-X)-axis 1
			set $mods(UnuAxinfo-X)-spacing [set $mods(NrrdInfo-full)-spacing0]
			set $mods(UnuAxinfo-X)-min [set $mods(NrrdInfo-full)-min0]
			set $mods(UnuAxinfo-X)-max [expr [set $mods(NrrdInfo-full)-min0] + [expr [set $mods(NrrdInfo-full)-spacing0] *$x]]
			
			set $mods(UnuAxinfo-Y)-axis 2
			set $mods(UnuAxinfo-Y)-spacing [set $mods(NrrdInfo-full)-spacing1]
			set $mods(UnuAxinfo-Y)-min [set $mods(NrrdInfo-full)-min1]
			set $mods(UnuAxinfo-Y)-max [expr [set $mods(NrrdInfo-full)-min1] + [expr [set $mods(NrrdInfo-full)-spacing1] * $y]]
			
			set $mods(UnuAxinfo-Z)-axis 3
			set $mods(UnuAxinfo-Z)-spacing [set $mods(NrrdInfo-full)-spacing2]
			set $mods(UnuAxinfo-Z)-min [set $mods(NrrdInfo-full)-min2]
			set $mods(UnuAxinfo-Z)-max [expr [set $mods(NrrdInfo-full)-min2] + [expr [set $mods(NrrdInfo-full)-spacing2] * $z]]
			
			disableModule $mods(UnuAxinsert) 0
			disableModule $mods(UnuReshape) 0
			disableModule $mods(UnuAxinfo-Z) 0
			disableModule $mods(ChooseNrrd-preprocess) 0
		    } else {
			tk_messageBox -message "Input data must be 3D nrrd (to pre-process) or 4D nrrd." -type ok -icon info -parent .standalone   
		    }
		    

		    # Crop out DWI
		    global $mods(UnuCrop-DWI)-num-axes
		    global $mods(UnuCrop-DWI)-minAxis0 $mods(UnuCrop-DWI)-minAxis1 $mods(UnuCrop-DWI)-minAxis2
		    global $mods(UnuCrop-DWI)-minAxis3 $mods(UnuCrop-DWI)-maxAxis0 $mods(UnuCrop-DWI)-maxAxis1
		    global $mods(UnuCrop-DWI)-maxAxis2 $mods(UnuCrop-DWI)-maxAxis3 $mods(UnuCrop-DWI)-absmaxAxis0
		    global $mods(UnuCrop-DWI)-absmaxAxis1 $mods(UnuCrop-DWI)-absmaxAxis2 
		    global $mods(UnuCrop-DWI)-absmaxAxis3
		    
		    set $mods(UnuCrop-DWI)-num-axes 4
		    
		    global B0_first
		    if {$B0_first == 1} {
			set $mods(UnuCrop-DWI)-minAxis0 1
		    } else {
			set $mods(UnuCrop-DWI)-minAxis0 0
		    }
		    set $mods(UnuCrop-DWI)-minAxis1 0
		    set $mods(UnuCrop-DWI)-minAxis2 0
		    set $mods(UnuCrop-DWI)-minAxis3 0
		    
		    if {$B0_first == 1} {
			set $mods(UnuCrop-DWI)-maxAxis0 [expr $v - 1]
		    } else {
			set $mods(UnuCrop-DWI)-maxAxis0 [expr $v - 2]
		    }
		    set $mods(UnuCrop-DWI)-maxAxis1 [expr $x - 1]
		    set $mods(UnuCrop-DWI)-maxAxis2 [expr $y - 1]
		    set $mods(UnuCrop-DWI)-maxAxis3 [expr $z - 1]
		    
		    if {$B0_first == 1} {
			set $mods(UnuCrop-DWI)-absmaxAxis0 [expr $v - 1]
		    } else {
			set $mods(UnuCrop-DWI)-absmaxAxis0 [expr $v - 2]
		    }
		    set $mods(UnuCrop-DWI)-absmaxAxis1 [expr $x - 1]
		    set $mods(UnuCrop-DWI)-absmaxAxis2 [expr $y - 1]
		    set $mods(UnuCrop-DWI)-absmaxAxis3 [expr $z - 1]
		    
		    # Slice out B0
		    global $mods(UnuSlice-B0)-axis $mods(UnuSlice-B0)-position
		    set $mods(UnuSlice-B0)-axis 0
		    if {$B0_first} {
			set $mods(UnuSlice-B0)-position 0
		    } else {
			set $mods(UnuSlice-B0)-position [expr $v - 1]
		    }
		    
		    disableModule $mods(UnuCrop-DWI) 0
		    disableModule $mods(ChooseNrrd-B0) 0
		    disableModule $mods(TendEpireg) 1
		    disableModule $mods(ChooseNrrd-ToReg) 1
		    $mods(ChooseNrrd-preprocess)-c needexecute
		} else {
		    disableModule $mods(ChooseNrrd-B0) 0
		    disableModule $mods(TendEpireg) 1
		    disableModule $mods(ChooseNrrd-ToReg) 1
		    disableModule $mods(UnuCrop-DWI) 1
		    $mods(ChooseNrrd-B0)-c needexecute
		}
	    }
	} elseif {$which == $mods(NrrdInfo1) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(NrrdInfo1) && $state == "Completed"} {
	    change_indicate_val 2
	    
	    global $mods(NrrdInfo1)-size1
	    
            global data_mode
	    if {[info exists $mods(NrrdInfo1)-size1]} {
		global $mods(NrrdInfo1)-size0
		global $mods(NrrdInfo1)-size1
		global $mods(NrrdInfo1)-size2
		global $mods(NrrdInfo1)-size3
		
		global $mods(NrrdInfo1)-spacing1
		global $mods(NrrdInfo1)-spacing2
		global $mods(NrrdInfo1)-spacing3
		
		global $mods(NrrdInfo1)-min1
		global $mods(NrrdInfo1)-min2
		global $mods(NrrdInfo1)-min3
		
		set volumes [set $mods(NrrdInfo1)-size0]
		set size_x [expr [set $mods(NrrdInfo1)-size1] - 1]
		set size_y [expr [set $mods(NrrdInfo1)-size2] - 1]
		set size_z [expr [set $mods(NrrdInfo1)-size3] - 1]
		
		set spacing_x [set $mods(NrrdInfo1)-spacing1]
		set spacing_y [set $mods(NrrdInfo1)-spacing2]
		set spacing_z [set $mods(NrrdInfo1)-spacing3]
		
		set average_spacing [expr [expr $spacing_x + $spacing_y + $spacing_z] / 3.0]
		
		set min_x [set $mods(NrrdInfo1)-min1]
		set min_y [set $mods(NrrdInfo1)-min2]
		set min_z [set $mods(NrrdInfo1)-min3]
		
		# configure fiber edges to be average spacing * 0.25
		global $mods(ShowField-Fibers)-edge_scale
		set $mods(ShowField-Fibers)-edge_scale [expr 0.125 * $average_spacing]
		
		global $mods(ShowField-Glyphs)-tensors_scale
		global glyph_scale_val
		if {!$loading} {
		    set $mods(ShowField-Glyphs)-tensors_scale [expr 0.5 * $average_spacing]
		    set glyph_scale_val 0.5
		}

		if {$data_mode == "DWI" || $data_mode == "DWIknownB0" || $data_mode == "B0DWI"} {
		    # new data has been loaded, configure
		    # the vis tabs and sync their values
		    
		    sync_variance_tabs
		    sync_planes_tabs
		    sync_isosurface_tabs
		    sync_glyphs_tabs
		    sync_fibers_tabs
		    
		    configure_sample_planes

		    # configure the variance offset
		    global $mods(ChangeFieldBounds-Variance)-outputcenterx
		    global $mods(ChangeFieldBounds-Variance)-outputcentery
		    set $mods(ChangeFieldBounds-Variance)-outputcenterx \
			[expr -($min_x + ($size_x*$spacing_x/2.0)*1.1)]
		    set $mods(ChangeFieldBounds-Variance)-outputcentery \
			[expr $min_y + ($size_y*$spacing_y/2.0)]


		    # reconfigure registration reference image slider
		    $ref_image1.s.ref configure -from 1 -to $volumes
		    $ref_image2.s.ref configure -from 1 -to $volumes
		} else {
		    sync_planes_tabs
		    sync_isosurface_tabs
		    sync_glyphs_tabs
		    sync_fibers_tabs
		    
		    configure_sample_planes
		    
		}
	    } else {
		puts "DATA DID NOT LOAD PROPERLY"
	    }
 	} elseif {$which == $mods(ShowField-X) && $state == "JustStarted"} {
	    change_indicate_val 1
 	} elseif {$which == $mods(ShowField-X) && $state == "Completed"} {
	    if { !$has_autoviewed && !$loading} {
		after 100 "$mods(Viewer)-ViewWindow_0-c autoview; global $mods(Viewer)-ViewWindow_0-pos; set $mods(Viewer)-ViewWindow_0-pos \"z0_y0\"; $mods(Viewer)-ViewWindow_0-c Views"
		set has_autoviewed 1
	    }
 	    change_indicate_val 2
 	} elseif {$which == $mods(ShowField-Y) && $state == "JustStarted"} {
	    change_indicate_val 1
 	} elseif {$which == $mods(ShowField-Y) && $state == "Completed"} {
 	    change_indicate_val 2
 	} elseif {$which == $mods(ShowField-Z) && $state == "JustStarted"} {
	    change_indicate_val 1
 	} elseif {$which == $mods(ShowField-Z) && $state == "Completed"} {
 	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Isosurface) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Isosurface) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Glyphs) && $state == "JustStarted"}  {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Glyphs) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Fibers) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Fibers) && $state == "Completed"} { 
	    change_indicate_val 2
	} elseif {$which == $mods(SampleField-GlyphSeeds) && $state == "Completed"} {
	    global $mods(ChooseField-GlyphSeeds)-port-selected-index
	    if {[set $mods(ShowField-Glyphs)-tensors-on] == 0 || [set $mods(ChooseField-GlyphSeeds)-port-selected-index] != 1}  {
		after 100 \
		    "uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	    }
	} elseif {$which == $mods(Probe-GlyphSeeds) && $state == "Completed"} {
	    global $mods(ChooseField-GlyphSeeds)-port-selected-index
	    if {[set $mods(ShowField-Glyphs)-tensors-on] == 0 || [set $mods(ChooseField-GlyphSeeds)-port-selected-index] != 0} {
		after 100 \
		    "uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	    }
	} elseif {$which == $mods(SampleField-FiberSeeds) && $state == "Completed"} {
	    global $mods(ChooseField-FiberSeeds)-port-selected-index
	    if {[set $mods(ShowField-Fibers)-edges-on] == 0 || [set $mods(ChooseField-FiberSeeds)-port-selected-index] != 1}  {
		after 100 \
		    "uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	    }
	} elseif {$which == $mods(Probe-FiberSeeds) && $state == "Completed"} {
	    global $mods(ChooseField-FiberSeeds)-port-selected-index
	    if {[set $mods(ShowField-Fibers)-edges-on] == 0 || [set $mods(ChooseField-FiberSeeds)-port-selected-index] != 0} {
		after 100 \
		    "uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	    }
	} 
    }

    
    ##########################
    ### indicate_error
    ##########################
    # This method should change the indicator and labels to
    # the error state.  This should be done using the change_indicate_val
    # and change_indicator_labels methods. We catch errors from
    # RescaleColorMap and ChooseColorMap modules and ignore them for this app.
    method indicate_error { which msg_state } {
	global mods
	
	# if it is an error from a ChooseColorMap module ignore it
	if {$which != $mods(ChooseColorMap-Isosurface) && \
		$which != $mods(ChooseColorMap-Planes) && \
		$which != $mods(ChooseColorMap-Glyphs) && \
		$which != $mods(ChooseColorMap-Fibers) && \
		$which != $mods(RescaleColorMap-ColorPlanes) && \
		$which != $mods(RescaleColorMap-Isosurface) && \
		$which != $mods(RescaleColorMap-Fibers) && \
		$which != $mods(RescaleColorMap-Glyphs) && \
		$which != $mods(RescaleColorMap2) && \
	        $which != $mods(SampleField-FiberSeeds) && \
	        $which != $mods(SampleField-GlyphSeeds)} {
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
		    
		    if {$dt_completed} {
			change_indicator_labels "Visualization..."
		    } elseif {$reg_completed} {
			change_indicator_labels "Building Diffusion Tensors..."
		    } elseif {$data_completed} {
			change_indicator_labels "Registration..."
		    } else {
			change_indicator_labels "Loading Data..."
		    }
		    change_indicate_val 0
		}
	    }
	}
    }


##########################################################################	
########################### PROCESSING STEPS #############################
##########################################################################


############# DATA ACQUISITION #############

    ######################
    ### configure_data_tab
    ######################
    # Configure next and execute buttons and modify if
    # loading in tensors directly
    method configure_data_tab {} {
	global data_mode
	foreach w [winfo children $data_tab1] {
	    enable_widget $w
	}
	foreach w [winfo children $data_tab2] {
	    enable_widget $w
	}

	# configure t2 reference image stuff if loading tensors directly
	if {$data_mode == "DWI" || $data_mode == "DWIknownB0" || $data_mode == "B0DWI"} {
	    toggle_data_mode
	} 

	if {$data_completed} {
	    $data_next_button1 configure -state normal -foreground black \
		-background $next_color
	    $data_next_button2 configure -state normal -foreground black \
		-background $next_color
	}

    }


    #######################
    ### execute_Data
    #######################
    # Method called when execute is clicked on Load Data Tab
    method execute_Data {} {
	global mods 
	global data_mode
	global $mods(ChooseNrrd1)-port-selected-index

	# Call toggle_data_mode to reset the data state which
	# helps maintain state of disabled/enabled modules when
	# the user executes twice
	if {!$loading} {
	    $this toggle_data_mode
	}

	if {$data_mode == "DWI" || $data_mode == "DWIknownB0" || $data_mode == "B0DWI"} {
	    # determine if we are loading nrrd, dicom, or analyze
	    # and check if both DWI and T2 files have been specified
	    if {[set $mods(ChooseNrrd1)-port-selected-index] == 0} {
		# Nrrd

		global $mods(NrrdReader1)-filename
		if {![file exists [set $mods(NrrdReader1)-filename]]} {
		    set answer [tk_messageBox -message \
				    "Please specify a valid nrrd file\nwith DWI volumes before executing." -type ok -icon info -parent .standalone] 
		    return
		}


		if {$data_mode == "DWIknownB0"} {
		    global $mods(NrrdReader-T2)-filename
		    if {![file exists [set $mods(NrrdReader-T2)-filename]]} {
			set answer [tk_messageBox -message \
					"Please specify an existing nrrd file\nwith a T2 reference image before\nexecuting." -type ok -icon info -parent .standalone] 
			return
		    }
		    
		}
	    } elseif {[set $mods(ChooseNrrd1)-port-selected-index] == 1} {
		# Dicom 
		global $mods(DicomToNrrd1)-num-entries
		global $mods(DicomToNrrd-T2)-num-entries

		if {[set $mods(DicomToNrrd1)-num-entries] == 0} {
		    set answer [tk_messageBox -message \
				    "Please specify existing Dicom files\nof DWI Volumes before\nexecuting." -type ok -icon info -parent .standalone] 
		    return
		}
		if {$data_mode == "DWIknownB0"} {
		    if {[set $mods(DicomToNrrd-T2)-num-entries] == 0} {
			set answer [tk_messageBox -message \
					"Please specify existing Dicom files\nof a T2 reference image before\nexecuting." -type ok -icon info -parent .standalone] 
			return
		    }
		}
	    } elseif {[set $mods(ChooseNrrd1)-port-selected-index] == 2} {
		# Analyze
		global $mods(AnalyzeToNrrd1)-num-files
		global $mods(AnalyzeToNrrd-T2)-num-files

		if {[set $mods(AnalyzeToNrrd1)-num-files] == 0} {
		    set answer [tk_messageBox -message \
				    "Please specify existing analyze files\nof DWI Volumes before\nexecuting." -type ok -icon info -parent .standalone] 
		    return
		}
		if {$data_mode == "DWIknownB0"} {
		    if {[set $mods(AnalyzeToNrrd-T2)-num-files] == 0} {
			set answer [tk_messageBox -message \
					"Please specify an existing analyze file\nof a T2 reference image before\nexecuting." -type ok -icon info -parent .standalone] 
			return
		    }
		}
	    } else {
		# shouldn't get here
		return
	    }
	    
	    set data_completed 1

	    if {!$dt_completed && !$loading} {
		disableModule $mods(ChooseNrrd-DT) 1
	    }

	    activate_registration

	    # enable Next button
	    $data_next_button1 configure -state normal \
		-foreground black -background $next_color
	    $data_next_button2 configure -state normal \
		-foreground black -background $next_color
	} else {
	    # Loading tensors
	    # determine if we are loading nrrd, dicom, or analyze
	    # and check if the tensors file has been specified
	    if {[set $mods(ChooseNrrd1)-port-selected-index] == 0} {
		# Nrrd
		global $mods(NrrdReader1)-filename
		if {![file exists [set $mods(NrrdReader1)-filename]]} {
		    set answer [tk_messageBox -message \
				    "Please specify a valid nrrd file\nwith Tensors before executing." -type ok -icon info -parent .standalone] 
		    return
		}
	    } elseif {[set $mods(ChooseNrrd1)-port-selected-index] == 1} {
		# Dicom 
		global $mods(DicomToNrrd1)-num-entries
		global $mods(DicomToNrrd-T2)-num-entries

		if {[set $mods(DicomToNrrd1)-num-entries] == 0} {
		    set answer [tk_messageBox -message \
				    "Please specify existing Dicom files\nofTensors before\nexecuting." -type ok -icon info -parent .standalone] 
		    return
		}
	    } elseif {[set $mods(ChooseNrrd1)-port-selected-index] == 2} {
		# Analyze
		global $mods(AnalyzeToNrrd1)-num-files

		if {[set $mods(AnalyzeToNrrd1)-num-files] == 0} {
		    set answer [tk_messageBox -message \
				    "Please specify existing analyze files\nof Tensors before\nexecuting." -type ok -icon info -parent .standalone] 
		    return
		}
	    } else {
		# shouldn't get here
		return
	    }

	    set data_completed 1

	    if {!$loading} {
		disableModule $mods(TendEpireg) 1
		disableModule $mods(ChooseNrrd-ToReg) 1
		disableModule $mods(ChooseNrrd-DT) 0
		disableModule $mods(TendEstim) 1
	    }

	    activate_vis
	}

	global data_mode
	if {!$loading && $data_mode == "B0DWI"} {
	    disableModule $mods(UnuSlice-B0) 0
	}
	
	# determine which type of data (nrrd, dicom, analyze)
	# and execute that module
	global $mods(ChooseNrrd1)-port-selected-index
	if {[set $mods(ChooseNrrd1)-port-selected-index] == 0} {
	    $mods(NrrdReader1)-c needexecute
	} elseif {[set $mods(ChooseNrrd1)-port-selected-index] == 1} {
	    $mods(DicomToNrrd1)-c needexecute
	} else {
	    $mods(AnalyzeToNrrd1)-c needexecute
	}
    }

    
    ###########################
    ### toggle_data_mode
    ###########################
    # Called when user selects to load tensors directly,
    # or DWI images.  If loading tensors, Registration and
    # Build Tensors steps skipped and Next button disabled.
    method toggle_data_mode { } {
	global data_mode
        global mods
        global $mods(ChooseNrrd-DT)-port-selected-index
	global $mods(NrrdReader1)-type
	global $mods(ChooseNrrd-B0)-port-selected-index
	global $mods(ChooseNrrd-T2)-port-selected-index
	global $mods(TendEstim)-knownB0
	
	if {$data_mode == "DWIknownB0"} {
	    configure_readers all
	    
	    disableModule $mods(UnuAxinsert) 1
	    disableModule $mods(UnuReshape) 1
	    disableModule $mods(UnuAxinfo-Z) 1
	    disableModule $mods(UnuCrop-DWI) 1
	    disableModule $mods(UnuSlice-B0) 1
	    disableModule $mods(ChooseNrrd-preprocess) 1
	    disableModule $mods(ChooseNrrd-B0) 1
	    
	    # configure text for DWI Volume
	    $nrrd_tab1.dwil configure -text "DWI Volume:"
	    $nrrd_tab2.dwil configure -text "DWI Volume:"
	    
	    $dicom_tab1.dwil configure -text "DWI Volume:"
	    $dicom_tab2.dwil configure -text "DWI Volume:"
	    
	    $analyze_tab1.dwil configure -text "DWI Volume:"
	    $analyze_tab2.dwil configure -text "DWI Volume:"
	    
	    # enable T2 stuff
	    $nrrd_tab1.t2l configure -state normal
	    $nrrd_tab2.t2l configure -state normal
	    $nrrd_tab1.file2 configure -state normal -foreground black
	    $nrrd_tab2.file2 configure -state normal -foreground black
	    $nrrd_tab1.load2 configure -state normal
	    $nrrd_tab2.load2 configure -state normal
	    
	    $dicom_tab1.t2l configure -state normal
	    $dicom_tab2.t2l configure -state normal
	    $dicom_tab1.load2 configure -state normal
	    $dicom_tab2.load2 configure -state normal
	    
	    $analyze_tab1.t2l configure -state normal
	    $analyze_tab2.t2l configure -state normal
	    $analyze_tab1.load2 configure -state normal
	    $analyze_tab2.load2 configure -state normal
	    
	    # configure ChooseNrrd
	    set $mods(ChooseNrrd-DT)-port-selected-index 0
	    
	    # enable registration and dt tabs
	    #activate_registration
	    
	    #activate_dt
	    
	    set $mods(TendEstim)-knownB0 1
	    
	    set $mods(ChooseNrrd-B0)-port-selected-index 0
	    set $mods(ChooseNrrd-T2)-port-selected-index $last_B0_port
	    
	    set $mods(ChooseNrrd-B0)-port-selected-index 0
	    
	} elseif {$data_mode == "DWI"} {
	    configure_readers all
	    
	    disableModule $mods(UnuAxinsert) 1
	    disableModule $mods(UnuReshape) 1
	    disableModule $mods(UnuAxinfo-Z) 1
	    disableModule $mods(UnuCrop-DWI) 1
	    disableModule $mods(UnuSlice-B0) 1
	    disableModule $mods(ChooseNrrd-preprocess) 1
	    disableModule $mods(ChooseNrrd-B0) 1
		
	    # configure text for DWI Volume
	    $nrrd_tab1.dwil configure -text "DWI Volume:"
	    $nrrd_tab2.dwil configure -text "DWI Volume:"
	    
	    $dicom_tab1.dwil configure -text "DWI Volume:"
	    $dicom_tab2.dwil configure -text "DWI Volume:"
	    
	    $analyze_tab1.dwil configure -text "DWI Volume:"
	    $analyze_tab2.dwil configure -text "DWI Volume:"
	    
	    # disable T2 stuff
	    $nrrd_tab1.t2l configure -state disabled
	    $nrrd_tab2.t2l configure -state disabled
	    $nrrd_tab1.file2 configure -state disabled -foreground grey64
	    $nrrd_tab2.file2 configure -state disabled -foreground grey64
	    $nrrd_tab1.load2 configure -state disabled
	    $nrrd_tab2.load2 configure -state disabled
	    
	    $dicom_tab1.t2l configure -state disabled
	    $dicom_tab2.t2l configure -state disabled
	    $dicom_tab1.load2 configure -state disabled
	    $dicom_tab2.load2 configure -state disabled
	    
	    $analyze_tab1.t2l configure -state disabled
	    $analyze_tab2.t2l configure -state disabled
	    $analyze_tab1.load2 configure -state disabled
	    $analyze_tab2.load2 configure -state disabled
	    
	    # configure ChooseNrrd
	    set $mods(ChooseNrrd-DT)-port-selected-index 0
	    
	    # enable registration and dt tabs
	    #activate_registration
	    
	    #activate_dt
	    
	    set $mods(TendEstim)-knownB0 0
	    
	    set $mods(ChooseNrrd-B0)-port-selected-index 0
	    set $mods(ChooseNrrd-T2)-port-selected-index $last_B0_port

	    set $mods(ChooseNrrd-B0)-port-selected-index 0
        } elseif {$data_mode == "B0DWI"} {
	    configure_readers all

	    if {!$loading} {
		disableModule $mods(UnuAxinsert) 1
		disableModule $mods(UnuReshape) 1
		disableModule $mods(UnuAxinfo-Z) 1
		disableModule $mods(UnuCrop-DWI) 1
		disableModule $mods(UnuSlice-B0) 1
		disableModule $mods(ChooseNrrd-preprocess) 1
		disableModule $mods(ChooseNrrd-B0) 1  
	    } 

	    # configure text for DWI Volume
	    $nrrd_tab1.dwil configure -text "B0/DWI Volume:"
	    $nrrd_tab2.dwil configure -text "B0/DWI Volume:"
	    
	    $dicom_tab1.dwil configure -text "B0/DWI Volume:"
	    $dicom_tab2.dwil configure -text "B0/DWI Volume:"
	    
	    $analyze_tab1.dwil configure -text "B0/DWI Volume:"
	    $analyze_tab2.dwil configure -text "B0/DWI Volume:"

	    # disable T2 stuff
	    $nrrd_tab1.t2l configure -state disabled
	    $nrrd_tab2.t2l configure -state disabled
	    $nrrd_tab1.file2 configure -state disabled -foreground grey64
	    $nrrd_tab2.file2 configure -state disabled -foreground grey64
	    $nrrd_tab1.load2 configure -state disabled
	    $nrrd_tab2.load2 configure -state disabled
	    
	    $dicom_tab1.t2l configure -state disabled
	    $dicom_tab2.t2l configure -state disabled
	    $dicom_tab1.load2 configure -state disabled
	    $dicom_tab2.load2 configure -state disabled
	    
	    $analyze_tab1.t2l configure -state disabled
	    $analyze_tab2.t2l configure -state disabled
	    $analyze_tab1.load2 configure -state disabled
	    $analyze_tab2.load2 configure -state disabled
	    
	    # configure ChooseNrrd
	    set $mods(ChooseNrrd-DT)-port-selected-index 0
	    
	    # enable registration and dt tabs
	    #activate_registration
	    
	    #activate_dt

	    set $mods(TendEstim)-knownB0 1

	    set $mods(ChooseNrrd-B0)-port-selected-index 1

	    set $mods(ChooseNrrd-T2)-port-selected-index 3
	} else {
	    configure_readers all

	    disableModule $mods(UnuAxinsert) 1
	    disableModule $mods(UnuReshape) 1
	    disableModule $mods(UnuAxinfo-Z) 1
	    disableModule $mods(UnuCrop-DWI) 1
	    disableModule $mods(UnuSlice-B0) 1
	    disableModule $mods(ChooseNrrd-preprocess) 1
	    disableModule $mods(ChooseNrrd-B0) 1 

	    # configure labels
	    $nrrd_tab1.dwil configure -text "Tensor Volume:"
	    $nrrd_tab2.dwil configure -text "Tensor Volume:"
	    
	    $dicom_tab1.dwil configure -text "Tensor Volume:"
	    $dicom_tab2.dwil configure -text "Tensor Volume:"
	    
	    $analyze_tab1.dwil configure -text "Tensor Volume:"
	    $analyze_tab2.dwil configure -text "Tensor Volume:"
	    
	    # disable T2 stuff
	    $nrrd_tab1.t2l configure -state disabled
	    $nrrd_tab2.t2l configure -state disabled
	    $nrrd_tab1.file2 configure -state disabled -foreground grey64
	    $nrrd_tab2.file2 configure -state disabled -foreground grey64
	    $nrrd_tab1.load2 configure -state disabled
	    $nrrd_tab2.load2 configure -state disabled
	    
	    $dicom_tab1.t2l configure -state disabled
	    $dicom_tab2.t2l configure -state disabled
	    $dicom_tab1.load2 configure -state disabled
	    $dicom_tab2.load2 configure -state disabled
	    
	    $analyze_tab1.t2l configure -state disabled
	    $analyze_tab2.t2l configure -state disabled
	    $analyze_tab1.load2 configure -state disabled
	    $analyze_tab2.load2 configure -state disabled
	    
	    # configure ChooseNrrd
	    set $mods(ChooseNrrd-DT)-port-selected-index 1
	    
	    # disable Next button
	    $data_next_button1 configure -state disabled \
		-background grey75 -foreground grey64
	    $data_next_button2 configure -state disabled \
		-background grey75 -foreground grey64
	    
	    # disable registation and dt tabs
	    foreach w [winfo children $reg_tab1] {
		disable_widget $w
	    }
	    foreach w [winfo children $reg_tab2] {
		disable_widget $w
	    }
	    
	    # fix next and execute in registration
	    $reg_tab1.last.ne configure -foreground grey64 -background grey75
	    $reg_tab2.last.ne configure -foreground grey64 -background grey75
	    
	    $reg_tab1.last.ex configure -foreground grey64 -background grey75
	    $reg_tab2.last.ex configure -foreground grey64 -background grey75
	    
	    foreach w [winfo children $dt_tab1] {
		disable_widget $w
	    }
	    foreach w [winfo children $dt_tab2] {
		disable_widget $w
	    }
	    
	    # fix execute in dt
	    $dt_tab1.last.ex configure -foreground grey64 -background grey75
	    $dt_tab2.last.ex configure -foreground grey64 -background grey75
	    
	    set $mods(ChooseNrrd-B0)-port-selected-index 0	
	    set $mods(ChooseNrrd-T2)-port-selected-index $last_B0_port
        }

	$this configure_data_mode

    }


    ##############################
    ### configure_readers
    ##############################
    # Keeps the readers in sync.  Every time a different
    # data tab is selected (Nrrd, Dicom, Analyze) the other
    # readers must be disabled to avoid errors.
    method configure_readers { which } {
        global mods
        global $mods(ChooseNrrd1)-port-selected-index
	global $mods(ChooseNrrd-T2)-port-selected-index
	global $mods(ChooseNrrd-ToProcess)-port-selected-index
        global data_mode

	if {$which == "Generic"} {
	    set $mods(ChooseNrrd1)-port-selected-index 0
	    set $mods(ChooseNrrd-T2)-port-selected-index 0
	    set last_B0_port 0	    
	    set $mods(ChooseNrrd-ToProcess)-port-selected-index 0

	    disableModule $mods(NrrdReader1) 0
	    disableModule $mods(NrrdReader-T2) 0

	    disableModule $mods(DicomToNrrd1) 1
	    disableModule $mods(DicomToNrrd-T2) 1

	    disableModule $mods(AnalyzeToNrrd1) 1
	    disableModule $mods(AnalyzeToNrrd-T2) 1

	    # disable T2 reader when loading single volume w/B0 and DWIs
	    global data_mode
	    if {$data_mode == "B0DWI"} {
		disableModule $mods(NrrdReader-T2) 1
		set $mods(ChooseNrrd-T2)-port-selected-index 3
	    }

	    if {$initialized != 0} {
		$data_tab1 view "Generic"
		$data_tab2 view "Generic"
		set c_data_tab "Generic"
	    }
        } elseif {$which == "Dicom"} {
	    set $mods(ChooseNrrd1)-port-selected-index 1
	    set $mods(ChooseNrrd-T2)-port-selected-index 1
	    set last_B0_port 1
	    set $mods(ChooseNrrd-ToProcess)-port-selected-index 1

	    disableModule $mods(NrrdReader1) 1
	    disableModule $mods(NrrdReader-T2) 1

	    disableModule $mods(DicomToNrrd1) 0
	    disableModule $mods(DicomToNrrd-T2) 0

	    disableModule $mods(AnalyzeToNrrd1) 1
	    disableModule $mods(AnalyzeToNrrd-T2) 1

	    # disable T2 reader when loading single volume w/B0 and DWIs
	    global data_mode
	    if {$data_mode == "B0DWI"} {
		disableModule $mods(DicomToNrrd-T2) 1
		set $mods(ChooseNrrd-T2)-port-selected-index 3
	    }

            if {$initialized != 0} {
		$data_tab1 view "Dicom"
		$data_tab2 view "Dicom"
		set c_data_tab "Dicom"
	    }
        } elseif {$which == "Analyze"} {
	    # Analyze
	    set $mods(ChooseNrrd1)-port-selected-index 2
	    set $mods(ChooseNrrd-T2)-port-selected-index 2
	    set last_B0_port 2
	    set $mods(ChooseNrrd-ToProcess)-port-selected-index 2

	    disableModule $mods(NrrdReader1) 1
	    disableModule $mods(NrrdReader-T2) 1

	    disableModule $mods(DicomToNrrd1) 1
	    disableModule $mods(DicomToNrrd-T2) 1

	    disableModule $mods(AnalyzeToNrrd1) 0
	    disableModule $mods(AnalyzeToNrrd-T2) 0

	    # disable T2 reader when loading single volume w/B0 and DWIs
	    global data_mode
	    if {$data_mode == "B0DWI"} {
		disableModule $mods(AnalyzeToNrrd-T2) 1
	    }

	    if {$initialized != 0} {
		$data_tab1 view "Analyze"
		$data_tab2 view "Analyze"
		set c_data_tab "Analyze"
	    }
        } elseif {$which == "all"} {
	    if {[set $mods(ChooseNrrd1)-port-selected-index] == 0} {
		# nrrd
		disableModule $mods(NrrdReader1) 0
		disableModule $mods(NrrdReader-T2) 0
		
		disableModule $mods(DicomToNrrd1) 1
		disableModule $mods(DicomToNrrd-T2) 1
		
		disableModule $mods(AnalyzeToNrrd1) 1
		disableModule $mods(AnalyzeToNrrd-T2) 1

		# disable T2 reader when loading single volume w/B0 and DWIs
		global data_mode
		if {$data_mode == "B0DWI"} {
		    disableModule $mods(NrrdReader-T2) 1
		    set $mods(ChooseNrrd-T2)-port-selected-index 3
		}
	    } elseif {[set $mods(ChooseNrrd1)-port-selected-index] == 1} {
		# dicom
		disableModule $mods(NrrdReader1) 1
		disableModule $mods(NrrdReader-T2) 1
		
		disableModule $mods(DicomToNrrd1) 0
		disableModule $mods(DicomToNrrd-T2) 0
		
		disableModule $mods(AnalyzeToNrrd1) 1
		disableModule $mods(AnalyzeToNrrd-T2) 1

		# disable T2 reader when loading single volume w/B0 and DWIs
		global data_mode
		if {$data_mode == "B0DWI"} {
		    disableModule $mods(DicomToNrrd-T2) 1
		    set $mods(ChooseNrrd-T2)-port-selected-index 3
		}
	    } else {
		# analyze
		disableModule $mods(NrrdReader1) 1
		disableModule $mods(NrrdReader-T2) 1
		
		disableModule $mods(DicomToNrrd1) 1
		disableModule $mods(DicomToNrrd-T2) 1
		
		disableModule $mods(AnalyzeToNrrd1) 0
		disableModule $mods(AnalyzeToNrrd-T2) 0

		# disable T2 reader when loading single volume w/B0 and DWIs
		global data_mode
		if {$data_mode == "B0DWI"} {
		    disableModule $mods(AnalyzeToNrrd-T2) 1
		    set $mods(ChooseNrrd-T2)-port-selected-index 3
		}
	    }
	}
    }


    #############################
    ### laod_nrrd_dwi
    #############################
    # Specify a nrrd file, set the tuple axis to 0
    method load_nrrd_dwi {} {
	global mods
	# disable execute button and change behavior of execute command
	$mods(NrrdReader1) initialize_ui
	.ui$mods(NrrdReader1).f7.execute configure -state disabled

	upvar #0 .ui$mods(NrrdReader1) data
	set data(-command) "wm withdraw .ui$mods(NrrdReader1)"
    }

    #############################
    ### load_nrrd_t2
    #############################
    # Specify a T2 nrrd file and set tuple axis 0
    method load_nrrd_t2 {} {
	global mods
	# disable execute button and change behavior of execute command
	$mods(NrrdReader-T2) initialize_ui
	.ui$mods(NrrdReader-T2).f7.execute configure -state disabled

	upvar #0 .ui$mods(NrrdReader-T2) data
	set data(-command) "wm withdraw .ui$mods(NrrdReader-T2)"

    } 

    method dicom_ui { m } {
	$m initialize_ui
	if {[winfo exists .ui$m]} {
	    # disable execute button 
	    .ui$m.buttonPanel.btnBox.execute configure -state disabled
	}
    }

    method analyze_ui { m } {
	$m initialize_ui
	if {[winfo exists .ui$m]} {
	    # disable execute button 
	    .ui$m.buttonPanel.btnBox.execute configure -state disabled
	}
    }



    ############# REGISTRATION ##############
    method configure_registration_tab {} {
	if {$data_completed} {
	    activate_registration
	}
    }
    
    method execute_Registration {} {
	global mods
	
	# Check gradient has been loaded
	global $mods(NrrdReader-Gradient)-filename
	
	if {[set $mods(NrrdReader-Gradient)-filename] != ""} {
	    # unblock modules
	    if {!$loading} {
		disableModule $mods(TendEpireg) 0
		disableModule $mods(ChooseNrrd-ToReg) 0
		disableModule $mods(UnuJoin) 0
		disableModule $mods(ChooseNrrd-ToReg) 0
		disableModule $mods(RescaleColorMap2) 0
	    }
	    
	    # activate reg variance checkbutton
	    $variance_tab1.reg configure -state normal
	    $variance_tab2.reg configure -state normal

	    # set ChooseNrrd-KnownB0 vals
	    global $mods(ChooseNrrd-KnownB0)-port-selected-index
	    global data_mode
	    if {$data_mode == "DWIknownB0"} {
		set $mods(ChooseNrrd-KnownB0)-port-selected-index 0
	    } elseif {$data_mode == "DWI"} {
		set $mods(ChooseNrrd-KnownB0)-port-selected-index 1
	    } elseif {$data_mode == "B0DWI"} {
		set $mods(ChooseNrrd-KnownB0)-port-selected-index 0
	    } else {
		set $mods(ChooseNrrd-KnownB0)-port-selected-index 0
	    }
	    
	    # execute
	    $mods(TendEpireg)-c needexecute

	    set reg_completed 1
	    set data_completed 1

	    # enable Next button (only if performing global epi registration)
	    global do_registration
	    if {$do_registration} {
		$reg_tab1.last.ne configure -state normal \
		    -foreground black -background $next_color
		$reg_tab2.last.ne configure -state normal \
		    -foreground black -background $next_color
	    }
	} else {
	    set answer [tk_messageBox -message \
			    "Please load a text file containing the gradients by clicking \"Load Gradients\"" -type ok -icon info -parent .standalone]
	    
	}
    }


    method activate_registration { } {
        global mods

	# configure Registrations next button
	if {$data_completed} {
	    foreach w [winfo children $reg_tab1] {
		enable_widget $w
	    }
	    
	    foreach w [winfo children $reg_tab2] {
		enable_widget $w
	    }

# 	    $reg_tab1.last.ne configure -state normal \
# 		-foreground black -background $next_color
# 	    $reg_tab2.last.ne configure -state normal \
# 		-foreground black -background $next_color
	    if {!$reg_completed} {
		$reg_tab1.last.ne configure -state disabled \
		    -foreground grey64 -background grey75
		$reg_tab2.last.ne configure -state disabled \
		    -foreground grey64 -background grey75
	    }
	} 
#        else {
# 	    $reg_tab1.last.ne configure -state disabled \
# 		-foreground grey64 -background grey75
# 	    $reg_tab2.last.ne configure -state disabled \
# 		-foreground grey64 -background grey75

# 	}
	
	toggle_reference_image_state

        $ref_image1.s.ref configure -from 1 -to $volumes
        $ref_image2.s.ref configure -from 1 -to $volumes


	# select appropriate resampling filter in optionmenu
        global $mods(TendEpireg)-kernel

        if {[set $mods(TendEpireg)-kernel] ==  "tent"} {
	    $reg_tab1.rf select "Linear"
	    $reg_tab2.rf select "Linear"
	} elseif {[set $mods(TendEpireg)-kernel] ==  "cubicCR"} {
	    $reg_tab1.rf select "Catmull-Rom"
	    $reg_tab2.rf select "Catmull-Rom"
	} elseif {[set $mods(TendEpireg)-kernel] ==  "hann"} {
	    $reg_tab1.rf select "Windowed Sinc"
	    $reg_tab2.rf select "Windowed Sinc"
        } 

	toggle_registration_threshold

    }


    method load_gradient {} {
        global mods
        #set theWindow [$mods(NrrdReader-Gradient) make_file_open_box]
	$mods(NrrdReader-Gradient) initialize_ui
	.ui$mods(NrrdReader-Gradient).f7.execute configure -state disabled
        #tkwait window $theWindow
	
        #update idletasks
    }

    method set_resampling_filter { w } {
        set value [$w get]

	$reg_tab1.rf select $value
	$reg_tab2.rf select $value

        set kern ""
        if {$value == "Linear"} {
          set kern "tent"
        } elseif {$value == "Catmull-Rom"} {
          set kern "cubicCR"
        } elseif {$value == "Windowed Sinc"} {
          set kern "hann"
        }

        global mods
        global $mods(TendEpireg)-kernel
        set $mods(TendEpireg)-kernel $kern
    }



    method configure_fitting_label { val } {
	$reg_tab1.fit.f configure -text "[expr round([expr $val * 100])]"
	$reg_tab2.fit.f configure -text "[expr round([expr $val * 100])]"
    }


    method toggle_registration_threshold {} {
       global mods
       global $mods(TendEpireg)-use-default-threshold
       if {[set $mods(TendEpireg)-use-default-threshold] == 0 } {
          $reg_thresh1.choose.entry configure -state normal -foreground black
          $reg_thresh2.choose.entry configure -state normal -foreground black
       } else {
          $reg_thresh1.choose.entry configure -state disabled -foreground grey64
          $reg_thresh2.choose.entry configure -state disabled -foreground grey64
       }
    }

    method toggle_reference_image_state {} {
       global mods
       global  $mods(TendEpireg)-reference
       global ref_image_state ref_image

       if {$ref_image_state == 0 } {
          # implicit reference image
          set $mods(TendEpireg)-reference "-1"
          $ref_image1.s.ref configure -state disabled
          $ref_image1.s.label configure -state disabled
          $ref_image2.s.ref configure -state disabled
          $ref_image2.s.label configure -state disabled
       } else {
          # choose reference image
          set $mods(TendEpireg)-reference [expr $ref_image - 1]
          $ref_image1.s.ref configure -state normal
          $ref_image1.s.label configure -state normal
          $ref_image2.s.ref configure -state normal
          $ref_image2.s.label configure -state normal
       }
    }

    method configure_reference_image { val } {
       global ref_image ref_image_state
       set ref_image $val
       if {$ref_image_state == 1} {
  	  global mods
          global $mods(TendEpireg)-reference
	  set $mods(TendEpireg)-reference [expr $val - 1]
       }
    }



################### BUILDING DIFFUSION TENSORS #################
    method configure_dt_tab {} {
	if {$reg_completed} {
	    activate_dt
	}
	
	toggle_do_smoothing

	toggle_dt_threshold

	toggle_b_matrix

    }


    method execute_DT {} {
	global mods
	
	# Check bmatrix has been loaded
	global $mods(NrrdReader-BMatrix)-filename
	global bmatrix
	
	if {$bmatrix == "load"} {
	    if {![file exists [set $mods(NrrdReader-BMatrix)-filename]]} {
		set answer [tk_messageBox -message \
				"Please load a B-Matrix file containing." -type ok -icon info -parent .standalone]
		return
	    }
	    
	} 
	
	# unblock modules
	if {!$loading} {
	    disableModule $mods(TendEstim) 0
	    disableModule $mods(ChooseNrrd-DT) 0
	    disableModule $mods(ChooseNrrd-ToReg) 0
	}

	# execute
	$mods(ChooseNrrd-ToSmooth)-c needexecute

	set dt_completed 1
	set reg_completed 1
	set data_completed 1
		
	view_Vis
    }


    method activate_dt { } {
	foreach w [winfo children $dt_tab1] {
	    enable_widget $w
        }

	foreach w [winfo children $dt_tab2] {
	    enable_widget $w
        }

        toggle_do_smoothing

        toggle_dt_threshold

        toggle_b_matrix

    }

    
    method load_bmatrix {} {
	global mods
        #set theWindow [$mods(NrrdReader-BMatrix) make_file_open_box]
	$mods(NrrdReader-BMatrix) initialize_ui
	.ui$mods(NrrdReader-BMatrix).f7.execute configure -state disabled
	
	#tkwait window $theWindow
	
	#update idletasks
	
	
	#        global $mods(NrrdReader-BMatrix)-axis
	#        set $mods(NrrdReader-BMatrix)-axis 0
    } 
    
    method toggle_do_registration {} {
        global mods
        global $mods(ChooseNrrd-ToReg)-port-selected-index
        global do_registration
	global bmatrix
	
	if {$do_registration == 1} {
	    disableModule $mods(ChooseNrrd-ToReg) 0
	    disableModule $mods(UnuJoin) 0
	    disableModule $mods(NrrdReader-Gradient) 0
	    disableModule $mods(TendEpireg) 0
	    
	    activate_registration

	    # change ChooseNrrd
	    set $mods(ChooseNrrd-ToReg)-port-selected-index 0

	    # set bmatrix selection to load compute
	    set bmatrix "compute"
	    $this toggle_b_matrix
        } else {
	    disableModule $mods(ChooseNrrd-ToReg) 0
	    disableModule $mods(UnuJoin) 1
	    disableModule $mods(NrrdReader-Gradient) 1
	    disableModule $mods(TendEpireg) 1
	    
	    # disable registration tab
	    foreach w [winfo children $reg_tab1] {
		disable_widget $w
	    }
	    foreach w [winfo children $reg_tab2] {
		disable_widget $w
	    }
	    
	    toggle_reference_image_state
	    toggle_registration_threshold
	    
	    # re-enable checkbutton 
	    $reg_tab1.doreg configure -state normal -foreground black
	    $reg_tab2.doreg configure -state normal -foreground black
	    
	    # re-enable next button
	    $reg_tab1.last.ne configure -state normal \
		-foreground black -background $next_color
	    $reg_tab2.last.ne configure -state normal \
		-foreground black -background $next_color

	    # grey out execute button
	    $reg_tab1.last.ex configure -background grey75 -foreground grey64
	    $reg_tab2.last.ex configure -background grey75 -foreground grey64
	    	    
	    # change ChooseNrrd
	    set $mods(ChooseNrrd-ToReg)-port-selected-index 1

	    # set bmatrix selection to load load
	    set bmatrix "load"
	    $this toggle_b_matrix

        }
    }



    
    method configure_sample_planes {} {
	global mods
	global $mods(SamplePlane-X)-sizex
	global $mods(SamplePlane-X)-sizey
	
	global $mods(SamplePlane-Y)-sizex
	global $mods(SamplePlane-Y)-sizey
	
	global $mods(SamplePlane-Z)-sizex
	global $mods(SamplePlane-Z)-sizey
	
	# X Axis
	set $mods(SamplePlane-X)-sizex $size_z
	set $mods(SamplePlane-X)-sizey $size_y
	
	# Y Axis
	set $mods(SamplePlane-Y)-sizex $size_x
	set $mods(SamplePlane-Y)-sizey $size_z
	
	# Z Axis
	set $mods(SamplePlane-Z)-sizex $size_x
	set $mods(SamplePlane-Z)-sizey $size_y
	
	global plane_x plane_y plane_z
	
	# configure SamplePlane positions
	global $mods(SamplePlane-X)-pos
	global $mods(SamplePlane-Y)-pos
	global $mods(SamplePlane-Z)-pos
	
	set result_x [expr [expr $plane_x / [expr $size_x / 2.0] ] - 1.0]
	set $mods(SamplePlane-X)-pos $result_x
	
	set result_y [expr [expr $plane_y / [expr $size_y / 2.0] ] - 1.0]
	set $mods(SamplePlane-Y)-pos $result_y
	
	set result_z [expr [expr $plane_z / [expr $size_z / 2.0] ] - 1.0]
	set $mods(SamplePlane-Z)-pos $result_z
	
	# configure ClipByFunction string
	configure_ClipByFunction            
    }

    method configure_ClipByFunction {} {
	global mods
	global $mods(ClipByFunction-Seeds)-clipfunction
	global $mods(Isosurface)-isoval
	global plane_x plane_y plane_z

	set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	set span_z [expr [expr $plane_z*$spacing_z]+$min_z]

	# The ClipByFunction module can now take arguments.
	#   -u0  $mods(Isosurface)-isoval
	#   -u1  $span_x
	#   -u2  $span_y
	#   -u3  $span_z

	# Only include axis information for planes that are turned on
	global show_plane_x
	global show_plane_y
	global show_plane_z
	
	global $mods(ClipByFunction-Seeds)-u0
	global $mods(ClipByFunction-Seeds)-u1
	global $mods(ClipByFunction-Seeds)-u2
	global $mods(ClipByFunction-Seeds)-u3

	set $mods(ClipByFunction-Seeds)-u0 [set $mods(Isosurface)-isoval]
	set function "(v > u0) &&"

	if {$show_plane_x} {
	    set $mods(ClipByFunction-Seeds)-u1 $span_x
	    set index [string last "&&" $function]
	    set function [string replace $function $index end "&& (x $clip_x u1) &&"]
	}
	if {$show_plane_y} {
	    set $mods(ClipByFunction-Seeds)-u2 $span_y
	    set index [string last "&&" $function]
	    set function [string replace $function $index end "&& (y $clip_y u2) &&"]
	}
	if {$show_plane_z} {
	    set $mods(ClipByFunction-Seeds)-u3 $span_z
	    set index [string last "&&" $function]
	    set function [string replace $function $index end "&& (z $clip_z u3) &&"]
	}
	set index [string last "&&" $function]
	set function [string replace $function $index end ""]

	set $mods(ClipByFunction-Seeds)-clipfunction $function

	#set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $span_x) && (y $clip_y $span_y) && (z $clip_z $span_z)"
	
    }


    method toggle_dt_threshold {} {
	global mods
        global $mods(TendEstim)-use-default-threshold

        if {[set $mods(TendEstim)-use-default-threshold] == 1} {
            $dt_tab1.thresh.childsite.choose.entry configure -state disabled -foreground grey64
            $dt_tab2.thresh.childsite.choose.entry configure -state disabled -foreground grey64
        } else {
            $dt_tab1.thresh.childsite.choose.entry configure -state normal -foreground black
            $dt_tab2.thresh.childsite.choose.entry configure -state normal -foreground black
        }
    }

    method toggle_b_matrix {} {
	global bmatrix
	global mods
	global $mods(ChooseNrrd-BMatrix)-port-selected-index
	
	if {$bmatrix == "compute"} {
            $dt_tab1.bm.childsite.load.e configure -state disabled \
                -foreground grey64
            $dt_tab1.bm.childsite.browse configure -state disabled
	    
            $dt_tab2.bm.childsite.load.e configure -state disabled \
                -foreground grey64
            $dt_tab2.bm.childsite.browse configure -state disabled

	    set $mods(ChooseNrrd-BMatrix)-port-selected-index 0
	    disableModule $mods(NrrdReader-BMatrix) 1
	    disableModule $mods(NrrdReader-Gradient) 0
	} else {
            $dt_tab1.bm.childsite.load.e configure -state normal \
                -foreground black
            $dt_tab1.bm.childsite.browse configure -state normal
	    
            $dt_tab2.bm.childsite.load.e configure -state normal \
                -foreground black
            $dt_tab2.bm.childsite.browse configure -state normal

	    set $mods(ChooseNrrd-BMatrix)-port-selected-index 1
	    disableModule $mods(NrrdReader-BMatrix) 0
	    disableModule $mods(NrrdReader-Gradient) 1
	}
    }

    method toggle_do_smoothing {} {
        global mods
        global $mods(ChooseNrrd-ToSmooth)-port-selected-index
        global do_smoothing

        if {$do_smoothing == 0} {
           # activate smoothing scrollbar
           $dt_tab1.blur.childsite.rad1.l configure -state disabled
           $dt_tab2.blur.childsite.rad1.l configure -state disabled

           $dt_tab1.blur.childsite.rad1.s configure -state disabled -foreground grey64
           $dt_tab2.blur.childsite.rad1.s configure -state disabled -foreground grey64

           $dt_tab1.blur.childsite.rad1.v configure -state disabled
           $dt_tab2.blur.childsite.rad1.v configure -state disabled

           $dt_tab1.blur.childsite.rad2.l configure -state disabled
           $dt_tab2.blur.childsite.rad2.l configure -state disabled

           $dt_tab1.blur.childsite.rad2.s configure -state disabled -foreground grey64
           $dt_tab2.blur.childsite.rad2.s configure -state disabled -foreground grey64

           $dt_tab1.blur.childsite.rad2.v configure -state disabled
           $dt_tab2.blur.childsite.rad2.v configure -state disabled

	   # disable resample modules
	   disableModule $mods(UnuResample-XY) 1
	   disableModule $mods(UnuResample-Z) 1

           set $mods(ChooseNrrd-ToSmooth)-port-selected-index 1
        } else {
           # disable smoothing scrollbar
           $dt_tab1.blur.childsite.rad1.l configure -state normal
           $dt_tab2.blur.childsite.rad1.l configure -state normal

           $dt_tab1.blur.childsite.rad1.s configure -state normal -foreground black
           $dt_tab2.blur.childsite.rad1.s configure -state normal -foreground black

           $dt_tab1.blur.childsite.rad1.v configure -state normal
           $dt_tab2.blur.childsite.rad1.v configure -state normal

           $dt_tab1.blur.childsite.rad2.l configure -state normal
           $dt_tab2.blur.childsite.rad2.l configure -state normal

           $dt_tab1.blur.childsite.rad2.s configure -state normal -foreground black
           $dt_tab2.blur.childsite.rad2.s configure -state normal -foreground black

           $dt_tab1.blur.childsite.rad2.v configure -state normal
           $dt_tab2.blur.childsite.rad2.v configure -state normal

	   # enable resample modules
	   disableModule $mods(UnuResample-XY) 0
	   disableModule $mods(UnuResample-Z) 0

           set $mods(ChooseNrrd-ToSmooth)-port-selected-index 0

        }
    }

    method change_xy_smooth { val } {
        global mods
        global $mods(UnuResample-XY)-sigma
        global $mods(UnuResample-XY)-extent
	
        set $mods(UnuResample-XY)-sigma $val
        set $mods(UnuResample-XY)-extent [expr $val*3.0]
    }	

    method change_z_smooth { val } {
        global mods
        global $mods(UnuResample-Z)-sigma
        global $mods(UnuResample-Z)-extent
	
        set $mods(UnuResample-Z)-sigma $val
        set $mods(UnuResample-Z)-extent [expr $val*3.0]
    }


    
    method view_Vis {} {
        if {$dt_completed && $c_vis_tab == "Variance"} {
            # view planes tab
            $vis_tab1 view "Planes"
            $vis_tab2 view "Planes"
        } 
    }



    method activate_vis {} {
	global mods
	
	# turn off variances
	global $mods(ShowField-Orig)-faces-on
	global $mods(ShowField-Reg)-faces-on
	set $mods(ShowField-Orig)-faces-on 0
	set $mods(ShowField-Reg)-faces-on 0
	$mods(ShowField-Orig)-c toggle_display_faces
	$mods(ShowField-Reg)-c toggle_display_faces

	$mods(ChooseField-FiberSeeds)-c needexecute
	$mods(ChooseField-GlyphSeeds)-c needexecute
	
	uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
	uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}" 0
	uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
	uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}" 0
	
	$mods(Viewer)-ViewWindow_0-c redraw
	
	# setup global clipping planes
	initialize_clip_info
	
	change_indicator_labels "Visualization..."
	
	configure_planes_tabs
	
	configure_isosurface_tabs
	
	configure_glyphs_tabs
	
	configure_fibers_tabs
	
	set vis_activated 1

	global exec_iso
        global $mods(Viewer)-ViewWindow_0-global-clip
	if { $exec_iso(global-clip) == "on"} {
	    set $mods(Viewer)-ViewWindow_0-global-clip 1
	} else {
	    set $mods(Viewer)-ViewWindow_0-global-clip 0
	}

	global exec_planes
	if {$exec_planes(update-X)} {
	    update_plane_x
	    set exec_planes(update-X) 0
	}
	if {$exec_planes(update-Y)} {
	    update_plane_y
	    set exec_planes(update-Y) 0
	}
	if {$exec_planes(update-Z)} {
	    update_plane_z
	    set exec_planes(update-Z) 0
	}
	
	# bring planes tab forward
	view_Vis
    }

##########################################################################	
######################### VISUALIZATION STEPS ############################
##########################################################################



######## VARIANCE #########
    method build_variance_tab { f } {
	global tips
	global mods
        global $mods(UnuSlice1)-position

           if {![winfo exists $f.instr]} {
	       set num_slices [expr $size_z - 1]
	       if {$num_slices < 0} {
		   set num_slices 0
	       }

	       # detached
	       checkbutton $f.orig -text "View Variance of Original Data" \
		   -variable $mods(ShowField-Orig)-faces-on \
		   -state disabled \
		   -command {
		       global mods
		       $mods(ShowField-Orig)-c toggle_display_faces
                   }
	       Tooltip $f.orig $tips(VarToggleOrig)
	       
	       checkbutton $f.reg -text "View Variance of Registered Data" \
		   -variable $mods(ShowField-Reg)-faces-on \
		   -state disabled \
		   -command {
		       global mods
		       $mods(ShowField-Reg)-c toggle_display_faces
                   }
	       Tooltip $f.reg $tips(VarToggleReg)
	       
	       pack $f.orig $f.reg -side top -anchor nw -padx 3 -pady 3
	       
	       label $f.instr -text "Select a slice in the Z direction to view the variance." \
		   -state disabled
	       pack $f.instr -side top -anchor n -padx 3 -pady 3
	       
	       ### Slice Slider 
	       scale $f.slice -label "Slice:" \
		   -variable $mods(UnuSlice1)-position \
		   -from 0 -to 100 \
		   -showvalue true \
	           -state disabled -foreground grey64 \
		   -orient horizontal \
		   -command "$this change_variance_slice" \
		   -width 15 \
		   -sliderlength 15 \
		   -length [expr $notebook_width - 60]
	       
	       pack $f.slice -side top -anchor n -padx 3 -pady 3
	       
	       bind $f.slice <ButtonRelease> "app update_variance_slice"
	   } 
    }
	
    method sync_variance_tabs {} {
	global mods
	global $mods(UnuSlice1)-position
	global $mods(UnuSlice2)-position

	# configure slice slider
	$variance_tab1.slice configure -from 0 -to $size_z
	$variance_tab2.slice configure -from 0 -to $size_z

	set $mods(UnuSlice1)-position [expr $size_z / 2]
	set $mods(UnuSlice2)-position [expr $size_z / 2]
	# give initial value in middle

	update_variance_slice
    }

    method configure_variance_tabs {} {
	if {$data_completed} {
	    $variance_tab1.orig configure -state normal
	    $variance_tab2.orig configure -state normal

	    $variance_tab1.instr configure -state normal
	    $variance_tab2.instr configure -state normal
	    
	    $variance_tab1.slice configure -state normal -foreground black
	    $variance_tab2.slice configure -state normal -foreground black
	}
	if {$reg_completed} {
	    $variance_tab1.reg configure -state normal
	    $variance_tab2.reg configure -state normal
	}
    }

    method update_variance_slice {} {
	# only update if not in loading mode, otherwise
	# everything will execute twice
	if {$data_completed && !$loading} {
	    global mods
	    $mods(UnuSlice1)-c needexecute
	    
	    if {$reg_completed} {
		$mods(UnuSlice2)-c needexecute
	    }
	}

    }


    method change_variance_slice { val } {
	if {$data_completed} {
	    global mods
	    global $mods(UnuSlice2)-position
	    set $mods(UnuSlice2)-position $val
	}
    }

    

######## PLANES ##########
    
    method build_planes_tab {f} {
	global tips
	global mods
	global show_planes
	global show_plane_x show_plane_y show_plane_z
	global plane_x plane_y plane_z

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Planes:" -variable show_planes \
		-command "$this toggle_show_planes" -foreground grey64
	    Tooltip $f.show $tips(PlanesToggle)

	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    frame $f.axis -relief groove -borderwidth 2
	    pack $f.axis -side top -anchor n -padx 3 -pady 3
	    
	    frame $f.axis.x
	    pack $f.axis.x -side top -anchor nw 
	    
	    checkbutton $f.axis.x.check -text "X" \
		-variable show_plane_x \
		-foreground grey64 \
		-command "$this toggle_plane X"
	    Tooltip $f.axis.x.check $tips(PlanesXToggle)

	    scale $f.axis.x.slider -from 0 -to 512 \
		-variable plane_x \
		-showvalue false \
		-length 150  -width 15 \
		-sliderlength 15 \
		-foreground grey64 \
		-orient horizontal 
	    Tooltip $f.axis.x.slider $tips(PlanesXSlider)
	    bind $f.axis.x.slider <ButtonRelease> "app update_plane_x"
	    label $f.axis.x.label -textvariable plane_x -foreground grey64
	    pack $f.axis.x.check $f.axis.x.slider $f.axis.x.label -side left -anchor nw \
		-padx 2 -pady 3
	    
	    frame $f.axis.y
	    pack $f.axis.y -side top -anchor nw 
	    checkbutton $f.axis.y.check -text "Y" \
		-variable show_plane_y \
		-foreground grey64 \
		-command "$this toggle_plane Y"
	    Tooltip $f.axis.y.check $tips(PlanesYToggle)
	    scale $f.axis.y.slider -from 0 -to 512 \
		-variable plane_y \
		-showvalue false \
		-length 150  -width 15 \
		-sliderlength 15 \
		-foreground grey64 \
		-orient horizontal 
	    Tooltip $f.axis.y.slider $tips(PlanesYSlider)
	    bind $f.axis.y.slider <ButtonRelease> "app update_plane_y"
	    label $f.axis.y.label -textvariable plane_y -foreground grey64
	    pack $f.axis.y.check $f.axis.y.slider $f.axis.y.label -side left -anchor nw \
		-padx 2 -pady 3
	    
	    frame $f.axis.z
	    pack $f.axis.z -side top -anchor nw 
	    checkbutton $f.axis.z.check -text "Z" \
		-variable show_plane_z \
		-foreground grey64 \
		-command "$this toggle_plane Z"
	    Tooltip $f.axis.x.check $tips(PlanesZToggle)
	    scale $f.axis.z.slider -from 0 -to 512 \
		-variable plane_z \
		-showvalue false \
		-length 150  -width 15 \
		-sliderlength 15 \
		-foreground grey64 \
		-orient horizontal 
	    Tooltip $f.axis.z.slider $tips(PlanesZSlider)
	    bind $f.axis.z.slider <ButtonRelease> "app update_plane_z"
	    label $f.axis.z.label -textvariable plane_z -foreground grey64
	    pack $f.axis.z.check $f.axis.z.slider $f.axis.z.label -side left -anchor nw \
		-padx 2 -pady 3
	    
	    iwidgets::labeledframe $f.color \
		-labelpos nw -labeltext "Color Planes Based On" -foreground grey64
	    pack $f.color -side top -anchor nw -padx 3 -pady 3
	    
	    set fr [$f.color childsite]
	    frame $fr.select
	    pack $fr.select -side top -anchor nw -padx 3 -pady 3
	    
	    iwidgets::optionmenu $fr.select.color -labeltext "" \
		-labelpos w \
	        -width 150 \
		-command "$this select_color_planes_color $fr.select" \
		-foreground grey64
	    pack $fr.select.color -side left -anchor n -padx 1 -pady 3
	    
	    addColorSelection $fr.select "Color" clip_to_isosurface_color "clip_color_change"
	    
	    $fr.select.color insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant"
	    
	    $fr.select.color select "Principle Eigenvector"
	    
	    iwidgets::labeledframe $fr.maps \
		-labelpos nw -labeltext "Color Maps" -foreground grey64
	    pack $fr.maps -side top -anchor n -padx 3 -pady 3
	    
	    set maps [$fr.maps childsite]
	    
	    global $mods(GenStandardColorMaps-ColorPlanes)-mapType
	    
	    # Gray
	    frame $maps.gray
	    pack $maps.gray -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.gray.b -text "Gray" \
		-variable $mods(GenStandardColorMaps-ColorPlanes)-mapType \
		-value 0 \
		-foreground grey64 \
		-command "$this update_planes_color_map"
	    Tooltip $maps.gray.b $tips(PlanesColorMap)
	    pack $maps.gray.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.gray.f -relief sunken -borderwidth 2
	    pack $maps.gray.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.gray.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.gray.f.canvas -anchor e \
		-fill both -expand 1
	    
	    draw_colormap Gray $maps.gray.f.canvas
	    
	    # Rainbow
	    frame $maps.rainbow
	    pack $maps.rainbow -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.rainbow.b -text "Rainbow" \
		-variable $mods(GenStandardColorMaps-ColorPlanes)-mapType \
		-value 2 \
		-foreground grey64 \
		-command "$this update_planes_color_map"
	    Tooltip $maps.rainbow.b $tips(PlanesColorMap)
	    pack $maps.rainbow.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.rainbow.f -relief sunken -borderwidth 2
	    pack $maps.rainbow.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.rainbow.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.rainbow.f.canvas -anchor e
	    
	    draw_colormap Rainbow $maps.rainbow.f.canvas
	    
	    # Darkhue
	    frame $maps.darkhue
	    pack $maps.darkhue -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.darkhue.b -text "Darkhue" \
		-variable $mods(GenStandardColorMaps-ColorPlanes)-mapType \
		-value 5 \
		-foreground grey64 \
		-command "$this update_planes_color_map"
	    Tooltip $maps.darkhue.b $tips(PlanesColorMap)
	    pack $maps.darkhue.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.darkhue.f -relief sunken -borderwidth 2
	    pack $maps.darkhue.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.darkhue.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.darkhue.f.canvas -anchor e
	    
	    draw_colormap Darkhue $maps.darkhue.f.canvas
	    
	    
	    # Blackbody
	    frame $maps.blackbody
	    pack $maps.blackbody -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.blackbody.b -text "Blackbody" \
		-variable $mods(GenStandardColorMaps-ColorPlanes)-mapType \
		-value 7 \
		-foreground grey64 \
		-command "$this update_planes_color_map"
	    Tooltip $maps.blackbody.b $tips(PlanesColorMap)
	    pack $maps.blackbody.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.blackbody.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.blackbody.f.canvas -anchor e
	    
	    draw_colormap Blackbody $maps.blackbody.f.canvas
	    
	    
	    # Blue-to-Red
	    frame $maps.bpseismic
	    pack $maps.bpseismic -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.bpseismic.b -text "Blue-to-Red" \
		-variable $mods(GenStandardColorMaps-ColorPlanes)-mapType \
		-value 17 \
		-foreground grey64 \
		-command "$this update_planes_color_map"
	    Tooltip $maps.bpseismic.b $tips(PlanesColorMap)
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side left -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "Blue-to-Red" $maps.bpseismic.f.canvas
	    
	    global clip_to_isosurface
	    global clip_to_isosurface_color
	    checkbutton $f.clipiso -text "Clip to Isosurface" \
		-variable clip_to_isosurface \
		-command "$this toggle_clip_to_isosurface" -foreground grey64
	    Tooltip $f.clipiso $tips(PlanesClipToIso)
	    pack $f.clipiso -side top -anchor nw -padx 5 -pady 5

	} 
    }

    method update_planes_color_map {} {
	global mods
	global show_planes

	if {$vis_activated && $show_planes} {
	    $mods(GenStandardColorMaps-ColorPlanes)-c needexecute
	} else {
	    global exec_planes
	    set exec_planes(GenStandardColorMaps-ColorPlanes) 1
	}
    }



    method sync_planes_tabs {} {
	global mods
	
	# configure sliders
	$planes_tab1.axis.x.slider configure -to $size_x
	$planes_tab2.axis.x.slider configure -to $size_x
	
	$planes_tab1.axis.y.slider configure -to $size_y
	$planes_tab2.axis.y.slider configure -to $size_y

	$planes_tab1.axis.z.slider configure -to $size_z
	$planes_tab2.axis.z.slider configure -to $size_z

	if {!$dt_completed} {
	    global plane_x plane_y plane_z
	    set plane_x [expr $size_x/2]
	    set plane_y [expr $size_y/2]
	    set plane_z [expr $size_z/2]
	}

	global $mods(ChooseField-ColorPlanes)-port-selected-index
	set port [set $mods(ChooseField-ColorPlanes)-port-selected-index]

	if {$plane_type == "Constant"} {
	    #Constant
	    $planes_tab1.color.childsite.select.color select "Constant"
	    $planes_tab2.color.childsite.select.color select "Constant"
	} elseif {$port == 0} {
	    #FA
	    $planes_tab1.color.childsite.select.color select "Fractional Anisotropy"
	    $planes_tab2.color.childsite.select.color select "Fractional Anisotropy"
	} elseif {$port == 1} {
	    #LA
	    $planes_tab1.color.childsite.select.color select "Linear Anisotropy"
	    $planes_tab2.color.childsite.select.color select "Linear Anisotropy"
	} elseif {$port == 2} {
	    #PA
	    $planes_tab1.color.childsite.select.color select "Planar Anisotropy"
	    $planes_tab2.color.childsite.select.color select "Planar Anisotropy"

	} elseif {$port == 3} {
	    #e1
	    $planes_tab1.color.childsite.select.color select "Principle Eigenvector"
	    $planes_tab2.color.childsite.select.color select "Principle Eigenvector"
	} 
    }

    method configure_planes_tabs {} {
	global mods
	global $mods(ChooseField-ColorPlanes)-port-selected-index
	global $mods(ChooseColorMap-Planes)-port-selected-index

	set port [set $mods(ChooseField-ColorPlanes)-port-selected-index]
	set color_port [set $mods(ChooseColorMap-Planes)-port-selected-index]

	foreach w [winfo children $planes_tab1] {
	    enable_widget $w
	}
	foreach w [winfo children $planes_tab2] {
	    enable_widget $w
	}

	if { $color_port == 1 && $port != 3} {
	    # rescale is disabled and it isn't Principle Eigenvector
	    $planes_tab1.color.childsite.select.colorFrame.set_color configure -state normal
	    $planes_tab2.color.childsite.select.colorFrame.set_color configure -state normal
	    disable_planes_colormaps
	} elseif {$port == 3} {
	    $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
	    $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disable_planes_colormaps
	} else {
	    $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
	    $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    enable_planes_colormaps
	}
    }

    method select_color_planes_color { w } {
        global mods
	global $mods(ChooseField-ColorPlanes)-port-selected-index
	global $mods(ChooseColorMap-Planes)-port-selected-index
	global $mods(ShowField-X)-faces-usedefcolor
	global $mods(ShowField-Y)-faces-usedefcolor
	global $mods(ShowField-Z)-faces-usedefcolor
	
        set which [$w.color get]

	set $mods(ShowField-X)-faces-usedefcolor 0
	set $mods(ShowField-Y)-faces-usedefcolor 0
	set $mods(ShowField-Z)-faces-usedefcolor 0
	
        if {$which == "Principle Eigenvector"} {
	    set plane_type "Principle Eigenvector"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(ChooseColorMap-Planes) 1
	    set $mods(ChooseColorMap-Planes)-port-selected-index 1
	    set $mods(ChooseField-ColorPlanes)-port-selected-index 3
	    disable_planes_colormaps
        } elseif {$which == "Fractional Anisotropy"} {
	    set plane_type "Fractional Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(ChooseColorMap-Planes) 0
	    set $mods(ChooseColorMap-Planes)-port-selected-index 0
	    set $mods(ChooseField-ColorPlanes)-port-selected-index 0
	    enable_planes_colormaps
        } elseif {$which == "Linear Anisotropy"} {
	    set plane_type "Linear Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(ChooseColorMap-Planes) 0
	    set $mods(ChooseColorMap-Planes)-port-selected-index 0
	    set $mods(ChooseField-ColorPlanes)-port-selected-index 1
	    enable_planes_colormaps
        } elseif {$which == "Planar Anisotropy"} {
	    set plane_type "Planar Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(ChooseColorMap-Planes) 0
	    set $mods(ChooseColorMap-Planes)-port-selected-index 0
	    set $mods(ChooseField-ColorPlanes)-port-selected-index 2
	    enable_planes_colormaps
        } else {
	    set plane_type "Constant"
	    # specified color
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state normal
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state normal
	    disableModule $mods(ChooseColorMap-Planes) 1
	    set $mods(ChooseColorMap-Planes)-port-selected-index 1
	    set $mods(ChooseField-ColorPlanes)-port-selected-index 0

	    set $mods(ShowField-X)-faces-usedefcolor 1
	    set $mods(ShowField-Y)-faces-usedefcolor 1
	    set $mods(ShowField-Z)-faces-usedefcolor 1
	    disable_planes_colormaps
        }

	configure_anisotropy_modules

	$planes_tab1.color.childsite.select.color select $which
	$planes_tab2.color.childsite.select.color select $which
	
        # execute 
	global show_planes
	
	if {$vis_activated && $show_planes == 1 && !$loading} {
	    $mods(ChooseField-ColorPlanes)-c needexecute
	    $mods(ShowField-X)-c rerender_faces
	    $mods(ShowField-Y)-c rerender_faces
	    $mods(ShowField-Z)-c rerender_faces
	}  else {
	    global exec_planes
	    set exec_planes(ChooseField-ColorPlanes) 1
	    set exec_planes(ShowField-X) 1
	    set exec_planes(ShowField-Y) 1
	    set exec_planes(ShowField-Z) 1
	}
    }

    method disable_planes_colormaps {} {
	foreach w [winfo children $planes_tab1.color.childsite.maps] {
	    grey_widget $w
	}

	foreach w [winfo children $planes_tab2.color.childsite.maps] {
	    grey_widget $w
	}
    }

    method enable_planes_colormaps {} {
	global show_planes
	
	if {$vis_activated && $show_planes} {
	    foreach w [winfo children $planes_tab1.color.childsite.maps] {
		enable_widget $w
	    }
	    
	    foreach w [winfo children $planes_tab2.color.childsite.maps] {
		enable_widget $w
	    }
	}
    }
  
    method initialize_clip_info {} {
        global mods
        global $mods(Viewer)-ViewWindow_0-global-clip
        set $mods(Viewer)-ViewWindow_0-global-clip 0

        global $mods(Viewer)-ViewWindow_0-clip
        set clip $mods(Viewer)-ViewWindow_0-clip

	global $clip-num
        set $clip-num 6

	global $clip-normal-x
	global $clip-normal-y
	global $clip-normal-z
	global $clip-normal-d
	global $clip-visible
	set $clip-visible 0
	set $clip-normal-d 0.0
	set $clip-normal-x 0.0
	set $clip-normal-y 0.0
	set $clip-normal-z 0.0
          
        # initialize to 0
	for {set i 1} {$i <= [set $clip-num]} {incr i 1} {
	    set mod $i

	    global $clip-normal-x-$mod
	    global $clip-normal-y-$mod
	    global $clip-normal-z-$mod
	    global $clip-normal-d-$mod
	    global $clip-visible-$mod

	    set $clip-visible-$mod 0
	    set $clip-normal-d-$mod 0.0
	    set $clip-normal-x-$mod 0.0
	    set $clip-normal-y-$mod 0.0
	    set $clip-normal-z-$mod 0.0
        }

        global plane_x plane_y plane_z

	global plane
	set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	set span_z [expr [expr $plane_z*$spacing_z]+$min_z]

        # 1
        set plane(-X) "on"
        global $clip-normal-x-1
        set $clip-normal-x-1 "-1.0"
        global $clip-normal-d-1 
        set $clip-normal-d-1 [expr -$span_x + $plane_inc]
        global $clip-visible-1
        set $clip-visible-1 1

        # 2
        set plane(+X) "off"
        global $clip-normal-x-2
        set $clip-normal-x-2 1.0
        global $clip-normal-d-2 
        set $clip-normal-d-2 [expr $span_x + $plane_inc]

        # 3
        set plane(-Y) "on"
        global $clip-normal-y-3
        set $clip-normal-y-3 "-1.0"
        global $clip-normal-d-3 
        set $clip-normal-d-3 [expr -$span_y + $plane_inc]
        global $clip-visible-3
        set $clip-visible-3 1

        # 4
        set plane(+Y) "off"
        global $clip-normal-y-4
        set $clip-normal-y-4 1.0
        global $clip-normal-d-4 
        set $clip-normal-d-4 [expr $span_y + $plane_inc]

        # 5
        set plane(-Z) "on"
        global $clip-normal-z-5
        set $clip-normal-z-5 "-1.0"
        global $clip-normal-d-5 
        set $clip-normal-d-5 [expr -$span_z + $plane_inc]
        global $clip-visible-5
        set $clip-visible-5 1

        # 6
        set plane(+Z) "off"
        global $clip-normal-z-6
        set $clip-normal-z-6 1.0
        global $clip-normal-d-6 
        set $clip-normal-d-6 [expr $span_z + $plane_inc]

        $mods(Viewer)-ViewWindow_0-c redraw
    }

    method toggle_clip_by_planes { w } {
	global mods
        global clip_by_planes
        global $mods(Viewer)-ViewWindow_0-global-clip

	if {$vis_activated} {
	    if {$clip_by_planes == 0} {
		set $mods(Viewer)-ViewWindow_0-global-clip 0
		$isosurface_tab1.clip.flipx configure -state disabled -foreground grey64
		$isosurface_tab2.clip.flipx configure -state disabled -foreground grey64
		
		$isosurface_tab1.clip.flipy configure -state disabled -foreground grey64
		$isosurface_tab2.clip.flipy configure -state disabled -foreground grey64
		
		$isosurface_tab1.clip.flipz configure -state disabled -foreground grey64
		$isosurface_tab2.clip.flipz configure -state disabled -foreground grey64
	    } else {
		set $mods(Viewer)-ViewWindow_0-global-clip 1
		
		$isosurface_tab1.clip.flipx configure -state normal -foreground black
		$isosurface_tab2.clip.flipx configure -state normal -foreground black
		
		$isosurface_tab1.clip.flipy configure -state normal -foreground black
		$isosurface_tab2.clip.flipy configure -state normal -foreground black
		
		$isosurface_tab1.clip.flipz configure -state normal -foreground black
		$isosurface_tab2.clip.flipz configure -state normal -foreground black
	    }
	    
	    $mods(Viewer)-ViewWindow_0-c redraw
	} else {
	    # need to set the global clipping planes when vis is activated
	    global exec_iso
	    if {$clip_by_planes == 0} {
		set exec_iso(global-clip) "off"
	    } else {
		set exec_iso(global-clip) "on"
	    }
	}
    }

    method flip_x_clipping_plane {} {
        global mods
        global show_plane_x
	global plane
        global $mods(Viewer)-ViewWindow_0-clip
        set clip $mods(Viewer)-ViewWindow_0-clip

        if {$show_plane_x == 1} {
           if {$plane(-X) == "on"} {
              global $clip-visible-1
              set $clip-visible-1 0
              set plane(-X) "off"

              global $clip-visible-2
              set $clip-visible-2 1
              set plane(+X) "on"

              set last_x 2
           } else {
              global $clip-visible-1
              set $clip-visible-1 1
              set plane(-X) "on"

              global $clip-visible-2
              set $clip-visible-2 0
              set plane(+X) "off"

              set last_x 1
           }
          
           global plane_x plane_y plane_z
           if {$clip_x == "<"} {
             set clip_x ">"
           } else {
             set clip_x "<"
           }

	    configure_ClipByFunction
	    
	    $mods(Viewer)-ViewWindow_0-c redraw
        }
    }

    method flip_y_clipping_plane {} {
        global mods
        global show_plane_y
	global plane
        global $mods(Viewer)-ViewWindow_0-clip
        set clip $mods(Viewer)-ViewWindow_0-clip

        if {$show_plane_y == 1} {
           if {$plane(-Y) == "on"} {
              global $clip-visible-3
              set $clip-visible-3 0
              set plane(-Y) "off"

              global $clip-visible-4
              set $clip-visible-4 1
              set plane(+Y) "on"

              set last_y 4
           } else {
              global $clip-visible-3
              set $clip-visible-3 1
              set plane(-Y) "on"

              global $clip-visible-4
              set $clip-visible-4 0
              set plane(+Y) "off"

              set last_y 3
           }

           global plane_x plane_y plane_z
           if {$clip_y == "<"} {
             set clip_y ">"
           } else {
             set clip_y "<"
           }
	    
	    configure_ClipByFunction

	    $mods(Viewer)-ViewWindow_0-c redraw
        }
    }

    method flip_z_clipping_plane {} {
        global mods
        global show_plane_z
	global plane
        global $mods(Viewer)-ViewWindow_0-clip
        set clip $mods(Viewer)-ViewWindow_0-clip

        if {$show_plane_z == 1} {
           if {$plane(-Z) == "on"} {
              global $clip-visible-5
              set $clip-visible-5 0
              set plane(-Z) "off"

              global $clip-visible-6
              set $clip-visible-6 1
              set plane(+Z) "on"

              set last_z 6
           } else {
              global $clip-visible-5
              set $clip-visible-5 1
              set plane(-Z) "on"

              global $clip-visible-6
              set $clip-visible-6 0
              set plane(+Z) "off"

              set last_z 5
           }

           global plane_x plane_y plane_z
           if {$clip_z == "<"} {
             set clip_z ">"
           } else {
             set clip_z "<"
           }

	    configure_ClipByFunction

	    $mods(Viewer)-ViewWindow_0-c redraw
        }
    }

    method toggle_clip_to_isosurface {} {
	global mods
	global clip_to_isosurface
	global $mods(ChooseField-X)-port-selected-index
	global $mods(ChooseField-Y)-port-selected-index
	global $mods(ChooseField-Z)-port-selected-index
	
	if {$clip_to_isosurface == 1} {
	    # change ChooseField port to 1
	    
	    set $mods(ChooseField-X)-port-selected-index 1
	    set $mods(ChooseField-Y)-port-selected-index 1
	    set $mods(ChooseField-Z)-port-selected-index 1
	} else {
	    # change ChooseField port to 0
	    
	    set $mods(ChooseField-X)-port-selected-index 0
	    set $mods(ChooseField-Y)-port-selected-index 0
	    set $mods(ChooseField-Z)-port-selected-index 0
	}
	
	# re-execute
	if {$vis_activated} {
	    $mods(ChooseField-ColorPlanes)-c needexecute
	}
    }
    
    method update_plane_x { } {
	global mods plane_x plane_y plane_z
	global $mods(SamplePlane-X)-pos
	
	if {$vis_activated} {
	    # set the sample plane position to be the normalized value
	    set result [expr [expr $plane_x / [expr $size_x / 2.0] ] - 1.0]
	    set $mods(SamplePlane-X)-pos $result
	    
	    # set the glabal clipping planes values
	    set clip $mods(Viewer)-ViewWindow_0-clip
	    global $clip-normal-d-1
	    global $clip-normal-d-2
	    set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	    set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	    set span_z [expr [expr $plane_z*$spacing_z]+$min_z]
	    
	    set $clip-normal-d-1 [expr -$span_x  + $plane_inc]
	    set $clip-normal-d-2 [expr $span_x  + $plane_inc]
	    
	    # configure ClipByFunction
	    configure_ClipByFunction

	    
            # only update if fibers or glyphs are on?
	    # and if they are seeding on grid
            global $mods(ShowField-Glyphs)-tensors-on
	    global $mods(ChooseField-GlyphSeeds)-port-selected-index
	    global $mods(ShowField-Fibers)-edges-on
	    global $mods(ChooseField-FiberSeeds)-port-selected-index
	    # if {([set $mods(ShowField-Glyphs)-tensors-on] && [set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 3) || ([set $mods(ChooseField-FiberSeeds)-port-selected-index] == 3 && [set $mods(ShowField-Fibers)-edges-on])} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    # }	    	   
	    
	    $mods(SamplePlane-X)-c needexecute
	    $mods(Viewer)-ViewWindow_0-c redraw
	} else {
	    global exec_planes
	    set exec_planes(update-X) 1
	}
    }
    
    method update_plane_y {} {
	global mods plane_x plane_y plane_z
	global $mods(SamplePlane-Y)-pos
	
	if {$vis_activated} {
	    # set the sample plane position to be the normalized value
	    set result [expr [expr $plane_y / [expr $size_y / 2.0] ] - 1.0]
	    set $mods(SamplePlane-Y)-pos $result
	    
	    # set the glabal clipping planes values
	    set clip $mods(Viewer)-ViewWindow_0-clip
	    global $clip-normal-d-3
	    global $clip-normal-d-4
	    set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	    set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	    set span_z [expr [expr $plane_z*$spacing_z]+$min_z]
	    
	    set $clip-normal-d-3 [expr -$span_y  + $plane_inc]
	    set $clip-normal-d-4 [expr $span_y  + $plane_inc]
	    
	    # configure ClipByFunction
	    configure_ClipByFunction

            # only update if fibers or glyphs are on?
	    # and if they are seeding on grid
            global $mods(ShowField-Glyphs)-tensors-on
	    global $mods(ChooseField-GlyphSeeds)-port-selected-index
	    global $mods(ShowField-Fibers)-edges-on
	    global $mods(ChooseField-FiberSeeds)-port-selected-index
	    # if {([set $mods(ShowField-Glyphs)-tensors-on] && [set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 3) || ([set $mods(ChooseField-FiberSeeds)-port-selected-index] == 3 && [set $mods(ShowField-Fibers)-edges-on])} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    # }	    
	    
	    $mods(SamplePlane-Y)-c needexecute
	    $mods(Viewer)-ViewWindow_0-c redraw
	} else {
	    global exec_planes
	    set exec_planes(update-Y) 1
	}
    }
    
    method update_plane_z {} {
	global mods plane_x plane_y plane_z
	global $mods(SamplePlane-Z)-pos
	
	if {$vis_activated} {
	    # set the sample plane position to be the normalized value
	    set result [expr [expr $plane_z / [expr $size_z / 2.0] ] - 1.0]
	    set $mods(SamplePlane-Z)-pos $result
	    
	    # set the glabal clipping planes values
	    set clip $mods(Viewer)-ViewWindow_0-clip
	    global $clip-normal-d-5
	    global $clip-normal-d-6
	    set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	    set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	    set span_z [expr [expr $plane_z*$spacing_z]+$min_z]
	    
	    set $clip-normal-d-5 [expr -$span_z  + $plane_inc]
	    set $clip-normal-d-6 [expr $span_z  + $plane_inc]
	    
	    # configure ClipByFunction
	    configure_ClipByFunction

            # only update if fibers or glyphs are on?
	    # and if they are seeding on grid
            global $mods(ShowField-Glyphs)-tensors-on
	    global $mods(ChooseField-GlyphSeeds)-port-selected-index
	    global $mods(ShowField-Fibers)-edges-on
	    global $mods(ChooseField-FiberSeeds)-port-selected-index
	    # if {([set $mods(ShowField-Glyphs)-tensors-on] && [set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 3) || ([set $mods(ChooseField-FiberSeeds)-port-selected-index] == 3 && [set $mods(ShowField-Fibers)-edges-on])} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    # }
	    $mods(SamplePlane-Z)-c needexecute
	    $mods(Viewer)-ViewWindow_0-c redraw
	} else {
	    global exec_planes
	    set exec_planes(update-Y) 1
	}
    }
    
    method toggle_plane { which } {
	global mods
	global show_plane_x show_plane_y show_plane_z
	global $mods(ShowField-X)-faces-on
	global $mods(ShowField-Y)-faces-on
	global $mods(ShowField-Z)-faces-on
        global $mods(ChooseField-GlyphSeeds)-port-selected-index
        global $mods(ChooseField-FiberSeeds)-port-selected-index
	global $mods(Viewer)-ViewWindow_0-clip
	set clip $mods(Viewer)-ViewWindow_0-clip
	
	
	# turn off showfields and configure global clipping planes
	
	if {$which == "X"} {
	    global $clip-visible-$last_x
	    if {$show_plane_x == 0} {
		# turn off plane face and global clipping plane
		set $mods(ShowField-X)-faces-on 0
		set $clip-visible-$last_x 0
		
		# disable connection
		block_connection $mods(ChooseField-X) 0 $mods(GatherPoints) 0
	    } else {
		set $mods(ShowField-X)-faces-on 1
		set $clip-visible-$last_x 1
		# enable connection
		unblock_connection $mods(ChooseField-X) 0 $mods(GatherPoints) 0
	    }  

	    configure_ClipByFunction

	    # only take the time to rexecute of glyphs or fibers are
	    # being seeded in the grid
	    if {[set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 3 || [set $mods(ChooseField-FiberSeeds)-port-selected-index] == 3} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    }

	    global show_planes
	    if {$vis_activated} {
		$mods(GatherPoints)-c needexecute
		
		if {$show_planes} {
		    $mods(ShowField-X)-c toggle_display_faces  
		    $mods(Viewer)-ViewWindow_0-c redraw
		}
	    } else {
		global exec_planes
		set exec_planes(GatherPoints) 1
		set exec_planes(ShowField-X) 1
	    }
	} elseif {$which == "Y"} {
	    global $clip-visible-$last_y
	    if {$show_plane_y == 0} {
		set $mods(ShowField-Y)-faces-on 0
		set $clip-visible-$last_y 0   
		block_connection $mods(ChooseField-Y) 0 $mods(GatherPoints) 1
	    } else {
		set $mods(ShowField-Y)-faces-on 1
		set $clip-visible-$last_y 1
		unblock_connection $mods(ChooseField-Y) 0 $mods(GatherPoints) 1
	    }   
	    configure_ClipByFunction

	    # only take the time to rexecute of glyphs or fibers are
	    # being seeded in the grid
	    if {[set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 3 || [set $mods(ChooseField-FiberSeeds)-port-selected-index] == 3} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    }

	    global show_planes
	    if {$vis_activated} {
		$mods(GatherPoints)-c needexecute

		if {$show_planes} {
		    $mods(ShowField-Y)-c toggle_display_faces
		    $mods(Viewer)-ViewWindow_0-c redraw
		}
	    } else {
		global exec_planes
		set exec_planes(GatherPoints) 1
		set exec_planes(ShowField-Y) 1
	    }
	} else {
	    # Z plane
	    global $clip-visible-$last_z
	    if {$show_plane_z == 0} {
		set $mods(ShowField-Z)-faces-on 0
		set $clip-visible-$last_z 0  
		block_connection $mods(ChooseField-Z) 0 $mods(GatherPoints) 2
	    } else {
		set $mods(ShowField-Z)-faces-on 1
		set $clip-visible-$last_z 1            
		unblock_connection $mods(ChooseField-Z) 0 $mods(GatherPoints) 2
	    }   
	    configure_ClipByFunction

	    # only take the time to rexecute of glyphs or fibers are
	    # being seeded in the grid
	    if {[set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 3 || [set $mods(ChooseField-FiberSeeds)-port-selected-index] == 3} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    }
	    
	    global show_planes
	    if {$vis_activated} {
		$mods(GatherPoints)-c needexecute

		if { $show_planes} {
		    $mods(ShowField-Z)-c toggle_display_faces
		    $mods(Viewer)-ViewWindow_0-c redraw
		}
	    } else {
		global exec_planes
		set exec_planes(GatherPoints) 1
		set exec_planes(ShowField-Z) 1
	    }
	}
    }
    
    method toggle_show_planes {} {
	global mods
	global show_planes
	
	global $mods(ShowField-X)-faces-on
	global $mods(ShowField-Y)-faces-on
	global $mods(ShowField-Z)-faces-on
	
	global $mods(Viewer)-ViewWindow_0-clip
	set clip $mods(Viewer)-ViewWindow_0-clip
	
	global $clip-visible-$last_x
	global $clip-visible-$last_y
	global $clip-visible-$last_z
	
	if {$show_planes == 0} {
	    # turn off global clipping planes
	    set $clip-visible-$last_x 0
	    set $clip-visible-$last_y 0
	    set $clip-visible-$last_z 0
	    
	    set $mods(ShowField-X)-faces-on 0
	    set $mods(ShowField-Y)-faces-on 0
	    set $mods(ShowField-Z)-faces-on 0
	    
	    if {$vis_activated} {
		$mods(ChooseField-ColorPlanes)-c needexecute
		set exec_planes(ChooseField-ColorPlanes) 0

		$mods(Viewer)-ViewWindow_0-c redraw
	    }
	} else {
	    global show_plane_x show_plane_y show_plane_z
	    
	    if {$show_plane_x} {
		set $mods(ShowField-X)-faces-on 1
		set $clip-visible-$last_x 1
	    }
	    if {$show_plane_y} {
		set $mods(ShowField-Y)-faces-on 1
		set $clip-visible-$last_y 1
	    }
	    if {$show_plane_z} {
		set $mods(ShowField-Z)-faces-on 1
		set $clip-visible-$last_z 1
	    }

	    if {$vis_activated} {
		# loop through array of planes modules and execute
		# ones that have had their gui modified but haven't
		# been executed
		global exec_planes
		if {$exec_planes(GatherPoints)} {
		    $mods(GatherPoints)-c needexecute
		    set exec_planes(GatherPoints) 0
		}
		if {$exec_planes(GenStandardColorMaps-ColorPlanes)} {
		    $mods(GenStandardColorMaps-ColorPlanes)-c needexecute
		    set exec_planes(GenStandardColorMaps-ColorPlanes) 0
		}
		if {$exec_planes(ShowField-X)} {
		    $mods(ShowField-X)-c toggle_display_faces
		    set exec_planes(ShowField-X) 0
		}
		if {$exec_planes(ShowField-Y)} {
		    $mods(ShowField-Y)-c toggle_display_faces
		    set exec_planes(ShowField-Y) 0
		}
		if {$exec_planes(ShowField-Z)} {
		    $mods(ShowField-Z)-c toggle_display_faces
		    set exec_planes(ShowField-Z) 0
		}
		
		$mods(ChooseField-ColorPlanes)-c needexecute
		set exec_planes(ChooseField-ColorPlanes) 0
		
		$mods(Viewer)-ViewWindow_0-c redraw
	    }
	}
    }
    


######## ISOSURFACE #########

    method build_isosurface_tab { f } {
	global tips
	global mods
	global $mods(ShowField-Isosurface)-faces-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Isosurface" \
		-variable $mods(ShowField-Isosurface)-faces-on \
		-command "$this toggle_show_isosurface" -foreground grey64
	    Tooltip $f.show $tips(IsoToggle)
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    # Isoval
	    frame $f.isoval
	    pack $f.isoval -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.isoval.l -text "Isoval:" -foreground grey64
	    scale $f.isoval.s -from 0.0 -to 1.0 \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution 0.0001 \
		-variable $mods(Isosurface)-isoval \
		-showvalue false \
		-foreground grey64\
		-orient horizontal \
		-command "$this update_isovals"
	    
	    bind $f.isoval.s <ButtonRelease> "$this execute_isoval_change"
	    
	    label $f.isoval.val -textvariable $mods(Isosurface)-isoval -foreground grey64
	    
	    pack $f.isoval.l $f.isoval.s $f.isoval.val -side left -anchor nw -padx 3      
	    
	    iwidgets::optionmenu $f.isovalcolor -labeltext "Isoval\nBased On:" \
		-labelpos w \
	        -width 150 \
		-foreground grey64 \
		-command "$this select_isoval_based_on $f"
	    pack $f.isovalcolor -side top -anchor nw -padx 1 -pady 3
	    
	    $f.isovalcolor insert end "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy"
	    
	    $f.isovalcolor select "Fractional Anisotropy"
	    
	    
	    global isosurface_color
	    iwidgets::labeledframe $f.isocolor \
		-labeltext "Color Isosurface Based On" \
		-labelpos nw -foreground grey64
	    pack $f.isocolor -side top -anchor nw -padx 3 -pady 5
	    
	    set isocolor [$f.isocolor childsite]
	    frame $isocolor.select
	    pack $isocolor.select -side top -anchor nw -padx 3 -pady 3
	    
	    iwidgets::optionmenu $isocolor.select.color -labeltext "" \
		-labelpos w \
	        -width 150 \
		-foreground grey64 \
		-command "$this select_isosurface_color $isocolor.select"
	    pack $isocolor.select.color -side left -anchor n -padx 1 -pady 3
	    
	    
	    addColorSelection $isocolor.select "Color" isosurface_color "clip_color_change"
	    
	    $isocolor.select.color insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant"
	    
	    $isocolor.select.color select "Principle Eigenvector"
	    
	    
	    iwidgets::labeledframe $isocolor.maps \
		-labeltext "Color Maps" \
		-labelpos nw -foreground grey64
	    pack $isocolor.maps -side top -anchor n -padx 3 -pady 0 -fill x
	    
	    set maps [$isocolor.maps childsite]
	    global $mods(GenStandardColorMaps-Isosurface)-mapType
	    
	    # Gray
	    frame $maps.gray
	    pack $maps.gray -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.gray.b -text "Gray" \
		-variable $mods(GenStandardColorMaps-Isosurface)-mapType \
		-value 0 \
		-foreground grey64 \
		-command "$this update_isoval_color_map"
	    Tooltip $maps.gray.b $tips(IsoColorMap)
	    pack $maps.gray.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.gray.f -relief sunken -borderwidth 2
	    pack $maps.gray.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.gray.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.gray.f.canvas -anchor e \
		-fill both -expand 1
	    
	    draw_colormap Gray $maps.gray.f.canvas
	    
	    # Rainbow
	    frame $maps.rainbow
	    pack $maps.rainbow -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.rainbow.b -text "Rainbow" \
		-variable $mods(GenStandardColorMaps-Isosurface)-mapType \
		-value 2 \
		-foreground grey64 \
		-command "$this update_isoval_color_map"
	    Tooltip $maps.rainbow.b $tips(IsoColorMap)
	    pack $maps.rainbow.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.rainbow.f -relief sunken -borderwidth 2
	    pack $maps.rainbow.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.rainbow.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.rainbow.f.canvas -anchor e
	    
	    draw_colormap Rainbow $maps.rainbow.f.canvas
	    
	    # Darkhue
	    frame $maps.darkhue
	    pack $maps.darkhue -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.darkhue.b -text "Darkhue" \
		-variable $mods(GenStandardColorMaps-Isosurface)-mapType \
		-value 5 \
		-foreground grey64 \
		-command "$this update_isoval_color_map"
	    Tooltip $maps.darkhue.b $tips(IsoColorMap)
	    pack $maps.darkhue.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.darkhue.f -relief sunken -borderwidth 2
	    pack $maps.darkhue.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.darkhue.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.darkhue.f.canvas -anchor e
	    
	    draw_colormap Darkhue $maps.darkhue.f.canvas
	    
	    
	    # Blackbody
	    frame $maps.blackbody
	    pack $maps.blackbody -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.blackbody.b -text "Blackbody" \
		-variable $mods(GenStandardColorMaps-Isosurface)-mapType \
		-value 7 \
		-foreground grey64 \
		-command "$this update_isoval_color_map"
	    Tooltip $maps.blackbody.b $tips(IsoColorMap)
	    pack $maps.blackbody.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.blackbody.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.blackbody.f.canvas -anchor e
	    
	    draw_colormap Blackbody $maps.blackbody.f.canvas
	    
	    # Blue-to-Red
	    frame $maps.bpseismic
	    pack $maps.bpseismic -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.bpseismic.b -text "Blue-to-Red" \
		-variable $mods(GenStandardColorMaps-Isosurface)-mapType \
		-value 17 \
		-foreground grey64 \
		-command "$this update_isoval_color_map"
	    Tooltip $maps.bpseismic.b $tips(IsoColorMap)
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "Blue-to-Red" $maps.bpseismic.f.canvas
	    
	    
	    global clip_by_planes
	    frame $f.clip
	    pack $f.clip -side top -anchor nw -padx 3 -pady 5
	    
	    checkbutton $f.clip.check -text "Clip by Planes" \
		-variable clip_by_planes -foreground grey64 \
		-command "$this toggle_clip_by_planes $f.clip"
	    Tooltip $f.clip.check $tips(ToggleClipPlanes)
	    
	    button $f.clip.flipx -text "Flip X" \
		-command "$this flip_x_clipping_plane" \
		-state disabled \
		-foreground grey64
	    Tooltip $f.clip.flipx $tips(FlipX)
	    button $f.clip.flipy -text "Flip Y" \
		-command "$this flip_y_clipping_plane" \
		-state disabled \
		-foreground grey64
	    Tooltip $f.clip.flipy $tips(FlipY)
	    button $f.clip.flipz -text "Flip Z" \
		-command "$this flip_z_clipping_plane" \
		-state disabled \
		-foreground grey64
	    Tooltip $f.clip.flipz $tips(FlipZ)
	    
	    pack $f.clip.check $f.clip.flipx $f.clip.flipy $f.clip.flipz \
		-side left -anchor nw -padx 3 -pady 3 -ipadx 2 
	} 
    }

    method update_isoval_color_map {} {
	global mods
	global $mods(ShowField-Isosurface)-faces-on

	if {$vis_activated && [set $mods(ShowField-Isosurface)-faces-on]==1} {
	    $mods(GenStandardColorMaps-Isosurface)-c needexecute
	} else {
	    global exec_iso
	    set exec_iso(GenStandardColorMaps-Isosurface) 1
	}
    }
    

    method sync_isosurface_tabs {} {
	global mods
	global $mods(ChooseField-Isoval)-port-selected-index
	set port [set $mods(ChooseField-Isoval)-port-selected-index]

	if {$port == 0} {
	    #FA
	    $isosurface_tab1.isovalcolor select "Fractional Anisotropy"
	    $isosurface_tab2.isovalcolor select "Fractional Anisotropy"
	} elseif {$port == 1} {
	    #LA
	    $isosurface_tab1.isovalcolor select "Linear Anisotropy"
	    $isosurface_tab2.isovalcolor select "Linear Anisotropy"
	} elseif {$port == 2} {
	    #PA
	    $isosurface_tab1.isovalcolor select "Planar Anisotropy"
	    $isosurface_tab2.isovalcolor select "Planar Anisotropy"
	} 

	global $mods(ChooseField-Isosurface)-port-selected-index
	set port [set $mods(ChooseField-Isosurface)-port-selected-index]
	global $mods(ChooseColorMap-Isosurface)-port-selected-index
	set color_port [set $mods(ChooseColorMap-Isosurface)-port-selected-index]


	if {$color_port == 1 && $port != 3} {
	    $isosurface_tab1.isocolor.childsite.select.color select "Constant"
	    $isosurface_tab2.isocolor.childsite.select.color select "Constant"
	} elseif {$port == 0} {
	    #FA
	    $isosurface_tab1.isocolor.childsite.select.color select "Fractional Anisotropy"
	    $isosurface_tab2.isocolor.childsite.select.color select "Fractional Anisotropy"
	} elseif {$port == 1} {
	    #LA
	    $isosurface_tab1.isocolor.childsite.select.color select "Linear Anisotropy"
	    $isosurface_tab2.isocolor.childsite.select.color select "Linear Anisotropy"
	} elseif {$port == 2} {
	    #PA
	    $isosurface_tab1.isocolor.childsite.select.color select "Planar Anisotropy"
	    $isosurface_tab2.isocolor.childsite.select.color select "Planar Anisotropy"
	} elseif {$port == 3} {
	    $isosurface_tab1.isocolor.childsite.select.color select "Principle Eigenvector"
	    $isosurface_tab2.isocolor.childsite.select.color select "Principle Eigenvector"
	} 
    }


    method configure_isosurface_tabs {} {
	global mods
	global $mods(ShowField-Isosurface)-faces-on

	if {$initialized != 0} {
	    if {$vis_activated} {
		foreach w [winfo children $isosurface_tab1] {
		    enable_widget $w
		}
		foreach w [winfo children $isosurface_tab2] {
		    enable_widget $w
		}
	    }
	    
	    # configure color button
	    if {$iso_type == "Constant"} {
		$isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state normal
		$isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state normal
		disable_isosurface_colormaps
	    } elseif {$iso_type == "Principle Eigenvector"} {
		$isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
		$isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled
		disable_isosurface_colormaps
		
	    } else {
		$isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
		$isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled
		enable_isosurface_colormaps
	    }
      

	    if {[set $mods(ShowField-Isosurface)-faces-on] == 0} {		
		foreach w [winfo children $isosurface_tab1] {
		    grey_widget $w
		}
		foreach w [winfo children $isosurface_tab2] {
		    grey_widget $w
		}
	    }

	    if {$vis_activated} {
		$isosurface_tab1.show configure -state normal -foreground black
		$isosurface_tab2.show configure -state normal -foreground black
		
		$isosurface_tab1.clip.check configure -state normal -foreground black
		$isosurface_tab2.clip.check configure -state normal -foreground black
		
		# configure flip buttons
		global clip_by_planes
		if {$clip_by_planes == 1} {
		    $isosurface_tab1.clip.flipx configure -state normal -foreground black
		    $isosurface_tab2.clip.flipx configure -state normal -foreground black
		    
		    $isosurface_tab1.clip.flipy configure -state normal -foreground black
		    $isosurface_tab2.clip.flipy configure -state normal -foreground black
		    
		    $isosurface_tab1.clip.flipz configure -state normal -foreground black
		    $isosurface_tab2.clip.flipz configure -state normal -foreground black
		} else {
		    $isosurface_tab1.clip.flipx configure -state disabled -foreground grey64
		    $isosurface_tab2.clip.flipx configure -state disabled -foreground grey64
		    
		    $isosurface_tab1.clip.flipy configure -state disabled -foreground grey64
		    $isosurface_tab2.clip.flipy configure -state disabled -foreground grey64
		    
		    $isosurface_tab1.clip.flipz configure -state disabled -foreground grey64
		    $isosurface_tab2.clip.flipz configure -state disabled -foreground grey64
		}
	    } 
	}
    }

    method toggle_show_isosurface {} {
	global mods
	global $mods(ShowField-Isosurface)-faces-on
	
	configure_isosurface_tabs

	if {$vis_activated} {
	    # loop through array of iso modules and execute
	    # ones that have had their gui modified but haven't
	    # been executed
	    global exec_iso
	    if {$exec_iso(ChooseField-Isoval)} {
		$mods(ChooseField-Isosurface)-c needexecute
		set exec_iso(ChooseField-Isosurface) 0
	    }
	    if {$exec_iso(Isosurface)} {
		$mods(Isosurface)-c needexecute
		set exec_iso(Isosurface) 0
	    }
	    if {$exec_iso(IsoClip-X)} {
		$mods(IsoClip-X)-c needexecute
		set exec_iso(IsoClip-X) 0
	    }
	    if {$exec_iso(IsoClip-Y)} {
		$mods(IsoClip-Y)-c needexecute
		set exec_iso(IsoClip-Y) 0
	    }
	    if {$exec_iso(IsoClip-Z)} {
		$mods(IsoClip-Z)-c needexecute
		set exec_iso(IsoClip-Z) 0
	    }
	    if {$exec_iso(GenStandardColorMaps-Isosurface)} {
		$mods(GenStandardColorMaps-Isosurface)-c needexecute
		set exec_iso(GenStandardColorMaps-Isosurface) 0
	    }
	    if {$exec_iso(ShowField-Isosurface)} {
		$mods(ShowField-Isosurface)-c rerender_faces
		set exec_iso(ShowField-Isosurface) 0
	    }
	    
	    
	    $mods(ShowField-Isosurface)-c toggle_display_faces
	}
    }


    method update_isovals { val } {
	# update all of the IsoClip modules
        global mods
        global $mods(IsoClip-X)-isoval
        global $mods(IsoClip-Y)-isoval
        global $mods(IsoClip-Z)-isoval
	
        set $mods(IsoClip-X)-isoval $val
        set $mods(IsoClip-Y)-isoval $val
        set $mods(IsoClip-Z)-isoval $val
    }
    

    method select_isosurface_color { w } {
	global mods
       	global $mods(ChooseField-Isosurface)-port-selected-index
	global $mods(ChooseColorMap-Isosurface)-port-selected-index
	
	set which [$w.color get]

	global $mods(ShowField-Isosurface)-faces-usedefcolor

	set $mods(ShowField-Isosurface)-faces-usedefcolor 0
	
        if {$which == "Principle Eigenvector"} {
	    set iso_type "Principle Eigenvector"
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(ChooseColorMap-Isosurface) 1
	    set $mods(ChooseColorMap-Isosurface)-port-selected-index 1
	    set $mods(ChooseField-Isosurface)-port-selected-index 3
	    disable_isosurface_colormaps
        } elseif {$which == "Fractional Anisotropy"} {
	    set iso_type "Fractional Anisotropy"
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	    
	    disableModule $mods(ChooseColorMap-Isosurface) 0
	    set $mods(ChooseColorMap-Isosurface)-port-selected-index 0
	    set $mods(ChooseField-Isosurface)-port-selected-index 0
	    enable_isosurface_colormaps
        } elseif {$which == "Linear Anisotropy"} {
	    set iso_type "Linear Anisotropy"
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	   
	    disableModule $mods(ChooseColorMap-Isosurface) 0
	    set $mods(ChooseColorMap-Isosurface)-port-selected-index 0
	    set $mods(ChooseField-Isosurface)-port-selected-index 1
	    enable_isosurface_colormaps
        } elseif {$which == "Planar Anisotropy"} {
	    set iso_type "Planar Anisotropy"
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	    
	    disableModule $mods(ChooseColorMap-Isosurface) 0
	    set $mods(ChooseColorMap-Isosurface)-port-selected-index 0
	    set $mods(ChooseField-Isosurface)-port-selected-index 2
	    enable_isosurface_colormaps
        } else {
	    set iso_type "Constant"
	    # constant color
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state normal
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state normal	   
	    disableModule $mods(ChooseColorMap-Isosurface) 1
	    set $mods(ChooseColorMap-Isosurface)-port-selected-index 1
	    set $mods(ChooseField-Isosurface)-port-selected-index 0
	    set $mods(ShowField-Isosurface)-faces-usedefcolor 1
	    disable_isosurface_colormaps
        }

	configure_anisotropy_modules

	$isosurface_tab1.isocolor.childsite.select.color select $which
	$isosurface_tab2.isocolor.childsite.select.color select $which
	
        # execute 
	global $mods(ShowField-Isosurface)-faces-on
	if {$vis_activated && [set $mods(ShowField-Isosurface)-faces-on]==1 \
		&& !$loading} {
	    $mods(ChooseField-Isosurface)-c needexecute
	    $mods(ShowField-Isosurface)-c rerender_faces
	} else {
	    global exec_iso
	    set exec_iso(ChooseField-Isosurface) 1
	    set exec_iso(ShowField-Isosurface) 1
	}
    }
    
    method disable_isosurface_colormaps {} {
	foreach w [winfo children $isosurface_tab1.isocolor.childsite.maps] {
	    grey_widget $w
	}

	foreach w [winfo children $isosurface_tab2.isocolor.childsite.maps] {
	    grey_widget $w
	}
    }

    method enable_isosurface_colormaps {} {
	global mods
	global $mods(ShowField-Isosurface)-faces-on
	if {$vis_activated && [set $mods(ShowField-Isosurface)-faces-on]==1} {	
	    foreach w [winfo children $isosurface_tab1.isocolor.childsite.maps] {
		enable_widget $w
	    }
	    
	    foreach w [winfo children $isosurface_tab2.isocolor.childsite.maps] {
		enable_widget $w
	    }
	}
    }
  

    method select_isoval_based_on { w } {
	global mods
       	global $mods(ChooseField-Isoval)-port-selected-index
	
	set which [$w.isovalcolor get]
	
        if {$which == "Fractional Anisotropy"} {
	    set $mods(ChooseField-Isoval)-port-selected-index 0
        } elseif {$which == "Linear Anisotropy"} {
	    set $mods(ChooseField-Isoval)-port-selected-index 1
        } else {
	    # Planar Anisotropy
	    set $mods(ChooseField-Isoval)-port-selected-index 2
        } 

	configure_anisotropy_modules

	$isosurface_tab1.isovalcolor select $which
	$isosurface_tab2.isovalcolor select $which
	
        # execute 
	global $mods(ShowField-Isosurface)-faces-on
	if {$vis_activated && [set $mods(ShowField-Isosurface)-faces-on]==1 \
		&& !$loading} {
	    $mods(ChooseField-Isoval)-c needexecute
	} else {
	    global exec_iso
	    set exec_iso(ChooseField-Isoval) 1
	}
    }
    

    method execute_isoval_change {} {
	global mods
	global $mods(ShowField-Isosurface)-faces-on

	if {$vis_activated && [set $mods(ShowField-Isosurface)-faces-on]==1} {
	    $mods(Isosurface)-c needexecute
	    $mods(IsoClip-X)-c needexecute
	    $mods(IsoClip-Y)-c needexecute
	    $mods(IsoClip-Z)-c needexecute
	} else {
	    global exec_iso
	    set exec_iso(Isosurface) 1
	    set exec_iso(IsoClip-X) 1
	    set exec_iso(IsoClip-Y) 1
	    set exec_iso(IsoClip-Z) 1

	}
    }




########### GLYPHS ############
    
    method build_glyphs_tab { f } {
	global tips
	global mods
        global $mods(ShowField-Glyphs)-tensors-on
	
	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Glyphs" \
		-variable $mods(ShowField-Glyphs)-tensors-on \
		-command "$this toggle_show_glyphs" -foreground grey64
	    Tooltip $f.show $tips(GlyphsToggle)
	    
	    pack $f.show -side top -anchor nw -padx 3 -pady 3	


	    global $mods(ShowField-Glyphs)-data-resolution
	    
	    frame $f.disc
	    pack $f.disc -side top -anchor nw -padx 8 -pady 0
	    
	    label $f.disc.la -text "Discretization: " -foreground grey64
	    
	    scale $f.disc.s -from 3 -to 20 \
                -resolution 1 \
  		-length 135  -width 15 \
		-sliderlength 15 \
                -orient horizontal \
		-showvalue false \
   	        -foreground grey64 \
	        -variable $mods(ShowField-Glyphs)-data-resolution
	    Tooltip $f.disc.s $tips(GlyphsRes)

	    label $f.disc.l -textvariable $mods(ShowField-Glyphs)-data-resolution -foreground grey64
	    bind $f.disc.s <ButtonRelease> {app change_glyph_disc}
	    
	    pack $f.disc.la $f.disc.s $f.disc.l -side left -anchor nw -padx 1 -pady 0


	    global scale_glyph
	    global $mods(TendNorm-Glyphs)-target
	    global glyph_scale_val
	    
	    frame $f.scale 
	    pack $f.scale -side top -anchor nw -padx 8 -pady 0
	    
	    checkbutton $f.scale.b -text "Normalize" \
		-variable scale_glyph \
		-foreground grey64 \
		-command "$this toggle_scale_glyph"
	    Tooltip $f.scale.b $tips(GlyphsNormalize)

	    label $f.scale.sc -text "   Scale:" -foreground grey64
	    
	    scale $f.scale.s -from 0.1 -to 5.0 \
                -resolution 0.01 \
  		-length 100  -width 15 \
		-sliderlength 15 \
                -orient horizontal \
		-showvalue false \
   	        -foreground grey64 \
	        -variable glyph_scale_val
	    Tooltip $f.scale.s $tips(GlyphsScale)
	    label $f.scale.l -textvariable glyph_scale_val -foreground grey64
	    bind $f.scale.s <ButtonRelease> {app change_glyph_scale}
	    
	    pack $f.scale.b $f.scale.sc $f.scale.s $f.scale.l -side left -anchor nw -padx 1 -pady 0

	    
	    global exag_glyph
	    global $mods(TendAnscale-Glyphs)-scale
	    
	    frame $f.exag 
	    pack $f.exag -side top -anchor nw -padx 8 -pady 0
	    
	    checkbutton $f.exag.b -text "Shape Exaggerate:" \
		-variable exag_glyph \
		-foreground grey64 \
		-command "$this toggle_exag_glyph"
	    Tooltip $f.exag.b $tips(GlyphsShape)
	    
	    scale $f.exag.s -from 0.2 -to 5.0 \
                -resolution 0.01 \
  		-length 100  -width 15 \
		-sliderlength 15 \
                -orient horizontal \
		-showvalue false \
   	        -foreground grey64 \
	        -variable $mods(TendAnscale-Glyphs)-scale
	    Tooltip $f.exag.s $tips(GlyphsShape)

	    label $f.exag.l -textvariable $mods(TendAnscale-Glyphs)-scale -foreground grey64
	    bind $f.exag.s <ButtonRelease> {app change_glyph_exag}
	    
	    pack $f.exag.b $f.exag.s $f.exag.l -side left -anchor nw -padx 1 -pady 0

	    
	    # Seed at
	    iwidgets::labeledframe $f.seed \
		-labeltext "Seed At" \
		-labelpos nw -foreground grey64
	    pack $f.seed -side top -anchor nw -padx 3 -pady 0 \
		-fill x
	    
	    set seed [$f.seed childsite]
	    
	    global $mods(ChooseField-GlyphSeeds)-port-selected-index
	    frame $seed.a
	    pack $seed.a -side left -anchor n -padx 3

	    frame $seed.a.pointf
	    pack $seed.a.pointf -side top\
		-anchor nw -padx 3 -pady 1
	    radiobutton $seed.a.pointf.point -text "Single Point" \
		-variable $mods(ChooseField-GlyphSeeds)-port-selected-index \
		-value 0 \
		-foreground grey64 \
		-command "$this update_glyph_seed_method"
	    Tooltip $seed.a.pointf.point $tips(GlyphsSeedPoint)

	    global glyph_point
	    checkbutton $seed.a.pointf.w -text "Widget" \
		-variable glyph_point \
		-foreground grey64 \
		-command "$this toggle_glyph_point"
	    Tooltip $seed.a.pointf.w $tips(GlyphsTogglePoint)

	    pack $seed.a.pointf.point $seed.a.pointf.w -side left -anchor nw -padx 0 -pady 0

	    frame $seed.a.rakef
	    pack $seed.a.rakef  -side top \
		-anchor nw -padx 3 -pady 1

	    radiobutton $seed.a.rakef.rake -text "Along Line  " \
		-variable $mods(ChooseField-GlyphSeeds)-port-selected-index \
		-value 1 \
		-foreground grey64 \
		-command "$this update_glyph_seed_method"
	    Tooltip $seed.a.rakef.rake $tips(GlyphsSeedLine)

	    global glyph_rake
	    checkbutton $seed.a.rakef.w -text "Widget" \
		-variable glyph_rake \
		-foreground grey64 \
		-command "$this toggle_glyph_rake"
	    Tooltip $seed.a.rakef.w $tips(GlyphsToggleRake)

	    pack $seed.a.rakef.rake $seed.a.rakef.w -side left -anchor nw -padx 0 -pady 0
	    
	    frame $seed.b
	    pack $seed.b -side right -anchor n -padx 3
	    radiobutton $seed.b.plane -text "On Planes" \
		-variable $mods(ChooseField-GlyphSeeds)-port-selected-index \
		-value 2 \
		-foreground grey64 \
		-command "$this update_glyph_seed_method"
	    Tooltip $seed.b.plane $tips(GlyphsSeedLine)
	    
	    radiobutton $seed.b.grid -text "On Grid" \
		-variable $mods(ChooseField-GlyphSeeds)-port-selected-index \
		-value 3 \
		-foreground grey64 \
		-command "$this update_glyph_seed_method"
	    Tooltip $seed.b.grid $tips(GlyphsSeedGrid)
	    
	    
	    pack $seed.b.plane $seed.b.grid -side top \
		-anchor nw -padx 3 -pady 1
	    
	    iwidgets::labeledframe $f.rep \
		-labeltext "Representation and Color" \
		-labelpos nw -foreground grey64
	    pack $f.rep -side top -anchor nw -padx 3 -pady 0 \
		-fill x
	    
	    set rep [$f.rep childsite]
	    
	    global glyph_display_type
	    frame $rep.f1 
	    pack $rep.f1 -side top -anchor nw -padx 3 -pady 1
	    
	    radiobutton $rep.f1.boxes -text "Boxes     " \
		-variable glyph_display_type \
		-value boxes \
		-foreground grey64 \
		-command "$this change_glyph_display_type radio $rep"
	    Tooltip $rep.f1.boxes $tips(GlyphsBoxes)
	    
	    iwidgets::optionmenu $rep.f1.type -labeltext "" \
		-width 150 -foreground grey64 \
		-command "$this change_glyph_display_type men $rep.f1"
	    pack $rep.f1.boxes $rep.f1.type -side left -anchor nw -padx 2 -pady 0
	    
	    $rep.f1.type insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant" "RGB"
	    $rep.f1.type select "Principle Eigenvector"
	    
	    frame $rep.f2
	    pack $rep.f2 -side top -anchor nw -padx 3 -pady 1
	    
	    radiobutton $rep.f2.ellips -text "Ellipsoids" \
		-variable glyph_display_type \
		-value ellipsoids \
		-foreground grey64 \
		-command "$this change_glyph_display_type radio $rep"
	    Tooltip $rep.f2.ellips $tips(GlyphsEllipsoids)
	    
	    iwidgets::optionmenu $rep.f2.type -labeltext "" \
		-width 150 \
		-command "$this change_glyph_display_type men $rep.f2" \
		-foreground grey64
	    pack $rep.f2.ellips $rep.f2.type -side left -anchor nw -padx 2 -pady 0
	    
	    $rep.f2.type insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant"
	    
	    $rep.f2.type select "Principle Eigenvector"

	    frame $rep.f3 
	    pack $rep.f3 -side top -anchor nw -padx 3 -pady 1
	    
	    radiobutton $rep.f3.quad -text "Super \nQuadrics " \
		-variable glyph_display_type \
		-value superquadrics \
		-foreground grey64 \
		-command "$this change_glyph_display_type radio $rep"
	    Tooltip $rep.f3.quad $tips(GlyphsSQ)
	    
	    iwidgets::optionmenu $rep.f3.type -labeltext "" \
		-width 150 -foreground grey64 \
		-command "$this change_glyph_display_type men $rep.f3"
	    pack $rep.f3.quad $rep.f3.type -side left -anchor nw -padx 2 -pady 0
	    
	    $rep.f3.type insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant"
	    $rep.f3.type select "Principle Eigenvector"
	    
	    global glyph_color
	    frame $rep.select
	    pack $rep.select -side top -anchor n -padx 3 -pady 3
	    addColorSelection $rep.select "Color" glyph_color \
		"default_color_change"
	    
	    iwidgets::labeledframe $rep.maps \
		-labeltext "Color Maps" \
		-labelpos nw -foreground grey64
	    pack $rep.maps -side top -anchor n -padx 3 -pady 3
	    
	    set maps [$rep.maps childsite]
	    global $mods(GenStandardColorMaps-Glyphs)-mapType
	    
	    # Gray
	    frame $maps.gray
	    pack $maps.gray -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.gray.b -text "Gray" \
		-variable $mods(GenStandardColorMaps-Glyphs)-mapType \
		-value 0 \
		-foreground grey64 \
		-command "$this update_glyphs_color_map"
	    Tooltip $maps.gray.b $tips(GlyphsColorMap)
	    pack $maps.gray.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.gray.f -relief sunken -borderwidth 2
	    pack $maps.gray.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.gray.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.gray.f.canvas -anchor e \
		-fill both -expand 1
	    
	    draw_colormap Gray $maps.gray.f.canvas
	    
	    # Rainbow
	    frame $maps.rainbow
	    pack $maps.rainbow -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.rainbow.b -text "Rainbow" \
		-variable $mods(GenStandardColorMaps-Glyphs)-mapType \
		-value 2 \
		-foreground grey64 \
		-command "$this update_glyphs_color_map"
	    Tooltip $maps.rainbow.b $tips(GlyphsColorMap)
	    pack $maps.rainbow.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.rainbow.f -relief sunken -borderwidth 2
	    pack $maps.rainbow.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.rainbow.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.rainbow.f.canvas -anchor e
	    
	    draw_colormap Rainbow $maps.rainbow.f.canvas
	    
	    # Darkhue
	    frame $maps.darkhue
	    pack $maps.darkhue -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.darkhue.b -text "Darkhue" \
		-variable $mods(GenStandardColorMaps-Glyphs)-mapType \
		-value 5 \
		-foreground grey64 \
		-command "$this update_glyphs_color_map"
	    Tooltip $maps.darkhue.b $tips(GlyphsColorMap)
	    pack $maps.darkhue.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.darkhue.f -relief sunken -borderwidth 2
	    pack $maps.darkhue.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.darkhue.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.darkhue.f.canvas -anchor e
	    
	    draw_colormap Darkhue $maps.darkhue.f.canvas
	    
	    
	    # Blackbody
	    frame $maps.blackbody
	    pack $maps.blackbody -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.blackbody.b -text "Blackbody" \
		-variable $mods(GenStandardColorMaps-Glyphs)-mapType \
		-value 7 \
		-foreground grey64 \
		-command "$this update_glyphs_color_map"
	    Tooltip $maps.blackbody.b $tips(GlyphsColorMap)
	    pack $maps.blackbody.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.blackbody.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.blackbody.f.canvas -anchor e
	    
	    draw_colormap Blackbody $maps.blackbody.f.canvas
	    
	    
	    # Blue-to-Red
	    frame $maps.bpseismic
	    pack $maps.bpseismic -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.bpseismic.b -text "Blue-to-Red" \
		-variable $mods(GenStandardColorMaps-Glyphs)-mapType \
		-value 17 \
		-foreground grey64 \
		-command "$this update_glyphs_color_map"
	    Tooltip $maps.bpseismic.b $tips(GlyphsColorMap)
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side left -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "Blue-to-Red" $maps.bpseismic.f.canvas	         
	    
	} 
    }

    method update_glyphs_color_map {} {
	global mods
	global $mods(ShowField-Glyphs)-tensors-on

	if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on]==1} {
	    $mods(GenStandardColorMaps-Glyphs)-c needexecute
	} else {
	    global exec_glyphs
	    set exec_glyphs(GenStandardColorMaps-Glyphs) 1
	}
    }

    method sync_glyphs_tabs {} {
	global mods
	global glyph_display_type
	global $mods(ChooseField-Glyphs)-port-selected-index
	global $mods(ShowField-Glyphs)-data_display_type
	global $mods(ChooseField-Glyphs)-port-selected-index

	set port [set $mods(ChooseField-Glyphs)-port-selected-index]
	set color_port [set $mods(ChooseField-Glyphs)-port-selected-index]

	set type ""
	if {$glyph_display_type == "boxes" && [set $mods(ShowField-Glyphs)-data_display_type] == "Colored Boxes"} {
	    # select boxes RGB and have ellipsoids default to Principle Eigenvector
	    $glyphs_tab1.rep.childsite.f1.type select "RGB"
	    $glyphs_tab2.rep.childsite.f1.type select "RGB"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Principle Eigenvector"
	    $glyphs_tab2.rep.childsite.f2.type select "Principle Eigenvector"
	    return
	} 

	# if not colored boxes, configure both optionmenus to be the same
	if {$color_port == 1 && $port != 3} {
	    # set optionmenu Constant and enable color button
	    $glyphs_tab1.rep.childsite.f1.type select "Constant"
	    $glyphs_tab2.rep.childsite.f1.type select "Constant"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Constant"
	    $glyphs_tab2.rep.childsite.f2.type select "Constant"

	    $glyphs_tab1.rep.childsite.f3.type select "Constant"
	    $glyphs_tab2.rep.childsite.f3.type select "Constant"
	} elseif {$port == 0} {
	    #FA - set option menu to Fractional Anisotropy disable Color button
	    $glyphs_tab1.rep.childsite.f1.type select "Fractional Anisotropy"
	    $glyphs_tab2.rep.childsite.f1.type select "Fractional Anisotropy"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Fractional Anisotropy"
	    $glyphs_tab2.rep.childsite.f2.type select "Fractional Anisotropy"

	    $glyphs_tab1.rep.childsite.f3.type select "Fractional Anisotropy"
	    $glyphs_tab2.rep.childsite.f3.type select "Fractional Anisotropy"
	} elseif {$port == 1} {
	    #LA -set optionmenu to LA and disable Color button
	    $glyphs_tab1.rep.childsite.f1.type select "Linear Anisotropy"
	    $glyphs_tab2.rep.childsite.f1.type select "Linear Anisotropy"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Linear Anisotropy"
	    $glyphs_tab2.rep.childsite.f2.type select "Linear Anisotropy"

	    $glyphs_tab1.rep.childsite.f3.type select "Linear Anisotropy"
	    $glyphs_tab2.rep.childsite.f3.type select "Linear Anisotropy"
	} elseif {$port == 2} {
	    #PA - set option menu to pa and disable color button
	    $glyphs_tab1.rep.childsite.f1.type select "Planar Anisotropy"
	    $glyphs_tab2.rep.childsite.f1.type select "Planar Anisotropy"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Planar Anisotropy"
	    $glyphs_tab2.rep.childsite.f2.type select "Planar Anisotropy"

	    $glyphs_tab1.rep.childsite.f3.type select "Planar Anisotropy"
	    $glyphs_tab2.rep.childsite.f3.type select "Planar Anisotropy"
	} elseif {$port == 3} {
	    #e1 - set option menu to e1 and disable color button
	    $glyphs_tab1.rep.childsite.f1.type select "Principle Eigenvector"
	    $glyphs_tab2.rep.childsite.f1.type select "Principle Eigenvector"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Principle Eigenvector"
	    $glyphs_tab2.rep.childsite.f2.type select "Principle Eigenvector"

	    $glyphs_tab1.rep.childsite.f3.type select "Principle Eigenvector"
	    $glyphs_tab2.rep.childsite.f3.type select "Principle Eigenvector"
	} 
    }

    method configure_glyphs_tabs {} {
        global mods
	global $mods(ShowField-Glyphs)-tensors-on        
	global glyph_display_type
	global scale_glyph exag_glyph
	
	if {$vis_activated} {
	    foreach w [winfo children $glyphs_tab1] {
		enable_widget $w
	    }
	    foreach w [winfo children $glyphs_tab2] {
		enable_widget $w
	    }
	}

	# configure boxes/ellipsoids optionmenus
	if {$glyph_display_type == "boxes"} {
	    $glyphs_tab1.rep.childsite.f1.type configure -state normal
	    $glyphs_tab2.rep.childsite.f1.type configure -state normal
	    $glyphs_tab1.rep.childsite.f2.type configure -state disabled
	    $glyphs_tab2.rep.childsite.f2.type configure -state disabled
	    $glyphs_tab1.rep.childsite.f3.type configure -state disabled
	    $glyphs_tab2.rep.childsite.f3.type configure -state disabled
	} elseif {$glyph_display_type == "superquadrics"} {
	    $glyphs_tab1.rep.childsite.f1.type configure -state disabled
	    $glyphs_tab2.rep.childsite.f1.type configure -state disabled
	    $glyphs_tab1.rep.childsite.f2.type configure -state disabled
	    $glyphs_tab2.rep.childsite.f2.type configure -state disabled
	    $glyphs_tab1.rep.childsite.f3.type configure -state normal
	    $glyphs_tab2.rep.childsite.f3.type configure -state normal
	} else {
	    $glyphs_tab1.rep.childsite.f1.type configure -state disabled
	    $glyphs_tab2.rep.childsite.f1.type configure -state disabled
	    $glyphs_tab1.rep.childsite.f2.type configure -state normal
	    $glyphs_tab2.rep.childsite.f2.type configure -state normal
	    $glyphs_tab1.rep.childsite.f3.type configure -state disabled
	    $glyphs_tab2.rep.childsite.f3.type configure -state disabled
	}       
	
	if {$scale_glyph == 0} {
	    $glyphs_tab1.scale.s configure -state disabled -foreground grey64
	    $glyphs_tab2.scale.s configure -state disabled -foreground grey64
	} else {
	    $glyphs_tab1.scale.s configure -state normal -foreground black
	    $glyphs_tab2.scale.s configure -state normal -foreground black
	}
	
	if {$exag_glyph == 0} {
	    $glyphs_tab1.exag.s configure -state disabled -foreground grey64
	    $glyphs_tab2.exag.s configure -state disabled -foreground grey64
	} else {
	    $glyphs_tab1.exag.s configure -state normal -foreground black
	    $glyphs_tab2.exag.s configure -state normal -foreground black
	}	
	
	# configure color swatch
	if {$glyph_type == "Constant"} {
	    $glyphs_tab1.rep.childsite.select.colorFrame.set_color configure -state normal
	    $glyphs_tab2.rep.childsite.select.colorFrame.set_color configure -state normal
	    disable_glyphs_colormaps
	} elseif {$glyph_type == "Principle Eigenvector"} {
	    $glyphs_tab1.rep.childsite.select.colorFrame.set_color configure -state disabled
	    $glyphs_tab2.rep.childsite.select.colorFrame.set_color configure -state disabled
	    disable_glyphs_colormaps
	} else {
	    $glyphs_tab1.rep.childsite.select.colorFrame.set_color configure -state disabled
	    $glyphs_tab2.rep.childsite.select.colorFrame.set_color configure -state disabled
	}
	
	# configure glyph rake
	global $mods(ChooseField-GlyphSeeds)-port-selected-index
	if {[set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 1} {
	    $glyphs_tab1.seed.childsite.a.rakef.w configure -state normal
	    $glyphs_tab2.seed.childsite.a.rakef.w configure -state normal
	    $glyphs_tab1.seed.childsite.a.pointf.w configure -state disabled
	    $glyphs_tab2.seed.childsite.a.pointf.w configure -state disabled
	} elseif {[set $mods(ChooseField-GlyphSeeds)-port-selected-index]== 0} {
	    $glyphs_tab1.seed.childsite.a.pointf.w configure -state normal
	    $glyphs_tab2.seed.childsite.a.pointf.w configure -state normal
	    $glyphs_tab1.seed.childsite.a.rakef.w configure -state disabled
	    $glyphs_tab2.seed.childsite.a.rakef.w configure -state disabled
	} else {
	    $glyphs_tab1.seed.childsite.a.rakef.w configure -state disabled
	    $glyphs_tab2.seed.childsite.a.rakef.w configure -state disabled
	    $glyphs_tab1.seed.childsite.a.pointf.w configure -state disabled
	    $glyphs_tab2.seed.childsite.a.pointf.w configure -state disabled
	}
	
	if {[set $mods(ShowField-Glyphs)-tensors-on] == 0 || !$vis_activated} {
	    foreach w [winfo children $glyphs_tab1] {
		grey_widget $w
	    }
	    foreach w [winfo children $glyphs_tab2] {
		grey_widget $w
	    }
	}
	
	if {$vis_activated} {
	    # activate checkbox
	    $glyphs_tab1.show configure -state normal -foreground black
	    $glyphs_tab2.show configure -state normal -foreground black
	}   
    }

    method toggle_scale_glyph {} {
	global mods
        global $mods(ChooseNrrd-Norm)-port-selected-index
        global scale_glyph
        global $mods(ShowField-Glyphs)-tensors-on

	
	if {$scale_glyph == 0} {
	    # $glyphs_tab1.scale.s configure -state disabled -foreground grey64
	    # $glyphs_tab2.scale.s configure -state disabled -foreground grey64
	    
	    set $mods(ChooseNrrd-Norm)-port-selected-index 1
		
	    if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1} {
		$mods(ChooseNrrd-Norm)-c needexecute
	    } else {
		global exec_glyphs
		set exec_glyphs(ChooseNrrd-Norm) 1
	    }
	} else {
	    #$glyphs_tab1.scale.s configure -state normal -foreground black
	    #$glyphs_tab2.scale.s configure -state normal -foreground black
	    
	    set $mods(ChooseNrrd-Norm)-port-selected-index 0
		
	    if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1} {
		$mods(TendNorm-Glyphs)-c needexecute
	    } else {
		global exec_glyphs
		set exec_glyphs(TendNorm-Glyphs) 1
	    }
	}
	
    }
    
    method toggle_exag_glyph {} {
	global mods
        global $mods(ChooseNrrd-Exag)-port-selected-index
        global exag_glyph

	if {$exag_glyph == 0} {
	    $glyphs_tab1.exag.s configure -state disabled -foreground grey64
	    $glyphs_tab2.exag.s configure -state disabled -foreground grey64
	    
	    set $mods(ChooseNrrd-Exag)-port-selected-index 1
	    
	    global $mods(ShowField-Glyphs)-tensors-on
	    if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1} {
		$mods(ChooseNrrd-Exag)-c needexecute
	    } else {
		global exec_glyphs
		set exec_glyphs(ChooseNrrd-Exag) 1
	    }
        } else {
	    $glyphs_tab1.exag.s configure -state normal -foreground black
	    $glyphs_tab2.exag.s configure -state normal -foreground black
	    
	    set $mods(ChooseNrrd-Exag)-port-selected-index 0

	    
	    global $mods(ShowField-Glyphs)-tensors-on
	    if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1} {
		$mods(TendAnscale-Glyphs)-c needexecute
	    } else {
		global exec_glyphs
		set exec_glyphs(TendAnscale-Glyphs) 1
	    }
        }
    }


    method change_glyph_display_type { change w } {
	global glyph_display_type
        global mods
        global $mods(ShowField-Glyphs)-tensor_display_type
        global $mods(ChooseField-Glyphs)-port-selected-index
	global $mods(ChooseColorMap-Glyphs)-port-selected-index
	global $mods(ShowField-Glyphs)-tensor-usedefcolor
	
        set type ""
	
	set $mods(ShowField-Glyphs)-tensor-usedefcolor 0
        if {$change == "radio"} {
	    # radio button changed  
	    if {$glyph_display_type == "boxes"} {
		set type [$w.f1.type get]
	    } elseif {$glyph_display_type == "superquadrics"} {
		set type [$w.f3.type get]
	    } elseif {$glyph_display_type == "ellipsoids"} {
		set type [$w.f2.type get]
	    }       	
	} else {
	    # optionmenu changed
	    set type [$w.type get]
        }
	
	global $mods(ShowField-Glyphs)-tensors-usedefcolor

	set $mods(ShowField-Glyphs)-tensors-usedefcolor 0

        # configure display type
        if {$glyph_display_type == "ellipsoids"} {
	    set $mods(ShowField-Glyphs)-tensor_display_type Ellipsoids
        } elseif {$glyph_display_type == "superquadrics"} {
	    set $mods(ShowField-Glyphs)-tensor_display_type Superquadrics
	} else {
	    # determine if normal boxes or colored boxes
	    if {$type == "RGB"} {
                set glyph_type "RGB"
		set $mods(ShowField-Glyphs)-tensor_display_type "Colored Boxes"
		disable_glyphs_colormaps
	    } else {
		set $mods(ShowField-Glyphs)-tensor_display_type Boxes
	    }
	}

	# configure color
	if {$type == "Principle Eigenvector"} {
	    set glyph_type "Principle Eigenvector"
	    set $mods(ChooseField-Glyphs)-port-selected-index 3
	    disableModule $mods(ChooseColorMap-Glyphs) 1
	    set $mods(ChooseColorMap-Glyphs)-port-selected-index 1
	    disable_glyphs_colormaps
	} elseif {$type == "Fractional Anisotropy"} {
	    set glyph_type "Fractional Anisotropy"
	    set $mods(ChooseField-Glyphs)-port-selected-index 0
	    disableModule $mods(ChooseColorMap-Glyphs) 0
	    set $mods(ChooseColorMap-Glyphs)-port-selected-index 0
	    enable_glyphs_colormaps
	} elseif {$type == "Linear Anisotropy"} {
	    set glyph_type "Linear Anisotropy"
	    set $mods(ChooseField-Glyphs)-port-selected-index 1
	    disableModule $mods(ChooseColorMap-Glyphs) 0
	    set $mods(ChooseColorMap-Glyphs)-port-selected-index 0
	    enable_glyphs_colormaps
	} elseif {$type == "Planar Anisotropy"} {
	    set glyph_type "Planar Anisotropy"
	    set $mods(ChooseField-Glyphs)-port-selected-index 2
	    disableModule $mods(ChooseColorMap-Glyphs) 0
	    set $mods(ChooseColorMap-Glyphs)-port-selected-index 0
	    enable_glyphs_colormaps
	} elseif {$type == "Constant"} {
	    set glyph_type "Constant"
	    disableModule $mods(ChooseColorMap-Glyphs) 1
	    set $mods(ChooseColorMap-Glyphs)-port-selected-index 0
	    set $mods(ShowField-Glyphs)-tensors-usedefcolor 1
	    disable_glyphs_colormaps
	}
	
	configure_anisotropy_modules

	# sync attached/detached optionmenus
	configure_glyphs_tabs

	global $mods(ShowField-Glyphs)-tensors-on
	if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1 \
	    && !$loading} {
	    $mods(ShowField-Glyphs)-c data_display_type
	    $mods(ChooseField-Glyphs)-c needexecute
	} else {
	    global exec_glyphs
	    set exec_glyphs(ShowField-Glyphs) 1
	    set exec_glyphs(ChooseField-Glyphs) 1
	}
    }

    method disable_glyphs_colormaps {} {
	foreach w [winfo children $glyphs_tab1.rep.childsite.maps] {
	    grey_widget $w
	}

	foreach w [winfo children $glyphs_tab2.rep.childsite.maps] {
	    grey_widget $w
	}
    }

    method enable_glyphs_colormaps {} {
	foreach w [winfo children $glyphs_tab1.rep.childsite.maps] {
	    enable_widget $w
	}

	foreach w [winfo children $glyphs_tab2.rep.childsite.maps] {
	    enable_widget $w
	}
    }

    method update_glyph_seed_method {} {
        global mods
        global $mods(ChooseField-GlyphSeeds)-port-selected-index

	if {[set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 0} {
	    # Point
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 1
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}" 0
	    $glyphs_tab1.seed.childsite.a.pointf.w configure -state normal
	    $glyphs_tab2.seed.childsite.a.pointf.w configure -state normal
	    
	    $glyphs_tab1.seed.childsite.a.rakef.w configure -state disabled
	    $glyphs_tab2.seed.childsite.a.rakef.w configure -state disabled
	} elseif {[set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 1} {
	    # Rake
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}" 1
	    $glyphs_tab1.seed.childsite.a.pointf.w configure -state disabled
	    $glyphs_tab2.seed.childsite.a.pointf.w configure -state disabled
	    
	    $glyphs_tab1.seed.childsite.a.rakef.w configure -state normal
	    $glyphs_tab2.seed.childsite.a.rakef.w configure -state normal
	} else {
	    # Grid or Planes
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}" 0
	    $glyphs_tab1.seed.childsite.a.pointf.w configure -state disabled
	    $glyphs_tab2.seed.childsite.a.pointf.w configure -state disabled
	    
	    $glyphs_tab1.seed.childsite.a.rakef.w configure -state disabled
	    $glyphs_tab2.seed.childsite.a.rakef.w configure -state disabled
	    if {$vis_activated && [set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 3} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    }
	}
	
	global $mods(ShowField-Glyphs)-tensors-on
	if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1} {	    
	    $mods(ChooseField-GlyphSeeds)-c needexecute       
	} else {
	    global exec_glyphs
	    set exec_glyphs(ChooseField-GlyphSeeds) 1
	}  
	$mods(Viewer)-ViewWindow_0-c redraw
    }

    method toggle_glyph_point {} {
	global glyph_point
	global mods
	
	if {$glyph_point} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 1
	} else {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
	}

	$mods(Viewer)-ViewWindow_0-c redraw
    }
    

    method toggle_glyph_rake {} {
	global glyph_rake
	global mods
	
	if {$glyph_rake} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}" 1
	} else {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}" 0
	}

	$mods(Viewer)-ViewWindow_0-c redraw
    }
    
    method toggle_show_glyphs {} {
	global mods
        global $mods(ShowField-Glyphs)-tensors-on
	
        if {[set $mods(ShowField-Glyphs)-tensors-on] == 0} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}" 0
        } else {
            global $mods(ChooseField-GlyphSeeds)-port-selected-index
            if {[set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 0} {
		# enable Probe Widget
		uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 1
            } elseif {[set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 1} {
		# enable rake
		uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (7)\}" 1
            } elseif {$vis_activated && [set $mods(ChooseField-GlyphSeeds)-port-selected-index] == 3} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    }
        }

	configure_glyphs_tabs

	# loop through array of iso modules and execute
	# ones that have had their gui modified but haven't
	# been executed
	global exec_glyphs
	if {$exec_glyphs(ChooseField-Glyphs)} {
	    $mods(ChooseField-Glyphs)-c needexecute
	    set exec_glyphs(ChooseField-Glyphs) 0
	}
	if {$exec_glyphs(ChooseField-GlyphSeeds)} {
	    $mods(ChooseField-GlyphSeeds)-c needexecute
	    set exec_glyphs(ChooseField-GlyphSeeds) 0
	}
	if {$exec_glyphs(GenStandardColorMaps-Glyphs)} {
	    $mods(GenStandardColorMaps-Glyphs)-c needexecute
	    set exec_glyphs(GenStandardColorMaps-Glyphs) 0
	}
	if {$exec_glyphs(ShowField-Glyphs)} {
	    $mods(ShowField-Glyphs)-c data_resolution_scale
	    $mods(ShowField-Glyphs)-c data_scale
	    $mods(ShowField-Glyphs)-c data_display_type
	    $mods(ShowField-Glyphs)-c needexecute
	    set exec_glyphs(ShowField-Glyphs) 0
	}
	if {$exec_glyphs(ChooseNrrd-Norm)} {
	    $mods(ChooseNrrd-Norm)-c needexecute
	    set exec_glyphs(ChooseNrrd-Norm) 0
	}
	if {$exec_glyphs(TendNorm-Glyphs)} {
	    $mods(TendNorm-Glyphs)-c needexecute
	    set exec_glyphs(TendNorm-Glyphs) 0
	}
	if {$exec_glyphs(TendAnscale-Glyphs)} {
	    $mods(TendAnscale-Glyphs)-c needexecute
	    set exec_glyphs(TendAnscale-Glyphs) 0
	}

	if {$vis_activated} {
	    $mods(ShowField-Glyphs)-c toggle_display_tensors
	}
	$mods(Viewer)-ViewWindow_0-c redraw
    }

    method change_glyph_scale {} {
	global scale_glyph
	global mods
	global $mods(ShowField-Glyphs)-tensors-on
	global $mods(ShowField-Glyphs)-tensors_scale

	global glyph_scale_val
	
	set $mods(ShowField-Glyphs)-tensors_scale [expr $average_spacing * $glyph_scale_val]
	
	if {$vis_activated && $scale_glyph && [set $mods(ShowField-Glyphs)-tensors-on] == 1 \
	    && !$loading} {

	    $mods(ShowField-Glyphs)-c data_scale
	} else {
	    global exec_glyphs
	    set exec_glyphs(ShowField-Glyphs) 1
	}
    }


    method change_glyph_disc {} {
	global mods
	global $mods(ShowField-Glyphs)-tensors-on

	if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1} {
	    global mods
	    $mods(ShowField-Glyphs)-c data_resolution_scale
	    $mods(ShowField-Glyphs)-c needexecute
	} else {
	    global exec_glyphs
	    set exec_glyphs(ShowField-Glyphs) 1
	}
    }

    method change_glyph_exag {} {
	global exag_glyph
	global mods
	global $mods(ShowField-Glyphs)-tensors-on

	if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1 && $exag_glyph} {
	    global mods
	    $mods(TendAnscale-Glyphs)-c needexecute
	} else {
	    global exec_glyphs
	    set exec_glyphs(TendAnscale-Glyphs) 1
	}
    }
    


############# FIBERS #############
    
    method build_fibers_tab { f } {
	global tips
	global mods
        global $mods(ShowField-Fibers)-edges-on
	
	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Fibers" \
		-variable $mods(ShowField-Fibers)-edges-on \
		-command "$this toggle_show_fibers" -foreground grey64
	    Tooltip $f.show $tips(FibersToggle)
	    
	    pack $f.show -side top -anchor nw -padx 3 -pady 0
	    
	    # Fiber Algorigthm
	    iwidgets::labeledframe $f.algo \
		-labeltext "Fiber Algorithm" \
		-labelpos nw -foreground grey64
	    
	    pack $f.algo -side top -anchor nw -padx 3 -pady 0 -fill x
	    set algo [$f.algo childsite]
	    
	    global $mods(TendFiber)-fibertype
	    frame $algo.f 
	    pack $algo.f -side top -anchor nw -padx 3 -pady 1
	    radiobutton $algo.f.evec1 -text "Major Eigenvector" \
		-variable $mods(TendFiber)-fibertype \
		-value evec1 \
		-command "$this execute_TendFiber" \
		-foreground grey64
	    
	    radiobutton $algo.f.tl -text "Tensorlines (TL)" \
		-variable $mods(TendFiber)-fibertype \
		-value tensorline \
		-command "$this execute_TendFiber" \
		-foreground grey64
	    
	    pack $algo.f.evec1 $algo.f.tl -side left -anchor nw -padx 5 -pady 1
	    
	    global fibers_stepsize
	    frame $algo.stepsize
	    pack $algo.stepsize -side top -anchor nw -padx 3 -pady 1
	    
	    label $algo.stepsize.l -text "Step Size:" -foreground grey64
	    scale $algo.stepsize.step -label "" \
		-from 0.1 -to 10 \
		-resolution 0.1 \
		-length 100  -width 15 \
		-sliderlength 15 \
		-orient horizontal \
		-showvalue false \
		-variable fibers_stepsize \
	        -foreground grey64
	    label $algo.stepsize.val -textvariable fibers_stepsize -foreground grey64
	    pack $algo.stepsize.l $algo.stepsize.step $algo.stepsize.val -side left -anchor nw -padx 3 -pady 1
	    bind $algo.stepsize.step <ButtonRelease> {app configure_fibers_stepsize}
	    
	    frame $algo.method 
	    pack $algo.method -side top -anchor nw -padx 3 -pady 0
	    
	    global $mods(TendFiber)-integration
	    label $algo.method.l -text "Integration Method: " -foreground grey64
	    radiobutton $algo.method.e -text "Euler" \
		-variable $mods(TendFiber)-integration \
		-value Euler \
		-foreground grey64 \
		-command "$this execute_TendFiber"
	    radiobutton $algo.method.rk -text "RK4" \
		-variable $mods(TendFiber)-integration \
		-value RK4 \
		-foreground grey64 \
		-command "$this execute_TendFiber"
	    
	    pack $algo.method.l $algo.method.e $algo.method.rk -side left -anchor nw \
		-padx 3 -pady 0
	    
	    # Resampling Filter
	    iwidgets::labeledframe $f.rs \
		-labeltext "Sampling Kernel" \
		-labelpos nw -foreground grey64
	    
	    pack $f.rs -side top -anchor nw -padx 3 -pady 0 -fill x
	    set rs [$f.rs childsite]
	    global $mods(TendFiber)-kernel
	    
	    frame $rs.f
	    pack $rs.f -side top -anchor n
	    
	    radiobutton $rs.f.tent -text "Tent" \
		-variable $mods(TendFiber)-kernel \
		-value tent \
		-foreground grey64 \
		-command "$this execute_TendFiber"
	    
	    radiobutton $rs.f.cat -text "Catmull-Rom" \
		-variable $mods(TendFiber)-kernel \
		-value cubicCR \
		-foreground grey64 \
		-command "$this execute_TendFiber"
	    
	    radiobutton $rs.f.b -text "B-Spline" \
		-variable $mods(TendFiber)-kernel \
		-value cubicBS \
		-foreground grey64 \
		-command "$this execute_TendFiber"
	    
	    pack $rs.f.tent $rs.f.cat $rs.f.b -side left -anchor nw -padx 3 -pady 0
	    
	    iwidgets::labeledframe $f.stop \
		-labeltext "Stopping Criteria" \
		-labelpos nw -foreground grey64
	    pack $f.stop -side top -anchor nw -padx 3 -pady 0 -fill x
	    set stop [$f.stop childsite]
	    
	    # Max Fiber Length
	    global $mods(TendFiber)-use-length
	    global fibers_length
	    frame $stop.fiber
	    pack $stop.fiber -side top -anchor nw
	    
	    checkbutton $stop.fiber.check -text "Max Fiber Length:" \
		-variable $mods(TendFiber)-use-length \
		-command "$this toggle_fibers_fiber_length; $this execute_TendFiber" \
		-foreground grey64 
	    scale $stop.fiber.val -label "" \
		-from 1 -to 400 \
		-resolution 1 \
		-length 80  -width 15 \
		-sliderlength 15 \
		-orient horizontal \
		-showvalue false \
		-variable fibers_length \
		-foreground grey64 
	    label $stop.fiber.l -textvariable fibers_length -foreground grey64
	    pack $stop.fiber.check $stop.fiber.val $stop.fiber.l -side left \
		-anchor nw -padx 3 -pady 0
	    bind $stop.fiber.val <ButtonRelease> {app change_fibers_fiber_length}
	    
	    # Number of Steps
	    global $mods(TendFiber)-use-steps
	    global fibers_steps
	    frame $stop.steps
	    pack $stop.steps -side top -anchor nw
	    
	    checkbutton $stop.steps.check -text "Number of Steps: " \
		-variable $mods(TendFiber)-use-steps \
		-command "$this toggle_fibers_steps; $this execute_TendFiber" \
		-foreground grey64 
	    scale $stop.steps.val -label "" \
		-from 10 -to 1000 \
		-resolution 10 \
		-length 80  -width 15 \
		-sliderlength 15 \
		-orient horizontal \
		-showvalue false \
		-variable fibers_steps \
		-foreground grey64 
	    label $stop.steps.l -textvariable fibers_steps -foreground grey64
	    pack $stop.steps.check $stop.steps.val $stop.steps.l -side left \
		-anchor nw -padx 3 -pady 0
	    bind $stop.steps.val <ButtonRelease> {app change_fibers_steps}
	    
	    # Anisotropy
	    global $mods(TendFiber)-use-aniso
	    global $mods(TendFiber)-aniso-metric
	    global $mods(TendFiber)-aniso-thresh
	    
	    frame $stop.aniso1
	    pack $stop.aniso1 -side top -anchor nw
	    
	    checkbutton $stop.aniso1.check -text "Anisotropy Threshold:" \
		-variable $mods(TendFiber)-use-aniso \
		-command "$this toggle_fibers_aniso; $this execute_TendFiber" \
		-foreground grey64 
	    scale $stop.aniso1.val -label "" \
		-from 0.0 -to 1.0 \
		-resolution 0.01 \
		-length 70  -width 15 \
		-sliderlength 15 \
		-orient horizontal \
		-showvalue false \
		-variable $mods(TendFiber)-aniso-thresh \
		-foreground grey64 
	    label $stop.aniso1.l -textvariable $mods(TendFiber)-aniso-thresh -foreground grey64
	    pack $stop.aniso1.check $stop.aniso1.val $stop.aniso1.l -side left \
		-anchor nw -padx 3 -pady 0

	    bind $stop.aniso1.val <ButtonRelease> "$this change_fibers_aniso"
	    
	    frame $stop.aniso2
	    pack $stop.aniso2 -side top -anchor e
	    
	    radiobutton $stop.aniso2.cl -text "Linear Anisotropy" \
		-variable $mods(TendFiber)-aniso-metric \
		-value tenAniso_Cl2 \
		-foreground grey64 \
		-command "$this execute_TendFiber"
	    
	    radiobutton $stop.aniso2.fa -text "Fractional Anisotropy" \
		-variable $mods(TendFiber)-aniso-metric \
		-value tenAniso_FA \
		-foreground grey64 \
		-command "$this execute_TendFiber"
	    pack $stop.aniso2.cl $stop.aniso2.fa -side left -anchor nw -padx 3 -pady 0
	    
	    
	    
	    # Seed at
	    iwidgets::labeledframe $f.seed \
		-labeltext "Seed At" \
		-labelpos nw -foreground grey64
	    pack $f.seed -side top -anchor nw -padx 3 -pady 0 \
		-fill x
	    
	    set seed [$f.seed childsite]
	    
	    global $mods(ChooseField-FiberSeeds)-port-selected-index
	    
	    frame $seed.a
	    pack $seed.a -side left -anchor n -padx 3

	    frame $seed.a.pointf
	    pack $seed.a.pointf -side top\
		-anchor nw -padx 3 -pady 1
	    radiobutton $seed.a.pointf.point -text "Single Point" \
		-variable $mods(ChooseField-FiberSeeds)-port-selected-index \
		-value 0 \
		-foreground grey64 \
		-command "$this update_fiber_seed_method"
	    Tooltip $seed.a.pointf.point $tips(FibersSeedPoint)

	    global fiber_point
	    checkbutton $seed.a.pointf.w -text "Widget" \
		-variable fiber_point \
		-foreground grey64 \
		-command "$this toggle_fiber_point"
	    Tooltip $seed.a.pointf.w $tips(FibersTogglePoint)

	    pack $seed.a.pointf.point $seed.a.pointf.w -side left -anchor nw -padx 0 -pady 0

	    frame $seed.a.rakef
	    pack $seed.a.rakef  -side top \
		-anchor nw -padx 3 -pady 1

	    radiobutton $seed.a.rakef.rake -text "Along Line  " \
		-variable $mods(ChooseField-FiberSeeds)-port-selected-index \
		-value 1 \
		-foreground grey64 \
		-command "$this update_fiber_seed_method"
	    Tooltip $seed.a.rakef.rake $tips(FibersSeedLine)

	    global fiber_rake
	    checkbutton $seed.a.rakef.w -text "Widget" \
		-variable fiber_rake \
		-foreground grey64 \
		-command "$this toggle_fiber_rake"
	    Tooltip $seed.a.rakef.w $tips(FibersToggleLine)

	    pack $seed.a.rakef.rake $seed.a.rakef.w -side left -anchor nw -padx 0 -pady 0

	    frame $seed.b
	    pack $seed.b -side right -anchor n -padx 3
	    radiobutton $seed.b.plane -text "On Planes" \
		-variable $mods(ChooseField-FiberSeeds)-port-selected-index \
		-value 2 \
		-foreground grey64 \
		-command "$this update_fiber_seed_method"
	    Tooltip $seed.b.plane $tips(FibersSeedPlanes)
	    
	    radiobutton $seed.b.grid -text "On Grid" \
		-variable $mods(ChooseField-FiberSeeds)-port-selected-index \
		-value 3 \
		-foreground grey64 \
		-command "$this update_fiber_seed_method"
	    Tooltip $seed.b.grid $tips(FibersSeedGrid)
	    
	    
	    pack $seed.b.plane $seed.b.grid -side top \
		-anchor nw -padx 3 -pady 1
	    
	    iwidgets::labeledframe $f.rep \
		-labeltext "Color Fibers Based On" \
		-labelpos nw -foreground grey64
	    pack $f.rep -side top -anchor nw -padx 3 -pady 0 \
		-fill x
	    
	    set rep [$f.rep childsite]
	    
	    frame $rep.f1 
	    pack $rep.f1 -side top -anchor nw -padx 3 -pady 1
	    
	    iwidgets::optionmenu $rep.f1.type -labeltext "" \
		-width 150 -foreground grey64 \
		-command "$this change_fiber_color_by $rep.f1"
	    pack $rep.f1.type -side left -anchor nw -padx 2 -pady 0
	    
	    $rep.f1.type insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant"
	    
	    $rep.f1.type select "Principle Eigenvector"
	    
	    
	    global fiber_color
	    
	    addColorSelection $rep.f1 "Color" fiber_color \
		"default_color_change"
	    
	    iwidgets::labeledframe $rep.maps \
		-labeltext "Color Maps" \
		-labelpos nw -foreground grey64
	    pack $rep.maps -side top -anchor n -padx 3 -pady 0
	    
	    set maps [$rep.maps childsite]
	    global $mods(GenStandardColorMaps-Fibers)-mapType

	    frame $maps.a
	    pack $maps.a -side left -anchor n -padx 0 -pady 0

	    # Grey
	    frame $maps.a.gray
	    pack $maps.a.gray -side top -anchor nw -padx 1 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.a.gray.b -text "Gray" \
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 0 \
		-foreground grey64 \
		-command "$this update_fibers_color_map"
	    Tooltip $maps.a.gray.b $tips(FibersColorMap)
	    pack $maps.a.gray.b -side left -anchor nw -padx 1 -pady 0
	    
	    frame $maps.a.gray.f -relief sunken -borderwidth 2
	    pack $maps.a.gray.f -padx 1 -pady 0 -side right -anchor e
	    canvas $maps.a.gray.f.canvas -bg "#ffffff" -height $colormap_height -width [expr $colormap_width/3]
	    pack $maps.a.gray.f.canvas -anchor e \
		-fill both -expand 1
	    
	    draw_mini_colormap Gray $maps.a.gray.f.canvas

	    
	    # Rainbow
	    frame $maps.a.rainbow
	    pack $maps.a.rainbow -side top -anchor nw -padx 1 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.a.rainbow.b -text "Rainbow" \
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 2 \
		-foreground grey64 \
		-command "$this update_fibers_color_map"
	    Tooltip $maps.a.rainbow.b $tips(FibersColorMap)
	    pack $maps.a.rainbow.b -side left -anchor nw -padx 1 -pady 0
	    
	    frame $maps.a.rainbow.f -relief sunken -borderwidth 2
	    pack $maps.a.rainbow.f -padx 1 -pady 0 -side right -anchor e
	    canvas $maps.a.rainbow.f.canvas -bg "#ffffff" -height $colormap_height -width [expr $colormap_width/3]
	    pack $maps.a.rainbow.f.canvas -anchor e
	    
	    draw_mini_colormap Rainbow $maps.a.rainbow.f.canvas
	    
	    # Darkhue
	    frame $maps.a.darkhue
	    pack $maps.a.darkhue -side top -anchor nw -padx 1 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.a.darkhue.b -text "Darkhue" \
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 5 \
		-foreground grey64 \
		-command "$this update_fibers_color_map"
	    Tooltip $maps.a.darkhue.b $tips(FibersColorMap)
	    pack $maps.a.darkhue.b -side left -anchor nw -padx 1 -pady 0
	    
	    frame $maps.a.darkhue.f -relief sunken -borderwidth 2
	    pack $maps.a.darkhue.f -padx 1 -pady 0 -side right -anchor e
	    canvas $maps.a.darkhue.f.canvas -bg "#ffffff" -height $colormap_height -width [expr $colormap_width/3]
	    pack $maps.a.darkhue.f.canvas -anchor e
	    
	    draw_mini_colormap Darkhue $maps.a.darkhue.f.canvas
	    


	    frame $maps.b
	    pack $maps.b -side right -anchor n -padx 0 -pady 0



	    
	    # Blackbody
	    frame $maps.b.blackbody
	    pack $maps.b.blackbody -side top -anchor nw -padx 1 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.b.blackbody.b -text "Blackbody" \
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 7 \
		-foreground grey64 \
		-command "$this update_fibers_color_map"
	    Tooltip $maps.b.blackbody.b $tips(FibersColorMap)
	    pack $maps.b.blackbody.b -side left -anchor nw -padx 1 -pady 0
	    
	    frame $maps.b.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.b.blackbody.f -padx 1 -pady 0 -side right -anchor e
	    canvas $maps.b.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width [expr $colormap_width/3]
	    pack $maps.b.blackbody.f.canvas -anchor e
	    
	    draw_mini_colormap Blackbody $maps.b.blackbody.f.canvas
	    
	    
	    # Blue-to-Red
	    frame $maps.b.bpseismic
	    pack $maps.b.bpseismic -side top -anchor nw -padx 1 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.b.bpseismic.b -text "Blue-to-Red" \
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 17 \
		-foreground grey64 \
		-command "$this update_fibers_color_map"
	    Tooltip $maps.b.bpseismic.b $tips(FibersColorMap)
	    pack $maps.b.bpseismic.b -side left -anchor nw -padx 1 -pady 0
	    
	    frame $maps.b.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.b.bpseismic.f -padx 1 -pady 0 -side left -anchor e
	    canvas $maps.b.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width [expr $colormap_width/3]
	    pack $maps.b.bpseismic.f.canvas -anchor e
	    
	    draw_mini_colormap "Blue-to-Red" $maps.b.bpseismic.f.canvas
	} 
    }

    method update_fibers_color_map {} {
	global mods
	global $mods(ShowField-Fibers)-edges-on

	if {$vis_activated && [set $mods(ShowField-Fibers)-edges-on] == 1} {
	    $mods(GenStandardColorMaps-Fibers)-c needexecute
	} else {
	    global exec_fibers
	    set exec_fibers(GenStandardColorMaps-Fibers) 1
	}
    }

    method execute_TendFiber {} {
	global mods 
	global $mods(ShowField-Fibers)-edges-on
	if {$vis_activated && [set $mods(ShowField-Fibers)-edges-on] == 1} {
	    $mods(TendFiber)-c needexecute
	} else {
	    global exec_fibers
	    set exec_fibers(TendFiber) 1
	}
    }

    method sync_fibers_tabs {} {
	global mods 
	global $mods(ChooseField-Fibers)-port-selected-index
	global $mods(ChooseColorMap-Fibers)-port-selected-index

	set port [set $mods(ChooseField-Fibers)-port-selected-index]
	set color_port [set $mods(ChooseColorMap-Fibers)-port-selected-index]

	if {$color_port == 1 && $port != 3} {
	    # set optionmenu Constant and enable color button
	    $fibers_tab1.rep.childsite.f1.type select "Constant"
	    $fibers_tab2.rep.childsite.f1.type select "Constant"
	} elseif {$port == 0} {
	    #FA - set option menu to Fractional Anisotropy disable Color button
	    $fibers_tab1.rep.childsite.f1.type select "Fractional Anisotropy"
	    $fibers_tab2.rep.childsite.f1.type select "Fractional Anisotropy"
	} elseif {$port == 1} {
	    #LA -set optionmenu to LA and disable Color button
	    $fibers_tab1.rep.childsite.f1.type select "Linear Anisotropy"
	    $fibers_tab2.rep.childsite.f1.type select "Linear Anisotropy"
	} elseif {$port == 2} {
	    #PA - set option menu to pa and disable color button
	    $fibers_tab1.rep.childsite.f1.type select "Planar Anisotropy"
	    $fibers_tab2.rep.childsite.f1.type select "Planar Anisotropy"
	} elseif {$port == 3} {
	    #e1 - set option menu to e1 and disable color button
	    $fibers_tab1.rep.childsite.f1.type select "Principle Eigenvector"
	    $fibers_tab2.rep.childsite.f1.type select "Principle Eigenvector"
	} 
    }

    method configure_fibers_tabs {} {
        global mods
	global $mods(ShowField-Fibers)-edges-on  
	
        if {$vis_activated && [set $mods(ShowField-Fibers)-edges-on] == 1} {
	    foreach w [winfo children $fibers_tab1] {
		enable_widget $w
	    }
	    foreach w [winfo children $fibers_tab2] {
		enable_widget $w
	    }

	    # configure checkbutton/radiobutton widgets

	    toggle_fibers_fiber_length
	    toggle_fibers_steps
	    toggle_fibers_aniso


	    # configure color swatch
	    if {$fiber_type == "Constant"} {
		$fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state normal
		$fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state normal
		disable_fibers_colormaps
	    } elseif {$fiber_type == "Principle Eigenvector"} {
		$fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state disabled
		$fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state disabled
		disable_fibers_colormaps
	    } else {
		$fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state disabled
		$fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state disabled
		enable_fibers_colormaps
	    }

	    # configure glyph rake
	    global $mods(ChooseField-FiberSeeds)-port-selected-index
	    if {[set $mods(ChooseField-FiberSeeds)-port-selected-index] == 1} {
		$fibers_tab1.seed.childsite.a.rakef.w configure -state normal
		$fibers_tab2.seed.childsite.a.rakef.w configure -state normal
		$fibers_tab1.seed.childsite.a.pointf.w configure -state disabled
		$fibers_tab2.seed.childsite.a.pointf.w configure -state disabled
	    } elseif {[set $mods(ChooseField-FiberSeeds)-port-selected-index]== 0} {
		$fibers_tab1.seed.childsite.a.pointf.w configure -state normal
		$fibers_tab2.seed.childsite.a.pointf.w configure -state normal
		$fibers_tab1.seed.childsite.a.rakef.w configure -state disabled
		$fibers_tab2.seed.childsite.a.rakef.w configure -state disabled
	    } else {
		$fibers_tab1.seed.childsite.a.rakef.w configure -state disabled
		$fibers_tab2.seed.childsite.a.rakef.w configure -state disabled
		$fibers_tab1.seed.childsite.a.pointf.w configure -state disabled
		$fibers_tab2.seed.childsite.a.pointf.w configure -state disabled
	    }
	} elseif {[set $mods(ShowField-Fibers)-edges-on] == 0 && $vis_activated} {
		foreach w [winfo children $fibers_tab1] {
		    grey_widget $w
		}
		foreach w [winfo children $fibers_tab2] {
		    grey_widget $w
		}
	    # enable checkbutton
	    $fibers_tab1.show configure -state normal -foreground black
	    $fibers_tab2.show configure -state normal -foreground black 
	} else {
	    foreach w [winfo children $fibers_tab1] {
		grey_widget $w
	    }
	    foreach w [winfo children $fibers_tab2] {
		grey_widget $w
	    }
	}
    }

    
    method change_fibers_fiber_length {} {
	global mods
	global $mods(TendFiber)-use-length
	global $mods(TendFiber)-length
	global fibers_length
	
	set $mods(TendFiber)-length [expr $fibers_length/100.0]

	if {$vis_activated && [set $mods(TendFiber)-use-length]} {	  
	    $mods(TendFiber)-c needexecute
	} else {
	    global exec_fibers
	    set exec_fibers(TendFiber) 1
	}
    }
    
    method toggle_fibers_fiber_length {} {
	global mods
	global $mods(TendFiber)-use-length

	if {[set $mods(TendFiber)-use-length] == 0} {
	    # disable scale
	    $fibers_tab1.stop.childsite.fiber.val configure -state disabled
	    $fibers_tab2.stop.childsite.fiber.val configure -state disabled

	    $fibers_tab1.stop.childsite.fiber.l configure -state disabled
	    $fibers_tab2.stop.childsite.fiber.l configure -state disabled
	} else {
	    # enable scale
	    $fibers_tab1.stop.childsite.fiber.val configure -state normal
	    $fibers_tab2.stop.childsite.fiber.val configure -state normal

	    $fibers_tab1.stop.childsite.fiber.l configure -state normal
	    $fibers_tab2.stop.childsite.fiber.l configure -state normal
	}
    }


    method change_fibers_steps {} {
	global mods
	global $mods(TendFiber)-use-steps
	global $mods(TendFiber)-steps
	global fibers_steps
	
	set $mods(TendFiber)-steps [expr $fibers_steps/100.0]
	
	if {$vis_activated && [set $mods(TendFiber)-use-steps]} {
	    $mods(TendFiber)-c needexecute
	} else {
	    global exec_fibers
	    set exec_fibers(TendFiber) 1
	}
    }


    method toggle_fibers_steps {} {
	global mods
	global $mods(TendFiber)-use-steps

	if {[set $mods(TendFiber)-use-steps] == 0} {
	    # disable scale
	    $fibers_tab1.stop.childsite.steps.val configure -state disabled
	    $fibers_tab2.stop.childsite.steps.val configure -state disabled

	    $fibers_tab1.stop.childsite.steps.l configure -state disabled
	    $fibers_tab2.stop.childsite.steps.l configure -state disabled
	} else {
	    # enable scale
	    $fibers_tab1.stop.childsite.steps.val configure -state normal
	    $fibers_tab2.stop.childsite.steps.val configure -state normal

	    $fibers_tab1.stop.childsite.steps.l configure -state normal
	    $fibers_tab2.stop.childsite.steps.l configure -state normal
	}
    }

    method toggle_fibers_aniso {} {
	global mods
	global $mods(TendFiber)-use-aniso

	if {[set $mods(TendFiber)-use-aniso] == 0} {
	    # disable scale
	    $fibers_tab1.stop.childsite.aniso1.val configure -state disabled
	    $fibers_tab2.stop.childsite.aniso1.val configure -state disabled

	    $fibers_tab1.stop.childsite.aniso1.l configure -state disabled
	    $fibers_tab2.stop.childsite.aniso1.l configure -state disabled

	    $fibers_tab1.stop.childsite.aniso2.cl configure -state disabled
	    $fibers_tab2.stop.childsite.aniso2.cl configure -state disabled

	    $fibers_tab1.stop.childsite.aniso2.fa configure -state disabled
	    $fibers_tab2.stop.childsite.aniso2.fa configure -state disabled
	} else {
	    # enable scale
	    $fibers_tab1.stop.childsite.aniso1.val configure -state normal
	    $fibers_tab2.stop.childsite.aniso1.val configure -state normal

	    $fibers_tab1.stop.childsite.aniso1.l configure -state normal
	    $fibers_tab2.stop.childsite.aniso1.l configure -state normal

	    $fibers_tab1.stop.childsite.aniso2.cl configure -state normal
	    $fibers_tab2.stop.childsite.aniso2.cl configure -state normal

	    $fibers_tab1.stop.childsite.aniso2.fa configure -state normal
	    $fibers_tab2.stop.childsite.aniso2.fa configure -state normal
	}
    }

    method change_fibers_aniso {} {
	global mods
	global $mods(TendFiber)-use-aniso
	global $mods(ShowField-Fibers)-edges-on

	if {$vis_activated && [set $mods(ShowField-Fibers)-edges-on] && [set $mods(TendFiber)-use-aniso]} {
	    $mods(TendFiber)-c needexecute
	} else {
	    global exec_fibers
	    set exec_fibers(ShowField-Fibers) 1
	}
    }


    method change_fiber_color_by { f } {
	global mods
	global $mods(ChooseField-Fibers)-port-selected-index
	global $mods(ChooseColorMap-Fibers)-port-selected-index
	global $mods(ShowField-Fibers)-edges-usedefcolor

	# get selection and change appropriate port
	set type [$f.type get]

	set $mods(ShowField-Fibers)-edges-usedefcolor 0

	# configure color
	if {$type == "Principle Eigenvector"} {
	    set fiber_type "Principle Eigenvector"
            $fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state disabled
            $fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state disabled

	    set $mods(ChooseField-Fibers)-port-selected-index 3
	    disableModule $mods(ChooseColorMap-Fibers) 1
	    set $mods(ChooseColorMap-Fibers)-port-selected-index 1
	    disable_fibers_colormaps
	} elseif {$type == "Fractional Anisotropy"} {
	    set fiber_type "Fractional Anisotropy"
            $fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state disabled
            $fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state disabled
	    set $mods(ChooseField-Fibers)-port-selected-index 0
	    disableModule $mods(ChooseColorMap-Fibers) 0
	    set $mods(ChooseColorMap-Fibers)-port-selected-index 0
	    enable_fibers_colormaps
	} elseif {$type == "Linear Anisotropy"} {
	    set fiber_type "Linear Anisotropy"
            $fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state disabled
            $fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state disabled
	    set $mods(ChooseField-Fibers)-port-selected-index 1
	    disableModule $mods(ChooseColorMap-Fibers) 0
	    set $mods(ChooseColorMap-Fibers)-port-selected-index 0
	    enable_fibers_colormaps
	} elseif {$type == "Planar Anisotropy"} {
	    set fiber_type "Planar Anisotropy"
            $fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state disabled
            $fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state disabled
	    set $mods(ChooseField-Fibers)-port-selected-index 2
	    disableModule $mods(ChooseColorMap-Fibers) 0
	    set $mods(ChooseColorMap-Fibers)-port-selected-index 0
	    enable_fibers_colormaps
	} elseif {$type == "Constant"} {
	    set fiber_type "Constant"
            $fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state normal
            $fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state normal
	    disableModule $mods(ChooseColorMap-Fibers) 1
	    set $mods(ChooseColorMap-Fibers)-port-selected-index 1
	    set $mods(ShowField-Fibers)-edges-usedefcolor 1
	    set $mods(ChooseField-Fibers)-port-selected-index 0
	    disable_fibers_colormaps
	}

	configure_anisotropy_modules
	
	# sync attached/detached optionmenus
	$fibers_tab1.rep.childsite.f1.type select $type
	$fibers_tab2.rep.childsite.f1.type select $type

	configure_fibers_tabs

	global $mods(ShowField-Fibers)-edges-on
	if {$vis_activated && [set $mods(ShowField-Fibers)-edges-on] \
	    && !$loading} {
	    $mods(ChooseField-Fibers)-c needexecute
	    $mods(ShowField-Fibers)-c rerender_edges
	} else {
	    global exec_fibers
	    set exec_fibers(ChooseField-Fibers) 1
	    set exec_fibers(ShowField-Fibers) 1
	}
    }
 
    method disable_fibers_colormaps {} {
	foreach w [winfo children $fibers_tab1.rep.childsite.maps] {
	    grey_widget $w
	}

	foreach w [winfo children $fibers_tab2.rep.childsite.maps] {
	    grey_widget $w
	}
    }

    method enable_fibers_colormaps {} {
	foreach w [winfo children $fibers_tab1.rep.childsite.maps] {
	    enable_widget $w
	}

	foreach w [winfo children $fibers_tab2.rep.childsite.maps] {
	    enable_widget $w
	}
    }


    method configure_fibers_stepsize {} {
	global mods
	global $mods(ShowField-Fibers)-edges-on
	
	global $mods(TendFiber)-stepsize
	global fibers_stepsize
	set $mods(TendFiber)-stepsize [expr $fibers_stepsize/100.0]

	if {$vis_activated && [set $mods(ShowField-Fibers)-edges-on]} {	    
	    $mods(TendFiber)-c needexecute
	} else {
	    global exec_fibers
	    set exec_fibers(TendFiber) 1
	}
    }


    method update_fiber_seed_method {} {
        global mods
        global $mods(ChooseField-FiberSeeds)-port-selected-index
	
	if {[set $mods(ChooseField-FiberSeeds)-port-selected-index] == 0} {
	    # Point
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 1
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}" 0
	    $fibers_tab1.seed.childsite.a.pointf.w configure -state normal
	    $fibers_tab2.seed.childsite.a.pointf.w configure -state normal
	    
	    $fibers_tab1.seed.childsite.a.rakef.w configure -state disabled
	    $fibers_tab2.seed.childsite.a.rakef.w configure -state disabled
	} elseif {[set $mods(ChooseField-FiberSeeds)-port-selected-index] == 1} {
	    # Rake
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}" 1
	    $fibers_tab1.seed.childsite.a.pointf.w configure -state disabled
	    $fibers_tab2.seed.childsite.a.pointf.w configure -state disabled
	    
	    $fibers_tab1.seed.childsite.a.rakef.w configure -state normal
	    $fibers_tab2.seed.childsite.a.rakef.w configure -state normal
	} else {
	    # Grid or Planes
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
	    uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}" 0
	    $fibers_tab1.seed.childsite.a.pointf.w configure -state disabled
	    $fibers_tab2.seed.childsite.a.pointf.w configure -state disabled
	    
	    $fibers_tab1.seed.childsite.a.rakef.w configure -state disabled
	    $fibers_tab2.seed.childsite.a.rakef.w configure -state disabled
	    if {$vis_activated && [set $mods(ChooseField-FiberSeeds)-port-selected-index] == 3} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    }
	}
	
	global $mods(ShowField-Fibers)-edges-on
	if {$vis_activated && [set $mods(ShowField-Fibers)-edges-on]} {
	    $mods(ChooseField-FiberSeeds)-c needexecute
	} else {
	    global exec_glyphs
	    set exec_glyphs(ChooseField-FiberSeeds) 1
	}
	$mods(Viewer)-ViewWindow_0-c redraw
    }

    method toggle_fiber_point {} {
	global fiber_point
	global mods
	
	if {$fiber_point} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 1
	} else {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
	}

	$mods(Viewer)-ViewWindow_0-c redraw
    }
    

    method toggle_fiber_rake {} {
	global fiber_rake
	global mods
	
	if {$fiber_rake} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}" 1
	} else {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}" 0
	}

	$mods(Viewer)-ViewWindow_0-c redraw
    }
	    


    method toggle_show_fibers {} {
	global mods
        global $mods(ShowField-Fibers)-edges-on
	
        if {[set $mods(ShowField-Fibers)-edges-on] == 0} {
	    # sync nodes
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}" 0

	    # disable rest of fibers tab except for checkbutton
        } else {
	    # sync nodes
            global $mods(ChooseField-FiberSeeds)-port-selected-index
            if {[set $mods(ChooseField-FiberSeeds)-port-selected-index] == 0} {
		# enable Probe Widget
		uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 1
            } elseif {[set $mods(ChooseField-FiberSeeds)-port-selected-index] == 1} {
		# enable rake
		uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-SampleField Rake (12)\}" 1
            } elseif {$vis_activated && [set $mods(ChooseField-FiberSeeds)-port-selected-index] == 3} {
		$mods(ClipByFunction-Seeds)-c needexecute
	    }
        }

	configure_fibers_tabs

	if {$vis_activated} {

	    global exec_fibers
	    if {$exec_fibers(ChooseField-Fibers)} {
		$mods(ChooseField-Fibers)-c needexecute
		set exec_fibers(ChooseField-Fibers) 0
	    }
	    if {$exec_fibers(ChooseField-FiberSeeds)} {
		$mods(ChooseField-FiberSeeds)-c needexecute
		set exec_fibers(ChooseField-FiberSeeds) 0
	    }
	    if {$exec_fibers(GenStandardColorMaps-Fibers)} {
		$mods(GenStandardColorMaps-Fibers)-c needexecute
		set exec_fibers(GenStandardColorMaps-Fibers) 0
	    }
	    if {$exec_fibers(TendFiber)} {
		$mods(TendFiber)-c needexecute
		set exec_fibers(TendFiber) 0
	    }
	    if {$exec_fibers(ShowField-Fibers)} {
		$mods(ShowField-Fibers)-c rerender_edges
		set exec_fibers(ShowField-Fibers) 0
	    }
	    $mods(ShowField-Fibers)-c toggle_display_edges
	}
	$mods(Viewer)-ViewWindow_0-c redraw
    }
    


    
#################### COLORMAPS ###################

    ###########################
    ### draw_mini_colormap
    ###########################
    # Just like draw_colormap but makes one only 1/3 of
    # the size.  I needed to scrunch things for the Fibers tab.
    method draw_mini_colormap { which canvas } {
	set color ""
	if {$which == "Gray"} {
	    set color { "Gray" { { 0 0 0 } { 255 255 255 } } }
	} elseif {$which == "Rainbow"} {
	    set color { "Rainbow" {	
		{ 255 0 0}  { 255 102 0}
		{ 255 204 0}  { 255 234 0}
		{ 204 255 0}  { 102 255 0}
		{ 0 255 0}    { 0 255 102}
		{ 0 255 204}  { 0 204 255}
		{ 0 102 255}  { 0 0 255}}}
	} elseif {$which == "Blackbody"} {
	    set color { "Blackbody" {	
		{0 0 0}   {52 0 0}
		{102 2 0}   {153 18 0}
		{200 41 0}   {230 71 0}
		{255 120 0}   {255 163 20}
		{255 204 55}   {255 228 80}
		{255 247 120}   {255 255 180}
		{255 255 255}}}
	} elseif {$which == "Darkhue"} {
	    set color { "Darkhue" {	
		{ 0  0  0 }  { 0 28 39 }
		{ 0 30 55 }  { 0 15 74 }
		{ 1  0 76 }  { 28  0 84 }
		{ 32  0 85 }  { 57  1 92 }
		{ 108  0 114 }  { 135  0 105 }
		{ 158  1 72 }  { 177  1 39 }
		{ 220  10 10 }  { 229 30  1 }
		{ 246 72  1 }  { 255 175 36 }
		{ 255 231 68 }  { 251 255 121 }
		{ 239 253 174 }}}
	} elseif {$which == "Blue-to-Red"} {
	    set color { "Blue-to-Red" { { 0 0 255 } { 255 255 255} { 255 0 0 } } }
	}

        set colorMap [$this set_color_map $color]
	
	set width [expr $colormap_width/3]
        set height $colormap_height
	
	set n [llength $colorMap]
	$canvas delete map
	set dx [expr $width/double($n)] 
	set x 0
	for {set i 0} {$i < $n} {incr i 1} {
	    set color [lindex $colorMap $i]
	    set r [lindex $color 0]
	    set g [lindex $color 1]
	    set b [lindex $color 2]
	    set c [format "#%02x%02x%02x" $r $g $b]
	    set oldx $x
	    set x [expr ($i+1)*$dx]
	    $canvas create rectangle \
		$oldx 0 $x $height -fill $c -outline $c -tags map
	}
    }
    
    
    method setColor {col color mode} {
	global $color
	global $color-r
	global $color-g
	global $color-b
	set ir [expr int([set $color-r] * 65535)]
	set ig [expr int([set $color-g] * 65535)]
	set ib [expr int([set $color-b] * 65535)]
	
	set window .standalone
	$col config -background [format #%04x%04x%04x $ir $ig $ib]
	
	if {$color == "clip_to_isosurface_color"} {
	    # set the default colors for the three ShowFields
	    global mods
	    global $mods(ShowField-X)-def-color-r
	    global $mods(ShowField-X)-def-color-g
	    global $mods(ShowField-X)-def-color-b
            set $mods(ShowField-X)-def-color-r [set $color-r]
            set $mods(ShowField-X)-def-color-g [set $color-g]
            set $mods(ShowField-X)-def-color-b [set $color-b]
	    
            global $mods(ShowField-Y)-def-color-r
            global $mods(ShowField-Y)-def-color-g
            global $mods(ShowField-Y)-def-color-b
            set $mods(ShowField-Y)-def-color-r [set $color-r]
            set $mods(ShowField-Y)-def-color-g [set $color-g]
            set $mods(ShowField-Y)-def-color-b [set $color-b]

            global $mods(ShowField-Z)-def-color-r
            global $mods(ShowField-Z)-def-color-g
            global $mods(ShowField-Z)-def-color-b
            set $mods(ShowField-Z)-def-color-r [set $color-r]
            set $mods(ShowField-Z)-def-color-g [set $color-g]
            set $mods(ShowField-Z)-def-color-b [set $color-b]

	    if {$mode == "load"}  {
		return
	    }
	    
	    global show_planes
	    if {$vis_activated && $show_planes} {
		$mods(ChooseField-ColorPlanes)-c needexecute
	    } else {
		global exec_planes
		set exec_planes(ChooseField-ColorPlanes) 1
	    }
         } elseif {$color == "isosurface_color"} {
            # set the default color for ShowField
            global mods
            global $mods(ShowField-Isosurface)-def-color-r
            global $mods(ShowField-Isosurface)-def-color-g
            global $mods(ShowField-Isosurface)-def-color-b
            set $mods(ShowField-Isosurface)-def-color-r [set $color-r]
            set $mods(ShowField-Isosurface)-def-color-g [set $color-g]
            set $mods(ShowField-Isosurface)-def-color-b [set $color-b]

	    if {$mode == "load"}  {
		return
	    }
	    
	    global $mods(ShowField-Isosurface)-faces-on
	    if {$vis_activated && [set $mods(ShowField-Isosurface)-faces-on] == 1} {
		$mods(Isosurface)-c needexecute
		$mods(ShowField-Isosurface)-c default_color_change
	    } else {
		global exec_iso
		set exec_iso(Isosurface) 1
	    }
         } elseif {$color == "glyph_color"} {
             # set the default color for ShowField
             global mods
             global $mods(ShowField-Glyphs)-def-color-r
             global $mods(ShowField-Glyphs)-def-color-g
             global $mods(ShowField-Glyphs)-def-color-b
             set $mods(ShowField-Glyphs)-def-color-r [set $color-r]
             set $mods(ShowField-Glyphs)-def-color-g [set $color-g]
             set $mods(ShowField-Glyphs)-def-color-b [set $color-b]
 
	     if {$mode == "load"}  {
		 return
	     }
	     
	     global $mods(ShowField-Glyphs)-tensors-on
	     if {$vis_activated && [set $mods(ShowField-Glyphs)-tensors-on] == 1} {
		 $mods(DirectInterpolate-Glyphs)-c needexecute
		 $mods(ShowField-Glyphs)-c default_color_change
	     } else {
		 global exec_glyphs
		 set exec_glyphs(ChooseField-GlyphSeeds) 1
	     }
         } elseif {$color == "fiber_color"} {
             # set the default color for ShowField
             global mods
             global $mods(ShowField-Fibers)-def-color-r
             global $mods(ShowField-Fibers)-def-color-g
             global $mods(ShowField-Fibers)-def-color-b
             set $mods(ShowField-Fibers)-def-color-r [set $color-r]
             set $mods(ShowField-Fibers)-def-color-g [set $color-g]
             set $mods(ShowField-Fibers)-def-color-b [set $color-b]
	     
	     if {$mode == "load"}  {
		 return
	     }
	     
	     global $mods(ShowField-Fibers)-edges-on
	     if {$vis_activated && [set $mods(ShowField-Fibers)-edges-on]} {
		 $mods(DirectInterpolate-Fibers)-c needexecute
		 $mods(ShowField-Fibers)-c default_color_change
	     } else {
		 global exec_fobers
		 set exec_fibers(ChooseField-FiberSeeds) 1
	     }
         }

    }

   
    
#########################################################
####################### TAB STATE #######################
#########################################################
	
    method change_vis_tab { which } {
	# change vis tab for attached/detached

        if {$initialized != 0} {
	    if {$which == "Variance"} {
		# Variance
		$vis_tab1 view "Variance"
		$vis_tab2 view "Variance"
		set c_vis_tab "Variance"
	    } elseif {$which == "Planes"} {
		# Planes
		$vis_tab1 view "Planes"
		$vis_tab2 view "Planes"
		set c_vis_tab "Planes"
	    } elseif {$which == "Isosurface"} {
		# Isosurface
		$vis_tab1 view "Isosurface"
		$vis_tab2 view "Isosurface"
		set c_vis_tab "Isosurface"
	    } elseif {$which == "Glyphs"} {
		# Glyphs
		$vis_tab1 view "Glyphs"
		$vis_tab2 view "Glyphs"
		set c_vis_tab "Glyphs"
	    } elseif {$which == "Fibers"} {
		# Fibers
		$vis_tab1 view "Fibers"
		$vis_tab2 view "Fibers"
		set c_vis_tab "Fibers"
	    }
	}
    }



    method change_vis_frame { which } {
	# change tabs for attached and detached

        if {$initialized != 0} {
	    if {$which == "Vis Options"} {
		# Vis Options
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
	

    method change_processing_tab { which } {
	global mods
	global do_registration

	if {$indicate != 1} {
	    change_indicate_val 0
	}
	if {$initialized} {
	    if {$which == "Load Data"} {
		# Data Acquisition step
		$proc_tab1 view "Load Data"
		$proc_tab2 view "Load Data"
		change_indicator_labels "Press Execute to Load Data..."
		set c_procedure_tab "Load Data"
	    } elseif {$which == "Registration"} {
		# Registration step
		if {$data_completed} {
		    $proc_tab1 view "Registration"
		    $proc_tab2 view "Registration"
		    change_indicator_labels "Press Execute to Perform EPI Registration..."
		} 
		set c_procedure_tab "Registration"
	    } elseif {$which == "Build Tensors"} {
		if {!$do_registration} {
		    set reg_completed 1
		    disableModule $mods(ChooseNrrd-ToReg) 0
		    # NOT SURE ABOUT THIS RESCALE
		    #disableModule $mods(RescaleColorMap2) 0
		    disableModule $mods(TendEpireg) 1
		    disableModule $mods(ChooseNrrd-ToReg) 1
                    global data_mode
                    # Enable the UnuJoin module if it hasn't already
		    # been enabled when the B0 volume has been provided.
		    # This could have been missed being enabled if
		    # registration was skipped.
		    if {($data_mode == "DWIknownB0" || $data_mode == "B0DWI") && $data_completed} {
			disableModule $mods(UnuJoin) 0
		    } else {
			disableModule $mods(UnuJoin) 1
		    }
		    #$mods(ChooseNrrd-ToReg)-c needexecute
		    activate_dt
		    $proc_tab1 view "Build Tensors"
		    $proc_tab2 view "Build Tensors"
		    change_indicator_labels "Press Execute to Build Diffusion Tensors..."
		} elseif {$reg_completed} {
		    # Building DTs step
		    $proc_tab1 view "Build Tensors"
		    $proc_tab2 view "Build Tensors"
		    change_indicator_labels "Press Execute to Build Diffusion Tensors..."
		}
		set c_procedure_tab "Build Tensors"
	    }
	    
	    set indicator 0
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

		    if {$loading} {
			set loading 0
			if {$dt_completed} {
			    change_indicator_labels "Visualization..."
			} elseif {$reg_completed} {
			    change_indicator_labels "Building Diffusion Tensors..."
			} elseif {$data_completed} {
			    change_indicator_labels "Registration..."
			} else {
			    change_indicator_labels "Loading Data..."
			}
		    }
		} elseif {$executing_modules < 0} {
		    # something wasn't caught, reset
		    set executing_modules 0
		    set indicate 2
		    change_indicator

		    if {$loading} {
			set loading 0
			if {$dt_completed} {
			    change_indicator_labels "Visualization..."
			} elseif {$reg_completed} {
			    change_indicator_labels "Building Diffusion Tensors..."
			} elseif {$data_completed} {
			    change_indicator_labels "Registration..."
			} else {
			    change_indicator_labels "Loading Data..."
			}
		    }

		}
	    }
	}
    }
    
    method change_indicator_labels { msg } {
	if {!$loading} {
	    if {($msg == "Visualization..." && $data_completed && $reg_completed && $dt_completed) || ($msg != "Visualization...")} {
		$indicatorL0 configure -text $msg
		$indicatorL1 configure -text $msg
	    }
	} else {
	    # $msg != "Dynamically Compiling Code..."
	    if {$msg != "E R R O R !"} {
		$indicatorL0 configure -text "Executing to save point..."
		$indicatorL1 configure -text "Executing to save point..."
	    } else {
		$indicatorL0 configure -text $msg
		$indicatorL1 configure -text $msg
	    }

	}
    }

    # This method will make sure that out of the 4 different
    # anisotropies available, only those needed are enabled
    method configure_anisotropy_modules {} {
	global mods
	global $mods(ChooseField-Isoval)-port-selected-index \
	    $mods(ChooseField-Isosurface)-port-selected-index \
	    $mods(ChooseField-ColorPlanes)-port-selected-index \
	    $mods(ChooseField-Glyphs)-port-selected-index \
	    $mods(ChooseField-Fibers)-port-selected-index

	set tmp1 [set $mods(ChooseField-Isoval)-port-selected-index]
	set tmp2 [set $mods(ChooseField-Isosurface)-port-selected-index]
	set tmp3 [set $mods(ChooseField-ColorPlanes)-port-selected-index]
	set tmp4 [set $mods(ChooseField-Glyphs)-port-selected-index]
	set tmp5 [set $mods(ChooseField-Fibers)-port-selected-index]

	# Fractional Anisotropy
	if {$tmp1 == 0 || $tmp2 == 0 || $tmp3 == 0 \
		|| $tmp4 == 0 || $tmp5 == 0} {
	    disableModule $mods(TendAnvol-0) 0
	    disableModule $mods(NrrdToField-0) 0	    
	} else {
	    disableModule $mods(TendAnvol-0) 1
	    disableModule $mods(NrrdToField-0) 1
	}

	# Linear Anisotropy
	if {$tmp1 == 1 || $tmp2 == 1 || $tmp3 == 1 \
		|| $tmp4 == 1 || $tmp5 == 1} {
	    disableModule $mods(TendAnvol-1) 0
	    disableModule $mods(NrrdToField-1) 0	    
	} else {
	    disableModule $mods(TendAnvol-1) 1
	    disableModule $mods(NrrdToField-1) 1
	}

	# Planar Anisotropy
	if {$tmp1 == 2 || $tmp2 == 2 || $tmp3 == 2 \
		|| $tmp4 == 2 || $tmp5 == 2} {
	    disableModule $mods(TendAnvol-2) 0
	    disableModule $mods(NrrdToField-2) 0	    
	} else {
	    disableModule $mods(TendAnvol-2) 1
	    disableModule $mods(NrrdToField-2) 1
	}

	# Principle Eigenvector
	if {$tmp1 == 3 || $tmp2 == 3 || $tmp3 == 3 \
		|| $tmp4 == 3 || $tmp5 == 3} {
	    disableModule $mods(TendEvecRGB) 0
	    disableModule $mods(NrrdToField-3) 0	    
	} else {
	    disableModule $mods(TendEvecRGB) 1
	    disableModule $mods(NrrdToField-3) 1
	}
    }

    method toggle_save_tensors {} {
	global mods save_tensors
	if {$save_tensors ==  0} {
	    # disable UnuSave
	    disableModule $mods(UnuSave-Tensors) 1
	    disableModule $mods(ChooseNrrd-Save) 1
	} else {
	    # enable Save module
	    disableModule $mods(UnuSave-Tensors) 0
	    disableModule $mods(ChooseNrrd-Save) 0

	    if {$dt_completed == 1} {
		$this execute_save
	    }
	}
    }

    method execute_save {} {
	global mods
	$mods(UnuSave-Tensors)-c needexecute
    }

    
    method configure_data_mode {} {
	global data_mode
	
	if {$initialized} {
	    if {$data_mode == "B0DWI"} {
		$proc_tab1.canvas.notebook.cs.page1.cs.mode3quest.ch configure \
		    -foreground black
		$proc_tab1.canvas.notebook.cs.page1.cs.mode3quest.vol configure \
		    -foreground black
		$proc_tab1.canvas.notebook.cs.page1.cs.mode3quest.z configure \
		    -foreground black
		$proc_tab1.canvas.notebook.cs.page1.cs.mode3quest.b0 configure \
		    -foreground black

		$proc_tab2.canvas.notebook.cs.page1.cs.mode3quest.ch configure \
		    -foreground black
		$proc_tab2.canvas.notebook.cs.page1.cs.mode3quest.vol configure \
		    -foreground black
		$proc_tab2.canvas.notebook.cs.page1.cs.mode3quest.z configure \
		    -foreground black
		$proc_tab2.canvas.notebook.cs.page1.cs.mode3quest.b0 configure \
		    -foreground black
	    } else {
		$proc_tab1.canvas.notebook.cs.page1.cs.mode3quest.ch configure \
		    -foreground grey64
		$proc_tab1.canvas.notebook.cs.page1.cs.mode3quest.vol configure \
		    -foreground grey64
		$proc_tab1.canvas.notebook.cs.page1.cs.mode3quest.z configure \
		    -foreground grey64
		$proc_tab1.canvas.notebook.cs.page1.cs.mode3quest.b0 configure \
		    -foreground grey64

		$proc_tab2.canvas.notebook.cs.page1.cs.mode3quest.ch configure \
		    -foreground grey64
		$proc_tab2.canvas.notebook.cs.page1.cs.mode3quest.vol configure \
		    -foreground grey64
		$proc_tab2.canvas.notebook.cs.page1.cs.mode3quest.z configure \
		    -foreground grey64
		$proc_tab2.canvas.notebook.cs.page1.cs.mode3quest.b0 configure \
		    -foreground grey64		
	    }
	}
    }


#############################################################
######################### VARIABLES #########################
#############################################################
    
    # Data size variables
    variable volumes

    variable size_x
    variable spacing_x
    variable min_x

    variable size_y
    variable spacing_y
    variable min_y

    variable size_z
    variable spacing_z
    variable min_z

    variable average_spacing


    # State
    variable data_completed
    variable reg_completed
    variable dt_completed
    variable vis_activated
	
    variable c_procedure_tab
    variable c_data_tab
    variable c_left_tab
    variable c_vis_tab

    variable last_B0_port

    
    # Procedures frame tabnotebook
    variable proc_tab1
    variable proc_tab2

    # Procedures
    variable data_tab1
    variable data_tab2

    variable reg_tab1
    variable reg_tab2

    variable dt_tab1
    variable dt_tab2

    # Data tabs
    variable nrrd_tab1
    variable nrrd_tab2
    variable dicom_tab1
    variable dicom_tab2
    variable analyze_tab1
    variable analyze_tab2
    variable data_next_button1
    variable data_next_button2
    variable data_ex_button1
    variable data_ex_button2

    # Visualization frame tabnotebook
    variable vis_frame_tab1
    variable vis_frame_tab2

    # Vis tabs notebook
    variable vis_tab1
    variable vis_tab2

    variable variance_tab1
    variable variance_tab2

    variable planes_tab1
    variable planes_tab2

    variable isosurface_tab1
    variable isosurface_tab2

    variable glyphs_tab1
    variable glyphs_tab2

    variable fibers_tab1
    variable fibers_tab2

    # pointers to widgets
    variable ref_image1
    variable ref_image2

    variable reg_thresh1
    variable reg_thresh2


    # Application placing and size
    variable notebook_width
    variable notebook_height


    # planes
    variable last_x
    variable last_y
    variable last_z
    variable plane_inc
    variable plane_type

    # isosurfaces
    variable iso_type

    # glyphs
    variable clip_x
    variable clip_y
    variable clip_z
    variable glyph_type
	
    # fibers
    variable fiber_type

    variable has_autoviewed

}

setProgressText "Building BioTensor Window..."

BioTensorApp app
app build_app

hideProgress


### Bind shortcuts - Must be after instantiation of App
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
