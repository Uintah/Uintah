# COLOR SCHEME
set basecolor grey

. configure -background $basecolor

option add *Frame*background black

option add *Button*padX 1
option add *Button*padY 1

option add *background $basecolor
option add *activeBackground $basecolor
option add *sliderForeground $basecolor
option add *troughColor $basecolor
option add *activeForeground white

option add *Scrollbar*activeBackground $basecolor
option add *Scrollbar*foreground $basecolor
option add *Scrollbar*width .35c
option add *Scale*width .35c

option add *selectBackground "white"
option add *selector red
option add *font "-Adobe-Helvetica-bold-R-Normal--*-120-75-*"
option add *highlightThickness 0


#######################################################################
# Check environment variables.  Ask user for input if not set:
set results [sourceSettingsFile]
set DATADIR [lindex $results 0]
#######################################################################


# global array indexed by module name to keep track of modules
global mods


############# NET ##############
::netedit dontschedule

set m0 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 14 9]
set m1 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 603 169]
set m2 [addModuleAtPosition "Teem" "Tend" "TendEpireg" 14 180]
set m3 [addModuleAtPosition "Teem" "Tend" "TendEstim" 14 718]
set m4 [addModuleAtPosition "Teem" "Tend" "TendBmat" 231 181]
set m5 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 992 516]
set m6 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 603 583]
set m7 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 992 649]
set m8 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 769 812]
set m9 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 992 812]
set m10 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 1121 928]
set m11 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 14 928]
set m12 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 14 994]
set m13 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 114 1378]
set m14 [addModuleAtPosition "Teem" "Tend" "TendEval" 1450 23]
set m15 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 1449 86]
set m16 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 545 995]
set m17 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 14 1143]
set m18 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 46 1220]
set m19 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 189 928]
set m20 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 364 928]
set m21 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 189 994]
set m22 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 364 994]
set m23 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 14 1072]
set m24 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 196 1081]
set m25 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 218 1222]
set m26 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 218 1156]
set m27 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 473 1081]
set m28 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 506 1405]
set m29 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 737 1057]
set m30 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 737 1158]
set m31 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 470 1650]
set m32 [addModuleAtPosition "SCIRun" "FieldsCreate" "ClipField" 1451 533]
set m33 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 488 1527]
set m34 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1050 1081]
set m35 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldMeasures" 1451 213]
set m36 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldMeasures" 1450 149]
set m37 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldMeasures" 1452 464]
set m38 [addModuleAtPosition "SCIRun" "FieldsData" "ManageFieldData" 1451 656]
set m39 [addModuleAtPosition "SCIRun" "FieldsData" "ManageFieldData" 1492 1414]
set m40 [addModuleAtPosition "SCIRun" "FieldsData" "ManageFieldData" 1492 1538]
set m41 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1452 339]
set m42 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1269 86]
set m43 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1268 26]
set m44 [addModuleAtPosition "SCIRun" "Render" "Viewer" 88 2390]
set m45 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 14 88]
set m46 [addModuleAtPosition "SCIRun" "FieldsGeometry" "ChangeFieldBounds" 603 645]
set m47 [addModuleAtPosition "Teem" "Unu" "UnuJoin" 14 455]
set m48 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 32 316]
set m49 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 231 558]
set m50 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 231 103]
set m51 [addModuleAtPosition "Teem" "Unu" "UnuJoin" 1269 149]
set m52 [addModuleAtPosition "Teem" "Tend" "TendEstim" 1269 212]
set m53 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 1269 275]
set m54 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 1102 1471]
set m55 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1269 463]
set m56 [addModuleAtPosition "SCIRun" "FieldsGeometry" "ChangeFieldBounds" 1269 400]
set m57 [addModuleAtPosition "Teem" "Unu" "UnuProject" 603 516]
set m58 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 787 648]
set m59 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 787 745]
set m60 [addModuleAtPosition "Teem" "Unu" "UnuProject" 992 580]
set m61 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1010 748]
set m62 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 774 169]
set m63 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 249 495]
set m64 [addModuleAtPosition "Teem" "Tend" "TendEvec" 1451 718]
set m65 [addModuleAtPosition "Teem" "Unu" "UnuCrop" 1448 594]
set m66 [addModuleAtPosition "Teem" "NrrdData" "EditTupleAxis" 1269 336]
set m67 [addModuleAtPosition "SCIRun" "FieldsCreate" "SamplePlane" 488 1264]
set m68 [addModuleAtPosition "SCIRun" "FieldsCreate" "SamplePlane" 674 1263]
set m69 [addModuleAtPosition "SCIRun" "FieldsCreate" "SamplePlane" 862 1263]
set m70 [addModuleAtPosition "SCIRun" "FieldsGeometry" "Unstructure" 1263 591]
set m71 [addModuleAtPosition "SCIRun" "FieldsGeometry" "Unstructure" 1270 529]
set m72 [addModuleAtPosition "SCIRun" "FieldsGeometry" "Unstructure" 1267 652]
set m73 [addModuleAtPosition "SCIRun" "FieldsGeometry" "QuadToTri" 524 1341]
set m74 [addModuleAtPosition "SCIRun" "FieldsGeometry" "QuadToTri" 710 1341]
set m75 [addModuleAtPosition "SCIRun" "FieldsGeometry" "QuadToTri" 898 1340]
set m76 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 470 1586]
set m77 [addModuleAtPosition "SCIRun" "FieldsCreate" "IsoClip" 506 1465]
set m78 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 692 1406]
set m79 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 880 1405]
set m80 [addModuleAtPosition "SCIRun" "FieldsCreate" "IsoClip" 692 1465]
set m81 [addModuleAtPosition "SCIRun" "FieldsCreate" "IsoClip" 880 1466]
set m82 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 674 1527]
set m83 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 862 1527]
set m84 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 656 1588]
set m85 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 844 1589]
set m86 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 656 1651]
set m87 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 844 1651]
set m88 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherPoints" 1451 400]
set m89 [addModuleAtPosition "SCIRun" "FieldsCreate" "ClipByFunction" 1492 1607]
set m90 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherPoints" 1157 1610]
set m91 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1050 1849]
set m92 [addModuleAtPosition "SCIRun" "FieldsCreate" "SampleField" 1138 1534]
set m93 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1120 1726]
set m94 [addModuleAtPosition "SCIRun" "FieldsCreate" "Probe" 1298 1611]
set m95 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1050 2313]
set m96 [addModuleAtPosition "Teem" "DataIO" "AnalyzeToNrrd" 446 8]
set m97 [addModuleAtPosition "Teem" "DataIO" "DicomToNrrd" 232 9]
set m98 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 32 392]
set m99 [addModuleAtPosition "Teem" "DataIO" "DicomToNrrd" 204 313]
set m100 [addModuleAtPosition "Teem" "DataIO" "AnalyzeToNrrd" 377 313]
set m101 [addModuleAtPosition "Teem" "Unu" "UnuResample" 1450 275]
set m102 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1102 1788]
set m103 [addModuleAtPosition "Teem" "Unu" "UnuResample" 14 517]
set m104 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 14 652]
set m105 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 14 248]
set m106 [addModuleAtPosition "Teem" "Unu" "UnuResample" 14 588]
set m107 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 1267 713]
set m108 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1068 1144]
set m109 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1068 1206]
set m110 [addModuleAtPosition "Teem" "DataIO" "FieldToNrrd" 1102 1913]
set m111 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 1086 2251]
set m112 [addModuleAtPosition "Teem" "Tend" "TendEvalClamp" 14 850]
set m113 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 410 89]
set m114 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 14 786]
set m115 [addModuleAtPosition "Teem" "Tend" "TendAnscale" 1086 2124]
set m116 [addModuleAtPosition "Teem" "Tend" "TendNorm" 1102 1979]
set m117 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 1102 2046]
set m118 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 1086 2190]
set m119 [addModuleAtPosition "SCIRun" "FieldsCreate" "GatherPoints" 1492 1475]
set m120 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1438 1725]
set m121 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1367 1926]
set m122 [addModuleAtPosition "SCIRun" "FieldsOther" "ChooseField" 1367 1081]
set m123 [addModuleAtPosition "Teem" "Tend" "TendFiber" 1420 1792]
set m124 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 1402 1863]
set m125 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 1367 1993]
set m126 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1385 1144]
set m127 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1385 1206]
set m128 [addModuleAtPosition "Teem" "Tend" "TendEvecRGB" 545 928]
set m129 [addModuleAtPosition "SCIRun" "FieldsOther" "FieldInfo" 1451 780]
set m130 [addModuleAtPosition "SCIRun" "FieldsCreate" "Probe" 1420 1288]
set m131 [addModuleAtPosition "SCIRun" "FieldsCreate" "SampleField" 1456 1348]
set m132 [addModuleAtPosition "SCIRun" "Visualization" "ChooseColorMap" 218 1294]
set m133 [addModuleAtPosition "SCIRun" "Visualization" "ChooseColorMap" 540 1161]
set m134 [addModuleAtPosition "SCIRun" "Visualization" "ChooseColorMap" 1068 1289]
set m135 [addModuleAtPosition "SCIRun" "Visualization" "ChooseColorMap" 1274 1358]

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
set c20 [addConnection $m40 0 $m89 0]
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
set c48 [addConnection $m110 0 $m116 0]
set c49 [addConnection $m0 0 $m45 0]
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
set c66 [addConnection $m45 0 $m62 0]
set c67 [addConnection $m45 0 $m2 0]
set c68 [addConnection $m45 0 $m1 0]
set c69 [addConnection $m98 0 $m47 0]
set c70 [addConnection $m104 0 $m3 0]
set c71 [addConnection $m105 0 $m5 0]
set c72 [addConnection $m114 0 $m112 0]
set c73 [addConnection $m117 0 $m115 0]
set c74 [addConnection $m118 0 $m111 0]
set c75 [addConnection $m115 0 $m118 0]
set c76 [addConnection $m11 0 $m12 0]
set c77 [addConnection $m19 0 $m21 0]
set c78 [addConnection $m20 0 $m22 0]
set c79 [addConnection $m4 0 $m49 0]
set c80 [addConnection $m2 0 $m105 0]
set c81 [addConnection $m3 0 $m114 0]
set c82 [addConnection $m112 0 $m10 0]
set c83 [addConnection $m112 0 $m54 0]
set c84 [addConnection $m112 0 $m11 0]
set c85 [addConnection $m112 0 $m19 0]
set c86 [addConnection $m112 0 $m20 0]
set c87 [addConnection $m112 0 $m128 0]
set c88 [addConnection $m112 0 $m123 0]
set c89 [addConnection $m128 0 $m16 0]
set c90 [addConnection $m116 0 $m117 0]
set c91 [addConnection $m47 0 $m103 0]
set c92 [addConnection $m57 0 $m6 0]
set c93 [addConnection $m60 0 $m7 0]
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
set c128 [addConnection $m97 0 $m45 1]
set c129 [addConnection $m97 0 $m113 1]
set c130 [addConnection $m99 0 $m98 1]
set c131 [addConnection $m110 0 $m117 1]
set c132 [addConnection $m50 0 $m2 1]
set c133 [addConnection $m63 0 $m49 1]
set c134 [addConnection $m6 0 $m59 1]
set c135 [addConnection $m7 0 $m61 1]
set c136 [addConnection $m21 0 $m23 1]
set c137 [addConnection $m21 0 $m24 1]
set c138 [addConnection $m21 0 $m27 1]
set c139 [addConnection $m21 0 $m34 1]
set c140 [addConnection $m21 0 $m122 1]
set c141 [addConnection $m45 0 $m105 1]
set c142 [addConnection $m49 0 $m3 1]
set c143 [addConnection $m105 0 $m47 1]
set c144 [addConnection $m113 0 $m114 1]
set c145 [addConnection $m117 0 $m118 1]
set c146 [addConnection $m123 0 $m124 1]
set c147 [addConnection $m123 0 $m121 1]
set c148 [addConnection $m47 0 $m104 1]
set c149 [addConnection $m90 0 $m93 2]
set c150 [addConnection $m90 0 $m120 2]
set c151 [addConnection $m124 0 $m125 2]
set c152 [addConnection $m83 0 $m90 2]
set c153 [addConnection $m13 0 $m44 2]
set c154 [addConnection $m96 0 $m45 2]
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


set $m0-notes {}
set $m0-label {unknown}
set $m0-type {Scalar}
set $m0-axis {axis0}
set $m0-add {0}
set $m0-filename $DATADIR/brain-dt/demo-DWI.nrrd
set $m1-notes {}
set $m1-axis {3}
set $m1-position {8}
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
set $m5-position {8}
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
set $m13-use-normals {0}
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
set $m14-notes {}
set $m14-threshold {}
set $m15-notes {}
set $m15-axis {}
set $m15-position {}
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
set $m23-port-index {0}
set $m24-notes {}
set $m24-port-index {3}
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
set $m27-port-index {3}
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
set $m32-notes {}
set $m32-clip-location {cell}
set $m32-clipmode {replace}
set $m32-autoexecute {0}
set $m32-autoinvert {0}
set $m32-execmode {0}
set $m33-notes {}
set $m33-port-index {0}
set $m34-notes {}
set $m34-port-index {3}
set $m35-notes {}
set $m35-simplexString {Node}
set $m35-xFlag {1}
set $m35-yFlag {1}
set $m35-zFlag {1}
set $m35-idxFlag {0}
set $m35-sizeFlag {0}
set $m35-numNbrsFlag {0}
set $m36-notes {}
set $m36-simplexString {Node}
set $m36-xFlag {1}
set $m36-yFlag {1}
set $m36-zFlag {1}
set $m36-idxFlag {0}
set $m36-sizeFlag {0}
set $m36-numNbrsFlag {0}
set $m37-notes {}
set $m37-simplexString {Node}
set $m37-xFlag {1}
set $m37-yFlag {1}
set $m37-zFlag {1}
set $m37-idxFlag {0}
set $m37-sizeFlag {0}
set $m37-numNbrsFlag {0}
set $m38-notes {}
set $m39-notes {}
set $m40-notes {}
set $m41-notes {}
set $m41-isoval {0}
set $m41-isoval-min {0}
set $m41-isoval-max {99}
set $m41-isoval-typed {0}
set $m41-isoval-quantity {1}
set $m41-quantity-range {colormap}
set $m41-quantity-min {0}
set $m41-quantity-max {100}
set $m41-isoval-list {0.0 1.0 2.0 3.0}
set $m41-extract-from-new-field {1}
set $m41-algorithm {0}
set $m41-build_trisurf {1}
set $m41-np {1}
set $m41-active-isoval-selection-tab {0}
set $m41-active_tab {MC}
set $m41-update_type {on release}
set $m41-color-r {0.4}
set $m41-color-g {0.2}
set $m41-color-b {0.9}
set $m42-notes {}
set $m42-isoval {0}
set $m42-isoval-min {0}
set $m42-isoval-max {99}
set $m42-isoval-typed {0}
set $m42-isoval-quantity {1}
set $m42-quantity-range {colormap}
set $m42-quantity-min {0}
set $m42-quantity-max {100}
set $m42-isoval-list {0.0 1.0 2.0 3.0}
set $m42-extract-from-new-field {1}
set $m42-algorithm {0}
set $m42-build_trisurf {1}
set $m42-np {1}
set $m42-active-isoval-selection-tab {0}
set $m42-active_tab {MC}
set $m42-update_type {on release}
set $m42-color-r {0.4}
set $m42-color-g {0.2}
set $m42-color-b {0.9}
set $m43-notes {}
set $m43-isoval {0}
set $m43-isoval-min {0}
set $m43-isoval-max {99}
set $m43-isoval-typed {0}
set $m43-isoval-quantity {1}
set $m43-quantity-range {colormap}
set $m43-quantity-min {0}
set $m43-quantity-max {100}
set $m43-isoval-list {0.0 1.0 2.0 3.0}
set $m43-extract-from-new-field {1}
set $m43-algorithm {0}
set $m43-build_trisurf {1}
set $m43-np {1}
set $m43-active-isoval-selection-tab {0}
set $m43-active_tab {MC}
set $m43-update_type {on release}
set $m43-color-r {0.4}
set $m43-color-g {0.2}
set $m43-color-b {0.9}
set $m44-notes {}
set $m44-ViewWindow_0-pos {z0_y0}
set $m44-ViewWindow_0-caxes {0}
set $m44-ViewWindow_0-raxes {1}
set $m44-ViewWindow_0-iaxes {}
set $m44-ViewWindow_0-have_collab_vis {0}
set $m44-ViewWindow_0-view-eyep-x {-7.6630625798236407}
set $m44-ViewWindow_0-view-eyep-y {75.028175882540012}
set $m44-ViewWindow_0-view-eyep-z {1015.1705499788859}
set $m44-ViewWindow_0-view-lookat-x {-3.0535172036584237}
set $m44-ViewWindow_0-view-lookat-y {89.966866425184293}
set $m44-ViewWindow_0-view-lookat-z {35.278357023344782}
set $m44-ViewWindow_0-view-up-x {-0.0064389303591822541}
set $m44-ViewWindow_0-view-up-y {-0.99986262175693541}
set $m44-ViewWindow_0-view-up-z {-0.015273434099017805}
set $m44-ViewWindow_0-view-fov {20}
set $m44-ViewWindow_0-view-eyep_offset-x {}
set $m44-ViewWindow_0-view-eyep_offset-y {}
set $m44-ViewWindow_0-view-eyep_offset-z {}
set $m44-ViewWindow_0-lightColors {{1.0 1.0 1.0} {1.0 1.0 1.0} {1.0 1.0 1.0} {1.0 1.0 1.0}}
set $m44-ViewWindow_0-lightVectors {{ 0 0 1 } { 0 0 1 } { 0 0 1 } { 0 0 1 }}
set $m44-ViewWindow_0-bgcolor-r {0}
set $m44-ViewWindow_0-bgcolor-g {0}
set $m44-ViewWindow_0-bgcolor-b {0}
set $m44-ViewWindow_0-shading {}
set $m44-ViewWindow_0-do_stereo {0}
set $m44-ViewWindow_0-ambient-scale {1.0}
set $m44-ViewWindow_0-diffuse-scale {1.0}
set $m44-ViewWindow_0-specular-scale {0.4}
set $m44-ViewWindow_0-emission-scale {1.0}
set $m44-ViewWindow_0-shininess-scale {1.0}
set $m44-ViewWindow_0-polygon-offset-factor {1.0}
set $m44-ViewWindow_0-polygon-offset-units {0.0}
set $m44-ViewWindow_0-point-size {1.0}
set $m44-ViewWindow_0-line-width {1.0}
set $m44-ViewWindow_0-sbase {0.40}
set $m44-ViewWindow_0-sr {1}
set $m44-ViewWindow_0-do_bawgl {0}
set $m44-ViewWindow_0-drawimg {}
set $m44-ViewWindow_0-saveprefix {}
set $m44-ViewWindow_0-resx {}
set $m44-ViewWindow_0-resy {}
set $m44-ViewWindow_0-aspect {}
set $m44-ViewWindow_0-aspect_ratio {}
set $m44-ViewWindow_0-global-light {1}
set $m44-ViewWindow_0-global-fog {0}
set $m44-ViewWindow_0-global-debug {0}
set $m44-ViewWindow_0-global-clip {0}
set $m44-ViewWindow_0-global-cull {0}
set $m44-ViewWindow_0-global-dl {0}
set $m44-ViewWindow_0-global-type {Gouraud}
set $m44-ViewWindow_0-ortho-view {1}
set $m44-ViewWindow_0-unused {1}
set $m44-ViewWindow_0-unused {1}
set $m45-notes {}
set $m45-port-index {0}
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
set $m48-label {unknown}
set $m48-type {Scalar}
set $m48-axis {axis0}
set $m48-add {1}
set $m48-filename $DATADIR/brain-dt/demo-B0.nrrd
set $m49-notes {}
set $m49-port-index {0}
set $m50-notes {}
set $m50-label {unknown}
set $m50-type {Scalar}
set $m50-axis {axis0}
set $m50-add {0}
set $m50-filename $DATADIR/brain-dt/demo-gradients.txt
set $m51-notes {}
set $m51-join-axis {0}
set $m51-incr-dim {0}
set $m51-dim {4}
set $m52-notes {}
set $m52-use-default-threshold {1}
set $m52-threshold {100}
set $m52-soft {0}
set $m52-scale {1}
set $m53-notes {}
set $m53-aniso_metric {tenAniso_FA}
set $m53-threshold {100}
set $m54-notes {}
set $m54-build-eigens {1}
set $m55-notes {}
set $m55-isoval {0.5000}
set $m55-isoval-min {0}
set $m55-isoval-max {1}
set $m55-isoval-typed {0}
set $m55-isoval-quantity {1}
set $m55-quantity-range {colormap}
set $m55-quantity-min {0}
set $m55-quantity-max {100}
set $m55-isoval-list {0.0 1.0 2.0 3.0}
set $m55-extract-from-new-field {1}
set $m55-algorithm {1}
set $m55-build_trisurf {0}
set $m55-np {1}
set $m55-active-isoval-selection-tab {0}
set $m55-active_tab {NOISE}
set $m55-update_type {on release}
set $m55-color-r {0.40}
set $m55-color-g {0.78}
set $m55-color-b {0.73}
set $m56-notes {}
set $m56-outputcenterx {-95.5}
set $m56-outputcentery {110.5}
set $m56-outputcenterz {51}
set $m56-outputsizex {171}
set $m56-outputsizey {221}
set $m56-outputsizez {102}
set $m56-useoutputcenter {1}
set $m56-useoutputsize {0}
set $m56-box-scale {-1}
set $m56-box-center-x {}
set $m56-box-center-y {}
set $m56-box-center-z {}
set $m56-box-right-x {}
set $m56-box-right-y {}
set $m56-box-right-z {}
set $m56-box-down-x {}
set $m56-box-down-y {}
set $m56-box-down-z {}
set $m56-box-in-x {}
set $m56-box-in-y {}
set $m56-box-in-z {}
set $m56-resetting {0}
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
set $m63-label {unknown}
set $m63-type {Scalar}
set $m63-axis {axis0}
set $m63-add {0}
set $m63-filename {}
set $m64-notes {}
set $m64-threshold {0.5}
set $m65-notes {}
set $m65-num-axes {4}
set $m65-minAxis0 {0}
set $m65-maxAxis0 {0}
set $m65-absmaxAxis0 {2}
set $m65-minAxis1 {0}
set $m65-maxAxis1 {0}
set $m65-absmaxAxis1 {0}
set $m65-minAxis2 {0}
set $m65-maxAxis2 {0}
set $m65-absmaxAxis2 {0}
set $m65-minAxis3 {0}
set $m65-maxAxis3 {0}
set $m65-absmaxAxis3 {0}
set $m65-minAxis0 {0}
set $m65-maxAxis0 {0}
set $m65-absmaxAxis0 {2}
set $m66-notes {}
set $m66-input-label {Unknown:Vector}
set $m66-output-label {unknown:Vector}
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
set $m70-notes {}
set $m71-notes {}
set $m72-notes {}
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
set $m82-port-index {0}
set $m83-notes {}
set $m83-port-index {0}
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
set $m88-notes {}
set $m89-notes {}
set $m89-clipmode {allnodes}
set $m89-clipfunction {v > 0.5}
set $m90-notes {}
set $m91-notes {}
set $m91-interpolation_basis {linear}
set $m91-map_source_to_single_dest {0}
set $m91-exhaustive_search {0}
set $m91-exhaustive_search_max_dist {-1}
set $m91-np {1}
set $m92-notes {}
set $m92-endpoints {1}
set $m92-endpoint0x {11.133508890092173}
set $m92-endpoint0y {85.711169630030724}
set $m92-endpoint0z {32.408377222523043}
set $m92-endpoint1x {159.86649110990783}
set $m92-endpoint1y {147.68324555495391}
set $m92-endpoint1z {75.788830369969276}
set $m92-widgetscale {4.4619894665944697}
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
set $m93-port-index {2}
set $m94-notes {}
set $m94-locx {1.3022900636159871e-296}
set $m94-locy {8.0169258393150308e-292}
set $m94-locz {8.0169640657928678e-292}
set $m94-value {[-0.466733 -0.282872 0.0540473 1.46286e-312 -0.626182 -0.0654924 2.82972e-254 1.06938e-315 -0.227296]}
set $m94-node {0}
set $m94-edge {160395}
set $m94-face {0}
set $m94-cell {0}
set $m94-show-value {1}
set $m94-show-node {1}
set $m94-show-edge {0}
set $m94-show-face {0}
set $m94-show-cell {1}
set $m94-probe_scale {5.0}
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
set $m98-port-index {0}
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
set $m101-notes {}
set $m101-filtertype {gaussian}
set $m101-dim {4}
set $m101-sigma {1}
set $m101-extent {4}
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
set $m104-port-index {1}
set $m105-notes {}
set $m105-port-index {0}
set $m106-notes {}
set $m106-filtertype {gaussian}
set $m106-dim {4}
set $m106-sigma {1}
set $m106-extent {3}
set $m106-resampAxis0 {x1}
set $m106-resampAxis1 {=}
set $m106-resampAxis2 {=}
set $m106-resampAxis3 {x1}
set $m107-notes {}
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
set $m113-port-index {0}
set $m114-notes {}
set $m114-port-index {0}
set $m115-notes {}
set $m115-scale {1.0}
set $m116-notes {}
set $m116-major-weight {1.0}
set $m116-medium-weight {1.0}
set $m116-minor-weight {1.0}
set $m116-amount {1.0}
set $m116-target {1.0}
set $m117-notes {}
set $m117-port-index {0}
set $m118-notes {}
set $m118-port-index {1}
set $m119-notes {}
set $m120-notes {}
set $m120-port-index {2}
set $m121-notes {}
set $m121-interpolation_basis {linear}
set $m121-map_source_to_single_dest {0}
set $m121-exhaustive_search {0}
set $m121-exhaustive_search_max_dist {-1}
set $m121-np {1}
set $m122-notes {}
set $m122-port-index {3}
set $m123-notes {}
set $m123-fibertype {tensorline}
set $m123-puncture {0.0}
set $m123-neighborhood {2.0}
set $m123-stepsize {0.005}
set $m123-integration {Euler}
set $m123-use-aniso {1}
set $m123-aniso-metric {tenAniso_Cl2}
set $m123-aniso-thresh {0.5}
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
set $m125-edge_scale {0.250}
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
set $m128-evec {major}
set $m128-aniso_metric {tenAniso_FA}
set $m128-background {0.0}
set $m128-gray {0.5}
set $m128-gamma {1.6}
set $m129-notes {}
set $m130-notes {}
set $m130-locx {89.4178080635233}
set $m130-locy {54.31557446344981}
set $m130-locz {22.94385244720455}
set $m130-value {}
set $m130-node {78734}
set $m130-edge {359078}
set $m130-face {347741}
set $m130-cell {67789}
set $m130-show-value {0}
set $m130-show-node {0}
set $m130-show-edge {0}
set $m130-show-face {0}
set $m130-show-cell {0}
set $m130-probe_scale {5.0}
set $m131-notes {}
set $m131-endpoints {1}
set $m131-endpoint0x {14.61881744818555}
set $m131-endpoint0y {86.87295581606185}
set $m131-endpoint0z {6.279754362046386}
set $m131-endpoint1x {156.3807825518145}
set $m131-endpoint1y {145.9404412759072}
set $m131-endpoint1z {47.62699418393815}
set $m131-widgetscale {4.252858953108867}
set $m131-maxseeds {75}
set $m131-numseeds {10}
set $m131-rngseed {1}
set $m131-rnginc {1}
set $m131-clamp {0}
set $m131-autoexecute {1}
set $m131-type {}
set $m131-dist {uniuni}
set $m131-whichtab {Widget}
set $m132-notes {}
set $m132-port-index {1}
set $m133-notes {}
set $m133-port-index {1}
set $m134-notes {}
set $m134-port-index {1}
set $m135-notes {}
set $m135-port-index {1}

::netedit scheduleok


set mods(NrrdReader1) $m0
set mods(DicomToNrrd1) $m97
set mods(AnalyzeToNrrd1) $m96
set mods(ChooseNrrd1) $m45
set mods(NrrdInfo1) $m62

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
set mods(ChooseNrrd2) $m49
set mods(ChooseNrrd-ToReg) $m105
set mods(ChooseNrrd-ToSmooth) $m104

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


### Viewer
set mods(Viewer) $m44

# Tooltips
global tips

global data_mode
set data_mode "DWI"

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

global exag_glyph
set exag_glyph 0

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


                                                                               
#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

class BioTensorApp {
    
    method modname {} {
	return "BioTensorApp"
    }
    
    constructor {} {
	toplevel .standalone
	wm title .standalone "BioTensor"	 
	set win .standalone
	
	set notebook_width 350
	set notebook_height 600
	
	set viewer_width 640
	set viewer_height 670
	
	set process_width 365
	set process_height $viewer_height
	
	set vis_width [expr $notebook_width + 40]
	set vis_height $viewer_height

	set screen_width [winfo screenwidth .]
	set screen_height [winfo screenheight .]

        set initialized 0
	set data_completed 0
	set reg_completed 0
	set dt_completed 0
	set vis_activated 0

	set c_procedure_tab "Data"
	set c_data_tab "Nrrd"
	set c_left_tab "Data Vis"
	set c_vis_tab "Variance"

        set indicator1 ""
        set indicator2 ""
        set indicatorL1 ""
        set indicatorL2 ""
        set indicate 0
        set cycle 0
	set darby 0
        set i_width 300
        set i_height 20
        set stripes 10
        set i_move [expr [expr $i_width/double($stripes)]/2.0]
        set i_back [expr $i_move*-3]

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

        set ref_image1 ""
        set ref_image2 ""

        set reg_thresh1 ""
        set reg_thresh2 ""

        set error_module ""

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

        set proc_color "dark red"
	set next_color "#cdc858"
	set execute_color "#5377b5"
        set feedback_color "dodgerblue4"
        set error_color "red4"

        # planes
        set last_x 2
        set last_y 4
        set last_z 6
        set plane_inc "-0.1"
        set plane_type "Principle Eigenvector"

        # glyphs
        set clip_x ">"
        set clip_y ">"
        set clip_z ">"
        set glyph_type "Principle Eigenvector"

	# fibers
        set fiber_type "Principle Eigenvector"

        # colormaps
        set colormap_width 150
        set colormap_height 15
        set colormap_res 64

        set indicatorID 0

        initialize_blocks
	
	### Define Tooltips
	##########################
	# General
	global tips
	set tips(IndicatorBar) \
	    "Indicates the status of\napplication. Click when\nred to view error\nmessages"
	set tips(IndicatorLabel) \
	    "Indicates the current event"

	# Data Acquisition Tab
	set tips(DataTab) "Select to access\nparameters for\nData Acquisition\nstep"
        set tips(Execute-DataAcquisition) "Select to execute the\nData Acquisition step"
	set tips(Next-DataAcquisition) "Select to proceed to\nthe Registration step"

	# Registration Tab
	set tips(Execute-Registration) "Select to execute the\nRegistration step"
	set tips(Next-Registration) "Select to build\ndiffusion tensors"

	# Build DTs Tab
	set tips(Execute-DT) "Select to execute building\nof diffusion tensors\nand start visualization"
	set tips(Next-DT) "Select to view first\nvisualization tab"

	# Attach/Detach Mouseovers
	set tips(PDetachedMsg) "Click hash marks to\nAttach to Viewer"
	set tips(PAttachedMsg) "Click hash marks to\nDetach from the Viewer"
	set tips(VDetachedMsg) "Click hash marks to\nAttach to Viewer"
	set tips(VAttachedMsg) "Click hash marks to\nDetach from the Viewer"

	# Global Options Tab

    }
    

    destructor {
	destroy $this
    }

    
    method initialize_blocks {} { 
	global mods
	
        # Blocking Data Section (Analyze and Dicom modules)
	
        disableModule $mods(DicomToNrrd1) 1
        disableModule $mods(AnalyzeToNrrd1) 1
	
        disableModule $mods(DicomToNrrd-T2) 1
        disableModule $mods(AnalyzeToNrrd-T2) 1

	# Blocking Registration
        disableModule $mods(TendEpireg) 1
	disableModule $mods(UnuJoin) 1
        disableModule $mods(ChooseNrrd-ToReg) 1
        #disableModule $mods(RescaleColorMap2) 1
  
        # Building Diffusion Tensors
        disableModule $mods(NrrdReader-BMatrix) 1
        disableModule $mods(TendEstim) 1
	disableModule $mods(ChooseNrrd-DT) 1

        # Planes
        #disableModule $mods(QuadToTri-X) 1
        #disableModule $mods(QuadToTri-Y) 1
        #disableModule $mods(QuadToTri-Z) 1
        #disableModule $mods(RescaleColorMap-ColorPlanes) 1

        # Isosurface
        #disableModule $mods(DirectInterpolate-Isosurface) 1
        #disableModule $mods(RescaleColorMap-Isosurface) 1


        # Glyphs
        #disableModule $mods(NrrdToField-GlyphSeeds) 1
        #disableModule $mods(Probe-GlyphSeeds) 1
        #disableModule $mods(SampleField-GlyphSeeds) 1
        ##disableModule $mods(DirectInterpolate-GlyphSeeds) 1
 	##disableModule $mods(ChooseField-Glyphs) 1
	#disableModule $mods(RescaleColorMap-Glyphs) 1

	# Fibers
        #disableModule $mods(Probe-FiberSeeds) 1
        #disableModule $mods(SampleField-FiberSeeds) 1
        #disableModule $mods(DirectInterpolate-FiberSeeds) 1
 	#disableModule $mods(ChooseField-Fibers) 1
	#disableModule $mods(RescaleColorMap-Fibers) 1
	
    }


    method build_app {} {
	global mods
	
	# Embed the Viewer
	set eviewer [$mods(Viewer) ui_embedded]
	$eviewer setWindow $win.viewer
	
	
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

	init_Pframe $detachedPFr.f 0
	init_Pframe $attachedPFr.f 1

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


	### pack 3 frames
	pack $attachedPFr $win.viewer $attachedVFr -side left \
	    -anchor n -fill both -expand 1

	set total_width [expr $process_width + $viewer_width + $vis_width]

	set total_height $viewer_height

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]
	set pos_y [expr [expr $screen_height / 2] - [expr $total_height / 2]]

	append geom $total_width x $total_height + $pos_x + $pos_y
	wm geometry .standalone $geom
	update	

	initialize_clip_info

        set initialized 1

    }


    method init_Pframe { m case } {
        global mods
	global tips
        
	if { [winfo exists $m] } {
	    ### Menu
	    frame $m.main_menu -relief raised -borderwidth 3
	    pack $m.main_menu -fill x -anchor nw


	    menubutton $m.main_menu.file -text "File" -underline 0 \
		-menu $m.main_menu.file.menu
	    
	    menu $m.main_menu.file.menu -tearoff false

	    $m.main_menu.file.menu add command -label "Load session...  Ctr+O" \
		-underline 1 -command "$this load_session" -state active
	    
	    $m.main_menu.file.menu add command -label "Save session... Ctr+S" \
		-underline 0 -command "$this save_session" -state active

#	    $m.main_menu.file.menu add command -label "Save image..." \
#		-underline 0 -command "$mods(Viewer)-ViewWindow_0 makeSaveImagePopup" -state active
	    
	    $m.main_menu.file.menu add command -label "Quit        Ctr+Q" \
		-underline 0 -command "$this exit_app" -state active
	    
	    pack $m.main_menu.file -side left

	    
	    global tooltipsOn
	    menubutton $m.main_menu.help -text "Help" -underline 0 \
		-menu $m.main_menu.help.menu
	    
	    menu $m.main_menu.help.menu -tearoff false

 	    $m.main_menu.help.menu add check -label "Show Tooltips" \
		-variable tooltipsOn \
 		-underline 0 -state active

	    $m.main_menu.help.menu add command -label "Help Contents" \
		-underline 0 -command "$this show_help" -state active

	    $m.main_menu.help.menu add command -label "About BioTensor" \
		-underline 0 -command "$this show_about" -state active
	    
	    pack $m.main_menu.help -side left
	    
	    tk_menuBar $m.main_menu $win.main_menu.file $win.main_menu.help
	    


	    ### Processing Steps
	    #####################
	    iwidgets::labeledframe $m.p \
		-labelpos n -labeltext "Processing Steps" 
	    pack $m.p -side left -fill both -anchor n -expand 1
	    
	    set process [$m.p childsite]
	    
            iwidgets::tabnotebook $process.tnb \
                -width [expr $process_width - 40] \
                -height [expr $process_height - 130] \
                -tabpos n
	    pack $process.tnb -side top -anchor n 
	    
            set step_tab [$process.tnb add -label "Data" -command "$this change_processing_tab Data"]

	    
            if {$case == 0} {
                set proc_tab1 $process.tnb
            } else {
                set proc_tab2 $process.tnb
            }


	    # Radiobuttons for by-passing any processing and loading
	    # the tensors directly
            global data_mode

            radiobutton $step_tab.mode1 -text "Load Diffusion Weighted Images (DWI)" \
                -variable data_mode -value "DWI" \
                -command "$this toggle_data_mode"

            radiobutton $step_tab.mode2 -text "Load Tensor Volumes" \
                -variable data_mode -value "tensor" \
                -command "$this toggle_data_mode"

            pack $step_tab.mode1 $step_tab.mode2 -side top -anchor nw -padx 3 -pady 3


	    ### Data Acquisition
            iwidgets::tabnotebook $step_tab.tnb \
		-width [expr $process_width - 65 ] -height 250 \
		-tabpos n 
            pack $step_tab.tnb -side top -anchor n \
		-padx 3 -pady 8

            if {$case == 0} {
                set data_tab1 $step_tab.tnb
            } else {
                set data_tab2 $step_tab.tnb
            }	      
	    
	    
            ### Nrrd
            set page [$step_tab.tnb add -label "Nrrd" -command {app configure_readers Nrrd}]

	    if {$case == 0} {
                set nrrd_tab1 $page
            } else {
                set nrrd_tab2 $page
            }
	    
            global $mods(NrrdReader1)-filename
            label $page.dwil -text "DWI Volume:"
            pack $page.dwil -side top -anchor nw -padx 3 -pady 3

            iwidgets::entryfield $page.file -labeltext "Nrrd File:" -labelpos w \
                -textvariable $mods(NrrdReader1)-filename \
                -command "$this execute_Data"
            pack $page.file -side top -padx 3 -pady 6 -anchor n \
	        -fill x 
	    
            button $page.load -text "Browse" \
                -command "$this load_nrrd_dwi" \
                -width 12
            pack $page.load -side top -anchor n -padx 3 -pady 6
	    
            label $page.space -text "    "
            pack $page.space -side top -anchor n -padx 3 -pady 3
            global $mods(NrrdReader-T2)-filename
            label $page.t2l -text "T2 Reference Image:"
            pack $page.t2l -side top -anchor nw -padx 3 -pady 3

            iwidgets::entryfield $page.file2 -labeltext "Nrrd File:" -labelpos w \
                -textvariable $mods(NrrdReader-T2)-filename 
            pack $page.file2 -side top -padx 3 -pady 6 -anchor n \
	        -fill x 
	    
            button $page.load2 -text "Browse" \
                -command "$this load_nrrd_t2" \
                -width 12
            pack $page.load2 -side top -anchor n -padx 3 -pady 6
	    
	    
            ### Dicom
            set page [$step_tab.tnb add -label "Dicom" -command {app configure_readers Dicom}]

	    if {$case == 0} {
                set dicom_tab1 $page
            } else {
                set dicom_tab2 $page
            }
	    
            label $page.dwil -text "DWI Volume:"
            pack $page.dwil -side top -anchor nw -padx 3 -pady 3

	    button $page.load -text "Dicom Loader" \
		-command "$mods(DicomToNrrd1) ui"
	    pack $page.load -side top -anchor n \
		-padx 3 -pady 5 -ipadx 2 -ipady 2
	    
            label $page.space -text "    "
            pack $page.space -side top -anchor n -padx 3 -pady 3

            label $page.t2l -text "T2 Reference Image:"
            pack $page.t2l -side top -anchor nw -padx 3 -pady 3

	    button $page.load2 -text "Dicom Loader" \
		-command "$mods(DicomToNrrd-T2) ui"
	    pack $page.load2 -side top -anchor n \
		-padx 3 -pady 5 -ipadx 2 -ipady 2
	    

            ### Analyze
            set page [$step_tab.tnb add -label "Analyze" -command {app configure_readers Analyze}]

	    if {$case == 0} {
                set analyze_tab1 $page
            } else {
                set analyze_tab2 $page
            }

	    
            label $page.dwil -text "DWI Volume:"
            pack $page.dwil -side top -anchor nw -padx 3 -pady 3

	    button $page.load -text "Analyze Loader" \
		-command "$mods(AnalyzeToNrrd1) ui"
	    pack $page.load -side top -anchor n \
		-padx 3 -pady 5 -ipadx 2 -ipady 2

            label $page.space -text "    "
            pack $page.space -side top -anchor n -padx 3 -pady 3

            label $page.t2l -text "T2 Reference Image:"
            pack $page.t2l -side top -anchor nw -padx 3 -pady 3
	    
	    button $page.load2 -text "Analyze Loader" \
		-command "$mods(AnalyzeToNrrd-T2) ui"
	    pack $page.load2 -side top -anchor n \
		-padx 3 -pady 5 -ipadx 2 -ipady 2
            

	    # Set default view to be Nrrd
            $step_tab.tnb view "Nrrd"
	    
	    
	    # Execute and Next buttons
            frame $step_tab.last
            pack $step_tab.last -side bottom -anchor ne \
		-padx 5 -pady 5
	    
            button $step_tab.last.ex -text "Execute" \
		-background $execute_color \
		-activebackground $execute_color \
		-width 8 \
		-command "$this execute_Data"
	    Tooltip $step_tab.last.ex $tips(Execute-DataAcquisition)

	    button $step_tab.last.ne -text "Next" \
                -command "$this change_processing_tab Registration" -width 8 \
                -activebackground $next_color \
                -background grey75 -state disabled 
	    Tooltip $step_tab.last.ne $tips(Next-DataAcquisition)

            pack $step_tab.last.ne $step_tab.last.ex -side right -anchor ne \
		-padx 2 -pady 0

	    if {$case == 0} {
		set data_next_button1 $step_tab.last.ne
		set data_ex_button1 $step_tab.last.ex
	    } else {
		set data_next_button2 $step_tab.last.ne
		set data_ex_button2 $step_tab.last.ex
	    }
	    
	    
	    ### Registration
            set step_tab [$process.tnb add -label "Registration" -command "$this change_processing_tab Registration"]          
	    
            if {$case == 0} {
		set reg_tab1 $step_tab
            } else {
		set reg_tab2 $step_tab
            }
	    
	    # Checkbutton to skip Registration entirely
            global do_registration
            checkbutton $step_tab.doreg -text "Perform Global EPI Registration" \
                -variable do_registration -state disabled \
                -command "$this toggle_do_registration"
            pack $step_tab.doreg -side top -anchor nw -padx 7 -pady 0
	    
	    # Gradient File
	    iwidgets::labeledframe $step_tab.gradients \
                -labeltext "Gradients" \
                -labelpos nw -foreground grey64
            pack $step_tab.gradients -side top -anchor n \
		-fill x -padx 3 -pady 2
	    
	    set gradients [$step_tab.gradients childsite]
	    
            iwidgets::entryfield $gradients.file -labeltext "Gradients File:" \
                -labelpos w \
                -textvariable $mods(NrrdReader-Gradient)-filename \
	        -state disabled -foreground grey64
            pack $gradients.file -side top -padx 3 -pady 3 -anchor n \
	        -fill x 
	    
            button $gradients.load -text "Browse" \
                -command "$this load_gradient" \
                -width 12 -state disabled
            pack $gradients.load -side top -anchor n -padx 3 -pady 0
	    
	    
            # Reference Image
            global $mods(TendEpireg)-reference
            global ref_image_state ref_image

            iwidgets::labeledframe $step_tab.refimg \
		-labeltext "Reference Image" \
		-labelpos nw -foreground grey64
            pack $step_tab.refimg -side top -anchor n -padx 3 -pady 0
	    
            set refimg [$step_tab.refimg childsite]
	    
	    if {$case == 0} {
		set ref_image1 $refimg
            } else {
		set ref_image2 $refimg
            }
	    
	    radiobutton $refimg.est -text "Implicit Reference: estimate distortion\nparameters from all images" \
		-state disabled \
		-variable ref_image_state -value 0 \
		-command "$this toggle_reference_image_state"
	    
            pack $refimg.est -side top -anchor nw -padx 2 -pady 0
	    
            frame $refimg.s
            pack $refimg.s -side top -anchor nw -padx 2 -pady 0
	    
            radiobutton $refimg.s.choose -text "Choose Reference Image:" \
		-state disabled \
		-variable ref_image_state -value 1  \
		-command "$this toggle_reference_image_state"
	    
            label $refimg.s.label -textvariable ref_image -state disabled
            pack $refimg.s.choose $refimg.s.label -side left -anchor n
	    
            scale $refimg.s.ref -label "" \
		-state disabled \
		-variable ref_image \
		-from 1 -to 7 \
		-showvalue false \
		-length 150  -width 15 \
		-sliderlength 15 \
		-command "$this configure_reference_image" \
		-orient horizontal
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
		-sliderlength 15 -width 15 -length 113 \
		-orient horizontal
            label $blur.labely -text "Y:" -state disabled
            scale $blur.entryy -label "" \
		-state disabled \
		-foreground grey64 \
		-variable $mods(TendEpireg)-blur_y \
		-from 0.0 -to 5.0 \
		-resolution 0.01 \
		-showvalue true \
		-sliderlength 15 -width 15 -length 113 \
		-orient horizontal
            pack $blur.labelx $blur.entryx \
                $blur.labely $blur.entryy \
                -side left -anchor w -padx 2 -pady 1 \
                -fill x -expand 1
	    
	    
            iwidgets::labeledframe $seg.thresh \
		-labeltext "Background/DWI Threshold" \
		-labelpos nw -foreground grey64
            pack $seg.thresh -side top -anchor n -padx 0 -pady 0 \
		-fill x -expand 1
	    
            set thresh [$seg.thresh childsite]
	    
            if {$case == 0} {
                set reg_thresh1 $thresh
            } else {
                set reg_thresh2 $thresh	  
            }
	    
	    global $mods(TendEpireg)-threshold
            global $mods(TendEpireg)-use-default-threshold
            radiobutton $thresh.auto -text "Automatically Determine Threshold" \
		-state disabled \
		-variable $mods(TendEpireg)-use-default-threshold -value 1 \
		-command "$this toggle_registration_threshold"
            pack $thresh.auto -side top -anchor nw -padx 3 -pady 0
            frame $thresh.choose
            pack $thresh.choose -side top -anchor nw -padx 0 -pady 0
	    
            radiobutton $thresh.choose.button -text "Specify Threshold:" \
		-state disabled \
		-variable $mods(TendEpireg)-use-default-threshold -value 0 \
		-command "$this toggle_registration_threshold"
            entry $thresh.choose.entry -width 10 \
		-textvariable $mods(TendEpireg)-threshold \
		-state disabled -foreground grey64
            pack $thresh.choose.button $thresh.choose.entry -side left -anchor n -padx 2 -pady 3
	    
            checkbutton $seg.cc -variable $mods(TendEpireg)-cc_analysis \
		-text "Use Connected Components"\
		-state disabled
            pack $seg.cc -side top -anchor nw -padx 6 -pady 0
	    
            # Fitting
            label $step_tab.fitl -text "Percentage of Slices for Parameter Estimation: " -state disabled
            pack $step_tab.fitl -side top -anchor nw -padx 8 -pady 0
	    
            frame $step_tab.fit
            pack $step_tab.fit -side top -anchor n -padx 10 -pady 3
	    
            global $mods(TendEpireg)-fitting
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
	    
            pack $step_tab.fit.f $step_tab.fit.p \
		$step_tab.fit.s  -side left \
		-anchor nw -padx 0 -pady 0
            
	    
	    iwidgets::optionmenu $step_tab.rf -labeltext "Resampling Filter" \
                -labelpos w -width 160  \
                -state disabled \
                -command "$this set_resampling_filter $step_tab.rf"
	    
            pack $step_tab.rf -side top \
		-anchor nw -padx 8 -pady 4
	    
            $step_tab.rf insert end Linear Catmull-Rom "Windowed Sinc"

	    $step_tab.rf select "Catmull-Rom"
	    
	    
	    # Execute and Next buttons
            frame $step_tab.last
            pack $step_tab.last -side bottom -anchor ne  \
		-padx 5 -pady 5
            button $step_tab.last.ex -text "Execute" -state disabled -width 8 \
		-command "$this execute_Registration"
	    Tooltip $step_tab.last.ex $tips(Execute-Registration)

	    button $step_tab.last.ne -text "Next" -state disabled -width 8 \
		-command "$this change_processing_tab \"Build DTs\"" 
	    Tooltip $step_tab.last.ne $tips(Next-Registration)

            pack $step_tab.last.ne $step_tab.last.ex -side right \
		-anchor ne -padx 2 -pady 0
	    
	    ### Build DT
            set step_tab [$process.tnb add -label "Build DTs" -command "$this change_processing_tab \"Build DTs\""]
	    
            if {$case == 0} {
		set dt_tab1 $step_tab
            } else {
		set dt_tab2 $step_tab
            }
	    
	    # DWI Smooting
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
		-length 100  -width 15 \
		-sliderlength 15 \
		-showvalue false \
		-command "$this change_xy_smooth"
            label $blur.rad1.v -textvariable xy_radius -state disabled
            pack $blur.rad1.l $blur.rad1.s $blur.rad1.v -side left -anchor nw \
		-padx 3 -pady 0
	    
            frame $blur.rad2
            pack $blur.rad2 -side top -anchor n -padx 3 -pady 0
	    
            label $blur.rad2.l -text "Radius in Z:         " -state disabled
            scale $blur.rad2.s -from 0.0 -to 5.0 \
		-resolution 0.01 \
		-state disabled \
		-variable z_radius \
		-orient horizontal \
		-length 100  -width 15 \
		-sliderlength 15 \
		-showvalue false \
		-command "$this change_z_smooth"
            label $blur.rad2.v -textvariable z_radius -state disabled
            pack $blur.rad2.l $blur.rad2.s $blur.rad2.v -side left -anchor nw \
		-padx 3 -pady 0
	    

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
            pack $thresh.choose -side top -anchor nw -padx 0 -pady 0
	    
            radiobutton $thresh.choose.button -text "Specify Threshold:" \
		-state disabled \
		-variable $mods(TendEstim)-use-default-threshold -value 0 \
		-command "$this toggle_dt_threshold" 
            entry $thresh.choose.entry -width 10 \
		-textvariable $mods(TendEstim)-threshold \
		-state disabled -foreground grey64
            pack $thresh.choose.button $thresh.choose.entry -side left \
		-anchor n -padx 2 -pady 3
	    
            $thresh.def select

	    # B-Matrix
            iwidgets::labeledframe $step_tab.bm \
		-labeltext "B-Matrix" \
		-labelpos nw -foreground grey64
            pack $step_tab.bm -side top \
		-fill x -padx 3 -pady 0 -anchor n
	    
            set bm [$step_tab.bm childsite]
	    
	    global bmatrix
            radiobutton $bm.computeb -text "Compute B-Matrix Using Gradients Provided" \
		-state disabled \
		-variable bmatrix \
		-value "compute" \
		-command "$this toggle_b_matrix"
            pack $bm.computeb  -side top -anchor nw -padx 2 -pady 0
	    
            frame $bm.load
            pack $bm.load -side top -anchor nw -padx 0 -pady 0
	    
            radiobutton $bm.load.b -text "Load B-Matrix" \
		-state disabled \
		-variable bmatrix \
		-value "load" \
		-command "$this toggle_b_matrix"
	    
            entry $bm.load.e -width 25 \
		-textvariable $mods(NrrdReader-BMatrix)-filename \
		-state disabled -foreground grey64
            pack $bm.load.b $bm.load.e -side left -anchor nw \
		-padx 2 -pady 0
	    
            button $bm.browse -text "Browse" \
		-command "$this load_bmatrix" \
		-state disabled -width 12
	    
            pack $bm.browse -side top -anchor ne -padx 35 -pady 5
	    
        
	    # Execute and Next
            frame $step_tab.last
            pack $step_tab.last -side bottom -anchor ne \
		-padx 5 -pady 5
	    
            button $step_tab.last.ex -text "Execute" \
		-width 16 -state disabled \
		-command "$this execute_DT"
	    Tooltip $step_tab.last.ex $tips(Execute-DT)

            pack $step_tab.last.ex -side right -anchor ne \
		-padx 2 -pady 0
	    
	    
            ### Indicator
	    frame $process.indicator -relief sunken -borderwidth 2
            pack $process.indicator -side bottom -anchor s -padx 3 -pady 5
	    
	    canvas $process.indicator.canvas -bg "white" -width $i_width \
	        -height $i_height 
	    pack $process.indicator.canvas -side top -anchor n -padx 3 -pady 3
	    
            bind $process.indicator <Button> {app display_module_error} 
	    
            label $process.indicatorL -text "Data Acquisition..."
            pack $process.indicatorL -side bottom -anchor sw -padx 5 -pady 3
	    
	    
            if {$case == 0} {
		set indicator1 $process.indicator.canvas
		set indicatorL1 $process.indicatorL
            } else {
		set indicator2 $process.indicator.canvas
		set indicatorL2 $process.indicatorL
            }
	    Tooltip $process.indicatorL $tips(IndicatorLabel)
	    
            construct_indicator $process.indicator.canvas
	    
            $process.tnb view "Data"
	    
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
		    Tooltip $m.d.cut$i $tips(PDetachedMsg)
		} else {
		    Tooltip $m.d.cut$i $tips(PAttachedMsg)
		}
            }
	    
	}
	
        wm protocol .standalone WM_DELETE_WINDOW { NiceQuit }  
    }
    
    

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
		-height [expr $vis_height - 30] -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            if {$case == 0} {
               set vis_frame_tab1 $vis.tnb
            } else {
               set vis_frame_tab2 $vis.tnb	    
            }
	    
	    set page [$vis.tnb add -label "Data Vis" -command "$this change_vis_frame \"Data Vis\""]
	    
	    ### Data Vis Tab
	    # Add tabs for each visualization
            # Variance, Planes, Isosurface, Glyphs, Fibers
	    iwidgets::tabnotebook $page.vis_tabs \
                -width $notebook_width \
                -height $notebook_height \
                -tabpos n -equaltabs 0
	    
            pack $page.vis_tabs -padx 4 -pady 4
	    
            if {$case == 0} {
		set vis_tab1 $page.vis_tabs
            } else {
		set vis_tab2 $page.vis_tabs
            }
	    
            ### Variance
            set vis_tab [$page.vis_tabs add -label "Variance" -command "$this change_vis_tab Variance"]
	    
	    if {$case == 0} {
		set variance_tab1 $vis_tab
		build_variance_tab $variance_tab1
	    } else {
		set variance_tab2 $vis_tab
		build_variance_tab $variance_tab2
	    }         

	    
	    ### Planes
            set vis_tab [$page.vis_tabs add -label "Planes" -command "$this change_vis_tab Planes"]
	    
	    if {$case == 0} {
		set planes_tab1 $vis_tab
		build_planes_tab $planes_tab1
	    } else {
		set planes_tab2 $vis_tab
		build_planes_tab $planes_tab2
	    } 
	    
	    
	    ### Isosurface
            set vis_tab [$page.vis_tabs add -label "Isosurface" -command "$this change_vis_tab Isosurface"]
	    if {$case == 0} {
		set isosurface_tab1 $vis_tab
		build_isosurface_tab $isosurface_tab1
	    } else {
		set isosurface_tab2 $vis_tab
		build_isosurface_tab $isosurface_tab2
	    } 


	    
	    
	    ### Glyphs
            set vis_tab [$page.vis_tabs add -label "Glyphs" -command "$this change_vis_tab Glyhps"]
	    
	    if {$case == 0} {
		set glyphs_tab1 $vis_tab
		build_glyphs_tab $glyphs_tab1
	    } else {
		set glyphs_tab2 $vis_tab
		build_glyphs_tab $glyphs_tab2
	    } 


	    
	    ### Fibers
            set vis_tab [$page.vis_tabs add -label "Fibers" -command "$this change_vis_tab Fibers"]
	    if {$case == 0} {
		set fibers_tab1 $vis_tab
		build_fibers_tab $fibers_tab1
	    } else {
		set fibers_tab2 $vis_tab
		build_fibers_tab $fibers_tab2
	    } 


	    
	    
            $page.vis_tabs view "Variance"
	    

	    ### Renderer Options Tab
	    create_viewer_tab $vis
	    
	    
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
		    Tooltip $m.d.cut$i $tips(VDetachedMsg)
		} else {
		    Tooltip $m.d.cut$i $tips(VAttachedMsg)
		}
            }
	}
    }
    

    method create_viewer_tab { vis } {
	global mods
	set page [$vis.tnb add -label "Global Options" -command "$this change_vis_frame \"Global Options\""]
	
	iwidgets::labeledframe $page.viewer_opts \
	    -labelpos nw -labeltext "Global Render Options"
	
	pack $page.viewer_opts -side top -anchor n -fill both -expand 1
	
	set view_opts [$page.viewer_opts childsite]
	
	frame $view_opts.eframe -relief groove -borderwidth 2
	pack $view_opts.eframe -side top -anchor n -padx 4 -pady 4
	
	checkbutton $view_opts.eframe.light -text "Lighting" \
	    -variable $mods(Viewer)-ViewWindow_0-global-light \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	checkbutton $view_opts.eframe.fog -text "Fog" \
	    -variable $mods(Viewer)-ViewWindow_0-global-fog \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	checkbutton $view_opts.eframe.bbox -text "BBox" \
	    -variable $mods(Viewer)-ViewWindow_0-global-debug \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	pack $view_opts.eframe.light $view_opts.eframe.fog \
	    $view_opts.eframe.bbox  \
	    -side left -anchor n -padx 4 -pady 4
	
	
	frame $view_opts.buttons -relief flat
	pack $view_opts.buttons -side top -anchor n -padx 4 -pady 4
	
	frame $view_opts.buttons.v1
	pack $view_opts.buttons.v1 -side left -anchor nw
	
	
	button $view_opts.buttons.v1.autoview -text "Autoview" \
	    -command "$mods(Viewer)-ViewWindow_0-c autoview" \
	    -width 12 -padx 3 -pady 3
	
	pack $view_opts.buttons.v1.autoview -side top -padx 3 -pady 3 \
	    -anchor n -fill x
	
	
	frame $view_opts.buttons.v1.views
	pack $view_opts.buttons.v1.views -side top -anchor nw -fill x -expand 1
	
	menubutton $view_opts.buttons.v1.views.def -text "Views" \
	    -menu $view_opts.buttons.v1.views.def.m -relief raised \
	    -padx 3 -pady 3  -width 12
	
	menu $view_opts.buttons.v1.views.def.m -tearoff 0

	$view_opts.buttons.v1.views.def.m add cascade -label "Look down +X Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.posx
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down +Y Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.posy
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down +Z Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.posz
	$view_opts.buttons.v1.views.def.m add separator
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down -X Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.negx
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down -Y Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.negy
	$view_opts.buttons.v1.views.def.m add cascade -label "Look down -Z Axis" \
	    -menu $view_opts.buttons.v1.views.def.m.negz
	
	pack $view_opts.buttons.v1.views.def -side left -pady 3 -padx 3 -fill x
	
	menu $view_opts.buttons.v1.views.def.m.posx -tearoff 0
	$view_opts.buttons.v1.views.def.m.posx add radiobutton -label "Up vector +Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x1_y1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posx add radiobutton -label "Up vector -Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x1_y0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posx add radiobutton -label "Up vector +Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x1_z1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posx add radiobutton -label "Up vector -Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x1_z0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.posy -tearoff 0
	$view_opts.buttons.v1.views.def.m.posy add radiobutton -label "Up vector +X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y1_x1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views" 
	$view_opts.buttons.v1.views.def.m.posy add radiobutton -label "Up vector -X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y1_x0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posy add radiobutton -label "Up vector +Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y1_z1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posy add radiobutton -label "Up vector -Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y1_z0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.posz -tearoff 0
	$view_opts.buttons.v1.views.def.m.posz add radiobutton -label "Up vector +X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z1_x1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views" 
	$view_opts.buttons.v1.views.def.m.posz add radiobutton -label "Up vector -X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z1_x0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posz add radiobutton -label "Up vector +Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z1_y1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.posz add radiobutton -label "Up vector -Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z1_y0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.negx -tearoff 0
	$view_opts.buttons.v1.views.def.m.negx add radiobutton -label "Up vector +Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x0_y1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negx add radiobutton -label "Up vector -Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x0_y0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negx add radiobutton -label "Up vector +Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x0_z1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negx add radiobutton -label "Up vector -Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value x0_z0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.negy -tearoff 0
	$view_opts.buttons.v1.views.def.m.negy add radiobutton -label "Up vector +X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y0_x1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views" 
	$view_opts.buttons.v1.views.def.m.negy add radiobutton -label "Up vector -X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y0_x0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negy add radiobutton -label "Up vector +Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y0_z1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negy add radiobutton -label "Up vector -Z" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value y0_z0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	menu $view_opts.buttons.v1.views.def.m.negz -tearoff 0
	$view_opts.buttons.v1.views.def.m.negz add radiobutton -label "Up vector +X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z0_x1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views" 
	$view_opts.buttons.v1.views.def.m.negz add radiobutton -label "Up vector -X" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z0_x0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negz add radiobutton -label "Up vector +Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z0_y1 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	$view_opts.buttons.v1.views.def.m.negz add radiobutton -label "Up vector -Y" \
	    -variable $mods(Viewer)-ViewWindow_0-pos -value z0_y0 \
	    -command "$mods(Viewer)-ViewWindow_0-c Views"
	
	
	frame $view_opts.buttons.v2 
	pack $view_opts.buttons.v2 -side left -anchor nw
	
	button $view_opts.buttons.v2.sethome -text "Set Home View" -padx 3 -pady 3 \
	    -command "$mods(Viewer)-ViewWindow_0-c sethome"
	
	button $view_opts.buttons.v2.gohome -text "Go Home" \
	    -command "$mods(Viewer)-ViewWindow_0-c gohome" \
	    -padx 3 -pady 3
	
	pack $view_opts.buttons.v2.sethome $view_opts.buttons.v2.gohome \
	    -side top -padx 2 -pady 2 -anchor ne -fill x
	
	$vis.tnb view "Data Vis"
    }


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
	    pack $attachedPFr -anchor n -side left -before $win.viewer \
	       -fill both -expand 1
	    set new_width [expr $c_width + $process_width]
            append geom $new_width x $c_height + [expr $x - $process_width] + $y
	    wm geometry $win $geom
	    set IsPAttached 1
	}
    }


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
	
	set types {
	    {{App Settings} {.set} }
	    {{Other} { * } }
	} 
	set savefile [ tk_getSaveFile -defaultextension {.set} \
			   -filetypes $types ]
	if { $savefile != "" } {
	    set fileid [open $savefile w]
	    
	    # Save out data information 
	    puts $fileid "# BioTensor Session\n"
	    puts $fileid "set app_version 1.0"

	    save_module_variables $fileid
	    save_class_variables $fileid
	    save_global_variables $fileid
	    save_disabled_modules $fileid
	    
	    close $fileid
	}
    }

    method save_module_variables { fileid } {
	# make globals accessible
	foreach g [info globals] {
	    global $g
	}
	
	puts $fileid "# Save out module variables\n"
	
	set searchID [array startsearch mods]
	while {[array anymore mods $searchID]} {
	    set m [array nextelement mods $searchID]
	    foreach v [info vars $mods($m)*] {
		set var [get_module_variable_name $v]
		if {$var != "msgStream" && ![array exists $v]} {
		    puts $fileid "set \$mods($m)-$var \{[set $mods($m)-$var]\}"
		}
	    }
	}
	array donesearch mods $searchID
    }
    
    method get_module_variable_name { var } {
	# take out the module part of the variable name
	set end [string length $var]
	set start [string first "-" $var]
	set start [expr 1 + $start]
	
	return [string range $var $start $end]
    }

    method save_disabled_modules { fileid } {
	global mods Disabled

	puts $fileid "\n# Disabled Modules\n"
	
	set searchID [array startsearch mods]
	while {[array anymore mods $searchID]} {
	    set m [array nextelement mods $searchID]
	    if {[info exists Disabled($mods($m))] && $Disabled($mods($m))} {
		puts $fileid "disableModule \$mods($m) 1"
	    }
	}
	array donesearch mods $searchID
    }

    method save_class_variables { fileid } {
	puts $fileid "\n# Class Variables\n"
	
	foreach v [info variable] {
	    set var [get_class_variable_name $v]
	    if {$var != "this" } {
		puts $fileid "set $var \{[set $var]\}"
	    }
	}

    }
    
    
    method save_global_variables { fileid } {
	global mods

	puts $fileid "\n# Global Variables\n"
	
	# Save out my globals by hand because otherwise they conflict with
	# the module variables
	global data_mode
	puts $fileid "set data_mode \{$data_mode\}"
	
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
    
    
    method get_class_variable_name { var } {
	# Remove the :: from the variables
	set end [string length $var]
	set start [string last "::" $var]
	set start [expr 2 + $start]
	
	return [string range $var $start $end]
    }

    
    method load_session {} {	
	set types {
	    {{App Settings} {.set} }
	    {{Other} { * }}
	}
	
	set file [tk_getOpenFile -filetypes $types]
	if {$file != ""} {
	    
	    # Reset application 
	    reset_app
	    
	    foreach g [info globals] {
		global $g
	    }
	    
	    source $file

	    # set a few variables that need to be reset
	    set indicate 0
	    set cycle 0
	    set vis_activated 0
	    set IsPAttached 1
	    set IsVAttached 1

	    # configure attach/detach

	    configure_variance_tabs
	    configure_planes_tabs
	    configure_isosurface_tabs
	    configure_glyphs_tabs
	    configure_fibers_tabs
	    
	    # bring tabs forward
	    $proc_tab1 view $c_procedure_tab
	    $proc_tab2 view $c_procedure_tab

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

	    set data_completed 0
	    set reg_completed 0
	    set dt_completed 0
	    change_indicator_labels "Data Acquisition..."
	}	
    }


    method reset_app {} {
	global mods
	# enable all modules
	set searchID [array startsearch mods]
	while {[array anymore mods $searchID]} {
	    set m [array nextelement mods $searchID]
	    disableModule $mods($m) 0
	}
	array donesearch mods $searchID
	
	# disable registration and building dt tabs
	
	# remove stuff on vis tabs if there???
    }
    
    
    method exit_app {} {
	NiceQuit
    }
    
    method show_help {} {
	#tk_messageBox -message "Please refer to the online BioTensor Tutorial\nhttp://software.sci.utah.edu/doc/User/BioTensorTutorial" -type ok -icon info -parent .standalone

	global SCIRUN_SRCDIR
	
	if {[winfo exists .splash]} {
	    wm deiconify .splash
	    raise .splash
	    return;
	}
	
	# CHANGE FILENAME HERE
	set filename [file join $SCIRUN_SRCDIR main scisplash.ppm]
	image create photo ::img::splash -file "$filename"
	toplevel .splash
	
	wm protocol .splash WM_DELETE_WINDOW "wm withdraw .splash"
	
	wm title .splash {Welcome to SCIRun}
	label .splash.splash -image ::img::splash
	pack .splash.splash

	wm geometry .splash 504x480+135+170
	label .splash.m1 -text "Please refer to the online BioTensor Tutorial"
	label .splash.m2 -text "http://software.sci.utah.edu/doc/User/BioTensorTutorial" 
	pack .splash.m1 .splash.m2 -anchor n

	button .splash.ok -text " OK " -command "wm withdraw .splash"
	pack .splash.ok -side bottom -padx 5 -pady 5 -fill none
	
	update idletasks
    }
    
    method show_about {} {
	puts "FIX ME: Insert BioTensor Intro here..."
	tk_messageBox -message "BioTensor About Box" -type ok -icon info -parent .standalone
    }
    
    method display_module_error {} {
        if {$error_module != ""} {
	    set result [$error_module displayLog]
        }
    }
    
    method indicate_dynamic_compile { which mode } {
	if {$mode == "start"} {
	    #change_indicate_val 1
	    #change_indicator_labels "Dynamically Compiling Code..."
        } else {
# 	    if {$dt_completed} {
# 		change_indicator_labels "Visualization..."
# 	    } elseif {$c_data_tab == "Build DTs"} {
# 		change_indicator_labels "Building Diffusion Tensors..."
# 	    } elseif {$c_data_tab == "Registration"} {
# 		change_indicator_labels "Registration..."
# 	    } else {
# 		change_indicator_labels "Data Acquisition..."
# 	    }
	    # change_indicate_val 0
	}
    }
    
    
    method update_progress { which state } {
	global mods
	global $mods(ShowField-Isosurface)-faces-on
	global $mods(ShowField-Glyphs)-tensors-on
	global $mods(ShowField-Fibers)-edges-on
	global show_plane_x show_plane_y show_plane_z
	
	
	if {$which == $mods(ChooseNrrd1) && $state == "JustStarted"} {
	    change_indicator_labels "Data Acquisition..."
	    change_indicate_val 1
	} elseif {$which == $mods(ChooseNrrd1) && $state == "Completed"} {
	    change_indicate_val 2
	    set data_completed 1	
	} elseif {$which == $mods(ShowField-Orig) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Orig) && $state == "Completed"} {
	    change_indicate_val 2
	    
	    configure_variance_tabs
	    
	    # Bring images into view
	    global $mods(ShowField-Orig)-faces-on
	    if {[set $mods(ShowField-Orig)-faces-on] == 1} {
		after 100 "$mods(Viewer)-ViewWindow_0-c autoview; global $mods(Viewer)-ViewWindow_0-pos; set $mods(Viewer)-ViewWindow_0-pos \"z0_y0\"; $mods(Viewer)-ViewWindow_0-c Views;"
	    }
	} elseif {$which == $mods(TendEpireg) && $state == "JustStarted"} {
	    if {$data_completed} {
		change_indicator_labels "Registration..."
	    }
	    change_indicate_val 1
	} elseif {$which == $mods(TendEpireg) && $state == "Completed"} {
	    change_indicate_val 2
	    set reg_completed 1

	    # activate next button
	    $reg_tab1.last.ne configure -state normal \
		-foreground black -background $next_color
	    $reg_tab2.last.ne configure -state normal \
		-foreground black -background $next_color
	    
#	    if {$reg_completed} {
		activate_dt
#	    }
	} elseif {$which == $mods(ShowField-Reg) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Reg) && $state == "Completed"} {
	    change_indicate_val 2
	    
	    configure_variance_tabs
	    
	    # Bring images into view
	    global $mods(ShowField-Reg)-faces-on
	    if {[set $mods(ShowField-Reg)-faces-on] == 1} {
		after 100 "$mods(Viewer)-ViewWindow_0-c autoview; global $mods(Viewer)-ViewWindow_0-pos; set $mods(Viewer)-ViewWindow_0-pos \"z0_y0\"; $mods(Viewer)-ViewWindow_0-c Views"
	    }
        } elseif {$which == $mods(TendEstim) && $state == "JustStarted"} {
	    if {$reg_completed} {
		change_indicator_labels "Building Diffusion Tensors..."
	    }
	    change_indicate_val 1
	} elseif {$which == $mods(TendEstim) && $state == "Completed"} {
	    change_indicate_val 2
	    set dt_completed 1

#	    if {$dt_completed} {
		activate_vis
#	    }
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
		
		set min_x [set $mods(NrrdInfo1)-min1]
		set min_y [set $mods(NrrdInfo1)-min2]
		set min_z [set $mods(NrrdInfo1)-min3]
		
		if {$data_mode == "DWI"} {
		    # new data has been loaded, configure
		    # the vis tabs and sync their values
		    
		    sync_variance_tabs
		    sync_planes_tabs
		    sync_isosurface_tabs
		    sync_glyphs_tabs
		    sync_fibers_tabs
		    
		    configure_sample_planes
		    
		    # reconfigure registration reference image slider
		    $ref_image1.s.ref configure -from 1 -to $volumes
		    $ref_image2.s.ref configure -from 1 -to $volumes
		} else {
		    sync_planes_tabs
		    sync_isosurface_tabs
		    sync_glyphs_tabs
		    sync_fibers_tabs
		    
		    configure_sample_planes
		    
		    set data_completed 1
		    set reg_completed 1
		    set dt_completed 1
		    activate_vis
		}
	    }
# 	} elseif {$which == $mods(SamplePlane-X) && $state == "JustStarted"} {
#  	    #change_indicator_labels "Visualization..."
#  	    change_indicate_val 1
# 	} elseif {$which == $mods(SamplePlane-X) && $state == "Completed"} {
#  	    change_indicate_val 2
 	} elseif {$which == $mods(ShowField-X) && $state == "JustStarted"} {
	    change_indicate_val 1
 	} elseif {$which == $mods(ShowField-X) && $state == "Completed"} {
 	    change_indicate_val 2
#  	} elseif {$which == $mods(SamplePlane-Y) && $state == "JustStarted"} {
#  	    #change_indicator_labels "Visualization..."
#  	    change_indicate_val 1
#  	} elseif {$which == $mods(SamplePlane-Y) && $state == "Completed"} {
#  	    change_indicate_val 2
 	} elseif {$which == $mods(ShowField-Y) && $state == "JustStarted"} {
	    change_indicate_val 1
 	} elseif {$which == $mods(ShowField-Y) && $state == "Completed"} {
 	    change_indicate_val 2
#  	} elseif {$which == $mods(SamplePlane-Z) && $state == "JustStarted"} {
#  	    #change_indicator_labels "Visualization..."
#  	    change_indicate_val 1
#  	} elseif {$which == $mods(SamplePlane-Z) && $state == "Completed"} {
# 	    change_indicate_val 2
 	} elseif {$which == $mods(ShowField-Z) && $state == "JustStarted"} {
	    change_indicate_val 1
 	} elseif {$which == $mods(ShowField-Z) && $state == "Completed"} {
 	    change_indicate_val 2
# 	} elseif {$which == $mods(Isosurface) && $state == "JustStarted"} {
# 	    #change_indicator_labels "Visualization..."
# 	    change_indicate_val 1
# 	} elseif {$which == $mods(Isosurface) && $state == "Completed"} {
# 	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Isosurface) && $state == "JustStarted"} {
	    #change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Isosurface) && $state == "Completed"} {
	    change_indicate_val 2
# 	} elseif {$which == $mods(ChooseField-GlyphSeeds) && $state == "JustStarted"} {
# 	    #change_indicator_labels "Visualization..."
# 	    change_indicate_val 1
# 	} elseif {$which == $mods(ChooseField-GlyphSeeds) && $state == "Completed"} {
# 	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Glyphs) && $state == "JustStarted"}  {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Glyphs) && $state == "Completed"} {
	    change_indicate_val 2
# 	} elseif {$which == $mods(ChooseField-FiberSeeds) && $state == "JustStarted"} {
# 	    #change_indicator_labels "Visualization..."
# 	    change_indicate_val 1
# 	} elseif {$which == $mods(ChooseField-FiberSeeds) && $state == "Completed"} {
# 	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Fibers) && $state == "JustStarted"} {
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Fibers) && $state == "Completed"} { 
	    change_indicate_val 2
	} elseif {$which == $mods(SampleField-GlyphSeeds) && $state == "Completed" && ![set $mods(ShowField-Glyphs)-tensors-on]} {
	    after 100 \
		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-StreamLines rake (7)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	} elseif {$which == $mods(Probe-GlyphSeeds) && $state == "Completed" && ![set $mods(ShowField-Glyphs)-tensors-on]} {
	    after 100 \
		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	} elseif {$which == $mods(SampleField-FiberSeeds) && $state == "Completed" && ![set $mods(ShowField-Fibers)-edges-on]} {
	    after 100 \
		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-StreamLines rake (12)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	} elseif {$which == $mods(Probe-FiberSeeds) && $state == "Completed" && ![set $mods(ShowField-Fibers)-edges-on]} {
	    after 100 \
		"uplevel \#0 set \"\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}\" 0; $mods(Viewer)-ViewWindow_0-c redraw"
	} 
    }

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
		$which != $mods(RescaleColorMap-Glyphs)} {
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
			change_indicator_labels "Data Acquisition..."
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
    method configure_data_tab {} {
	global data_mode
	foreach w [winfo children $data_tab1] {
	    activate_widget $w
	}
	foreach w [winfo children $data_tab2] {
	    activate_widget $w
	}

	# configure t2 reference image stuff if loading tensors directly
	if {$data_mode == "DWI"} {
	    toggle_data_mode
	} 

	if {$data_completed} {
	    $data_next_button1 configure -state normal -foreground black \
		-background $next_color
	    $data_next_button2 configure -state normal -foreground black \
		-background $next_color
	}

    }


    method execute_Data {} {
	global mods 
	global data_mode
	
	if {$data_mode == "tensor"} {
	    disableModule $mods(ChooseNrrd-DT) 0
	    disableModule $mods(TendEstim) 1
	}

	$mods(ChooseNrrd1)-c needexecute

	activate_registration

	# enable Next button
	$data_next_button1 configure -state normal \
	    -foreground black -background $next_color
	$data_next_button2 configure -state normal \
	    -foreground black -background $next_color
    }


    method toggle_data_mode { } {
	global data_mode
        global mods
        global $mods(ChooseNrrd-DT)-port-index
	
        if {$data_mode == "DWI"} {
           configure_readers all

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
           set $mods(ChooseNrrd-DT)-port-index 0

           # enable registration and dt tabs
	    #activate_registration

	    #activate_dt
	
        } else {
           configure_readers all

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
           set $mods(ChooseNrrd-DT)-port-index 1

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
	    
        }
    }


    method configure_readers { which } {
        global mods
        global $mods(ChooseNrrd1)-port-index
	global $mods(ChooseNrrd-T2)-port-index
	global $mods(ChooseNrrd-ToProcess)-port-index
        global data_mode

	if {$which == "Nrrd"} {
	    set $mods(ChooseNrrd1)-port-index 0
	    set $mods(ChooseNrrd-T2)-port-index 0
	    set $mods(ChooseNrrd-ToProcess)-port-index 0

	    disableModule $mods(NrrdReader1) 0
	    disableModule $mods(NrrdReader-T2) 0

	    disableModule $mods(DicomToNrrd1) 1
	    disableModule $mods(DicomToNrrd-T2) 1

	    disableModule $mods(AnalyzeToNrrd1) 1
	    disableModule $mods(AnalyzeToNrrd-T2) 1

	    if {$initialized != 0} {
		$data_tab1 view "Nrrd"
		$data_tab2 view "Nrrd"
		set c_data_tab "Nrrd"
	    }
        } elseif {$which == "Dicom"} {
	    set $mods(ChooseNrrd1)-port-index 1
	    set $mods(ChooseNrrd-T2)-port-index 1
	    set $mods(ChooseNrrd-ToProcess)-port-index 1

	    disableModule $mods(NrrdReader1) 1
	    disableModule $mods(NrrdReader-T2) 1

	    disableModule $mods(DicomToNrrd1) 0
	    disableModule $mods(DicomToNrrd-T2) 0

	    disableModule $mods(AnalyzeToNrrd1) 1
	    disableModule $mods(AnalyzeToNrrd-T2) 1

            if {$initialized != 0} {
		$data_tab1 view "Dicom"
		$data_tab2 view "Dicom"
		set c_data_tab "Dicom"
	    }
        } elseif {$which == "Analyze"} {
	    # Analyze
	    set $mods(ChooseNrrd1)-port-index 2
	    set $mods(ChooseNrrd-T2)-port-index 2
	    set $mods(ChooseNrrd-ToProcess)-port-index 2

	    disableModule $mods(NrrdReader1) 1
	    disableModule $mods(NrrdReader-T2) 1

	    disableModule $mods(DicomToNrrd1) 1
	    disableModule $mods(DicomToNrrd-T2) 1

	    disableModule $mods(AnalyzeToNrrd1) 0
	    disableModule $mods(AnalyzeToNrrd-T2) 0

	    if {$initialized != 0} {
		$data_tab1 view "Analyze"
		$data_tab2 view "Analyze"
		set c_data_tab "Analyze"
	    }
        } elseif {$which == "all"} {
	    if {[set $mods(ChooseNrrd1)-port-index] == 0} {
		# nrrd
		disableModule $mods(NrrdReader1) 0
		disableModule $mods(NrrdReader-T2) 0
		
		disableModule $mods(DicomToNrrd1) 1
		disableModule $mods(DicomToNrrd-T2) 1
		
		disableModule $mods(AnalyzeToNrrd1) 1
		disableModule $mods(AnalyzeToNrrd-T2) 1
	    } elseif {[set $mods(ChooseNrrd1)-port-index] == 1} {
		# dicom
		disableModule $mods(NrrdReader1) 1
		disableModule $mods(NrrdReader-T2) 1
		
		disableModule $mods(DicomToNrrd1) 0
		disableModule $mods(DicomToNrrd-T2) 0
		
		disableModule $mods(AnalyzeToNrrd1) 1
		disableModule $mods(AnalyzeToNrrd-T2) 1
	    } else {
		# analyze
		disableModule $mods(NrrdReader1) 1
		disableModule $mods(NrrdReader-T2) 1
		
		disableModule $mods(DicomToNrrd1) 1
		disableModule $mods(DicomToNrrd-T2) 1
		
		disableModule $mods(AnalyzeToNrrd1) 0
		disableModule $mods(AnalyzeToNrrd-T2) 0
	    }
	}
    }



    method load_nrrd_dwi {} {
	global mods
        $mods(NrrdReader1) make_file_open_box
	
	tkwait window .ui$mods(NrrdReader1)-fb
	
	update idletasks

 	global $mods(NrrdReader1)-axis
	set $mods(NrrdReader1)-axis axis0
	
        #execute_Data
    }
    
    method load_nrrd_t2 {} {
	global mods
        $mods(NrrdReader-T2) make_file_open_box
	
	tkwait window .ui$mods(NrrdReader-T2)-fb

 	global $mods(NrrdReader-T2)-axis
	set $mods(NrrdReader-T2)-axis axis0
	
	update idletasks
	
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
	    disableModule $mods(TendEpireg) 0
	    disableModule $mods(UnuJoin) 0
	    disableModule $mods(ChooseNrrd-ToReg) 0
	    #disableModule $mods(RescaleColorMap2) 0
		
	    # activate reg variance checkbutton
	    $variance_tab1.reg configure -state normal
	    $variance_tab2.reg configure -state normal

	    # execute
	    $mods(TendEpireg)-c needexecute

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
		activate_widget $w
	    }
	    
	    foreach w [winfo children $reg_tab2] {
		activate_widget $w
	    }

	    $reg_tab1.last.ne configure -state normal \
		-foreground black -background $next_color
	    $reg_tab2.last.ne configure -state normal \
		-foreground black -background $next_color
	} else {
	    $reg_tab1.last.ne configure -state disabled \
		-foreground grey64 -background grey75
	    $reg_tab2.last.ne configure -state disabled \
		-foreground grey64 -background grey75

	}
	

        $ref_image1.s.ref configure -from 1 -to $volumes
        $ref_image2.s.ref configure -from 1 -to $volumes

    }


    

    method load_gradient {} {
        global mods
        $mods(NrrdReader-Gradient) make_file_open_box
	
        tkwait window .ui$mods(NrrdReader-Gradient)-fb
	
        update idletasks
	
 	global $mods(NrrdReader-Gradient)-axis
	set $mods(NrrdReader-Gradient)-axis axis0
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
    }


    method execute_DT {} {
       global mods
 
       # Check bmatrix has been loaded
       global $mods(NrrdReader-BMatrix)-filename
       global bmatrix

	if {$bmatrix == "load"} {
	    if {[set $mods(NrrdReader-BMatrix)-filename] == ""} {
		set answer [tk_messageBox -message \
				"Please load a B-Matrix file containing." -type ok -icon info -parent .standalone]
		return
	    }
	} 
	
	# unblock modules
	disableModule $mods(TendEstim) 0
	disableModule $mods(ChooseNrrd-DT) 0
	
	# unblock modules
	disableModule $mods(TendEstim) 0
	disableModule $mods(ChooseNrrd-DT) 0
	
	# execute
	$mods(ChooseNrrd-ToSmooth)-c needexecute
		
	view_Vis
    }


    method activate_dt { } {
	foreach w [winfo children $dt_tab1] {
	    activate_widget $w
        }

	foreach w [winfo children $dt_tab2] {
	    activate_widget $w
        }

        toggle_do_smoothing

        toggle_dt_threshold

        toggle_b_matrix

    }

    
    method load_bmatrix {} {
	global mods
        $mods(NrrdReader-BMatrix) make_file_open_box
	
	tkwait window .ui$mods(NrrdReader-BMatrix)-fb
	
	update idletasks
	
        global $mods(NrrdReader-BMatrix)-axis
        set $mods(NrrdReader-BMatrix)-axis 0
    } 
    
    method toggle_do_registration {} {
        global mods
        global $mods(ChooseNrrd-ToReg)-port-index
        global do_registration
	
	if {$do_registration == 1} {
	    disableModule $mods(TendEpireg) 0
	    disableModule $mods(UnuJoin) 0
	    
	    activate_registration

	    # change ChooseNrrd
	    set $mods(ChooseNrrd-ToReg)-port-index 0
        } else {
	    disableModule $mods(TendEpireg) 1
	    disableModule $mods(UnuJoin) 1
	    
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
	    set $mods(ChooseNrrd-ToReg)-port-index 1

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
	global $mods(ClipByFunction-Seeds)-clipfunction
	global $mods(Isosurface)-isoval
	set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	set span_z [expr [expr $plane_z*$spacing_z]+$min_z]

	set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $span_x) && (y $clip_y $span_y) && (z $clip_z $span_z)"
        
	
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
	
	if {$bmatrix == "compute"} {
            $dt_tab1.bm.childsite.load.e configure -state disabled \
                -foreground grey64
            $dt_tab1.bm.childsite.browse configure -state disabled
	    
            $dt_tab2.bm.childsite.load.e configure -state disabled \
                -foreground grey64
            $dt_tab2.bm.childsite.browse configure -state disabled
	} else {
            $dt_tab1.bm.childsite.load.e configure -state normal \
                -foreground black
            $dt_tab1.bm.childsite.browse configure -state normal
	    
            $dt_tab2.bm.childsite.load.e configure -state normal \
                -foreground black
            $dt_tab2.bm.childsite.browse configure -state normal
	}
    }

    method toggle_do_smoothing {} {
        global mods
        global $mods(ChooseNrrd-ToSmooth)-port-index
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

           set $mods(ChooseNrrd-ToSmooth)-port-index 1
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

           set $mods(ChooseNrrd-ToSmooth)-port-index 0

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
        if {$dt_completed} {
            # view planes tab
            $vis_tab1 view "Planes"
            $vis_tab2 view "Planes"
        } 
# 	else {
#             set answer [tk_messageBox -message \
#                  "Please finish constructing the Diffusion Tensors." -type ok -icon info -parent .standalone]
#         }
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
	
	# enable glyph modules but turn off stuff in viewer
	#disableModule $mods(NrrdToField-GlyphSeeds) 0
	#disableModule $mods(Probe-GlyphSeeds) 0
	#disableModule $mods(SampleField-GlyphSeeds) 0
	#disableModule $mods(DirectInterpolate-GlyphSeeds) 0
	#disableModule $mods(ChooseField-Glyphs) 0
	#$mods(Probe-GlyphSeeds)-c needexecute
	
	# enable fiber modules but turn off stuff in viewer
	#disableModule $mods(Probe-FiberSeeds) 0
	#disableModule $mods(SampleField-FiberSeeds) 0
	#disableModule $mods(DirectInterpolate-FiberSeeds) 0
	#disableModule $mods(ChooseField-Fibers) 0
	#$mods(Probe-FiberSeeds)-c needexecute
	
	#$mods(Viewer)-ViewWindow_0-c autoview
	#global $mods(Viewer)-ViewWindow_0-pos
	#set $mods(Viewer)-ViewWindow_0-pos "z0_y0"
	#$mods(Viewer)-ViewWindow_0-c Views
	
	uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
	uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (7)\}" 0
	uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
	uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (12)\}" 0
	
	$mods(Viewer)-ViewWindow_0-c redraw
	
	# setup global clipping planes
	initialize_clip_info
	
	change_indicator_labels "Visualization..."
	
	configure_planes_tabs
	
	configure_isosurface_tabs
	
	configure_glyphs_tabs
	
	configure_fibers_tabs
	
	set vis_activated 1

	# bring planes tab forward
	view_Vis
    }

##########################################################################	
######################### VISUALIZATION STEPS ############################
##########################################################################



######## VARIANCE #########
    method build_variance_tab { f } {
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
	       
	       checkbutton $f.reg -text "View Variance of Registered Data" \
		   -variable $mods(ShowField-Reg)-faces-on \
		   -state disabled \
		   -command {
		       global mods
		       $mods(ShowField-Reg)-c toggle_display_faces
                   }
	       
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
	$variance_tab1.slice configure -from 0 -to [expr $size_z - 1]
	$variance_tab2.slice configure -from 0 -to [expr $size_z - 1]

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
	if {$data_completed} {
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
	global mods
	global show_planes
	global show_plane_x show_plane_y show_plane_z
	global plane_x plane_y plane_z

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Planes:" -variable show_planes \
		-command "$this toggle_show_planes" -state disabled
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    frame $f.axis -relief groove -borderwidth 2
	    pack $f.axis -side top -anchor n -padx 3 -pady 3
	    
	    frame $f.axis.x
	    pack $f.axis.x -side top -anchor nw 
	    
	    checkbutton $f.axis.x.check -text "X" \
		-variable show_plane_x \
		-state disabled \
		-command "$this toggle_plane X"
	    scale $f.axis.x.slider -from 0 -to 512 \
		-variable plane_x \
		-showvalue false \
		-length 200  -width 15 \
		-sliderlength 15 \
		-state disabled -foreground grey64 \
		-orient horizontal  
	    bind $f.axis.x.slider <ButtonRelease> "app update_plane_x"
	    label $f.axis.x.label -textvariable plane_x -state disabled
	    pack $f.axis.x.check $f.axis.x.slider $f.axis.x.label -side left -anchor nw \
		-padx 2 -pady 3
	    
	    frame $f.axis.y
	    pack $f.axis.y -side top -anchor nw 
	    checkbutton $f.axis.y.check -text "Y" \
		-variable show_plane_y \
		-state disabled \
		-command "$this toggle_plane Y"
	    scale $f.axis.y.slider -from 0 -to 512 \
		-variable plane_y \
		-showvalue false \
		-length 200  -width 15 \
		-sliderlength 15 \
		-state disabled -foreground grey64 \
		-orient horizontal 
	    bind $f.axis.y.slider <ButtonRelease> "app update_plane_y"
	    label $f.axis.y.label -textvariable plane_y -state disabled
	    pack $f.axis.y.check $f.axis.y.slider $f.axis.y.label -side left -anchor nw \
		-padx 2 -pady 3
	    
	    frame $f.axis.z
	    pack $f.axis.z -side top -anchor nw 
	    checkbutton $f.axis.z.check -text "Z" \
		-variable show_plane_z \
		-state disabled \
		-command "$this toggle_plane Z"
	    scale $f.axis.z.slider -from 0 -to 512 \
		-variable plane_z \
		-showvalue false \
		-length 200  -width 15 \
		-sliderlength 15 \
		-state disabled -foreground grey64 \
		-orient horizontal 
	    bind $f.axis.z.slider <ButtonRelease> "app update_plane_z"
	    label $f.axis.z.label -textvariable plane_z -state disabled
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
		-command "$this select_color_planes_color $fr.select" \
		-state disabled
	    pack $fr.select.color -side left -anchor n -padx 3 -pady 3
	    
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-ColorPlanes)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-ColorPlanes)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-ColorPlanes)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-ColorPlanes)-c needexecute"
	    pack $maps.blackbody.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.blackbody.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.blackbody.f.canvas -anchor e
	    
	    draw_colormap Blackbody $maps.blackbody.f.canvas
	    
	    
	    # BP Seismic
	    frame $maps.bpseismic
	    pack $maps.bpseismic -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.bpseismic.b -text "BP Seismic" \
		-variable $mods(GenStandardColorMaps-ColorPlanes)-mapType \
		-value 17 \
		-state disabled \
		-command "$mods(GenStandardColorMaps-ColorPlanes)-c needexecute"
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side left -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "BP Seismic" $maps.bpseismic.f.canvas
	    
	    global clip_to_isosurface
	    global clip_to_isosurface_color
	    checkbutton $f.clipiso -text "Clip to Isosurface" \
		-variable clip_to_isosurface \
		-command "$this toggle_clip_to_isosurface" -state disabled
	    pack $f.clipiso -side top -anchor nw -padx 5 -pady 5

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
	    # FIX ME
	    set plane_x [expr $size_x/2]
	    set plane_y [expr $size_y/2]
	    set plane_z [expr $size_z/2]
	}

	global $mods(ChooseField-ColorPlanes)-port-index
	set port [set $mods(ChooseField-ColorPlanes)-port-index]

	if {$port == 0} {
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
	global $mods(ChooseField-ColorPlanes)-port-index
	global $mods(ChooseColorMap-Planes)-port-index

	set port [set $mods(ChooseField-ColorPlanes)-port-index]
	set color_port [set $mods(ChooseColorMap-Planes)-port-index]

	foreach w [winfo children $planes_tab1] {
	    activate_widget $w
	}
	foreach w [winfo children $planes_tab2] {
	    activate_widget $w
	}

	if { $color_port == 1 && $port != 3} {
	    # rescale is disabled and it isn't Principle Eigenvector
	    $planes_tab1.color.childsite.select.colorFrame.set_color configure -state normal
	    $planes_tab2.color.childsite.select.colorFrame.set_color configure -state normal
	} else {
	    $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
	    $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	}
    }

    method select_color_planes_color { w } {
        global mods
	global $mods(ChooseField-ColorPlanes)-port-index
	global $mods(ChooseColorMap-Planes)-port-index
	global $mods(ShowField-X)-tensor-usedefcolor
	global $mods(ShowField-Y)-tensor-usedefcolor
	global $mods(ShowField-Z)-tensor-usedefcolor
	
        set which [$w.color get]

	set $mods(ShowField-X)-tensor-usedefcolor 0
	set $mods(ShowField-Y)-tensor-usedefcolor 0
	set $mods(ShowField-Z)-tensor-usedefcolor 0
	
        if {$which == "Principle Eigenvector"} {
	    set plane_type "Principle Eigenvector"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    #disableModule $mods(RescaleColorMap-ColorPlanes) 1
	    set $mods(ChooseColorMap-Planes)-port-index 1
	    set $mods(ChooseField-ColorPlanes)-port-index 3
        } elseif {$which == "Fractional Anisotropy"} {
	    set plane_type "Fractional Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    #disableModule $mods(RescaleColorMap-ColorPlanes) 0
	    set $mods(ChooseColorMap-Planes)-port-index 0
	    set $mods(ChooseField-ColorPlanes)-port-index 0
        } elseif {$which == "Linear Anisotropy"} {
	    set plane_type "Linear Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    #disableModule $mods(RescaleColorMap-ColorPlanes) 0
	    set $mods(ChooseColorMap-Planes)-port-index 0
	    set $mods(ChooseField-ColorPlanes)-port-index 1
        } elseif {$which == "Planar Anisotropy"} {
	    set plane_type "Planar Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    #disableModule $mods(RescaleColorMap-ColorPlanes) 0
	    set $mods(ChooseColorMap-Planes)-port-index 0
	    set $mods(ChooseField-ColorPlanes)-port-index 2
        } else {
	    set plane_type "Constant"
	    # specified color
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state normal
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state normal
	    #disableModule $mods(RescaleColorMap-ColorPlanes) 1
	    set $mods(ChooseColorMap-Planes)-port-index 1
	    set $mods(ChooseField-ColorPlanes)-port-index 0

	    set $mods(ShowField-X)-tensor-usedefcolor 1
	    set $mods(ShowField-Y)-tensor-usedefcolor 1
	    set $mods(ShowField-Z)-tensor-usedefcolor 1
        }

	$planes_tab1.color.childsite.select.color select $which
	$planes_tab2.color.childsite.select.color select $which
	
        # execute 
        $mods(ChooseField-ColorPlanes)-c needexecute
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
           global $mods(ClipByFunction-Seeds)-clipfunction
           global $mods(Isosurface)-isoval
	    set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	    set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	    set span_z [expr [expr $plane_z*$spacing_z]+$min_z]
	    
	    set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $span_x) && (y $clip_y $span_y) && (z $clip_z $span_z)"
	    
	    
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
           global $mods(ClipByFunction-Seeds)-clipfunction
           global $mods(Isosurface)-isoval
	    set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	    set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	    set span_z [expr [expr $plane_z*$spacing_z]+$min_z]
	    
	    set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $span_x) && (y $clip_y $span_y) && (z $clip_z $span_z)"
	    

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
           global $mods(ClipByFunction-Seeds)-clipfunction
           global $mods(Isosurface)-isoval
	    set span_x [expr [expr $plane_x*$spacing_x]+$min_x]
	    set span_y [expr [expr $plane_y*$spacing_y]+$min_y]
	    set span_z [expr [expr $plane_z*$spacing_z]+$min_z]
	    
	    set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $span_x) && (y $clip_y $span_y) && (z $clip_z $span_z)"


           $mods(Viewer)-ViewWindow_0-c redraw
        }
    }

    method toggle_clip_to_isosurface {} {
       global mods
       global clip_to_isosurface
       global $mods(ChooseField-X)-port-index
       global $mods(ChooseField-Y)-port-index
       global $mods(ChooseField-Z)-port-index

       if {$clip_to_isosurface == 1} {
	# enable Unstructure modules and change ChooseField port to 1
        #disableModule $mods(QuadToTri-X) 0
        #disableModule $mods(QuadToTri-Y) 0
        #disableModule $mods(QuadToTri-Z) 0
 
        set $mods(ChooseField-X)-port-index 1
        set $mods(ChooseField-Y)-port-index 1
        set $mods(ChooseField-Z)-port-index 1
       } else {
	# disable Unstructure modules and change ChooseField port to 0
        #disableModule $mods(QuadToTri-X) 1
        #disableModule $mods(QuadToTri-Y) 1
        #disableModule $mods(QuadToTri-Z) 1
 
        set $mods(ChooseField-X)-port-index 0
        set $mods(ChooseField-Y)-port-index 0
        set $mods(ChooseField-Z)-port-index 0
       }

       # re-execute
       $mods(ChooseField-ColorPlanes)-c needexecute
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
          global $mods(ClipByFunction-Seeds)-clipfunction
          global $mods(Isosurface)-isoval
	  
	  set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $span_x) && (y $clip_y $span_y) && (z $clip_z $span_z)"
         

          $mods(SamplePlane-X)-c needexecute
          $mods(Viewer)-ViewWindow_0-c redraw
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
          global $mods(Isosurface)-isoval
          global $mods(ClipByFunction-Seeds)-clipfunction
	  
	  set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $span_x) && (y $clip_y $span_y) && (z $clip_z $span_z)"
          

          $mods(SamplePlane-Y)-c needexecute
          $mods(Viewer)-ViewWindow_0-c redraw
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
          global $mods(Isosurface)-isoval
          global $mods(ClipByFunction-Seeds)-clipfunction

	set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $span_x) && (y $clip_y $span_y) && (z $clip_z $span_z)"

          $mods(SamplePlane-Z)-c needexecute
          $mods(Viewer)-ViewWindow_0-c redraw
       }
    }

    method toggle_plane { which } {
       global mods
       global show_plane_x show_plane_y show_plane_z
       global $mods(ShowField-X)-faces-on
       global $mods(ShowField-Y)-faces-on
       global $mods(ShowField-Z)-faces-on
       global $mods(Viewer)-ViewWindow_0-clip
       set clip $mods(Viewer)-ViewWindow_0-clip


       # turn off showfields and configure global clipping planes

       if {$which == "X"} {
          global $clip-visible-$last_x
          if {$show_plane_x == 0} {
              # turn off 
              set $mods(ShowField-X)-faces-on 0
              set $clip-visible-$last_x 0
          } else {
              set $mods(ShowField-X)-faces-on 1
              set $clip-visible-$last_x 1
          }  
          $mods(ShowField-X)-c toggle_display_faces  
          $mods(Viewer)-ViewWindow_0-c redraw
       } elseif {$which == "Y"} {
          global $clip-visible-$last_y
          if {$show_plane_y == 0} {
              set $mods(ShowField-Y)-faces-on 0
              set $clip-visible-$last_y 0             
          } else {
              set $mods(ShowField-Y)-faces-on 1
              set $clip-visible-$last_y 1              
          }   
          $mods(ShowField-Y)-c toggle_display_faces
          $mods(Viewer)-ViewWindow_0-c redraw
       } else {
	   # Z plane
          global $clip-visible-$last_z
          if {$show_plane_z == 0} {
              set $mods(ShowField-Z)-faces-on 0
              set $clip-visible-$last_z 0              
          } else {
              set $mods(ShowField-Z)-faces-on 1
              set $clip-visible-$last_z 1             
          }   

          $mods(ShowField-Z)-c toggle_display_faces
          $mods(Viewer)-ViewWindow_0-c redraw
       }
    }

    method toggle_plane_y {} {
       global mods
       global show_plane_y
       global $mods(ShowField-Y)-faces-on

       if {$show_plane_y == 0} {
           set $mods(ShowField-Y)-faces-on 0
       } else {
           set $mods(ShowField-X)-faces-on 1
       }     
 
       # execute showfield
       $mods(ShowField-X)-c toggle_display_faces
    }

    method toggle_plane_x {} {
       global mods
       global show_plane_x
       global $mods(ShowField-X)-faces-on

       if {$show_plane_x == 0} {
           set $mods(ShowField-X)-faces-on 0
       } else {
           set $mods(ShowField-X)-faces-on 1
       }     
 
       # execute showfield
       $mods(ShowField-X)-c toggle_display_faces
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

         $mods(ChooseField-ColorPlanes)-c needexecute
         $mods(Viewer)-ViewWindow_0-c redraw
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
         $mods(ChooseField-ColorPlanes)-c needexecute
         $mods(Viewer)-ViewWindow_0-c redraw
      }
    }



######## ISOSURFACE #########

    method build_isosurface_tab { f } {
	global mods
	global $mods(ShowField-Isosurface)-faces-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Isosurface" \
		-variable $mods(ShowField-Isosurface)-faces-on \
		-command "$this toggle_show_isosurface" -state disabled
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    # Isoval
	    frame $f.isoval
	    pack $f.isoval -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.isoval.l -text "Isoval:" -state disabled -state disabled
	    scale $f.isoval.s -from 0.0 -to 1.0 \
		-length 200 -width 15 \
		-sliderlength 15 \
		-resolution 0.0001 \
		-variable $mods(Isosurface)-isoval \
		-showvalue false \
		-state disabled -foreground grey64\
		-orient horizontal \
		-command "$this update_isovals"
	    
	    bind $f.isoval.s <ButtonRelease> "$this execute_isoval_change"
	    
	    label $f.isoval.val -textvariable $mods(Isosurface)-isoval -state disabled
	    
	    pack $f.isoval.l $f.isoval.s $f.isoval.val -side left -anchor nw -padx 3      
	    
	    iwidgets::optionmenu $f.isovalcolor -labeltext "Isoval Based On:" \
		-labelpos w \
		-state disabled \
		-command "$this select_isoval_based_on $f"
	    pack $f.isovalcolor -side top -anchor nw -padx 3 -pady 5
	    
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
		-state disabled \
		-command "$this select_isosurface_color $isocolor.select"
	    pack $isocolor.select.color -side left -anchor n -padx 3 -pady 3
	    
	    
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-Isosurface)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-Isosurface)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-Isosurface)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-Isosurface)-c needexecute"
	    pack $maps.blackbody.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.blackbody.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.blackbody.f.canvas -anchor e
	    
	    draw_colormap Blackbody $maps.blackbody.f.canvas
	    
	    # BP Seismic
	    frame $maps.bpseismic
	    pack $maps.bpseismic -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.bpseismic.b -text "BP Seismic" \
		-variable $mods(GenStandardColorMaps-Isosurface)-mapType \
		-value 17 \
		-state disabled \
		-command "$mods(GenStandardColorMaps-Isosurface)-c needexecute"
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "BP Seismic" $maps.bpseismic.f.canvas
	    
	    
	    global clip_by_planes
	    frame $f.clip
	    pack $f.clip -side top -anchor nw -padx 3 -pady 5
	    
	    checkbutton $f.clip.check -text "Clip by Planes" \
		-variable clip_by_planes -state disabled \
		-command "$this toggle_clip_by_planes $f.clip"
	    
	    button $f.clip.flipx -text "Flip X" \
		-command "$this flip_x_clipping_plane" \
		-state disabled
	    button $f.clip.flipy -text "Flip Y" \
		-command "$this flip_y_clipping_plane" \
		-state disabled
	    button $f.clip.flipz -text "Flip Z" \
		-command "$this flip_z_clipping_plane" \
		-state disabled
	    
	    pack $f.clip.check $f.clip.flipx $f.clip.flipy $f.clip.flipz \
		-side left -anchor nw -padx 3 -pady 3 -ipadx 2 
	} 
    }
    

    method sync_isosurface_tabs {} {
	global mods
	global $mods(ChooseField-Isoval)-port-index
	set port [set $mods(ChooseField-Isoval)-port-index]

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

	global $mods(ChooseField-Isosurface)-port-index
	set port [set $mods(ChooseField-Isosurface)-port-index]
	global $mods(ChooseColorMap-Isosurface)-port-index
	set color_port [set $mods(ChooseColorMap-Isosurface)-port-index]


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
	    if {[set $mods(ShowField-Isosurface)-faces-on] == 1} {
		foreach w [winfo children $isosurface_tab1] {
		    activate_widget $w
		}
		foreach w [winfo children $isosurface_tab2] {
		    activate_widget $w
		}
		
		# configure color button
		if {$plane_type == "Constant"} {
		    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state normal
		    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state normal
		} else {
		    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
		    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled
		}
	    } else {
		foreach w [winfo children $isosurface_tab1] {
		    disable_widget $w
		}
		foreach w [winfo children $isosurface_tab2] {
		    disable_widget $w
		}
		$isosurface_tab1.show configure -state normal -foreground black
		$isosurface_tab2.show configure -state normal -foreground black
		
		$isosurface_tab1.clip.check configure -state normal -foreground black
		$isosurface_tab2.clip.check configure -state normal -foreground black
	    }
	    
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

    method toggle_show_isosurface {} {
       global mods
       global $mods(ShowField-Isosurface)-faces-on
 
	if {[set $mods(ShowField-Isosurface)-faces-on] == 1} {
	    #disableModule $mods(DirectInterpolate-Isosurface) 0
	} else {
	    #disableModule $mods(DirectInterpolate-Isosurface) 1
	}
	
	configure_isosurface_tabs
	
	$mods(ShowField-Isosurface)-c toggle_display_faces
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
       	global $mods(ChooseField-Isosurface)-port-index
	global $mods(ChooseColorMap-Isosurface)-port-index
	
	set which [$w.color get]

	#disableModule $mods(ChooseField-Isosurface) 0
	
        if {$which == "Principle Eigenvector"} {
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    #disableModule $mods(RescaleColorMap-Isosurface) 1
	    set $mods(ChooseColorMap-Isosurface)-port-index 1
	    set $mods(ChooseField-Isosurface)-port-index 3
        } elseif {$which == "Fractional Anisotropy"} {
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	    
  	    #disableModule $mods(RescaleColorMap-Isosurface) 0
	    set $mods(ChooseColorMap-Isosurface)-port-index 0
	    set $mods(ChooseField-Isosurface)-port-index 0
        } elseif {$which == "Linear Anisotropy"} {
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	   
	    #disableModule $mods(RescaleColorMap-Isosurface) 0
	    set $mods(ChooseColorMap-Isosurface)-port-index 0
	    set $mods(ChooseField-Isosurface)-port-index 1
        } elseif {$which == "Planar Anisotropy"} {
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	    
	    #disableModule $mods(RescaleColorMap-Isosurface) 0
	    set $mods(ChooseColorMap-Isosurface)-port-index 0
	    set $mods(ChooseField-Isosurface)-port-index 2
        } else {
	    # constant color
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state normal
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state normal	   
	    #disableModule $mods(RescaleColorMap-Isosurface) 1
	    set $mods(ChooseColorMap-Isosurface)-port-index 1
	    set $mods(ChooseField-Isosurface)-port-index 0
        }

	$isosurface_tab1.isocolor.childsite.select.color select $which
	$isosurface_tab2.isocolor.childsite.select.color select $which
	
        # execute 
        $mods(ChooseField-Isosurface)-c needexecute
    }
    

    method select_isoval_based_on { w } {
	global mods
       	global $mods(ChooseField-Isoval)-port-index
	
	set which [$w.isovalcolor get]
	
        if {$which == "Fractional Anisotropy"} {
	    set $mods(ChooseField-Isoval)-port-index 0
        } elseif {$which == "Linear Anisotropy"} {
	    set $mods(ChooseField-Isoval)-port-index 1
        } else {
	    # Planar Anisotropy
	    set $mods(ChooseField-Isoval)-port-index 2
        } 

	$isosurface_tab1.isovalcolor select $which
	$isosurface_tab2.isovalcolor select $which
	
        # execute 
        $mods(ChooseField-Isoval)-c needexecute
    }
    

    method execute_isoval_change {} {
	global mods
	
	if {$vis_activated} {
	    $mods(Isosurface)-c needexecute
	    $mods(IsoClip-X)-c needexecute
	    $mods(IsoClip-Y)-c needexecute
	    $mods(IsoClip-Z)-c needexecute
	}
    }




########### GLYPHS ############
    
    method build_glyphs_tab { f } {
	global mods
        global $mods(ShowField-Glyphs)-tensors-on
	
	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Glyphs" \
		-variable $mods(ShowField-Glyphs)-tensors-on \
		-command "$this toggle_show_glyphs" -state disabled
	    
	    pack $f.show -side top -anchor nw -padx 3 -pady 3	
	    
	    # Seed at
	    iwidgets::labeledframe $f.seed \
		-labeltext "Seed At" \
		-labelpos nw -foreground grey64
	    pack $f.seed -side top -anchor nw -padx 3 -pady 0 \
		-fill x
	    
	    set seed [$f.seed childsite]
	    
	    global $mods(ChooseField-GlyphSeeds)-port-index
	    frame $seed.a
	    pack $seed.a -side left -anchor n -padx 10
	    radiobutton $seed.a.point -text "Single Point" \
		-variable $mods(ChooseField-GlyphSeeds)-port-index \
		-value 0 \
		-state disabled \
		-command "$this update_glyph_seed_method"
	    
	    radiobutton $seed.a.rake -text "Along Line" \
		-variable $mods(ChooseField-GlyphSeeds)-port-index \
		-value 1 \
		-state disabled \
		-command "$this update_glyph_seed_method"
	    
	    frame $seed.b
	    pack $seed.b -side right -anchor n -padx 10
	    radiobutton $seed.b.plane -text "On Planes" \
		-variable $mods(ChooseField-GlyphSeeds)-port-index \
		-value 2 \
		-state disabled \
		-command "$this update_glyph_seed_method"
	    
	    radiobutton $seed.b.grid -text "On Grid" \
		-variable $mods(ChooseField-GlyphSeeds)-port-index \
		-value 3 \
		-state disabled \
		-command "$this update_glyph_seed_method"
	    
	    pack $seed.a.point $seed.a.rake  -side top \
		-anchor nw -padx 3 -pady 1
	    
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
		-state disabled \
		-command "$this change_glyph_display_type radio $rep"
	    
	    iwidgets::optionmenu $rep.f1.type -labeltext "" \
		-width 180 -state disabled \
		-command "$this change_glyph_display_type men $rep.f1"
	    pack $rep.f1.boxes $rep.f1.type -side left -anchor nw -padx 2 -pady 0
	    
	    $rep.f1.type insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant" "RGB"
	    $rep.f1.type select "Principle Eigenvector"
	    
	    frame $rep.f2
	    pack $rep.f2 -side top -anchor nw -padx 3 -pady 1
	    
	    radiobutton $rep.f2.ellips -text "Ellipsoids" \
		-variable glyph_display_type \
		-value ellipsoids \
		-state disabled \
		-command "$this change_glyph_display_type radio $rep"
	    
	    iwidgets::optionmenu $rep.f2.type -labeltext "" \
		-width 180 \
		-command "$this change_glyph_display_type men $rep.f2" \
		-state disabled
	    pack $rep.f2.ellips $rep.f2.type -side left -anchor nw -padx 2 -pady 0
	    
	    $rep.f2.type insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant"
	    
	    $rep.f2.type select "Principle Eigenvector"
	    
	    global glyph_color
	    frame $rep.select
	    pack $rep.select -side top -anchor n -padx 3 -pady 3
	    addColorSelection $rep.select "Color" glyph_color "glyph_color_change"
	    
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-Glyphs)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-Glyphs)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-Glyphs)-c needexecute"
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
		-state disabled \
		-command "$mods(GenStandardColorMaps-Glyphs)-c needexecute"
	    pack $maps.blackbody.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.blackbody.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.blackbody.f.canvas -anchor e
	    
	    draw_colormap Blackbody $maps.blackbody.f.canvas
	    
	    
	    # BP Seismic
	    frame $maps.bpseismic
	    pack $maps.bpseismic -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.bpseismic.b -text "BP Seismic" \
		-variable $mods(GenStandardColorMaps-Glyphs)-mapType \
		-value 17 \
		-state disabled \
		-command "$mods(GenStandardColorMaps-Glyphs)-c needexecute"
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side left -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "BP Seismic" $maps.bpseismic.f.canvas
	    
	    
	    global scale_glyph
	    global $mods(TendNorm-Glyphs)-target
	    global $mods(ShowField-Glyphs)-tensors_scale
	    
	    frame $f.scale 
	    pack $f.scale -side top -anchor nw -padx 3 -pady 0
	    
	    checkbutton $f.scale.b -text "Glyph Size:          " \
		-variable scale_glyph \
		-state disabled \
		-command "$this toggle_scale_glyph"
	    
	    scale $f.scale.s -from 0.2 -to 5.0 \
                -resolution 0.01 \
  		-length 150  -width 15 \
		-sliderlength 15 \
                -orient horizontal \
   	        -state disabled \
   	        -foreground grey64 \
	        -variable $mods(ShowField-Glyphs)-tensors_scale
	    bind $f.scale.s <ButtonRelease> {app change_glyph_scale}
	    
	    pack $f.scale.b -side left -anchor w -padx 3 -pady 0
	    pack  $f.scale.s -side left -anchor ne -padx 3 -pady 0
	    
	    global exag_glyph
	    global $mods(TendAnscale-Glyphs)-scale
	    
	    frame $f.exag 
	    pack $f.exag -side top -anchor nw -padx 3 -pady 0
	    
	    checkbutton $f.exag.b -text "Shape Exaggerate:" \
		-variable exag_glyph \
		-state disabled \
		-command "$this toggle_exag_glyph"
	    
	    scale $f.exag.s -from 0.2 -to 5.0 \
                -resolution 0.01 \
  		-length 150  -width 15 \
		-sliderlength 15 \
                -orient horizontal \
   	        -state disabled \
   	        -foreground grey64 \
	        -variable $mods(TendAnscale-Glyphs)-scale
	    bind $f.exag.s <ButtonRelease> {app change_glyph_exag}
	    
	    pack $f.exag.b -side left -anchor w -padx 3 -pady 0
	    pack  $f.exag.s -side left -anchor ne -padx 3 -pady 0
	    
	    message $f.exagm -text "A value less than 1.0 will make the glyphs more isotropic while a value greater than 1.0 will make them more anisotropic.  Setting the value to 1.0 will not change the glyphs." -width 300 -foreground grey64
	    
	    pack $f.exagm -side top -anchor n -padx 3 -pady 0
	} 
    }

    method sync_glyphs_tabs {} {
	global mods
	global glyph_display_type
	global $mods(ChooseField-Glyphs)-port-index
	global $mods(ShowField-Glyphs)-data_display_type
	global $mods(ChooseField-Glyphs)-port-index

	set port [set $mods(ChooseField-Glyphs)-port-index]
	set color_port [set $mods(ChooseField-Glyphs)-port-index]

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
	} elseif {$port == 0} {
	    #FA - set option menu to Fractional Anisotropy disable Color button
	    $glyphs_tab1.rep.childsite.f1.type select "Fractional Anisotropy"
	    $glyphs_tab2.rep.childsite.f1.type select "Fractional Anisotropy"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Fractional Anisotropy"
	    $glyphs_tab2.rep.childsite.f2.type select "Fractional Anisotropy"
	} elseif {$port == 1} {
	    #LA -set optionmenu to LA and disable Color button
	    $glyphs_tab1.rep.childsite.f1.type select "Linear Anisotropy"
	    $glyphs_tab2.rep.childsite.f1.type select "Linear Anisotropy"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Linear Anisotropy"
	    $glyphs_tab2.rep.childsite.f2.type select "Linear Anisotropy"
	    
	} elseif {$port == 2} {
	    #PA - set option menu to pa and disable color button
	    $glyphs_tab1.rep.childsite.f1.type select "Planar Anisotropy"
	    $glyphs_tab2.rep.childsite.f1.type select "Planar Anisotropy"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Planar Anisotropy"
	    $glyphs_tab2.rep.childsite.f2.type select "Planar Anisotropy"
	} elseif {$port == 3} {
	    #e1 - set option menu to e1 and disable color button
	    $glyphs_tab1.rep.childsite.f1.type select "Principle Eigenvector"
	    $glyphs_tab2.rep.childsite.f1.type select "Principle Eigenvector"
	    
	    $glyphs_tab1.rep.childsite.f2.type select "Principle Eigenvector"
	    $glyphs_tab2.rep.childsite.f2.type select "Principle Eigenvector"
	} 
    }

    method configure_glyphs_tabs {} {
        global mods
	global $mods(ShowField-Glyphs)-tensors-on        
	global glyph_display_type
	global scale_glyph exag_glyph

        if {[set $mods(ShowField-Glyphs)-tensors-on] == 1} {
	    foreach w [winfo children $glyphs_tab1] {
		activate_widget $w
	    }
	    foreach w [winfo children $glyphs_tab2] {
		activate_widget $w
	    }

            # configure boxes/ellipsoids optionmenus
	    if {$glyph_display_type == "boxes"} {
		$glyphs_tab1.rep.childsite.f1.type configure -state normal
		$glyphs_tab2.rep.childsite.f1.type configure -state normal
		$glyphs_tab1.rep.childsite.f2.type configure -state disabled
		$glyphs_tab2.rep.childsite.f2.type configure -state disabled
	    } else {
		$glyphs_tab1.rep.childsite.f1.type configure -state disabled
		$glyphs_tab2.rep.childsite.f1.type configure -state disabled
		$glyphs_tab1.rep.childsite.f2.type configure -state normal
		$glyphs_tab2.rep.childsite.f2.type configure -state normal
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
	    } else {
		$glyphs_tab1.rep.childsite.select.colorFrame.set_color configure -state disabled
		$glyphs_tab2.rep.childsite.select.colorFrame.set_color configure -state disabled
	    }
        } else {
	    foreach w [winfo children $glyphs_tab1] {
		disable_widget $w
	    }
	    foreach w [winfo children $glyphs_tab2] {
		disable_widget $w
	    }

	    # activate checkbox
	    $glyphs_tab1.show configure -state normal -foreground black
	    $glyphs_tab2.show configure -state normal -foreground black
	}
	

    }

    method toggle_scale_glyph {} {
	global mods
        global $mods(ChooseNrrd-Norm)-port-index
        global scale_glyph

	if {$scale_glyph == 0} {
	   $glyphs_tab1.scale.s configure -state disabled -foreground grey64
	   $glyphs_tab2.scale.s configure -state disabled -foreground grey64

	   set $mods(ChooseNrrd-Norm)-port-index 1

           $mods(ChooseNrrd-Norm)-c needexecute
        } else {
	   $glyphs_tab1.scale.s configure -state normal -foreground black
	   $glyphs_tab2.scale.s configure -state normal -foreground black

	   set $mods(ChooseNrrd-Norm)-port-index 0

           $mods(TendNorm-Glyphs)-c needexecute
        }

    }

    method toggle_exag_glyph {} {
	global mods
        global $mods(ChooseNrrd-Exag)-port-index
        global exag_glyph

	if {$exag_glyph == 0} {
	   $glyphs_tab1.exag.s configure -state disabled -foreground grey64
	   $glyphs_tab2.exag.s configure -state disabled -foreground grey64

	   set $mods(ChooseNrrd-Exag)-port-index 1
           $mods(ChooseNrrd-Exag)-c needexecute
        } else {
	   $glyphs_tab1.exag.s configure -state normal -foreground black
	   $glyphs_tab2.exag.s configure -state normal -foreground black

	   set $mods(ChooseNrrd-Exag)-port-index 0
           $mods(TendAnscale-Glyphs)-c needexecute
        }
    }


    method change_glyph_display_type { change w } {
	global glyph_display_type
        global mods
        global $mods(ShowField-Glyphs)-tensor_display_type
        global $mods(ChooseField-Glyphs)-port-index
	global $mods(ChooseColorMap-Glyphs)-port-index
	global $mods(ShowField-Glyphs)-tensor-usedefcolor
	
        set type ""
	
	set $mods(ShowField-Glyphs)-tensor-usedefcolor 0
        if {$change == "radio"} {
	    # radio button changed  
	    if {$glyph_display_type == "boxes"} {
		set type [$w.f1.type get]
	    } else {
		set type [$w.f2.type get]
	    }       	
        } else {
	    # optionmenu changed
	    set type [$w.type get]
        }
	
        # configure display type
        if {$glyph_display_type == "ellipsoids"} {
	    set $mods(ShowField-Glyphs)-tensor_display_type Ellipsoids
        } else {
	    # determine if normal boxes or colored boxes
	    if {$type == "RGB"} {
                set glyph_type "RGB"
		set $mods(ShowField-Glyphs)-tensor_display_type "Colored Boxes"
   	        #disableModule $mods(RescaleColorMap-Glyphs) 1
		set $mods(ChooseColorMap-Glyphs)-port-index 1
		set $mods(ChooseColorMap-Glyphs)-port-index 0
	    } else {
		set $mods(ShowField-Glyphs)-tensor_display_type Boxes
	    }
	}

	# configure color
	if {$type == "Principle Eigenvector"} {
	    set glyph_type "Principle Eigenvector"
	    set $mods(ChooseField-Glyphs)-port-index 3
	    #disableModule $mods(RescaleColorMap-Glyphs) 1
	    set $mods(ChooseColorMap-Glyphs)-port-index 1
	} elseif {$type == "Fractional Anisotropy"} {
	    set glyph_type "Fractional Anisotropy"
	    set $mods(ChooseField-Glyphs)-port-index 0
	    #disableModule $mods(RescaleColorMap-Glyphs) 0
	    set $mods(ChooseColorMap-Glyphs)-port-index 0
	} elseif {$type == "Linear Anisotropy"} {
	    set glyph_type "Linear Anisotropy"
	    set $mods(ChooseField-Glyphs)-port-index 1
	    #disableModule $mods(RescaleColorMap-Glyphs) 0
	    set $mods(ChooseColorMap-Glyphs)-port-index 0
	} elseif {$type == "Planar Anisotropy"} {
	    set glyph_type "Planar Anisotropy"
	    set $mods(ChooseField-Glyphs)-port-index 2
	    #disableModule $mods(RescaleColorMap-Glyphs) 0
	    set $mods(ChooseColorMap-Glyphs)-port-index 0
	} elseif {$type == "Constant"} {
	    set glyph_type "Constant"
	    #disableModule $mods(RescaleColorMap-Glyphs) 1
	    set $mods(ChooseColorMap-Glyphs)-port-index 1
	    set $mods(ShowField-Glyphs)-tensor-usedefcolor 1
	    set $mods(ChooseColorMap-Glyphs)-port-index 0
	}
	
	# sync attached/detached optionmenus
	configure_glyphs_tabs

	$mods(ShowField-Glyphs)-c data_display_type
	$mods(ChooseField-Glyphs)-c needexecute
    }

    method update_glyph_seed_method {} {
        global mods
        global $mods(ChooseField-GlyphSeeds)-port-index

        if {[set $mods(ChooseField-GlyphSeeds)-port-index] == 0} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 1
        } elseif {[set $mods(ChooseField-GlyphSeeds)-port-index] == 1} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (7)\}" 1
        } else {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (7)\}" 0
        }
	
        $mods(ChooseField-GlyphSeeds)-c needexecute
	
	after 100 "$mods(Viewer)-ViewWindow_0-c redraw"
    }
    
    method toggle_show_glyphs {} {
	global mods
        global $mods(ShowField-Glyphs)-tensors-on
	
        if {[set $mods(ShowField-Glyphs)-tensors-on] == 0} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (7)\}" 0
        } else {
            global $mods(ChooseField-GlyphSeeds)-port-index
            if {[set $mods(ChooseField-GlyphSeeds)-port-index] == 0} {
		# enable Probe Widget
		uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 1
            } elseif {[set $mods(ChooseField-GlyphSeeds)-port-index] == 1} {
		# enable rake
		uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (7)\}" 1
            }
        }

	configure_glyphs_tabs
	
        $mods(ShowField-Glyphs)-c toggle_display_tensors
        after 100 "$mods(Viewer)-ViewWindow_0-c redraw"
    }

    method change_glyph_scale {} {
	global scale_glyph
	if {$vis_activated && $scale_glyph} {
	    global mods
	    $mods(ShowField-Glyphs)-c data_display_tensors
	}
    }

    method change_glyph_exag {} {
	global exag_glyph
	if {$vis_activated && $exag_glyph} {
	    global mods
	    $mods(TendAnscale-Glyphs)-c needexecute
	}
    }
    


############# FIBERS #############
    
    method build_fibers_tab { f } {
	global mods
        global $mods(ShowField-Fibers)-edges-on
	
	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Fibers" \
		-variable $mods(ShowField-Fibers)-edges-on \
		-command "$this toggle_show_fibers" -state disabled
	    
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
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
		-command "$mods(TendFiber)-c needexecute" \
		-state disabled
	    
	    radiobutton $algo.f.tl -text "Tensorlines (TL)" \
		-variable $mods(TendFiber)-fibertype \
		-value tensorline \
		-command "$mods(TendFiber)-c needexecute" \
		-state disabled
	    
	    pack $algo.f.evec1 $algo.f.tl -side left -anchor nw -padx 5 -pady 1
	    
	    global fibers_stepsize
	    frame $algo.stepsize
	    pack $algo.stepsize -side top -anchor nw -padx 3 -pady 1
	    
	    label $algo.stepsize.l -text "Step Size:" -state disabled
	    scale $algo.stepsize.step -label "" \
		-from 0.1 -to 10 \
		-resolution 0.1 \
		-length 150  -width 15 \
		-sliderlength 15 \
		-orient horizontal \
		-showvalue false \
		-variable fibers_stepsize \
		-state disabled -foreground grey64
	    label $algo.stepsize.val -textvariable fibers_stepsize -state disabled
	    pack $algo.stepsize.l $algo.stepsize.step $algo.stepsize.val -side left -anchor nw -padx 3 -pady 1
	    bind $algo.stepsize.step <ButtonRelease> {app configure_fibers_stepsize}
	    
	    frame $algo.method 
	    pack $algo.method -side top -anchor nw -padx 3 -pady 0
	    
	    global $mods(TendFiber)-integration
	    label $algo.method.l -text "Integration Method: " -state disabled
	    radiobutton $algo.method.e -text "Euler" \
		-variable $mods(TendFiber)-integration \
		-value Euler \
		-state disabled \
		-command "$mods(TendFiber)-c needexecute"
	    radiobutton $algo.method.rk -text "RK4" \
		-variable $mods(TendFiber)-integration \
		-value RK4 \
		-state disabled \
		-command "$mods(TendFiber)-c needexecute"
	    
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
		-state disabled \
		-command "$mods(TendFiber)-c needexecute"
	    
	    radiobutton $rs.f.cat -text "Catmull-Rom" \
		-variable $mods(TendFiber)-kernel \
		-value cubicCR \
		-state disabled \
		-command "$mods(TendFiber)-c needexecute"
	    
	    radiobutton $rs.f.b -text "B-Spline" \
		-variable $mods(TendFiber)-kernel \
		-value cubicBS \
		-state disabled \
		-command "$mods(TendFiber)-c needexecute"
	    
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
		-command "$this toggle_fibers_fiber_length; $mods(TendFiber)-c needexecute" \
		-state disabled -foreground grey64
	    scale $stop.fiber.val -label "" \
		-from 1 -to 400 \
		-resolution 1 \
		-length 120  -width 15 \
		-sliderlength 15 \
		-orient horizontal \
		-showvalue false \
		-variable fibers_length \
		-state disabled -foreground grey64
	    label $stop.fiber.l -textvariable fibers_length -state disabled
	    pack $stop.fiber.check $stop.fiber.val $stop.fiber.l -side left \
		-anchor nw -padx 3 -pady 0
	    bind $stop.fiber.val <ButtonRelease> {app change_fibers_fiber_length}
	    
	    # Number of Steps
	    global $mods(TendFiber)-use-steps
	    global fibers_steps
	    frame $stop.steps
	    pack $stop.steps -side top -anchor nw
	    
	    checkbutton $stop.steps.check -text "Number of Steps:" \
		-variable $mods(TendFiber)-use-steps \
		-command "$this toggle_fibers_steps; $mods(TendFiber)-c needexecute" \
		-state disabled -foreground grey64
	    scale $stop.steps.val -label "" \
		-from 10 -to 1000 \
		-resolution 10 \
		-length 120  -width 15 \
		-sliderlength 15 \
		-orient horizontal \
		-showvalue false \
		-variable fibers_steps \
		-state disabled -foreground grey64
	    label $stop.steps.l -textvariable fibers_steps -state disabled
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
		-command "$this toggle_fibers_aniso; $mods(TendFiber)-c needexecute" \
		-state disabled -foreground grey64
	    scale $stop.aniso1.val -label "" \
		-from 0.0 -to 1.0 \
		-resolution 0.01 \
		-length 95  -width 15 \
		-sliderlength 15 \
		-orient horizontal \
		-showvalue false \
		-variable $mods(TendFiber)-aniso-thresh \
		-state disabled -foreground grey64
	    label $stop.aniso1.l -textvariable $mods(TendFiber)-aniso-thresh -state disabled
	    pack $stop.aniso1.check $stop.aniso1.val $stop.aniso1.l -side left \
		-anchor nw -padx 3 -pady 0
	    
	    frame $stop.aniso2
	    pack $stop.aniso2 -side top -anchor e
	    
	    radiobutton $stop.aniso2.cl -text "Linear Anisotropy" \
		-variable $mods(TendFiber)-aniso-metric \
		-value tenAniso_Cl2 \
		-state disabled \
		-command "$mods(TendFiber)-c needexecute"
	    
	    radiobutton $stop.aniso2.fa -text "Fractional Anisotropy" \
		-variable $mods(TendFiber)-aniso-metric \
		-value tenAniso_FA \
		-state disabled \
		-command "$mods(TendFiber)-c needexecute"
	    pack $stop.aniso2.cl $stop.aniso2.fa -side left -anchor nw -padx 3 -pady 0
	    
	    
	    
	    # Seed at
	    iwidgets::labeledframe $f.seed \
		-labeltext "Seed At" \
		-labelpos nw -foreground grey64
	    pack $f.seed -side top -anchor nw -padx 3 -pady 0 \
		-fill x
	    
	    set seed [$f.seed childsite]
	    
	    global $mods(ChooseField-FiberSeeds)-port-index
	    
	    frame $seed.a
	    pack $seed.a -side left -anchor n -padx 10 
	    radiobutton $seed.a.point -text "Single Point" \
		-variable $mods(ChooseField-FiberSeeds)-port-index \
		-value 0 \
		-state disabled \
		-command "$this update_fiber_seed_method"
	    
	    radiobutton $seed.a.rake -text "Along Line" \
	    -variable $mods(ChooseField-FiberSeeds)-port-index \
		-value 1 \
		-state disabled \
		-command "$this update_fiber_seed_method"
	    
	    frame $seed.b
	    pack $seed.b -side right -anchor n -padx 10
	    radiobutton $seed.b.plane -text "On Planes" \
		-variable $mods(ChooseField-FiberSeeds)-port-index \
		-value 2 \
		-state disabled \
		-command "$this update_fiber_seed_method"
	    
	    radiobutton $seed.b.grid -text "On Grid" \
		-variable $mods(ChooseField-FiberSeeds)-port-index \
		-value 3 \
		-state disabled \
		-command "$this update_fiber_seed_method"
	    
	    pack $seed.a.point $seed.a.rake  -side top \
		-anchor nw -padx 5 -pady 1
	    
	    pack $seed.b.plane $seed.b.grid -side top \
		-anchor nw -padx 5 -pady 1
	    
	    iwidgets::labeledframe $f.rep \
		-labeltext "Color Fibers Based On" \
		-labelpos nw -foreground grey64
	    pack $f.rep -side top -anchor nw -padx 3 -pady 0 \
		-fill x
	    
	    set rep [$f.rep childsite]
	    
	    frame $rep.f1 
	    pack $rep.f1 -side top -anchor nw -padx 3 -pady 1
	    
	    iwidgets::optionmenu $rep.f1.type -labeltext "" \
		-width 180 -state disabled \
		-command "$this change_fiber_color_by $rep.f1"
	    pack $rep.f1.type -side left -anchor nw -padx 2 -pady 0
	    
	    $rep.f1.type insert end "Principle Eigenvector" "Fractional Anisotropy" "Linear Anisotropy" "Planar Anisotropy" "Constant"
	    
	    $rep.f1.type select "Principle Eigenvector"
	    
	    
	    global fiber_color
	    
	    addColorSelection $rep.f1 "Color" fiber_color "fiber_color_change"
	    
	    iwidgets::labeledframe $rep.maps \
		-labeltext "Color Maps" \
		-labelpos nw -foreground grey64
	    pack $rep.maps -side top -anchor n -padx 3 -pady 3
	    
	    set maps [$rep.maps childsite]
	    global $mods(GenStandardColorMaps-Fibers)-mapType
	    
	    # Gray
	    frame $maps.gray
	    pack $maps.gray -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.gray.b -text "Gray" \
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 0 \
		-state disabled \
		-command "$mods(GenStandardColorMaps-Fibers)-c needexecute"
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
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 2 \
		-state disabled \
		-command "$mods(GenStandardColorMaps-Fibers)-c needexecute"
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
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 5 \
		-state disabled \
		-command "$mods(GenStandardColorMaps-Fibers)-c needexecute"
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
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 7 \
		-state disabled \
		-command "$mods(GenStandardColorMaps-Fibers)-c needexecute"
	    pack $maps.blackbody.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.blackbody.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.blackbody.f.canvas -anchor e
	    
	    draw_colormap Blackbody $maps.blackbody.f.canvas
	    
	    
	    # BP Seismic
	    frame $maps.bpseismic
	    pack $maps.bpseismic -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.bpseismic.b -text "BP Seismic" \
		-variable $mods(GenStandardColorMaps-Fibers)-mapType \
		-value 17 \
		-state disabled \
		-command "$mods(GenStandardColorMaps-Fibers)-c needexecute"
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side left -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "BP Seismic" $maps.bpseismic.f.canvas
	} 
    }

    method sync_fibers_tabs {} {
	global mods 
	global $mods(ChooseField-Fibers)-port-index
	global $mods(ChooseColorMap-Fibers)-port-index

	set port [set $mods(ChooseField-Fibers)-port-index]
	set color_port [set $mods(ChooseColorMap-Fibers)-port-index]

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
	
        if {[set $mods(ShowField-Fibers)-edges-on] == 1} {
	    foreach w [winfo children $fibers_tab1] {
		activate_widget $w
	    }
	    foreach w [winfo children $fibers_tab2] {
		activate_widget $w
	    }

	    # configure checkbutton/radiobutton widgets

	    toggle_fibers_fiber_length
	    toggle_fibers_steps
	    toggle_fibers_aniso


	    # configure color swatch
	    if {$fiber_type == "Constant"} {
		$fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state normal
		$fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state normal
	    } else {
		$fibers_tab1.rep.childsite.f1.colorFrame.set_color configure -state disabled
		$fibers_tab2.rep.childsite.f1.colorFrame.set_color configure -state disabled
	    }

	} else {
	    foreach w [winfo children $fibers_tab1] {
		disable_widget $w
	    }
	    foreach w [winfo children $fibers_tab2] {
		disable_widget $w
	    }

	    # enable checkbutton
	    $fibers_tab1.show configure -state normal -foreground black
	    $fibers_tab2.show configure -state normal -foreground black
	}
    }

    
    method change_fibers_fiber_length {} {
	if {$vis_activated} {
	    global mods
	    global $mods(TendFiber)-length
	    global fibers_length
	    
	    set $mods(TendFiber)-length [expr $fibers_length/100.0]
	    
	    $mods(TendFiber)-c needexecute
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
	if {$vis_activated} {
	    global mods
	    global $mods(TendFiber)-steps
	    global fibers_steps
	    
	    set $mods(TendFiber)-steps [expr $fibers_steps/100.0]
	    
	    $mods(TendFiber)-c needexecute
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


    method change_fiber_color_by { f } {
	global mods
	global $mods(ChooseField-Fibers)-port-index
	global $mods(ChooseColorMap-Fibers)-port-index
	global $mods(ShowField-Fibers)-tensor-usedefcolor

	# get selection and change appropriate port
	set type [$f.type get]

	set $mods(ShowField-Fibers)-tensor-usedefcolor 0
	# configure color
	if {$type == "Principle Eigenvector"} {
	    set fiber_type "Principle Eigenvector"
	    set $mods(ChooseField-Fibers)-port-index 3
	    #disableModule $mods(RescaleColorMap-Fibers) 1
	    set $mods(ChooseColorMap-Fibers)-port-index 1
	} elseif {$type == "Fractional Anisotropy"} {
	    set fiber_type "Fractional Anisotropy"
	    set $mods(ChooseField-Fibers)-port-index 0
	    #disableModule $mods(RescaleColorMap-Fibers) 0
	    set $mods(ChooseColorMap-Fibers)-port-index 0
	} elseif {$type == "Linear Anisotropy"} {
	    set fiber_type "Linear Anisotropy"
	    set $mods(ChooseField-Fibers)-port-index 1
	    #disableModule $mods(RescaleColorMap-Fibers) 0
	    set $mods(ChooseColorMap-Fibers)-port-index 0
	} elseif {$type == "Planar Anisotropy"} {
	    set fiber_type "Planar Anisotropy"
	    set $mods(ChooseField-Fibers)-port-index 2
	    #disableModule $mods(RescaleColorMap-Fibers) 0
	    set $mods(ChooseColorMap-Fibers)-port-index 0
	} elseif {$type == "Constant"} {
	    set fiber_type "Constant"
	    #disableModule $mods(RescaleColorMap-Fibers) 1
	    set $mods(ChooseColorMap-Fibers)-port-index 1
	    set $mods(ShowField-Fibers)-tensor-usedefcolor 1
	    set $mods(ChooseField-Fibers)-port-index 0
	}
	
	# sync attached/detached optionmenus
	$fibers_tab1.rep.childsite.f1.type select $type
	$fibers_tab2.rep.childsite.f1.type select $type

	configure_fibers_tabs

	$mods(ShowField-Fibers)-c data_display_type
	$mods(ChooseField-Fibers)-c needexecute
    }
 

    method configure_fibers_stepsize {} {
	if {$vis_activated} {
	    global mods
	    global $mods(TendFiber)-stepsize
	    
	    global fibers_stepsize
	    
	    set $mods(TendFiber)-stepsize [expr $fibers_stepsize/100.0]
	    
	    $mods(TendFiber)-c needexecute
	}
    }
   


    method update_fiber_seed_method {} {
        global mods
        global $mods(ChooseField-FiberSeeds)-port-index

        if {[set $mods(ChooseField-FiberSeeds)-port-index] == 0} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 1
        } elseif {[set $mods(ChooseField-FiberSeeds)-port-index] == 1} {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (12)\}" 1
        } else {
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (12)\}" 0
        }
	
        $mods(ChooseField-FiberSeeds)-c needexecute
	
	after 100 "$mods(Viewer)-ViewWindow_0-c redraw"
    }
	    


    method toggle_show_fibers {} {
	global mods
        global $mods(ShowField-Fibers)-edges-on
        global $mods(ShowField-Fibers)-nodes-on
	
        if {[set $mods(ShowField-Fibers)-edges-on] == 0} {
	    # sync nodes
	    set $mods(ShowField-Fibers)-nodes-on 0 
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
            uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (12)\}" 0

	    # disable rest of fibers tab except for checkbutton
        } else {
	    # sync nodes
	    set $mods(ShowField-Fibers)-nodes-on 1
            global $mods(ChooseField-FiberSeeds)-port-index
            if {[set $mods(ChooseField-FiberSeeds)-port-index] == 0} {
		# enable Probe Widget
		uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 1
            } elseif {[set $mods(ChooseField-FiberSeeds)-port-index] == 1} {
		# enable rake
		uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (12)\}" 1
            }
        }

	configure_fibers_tabs
	
        $mods(ShowField-Fibers)-c toggle_display_edges
        $mods(ShowField-Fibers)-c toggle_display_nodes
        after 100 "$mods(Viewer)-ViewWindow_0-c redraw"
    }
    


    
#################### COLORMAPS ###################

    method draw_colormap { which canvas } {
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
	} elseif {$which == "BP Seismic"} {
	    set color { "BP Seismic" { { 0 0 255 } { 255 255 255} { 255 0 0 } } }
	}

        set colorMap [$this set_color_map $color]
	
	set width $colormap_width
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
    
    
    
    method set_color_map { map } {
        set resolution $colormap_res
	set colorMap {}
	set currentMap {}
	set currentMap [$this make_new_map [ lindex $map 1 ]]
	set n [llength $currentMap]
	if { $resolution < $n } {
	    set resolution $n
	}
	set m $resolution
	
	set frac [expr ($n-1)/double($m-1)]
	for { set i 0 } { $i < $m  } { incr i} {
	    if { $i == 0 } {
		set color [lindex $currentMap 0]
		lappend color 0.5
	    } elseif { $i == [expr ($m -1)] } {
		set color [lindex $currentMap [expr ($n - 1)]]
		lappend color 0.5
	    } else {
		set index [expr int($i * $frac)]
		set t [expr ($i * $frac)-$index]
		set c1 [lindex $currentMap $index]
		set c2 [lindex $currentMap [expr $index + 1]]
		set color {}
		for { set j 0} { $j < 3 } { incr j} {
		    set v1 [lindex $c1 $j]
		    set v2 [lindex $c2 $j]
		    lappend color [expr int($v1 + $t*($v2 - $v1))]
		}
		lappend color 0.5
	    }
	    lappend colorMap $color
	}	
        return $colorMap
    }
    
    
    method make_new_map { currentMap } {
        set gamma 0
	set res $colormap_res
	set newMap {}
	set m [expr int($res + abs( $gamma )*(255 - $res))]
	set n [llength $currentMap]
	if { $m < $n } { set m $n }
	set frac [expr double($n-1)/double($m - 1)]
	for { set i 0 } { $i < $m  } { incr i} {
	    if { $i == 0 } {
		set color [lindex $currentMap 0]
	    } elseif { $i == [expr ($m -1)] } {
		set color [lindex $currentMap [expr ($n - 1)]]
	    } else {
		set index_double [$this modify [expr $i * $frac] [expr $n-1]]
		
		set index [expr int($index_double)]
		set t  [expr $index_double - $index]
		set c1 [lindex $currentMap $index]
		set c2 [lindex $currentMap [expr $index + 1]]
		set color {}
		for { set j 0} { $j < 3 } { incr j} {
		    set v1 [lindex $c1 $j]
		    set v2 [lindex $c2 $j]
		    lappend color [expr int($v1 + $t*($v2 - $v1))]
		}
	    }
	    lappend newMap $color
	}
	return $newMap
    }
    

    method modify { i range } {
	set gamma 0
	
	set val [expr $i/double($range)]
	set bp [expr tan( 1.570796327*(0.5 + $gamma*0.49999))]
	set index [expr pow($val,$bp)]
	return $index*$range
    }
    
 

    method addColorSelection {frame text color mod} {
	#add node color picking 
	global $color
	global $color-r
	global $color-g
	global $color-b
	#add node color picking 
	set ir [expr int([set $color-r] * 65535)]
	set ig [expr int([set $color-g] * 65535)]
	set ib [expr int([set $color-b] * 65535)]
	
	frame $frame.colorFrame
	frame $frame.colorFrame.col -relief ridge -borderwidth \
	    4 -height 0.6c -width 1.0c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
			 
	set cmmd "$this raiseColor $frame.colorFrame.col $color $mod"
	button $frame.colorFrame.set_color \
	    -state disabled \
	    -text $text -command $cmmd
	
	#pack the node color frame
	pack $frame.colorFrame.set_color \
	    -side left -ipadx 3 -ipady 3
	pack $frame.colorFrame.col -side left 
	pack $frame.colorFrame -side left -padx 3
    }
    
    
    method raiseColor {col color mod} {
	global $color
	set window .standalone
	if {[winfo exists $window.color]} {
	    raise $window.color
	    return;
	} else {
	    toplevel $window.color
	    makeColorPicker $window.color $color \
		"$this setColor $col $color $mod" \
		"destroy $window.color"
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

            $mods(ChooseField-ColorPlanes)-c needexecute
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

            $mods(Isosurface)-c needexecute
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

             $mods(DirectInterpolate-Glyphs)-c needexecute
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
	     
             $mods(DirectInterpolate-Fibers)-c needexecute
         }

    }

   
    
####################################################################
######################### HELPER FUNCTIONS #########################
####################################################################

    method activate_widget {w} {
    	set has_state_option 0
    	set has_foreground_option 0
        set has_text_option 0
    	foreach opt [$w configure ] {
	    set temp1 [lsearch -exact $opt "state"]
	    set temp2 [lsearch -exact $opt "foreground"]
	    set temp3 [lsearch -exact $opt "text"]

	    if {$temp1 > -1} {
	       set has_state_option 1
	    }
            if {$temp2 > -1} {
               set has_foreground_option 1
            }
            if {$temp3 > -1} {
               set has_text_option 1
            }
        }

        if {$has_state_option} {
	    $w configure -state normal
        }

        if {$has_foreground_option} {
            $w configure -foreground black
        }
      
        if {$has_text_option} {
           # if it is a next button configure the background 
           set t [$w configure -text]
           if {[lindex $t 4]== "Next"} {
             $w configure -background $next_color
             $w configure -activebackground $next_color
           } elseif {[lindex $t 4] == "Execute"} {
             $w configure -background $execute_color
             $w configure -activebackground $execute_color
           }
        }

        foreach widg [winfo children $w] {
	     activate_widget $widg
        }
    }


    method disable_widget {w} {
    	set has_state_option 0
    	set has_foreground_option 0
    	foreach opt [$w configure ] {
	    set temp1 [lsearch -exact $opt "state"]
	    set temp2 [lsearch -exact $opt "foreground"]
	    if {$temp1 > -1} {
	       set has_state_option 1
	    }
            if {$temp2 > -1} {
               set has_foreground_option 1
            }
        }

        if {$has_state_option} {
	    $w configure -state disabled
        }
        if {$has_foreground_option} {
            $w configure -foreground grey64
        }


        foreach widg [winfo children $w] {
	     disable_widget $widg
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
	    if {$which == "Data Vis"} {
		# Data Vis
		$vis_frame_tab1 view "Data Vis"
		$vis_frame_tab2 view "Data Vis"
		set c_left_tab "Data Vis"
	    } else {
 		$vis_frame_tab1 view "Global Options"
 		$vis_frame_tab2 view "Global Options"
		set c_left_tab "Global Options"
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
	    if {$which == "Data"} {
		# Data Acquisition step
		$proc_tab1 view "Data"
		$proc_tab2 view "Data"
		change_indicator_labels "Data Acquisition..."
		set c_procedure_tab "Data"
	    } elseif {$which == "Registration"} {
		# Registration step
		if {$data_completed} {
		    $proc_tab1 view "Registration"
		    $proc_tab2 view "Registration"
		    change_indicator_labels "Registration..."
		} 
		set c_procedure_tab "Registration"
	    } elseif {$which == "Build DTs"} {
		if {!$do_registration} {
		    set reg_completed 1
		    disableModule $mods(ChooseNrrd-ToReg) 0
		    # NOT SURE ABOUT THIS RESCALE
		    #disableModule $mods(RescaleColorMap2) 0
		    disableModule $mods(TendEpireg) 1
		    disableModule $mods(UnuJoin) 1
		    $mods(ChooseNrrd-ToReg)-c needexecute
		    activate_dt
		    $proc_tab1 view "Build DTs"
		    $proc_tab2 view "Build DTs"
		} elseif {$reg_completed} {
		    # Building DTs step
		    $proc_tab1 view "Build DTs"
		    $proc_tab2 view "Build DTs"
		    change_indicator_labels "Building Diffusion Tensors..."
		}
		set c_procedure_tab "Build DTs"
	    }
	    
	    set indicator 0
	}
    }
	
	
#############################################################
################# PROGRESS INDICATOR ########################
#############################################################


    method change_indicator {} {
       if {[winfo exists $indicator2] == 1} {
	   
	   if {$indicatorID != 0} {
	       after cancel $indicatorID
	       set indicatorID 0
	   }

	   if {$indicate == 0} {
	       # reset and do nothing
	       $indicator1 raise res all
	       $indicator2 raise res all
	       after cancel $indicatorID
           } elseif {$indicate == 1} {
	       # indicate something is happening
	       if {$cycle == 0} { 
		   $indicator1 raise swirl all
		   $indicator2 raise swirl all
		   $indicator1 move swirl $i_back 0
		   $indicator2 move swirl $i_back 0		  
		   set cycle 1
	       } elseif {$cycle == 1} {
		   $indicator1 move swirl $i_move 0
		   $indicator2 move swirl $i_move 0
		   set cycle 2
	       } elseif {$cycle == 2} {
		   $indicator1 move swirl $i_move 0
		   $indicator2 move swirl $i_move 0
		   set cycle 3
	       } else {
		   $indicator1 move swirl $i_move 0
		   $indicator2 move swirl $i_move 0
		   set cycle 0
	       } 
	       set indicatorID [after 200 "$this change_indicator"]
           } elseif {$indicate == 2} {
	       # indicate complete
	       $indicator1 raise comp1 all
	       $indicator2 raise comp1 all
	       
	       $indicator1 raise comp2 all
	       $indicator2 raise comp2 all
           } else {
	       $indicator1 raise error1 all
	       $indicator2 raise error1 all
	       
	       $indicator1 raise error2 all
	       $indicator2 raise error2 all
	       after cancel $indicatorID
           }
       }
    }
	

    method construct_indicator { canvas } {
	global tips

       # make image swirl
       set dx [expr $i_width/double($stripes)]
       set x 0
       set longer [expr $stripes+10]
       for {set i 0} {$i <= $longer} {incr i 1} {
	  if {[expr $i % 2] != 0} {
	     set r 83
 	     set g 119
             set b 181
	     set c [format "#%02x%02x%02x" $r $g $b]
             set oldx $x
             set x [expr ($i+1)*$dx]
             set prevx [expr $oldx - $dx]
             $canvas create polygon \
  	        $oldx 0 $x 0 $oldx $i_height $prevx $i_height \
	        -fill $c -outline $c -tags swirl
          } else {
	     set r 237
   	     set g 240
             set b 242
	     set c [format "#%02x%02x%02x" $r $g $b]
             set oldx $x
             set x [expr ($i+1)*$dx]
             set prevx [expr $oldx - $dx]
             $canvas create polygon \
	        $oldx 0 $x 0 $oldx $i_height $prevx $i_height \
	        -fill $c -outline $c -tags swirl
          }
       }

       set i_font "-Adobe-Helvetica-Bold-R-Normal-*-16-120-75-*"

       # make completed
       set s [expr $i_width/2]
       set dx [expr $i_width/double($s)]
       set x 0
       for {set i 0} {$i <= $s} {incr i 1} {
	  if {[expr $i % 2] != 0} {
	     set r 0
 	     set g 139
             set b 69
	     set c [format "#%02x%02x%02x" $r $g $b]
             set oldx $x
             set x [expr ($i+1)*$dx]
             $canvas create rectangle \
  	        $oldx 0 $x $i_height \
	        -fill $c -outline $c -tags comp1
          } else {
	     set r 49
   	     set g 160
             set b 101
	     set c [format "#%02x%02x%02x" $r $g $b]
             set oldx $x
             set x [expr ($i+1)*$dx]
             $canvas create rectangle \
	        $oldx 0 $x $i_height  \
	        -fill $c -outline $c -tags comp1
          }
       }

       $canvas create text [expr $i_width/2] [expr $i_height/2] -text "C O M P L E T E" \
   	  -font $i_font -fill "black" -tags comp2

       # make error
       set s [expr $i_width/2]
       set dx [expr $i_width/double($s)]
       set x 0
       for {set i 0} {$i <= $s} {incr i 1} {
	  if {[expr $i % 2] == 0} {
	     set r 191
	     set g 59
	     set b 59
	     set c [format "#%02x%02x%02x" $r $g $b]
             set oldx $x
             set x [expr ($i+1)*$dx]
             $canvas create rectangle \
  	        $oldx 0 $x $i_height \
	        -fill $c -outline $c -tags error1
          } else {
	     set r 206
	     set g 78
	     set b 78
	     set c [format "#%02x%02x%02x" $r $g $b]
             set oldx $x
             set x [expr ($i+1)*$dx]
             $canvas create rectangle \
	        $oldx 0 $x $i_height  \
	        -fill $c -outline $c -tags error1
          }
       }


       $canvas create text [expr $i_width/2] [expr $i_height/2] -text "E R R O R" \
   	  -font $i_font -fill "black" -tags error2

       # make reset
       set r 237
       set g 240
       set b 242
       set c [format "#%02x%02x%02x" $r $g $b]
       $canvas create rectangle \
	   0 0 $i_width $i_height -fill $c -outline $c -tags res
       
       bind $canvas <ButtonPress> {app display_module_error}

       Tooltip $canvas $tips(IndicatorBar)
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
		set darby [expr $darby + 1]
		set indicate 1
		change_indicator
	    } elseif {$v == 2} {
		# Complete
		set darby [expr $darby - 1]
		if {$darby == 0} {
		    # only change indicator if progress isn't running
		    set indicate 2
		    change_indicator
		} elseif {$darby < 0} {
		    # something wasn't caught, reset
		    set darby 0
		    set indicate 2
		    change_indicator
		}
	    }
	}
    }
    
    method change_indicator_labels { msg } {
	if {($msg == "Visualization..." && $data_completed && $reg_completed && $dt_completed) || ($msg != "Visualization...")} {
	    $indicatorL1 configure -text $msg
	    $indicatorL2 configure -text $msg
	}
    }
    



#############################################################
######################### VARIABLES #########################
#############################################################
    
    # Embedded Viewer
    variable eviewer

    # Standalone
    variable win

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

    # Flag to indicate whether entire gui has been built
    variable initialized

    # State
    variable data_completed
    variable reg_completed
    variable dt_completed
    variable vis_activated
	
    variable c_procedure_tab
    variable c_data_tab
    variable c_left_tab
    variable c_vis_tab

    
    variable IsPAttached
    variable detachedPFr
    variable attachedPFr

    variable IsVAttached
    variable detachedVFr
    variable attachedVFr


    # Indicator
    variable indicatorID
    variable indicator1
    variable indicator2
    variable indicatorL1
    variable indicatorL2
    variable indicate
    variable cycle
    variable darby
    variable i_width
    variable i_height
    variable stripes
    variable i_move
    variable i_back
    variable error_module

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

    # Visualiztion frame tabnotebook
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

    variable process_width
    variable process_height

    variable viewer_width
    variable viewer_height

    variable vis_width
    variable vis_height

    variable screen_width
    variable screen_height


    # Colors
    variable proc_color
    variable next_color
    variable execute_color
    variable feedback_color
    variable error_color

    # planes
    variable last_x
    variable last_y
    variable last_z
    variable plane_inc
    variable plane_type

    # glyphs
    variable clip_x
    variable clip_y
    variable clip_z
    variable glyph_type

    # fibers
    variable fiber_type

    # colormaps
    variable colormap_width
    variable colormap_height
    variable colormap_res


}

BioTensorApp app

app build_app



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

bind all <Control-a> {
    global mods
    $mods(Viewer)-ViewWindow_0-c autoview
}
