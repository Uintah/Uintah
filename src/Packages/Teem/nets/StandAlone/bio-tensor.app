# global array indexed by module name to keep track of modules
global mods


############# NET ##############
::netedit dontschedule

set m0 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 31 10]
set m1 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 121 180]
set m2 [addModuleAtPosition "Teem" "Tend" "TendEpireg" 31 374]
set m3 [addModuleAtPosition "Teem" "Tend" "TendEstim" 31 799]
set m4 [addModuleAtPosition "Teem" "Tend" "TendBmat" 49 731]
set m5 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 133 571]
set m6 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 564 331]
set m7 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 579 717]
set m8 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 692 495]
set m9 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 903 886]
set m10 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 31 1102]
set m11 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 176 1038]
set m12 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 176 1104]
set m13 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 458 1449]
set m14 [addModuleAtPosition "Teem" "Tend" "TendEval" 704 971]
set m15 [addModuleAtPosition "Teem" "Unu" "UnuSlice" 704 1038]
set m16 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 704 1103]
set m17 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 286 1293]
set m18 [addModuleAtPosition "SCIRun" "Fields" "DirectInterpolate" 458 1371]
set m19 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 351 1038]
set m20 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 526 1038]
set m21 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 351 1104]
set m22 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 526 1104]
set m23 [addModuleAtPosition "SCIRun" "Fields" "ChooseField" 286 1230]
set m24 [addModuleAtPosition "SCIRun" "Fields" "ChooseField" 458 1233]
set m25 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 628 1371]
set m26 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 628 1305]
set m27 [addModuleAtPosition "SCIRun" "Fields" "ChooseField" 862 1232]
set m28 [addModuleAtPosition "SCIRun" "Fields" "DirectInterpolate" 862 1489]
set m29 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 1014 1564]
set m30 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 1014 1633]
set m31 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 996 1713]
set m32 [addModuleAtPosition "SCIRun" "Fields" "ClipField" 827 1570]
set m33 [addModuleAtPosition "SCIRun" "Fields" "ChooseField" 845 1634]
set m34 [addModuleAtPosition "SCIRun" "Fields" "ChooseField" 76 1245]
set m35 [addModuleAtPosition "SCIRun" "Fields" "FieldMeasures" 1044 1232]
set m36 [addModuleAtPosition "SCIRun" "Fields" "FieldMeasures" 1219 1234]
set m37 [addModuleAtPosition "SCIRun" "Fields" "FieldMeasures" 1395 1236]
set m38 [addModuleAtPosition "SCIRun" "Fields" "ManageFieldData" 1026 1302]
set m39 [addModuleAtPosition "SCIRun" "Fields" "ManageFieldData" 1201 1304]
set m40 [addModuleAtPosition "SCIRun" "Fields" "ManageFieldData" 1377 1307]
set m41 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1026 1371]
set m42 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1426 752]
set m43 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1602 752]
set m44 [addModuleAtPosition "SCIRun" "Render" "Viewer" 885 1133]
set m45 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 31 96]
set m46 [addModuleAtPosition "SCIRun" "Fields" "ChangeFieldBounds" 564 409]
set m47 [addModuleAtPosition "Teem" "Unu" "UnuJoin" 276 825]
set m48 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 276 745]
set m49 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 105 485]
set m50 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 280 379]
set m51 [addModuleAtPosition "Teem" "Unu" "UnuJoin" 1085 437]
set m52 [addModuleAtPosition "Teem" "Tend" "TendEstim" 1085 499]
set m53 [addModuleAtPosition "Teem" "Tend" "TendAnvol" 1085 562]
set m54 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 1085 626]
set m55 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 1085 750]
set m56 [addModuleAtPosition "SCIRun" "Fields" "ChangeFieldBounds" 1085 687]
set m57 [addModuleAtPosition "Teem" "Unu" "UnuProject" 121 247]
set m58 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 746 330]
set m59 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 747 409]
set m60 [addModuleAtPosition "Teem" "Unu" "UnuProject" 133 636]
set m61 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 921 822]
set m62 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 393 179]
set m63 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 356 628]

addConnection $m3 0 $m10 0
addConnection $m3 0 $m11 0
addConnection $m11 0 $m12 0
addConnection $m3 0 $m14 0
addConnection $m14 0 $m15 0
addConnection $m15 0 $m16 0
addConnection $m19 0 $m21 0
addConnection $m20 0 $m22 0
addConnection $m3 0 $m19 0
addConnection $m3 0 $m20 0
addConnection $m12 0 $m23 0
addConnection $m21 0 $m23 1
addConnection $m22 0 $m23 2
addConnection $m16 0 $m23 3
addConnection $m12 0 $m24 0
addConnection $m21 0 $m24 1
addConnection $m22 0 $m24 2
addConnection $m16 0 $m24 3
addConnection $m23 0 $m17 0
addConnection $m17 0 $m18 1
addConnection $m24 0 $m18 0
addConnection $m26 0 $m25 0
addConnection $m24 0 $m25 1
addConnection $m18 0 $m13 0
addConnection $m25 0 $m13 1
addConnection $m12 0 $m27 0
addConnection $m21 0 $m27 1
addConnection $m22 0 $m27 2
addConnection $m16 0 $m27 3
addConnection $m27 0 $m28 0
addConnection $m27 0 $m30 1
addConnection $m29 0 $m30 0
addConnection $m30 0 $m31 1
addConnection $m28 0 $m32 0
addConnection $m17 0 $m32 1
addConnection $m32 1 $m33 0
addConnection $m28 0 $m33 1
addConnection $m33 0 $m31 0
addConnection $m10 0 $m35 0
addConnection $m37 0 $m40 1
addConnection $m36 0 $m39 1
addConnection $m35 0 $m38 1
addConnection $m10 0 $m38 0
addConnection $m10 0 $m39 0
addConnection $m10 0 $m40 0
addConnection $m10 0 $m36 0
addConnection $m10 0 $m37 0
addConnection $m38 0 $m41 0
addConnection $m39 0 $m42 0
addConnection $m40 0 $m43 0
addConnection $m8 0 $m44 0
addConnection $m0 0 $m45 0
addConnection $m45 0 $m2 0
addConnection $m9 0 $m44 1
addConnection $m48 0 $m47 0
addConnection $m13 0 $m44 2
addConnection $m47 0 $m3 0
addConnection $m17 1 $m44 3
addConnection $m46 0 $m8 0
addConnection $m48 0 $m51 0
addConnection $m45 0 $m51 1
addConnection $m51 0 $m52 0
addConnection $m52 0 $m53 0
addConnection $m53 0 $m54 0
addConnection $m54 0 $m56 0
addConnection $m56 0 $m55 0
addConnection $m55 1 $m44 4
addConnection $m45 0 $m1 0
addConnection $m1 0 $m57 0
addConnection $m57 0 $m6 0
addConnection $m6 0 $m46 0
addConnection $m58 0 $m59 0
addConnection $m6 0 $m59 1
addConnection $m59 0 $m8 1
addConnection $m5 0 $m60 0
addConnection $m58 0 $m61 0
addConnection $m7 0 $m61 1
addConnection $m60 0 $m7 0
addConnection $m61 0 $m9 1
addConnection $m7 0 $m9 0
addConnection $m45 0 $m62 0
addConnection $m2 0 $m5 0
addConnection $m2 0 $m47 1
addConnection $m50 0 $m2 1
addConnection $m50 0 $m4 0
addConnection $m4 0 $m49 0
addConnection $m63 0 $m49 1
addConnection $m49 0 $m3 1
addConnection $m49 0 $m52 1


set $m0-notes {}
set $m0-label {unknown}
set $m0-type {Scalar}
set $m0-axis {axis0}
set $m0-add {0}
set $m0-filename {/home/darbyb/sci/SCIRunData/A.nrrd}
set $m1-notes {}
set $m1-axis {3}
set $m1-position {15}
set $m2-notes {}
set $m2-gradient_list {}
set $m2-reference {-1}
set $m2-blur_x {1.0}
set $m2-blur_y {2.0}
set $m2-threshold {100}
set $m2-cc_analysis {1}
set $m2-fitting {0.7}
set $m2-kernel {cubicCR}
set $m2-sigma {1}
set $m2-extent {1}
set $m3-notes {}
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
set $m5-position {15}
set $m6-notes {}
set $m7-notes {}
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
set $m11-notes {}
set $m11-aniso_metric {tenAniso_FA}
set $m11-threshold {0.5}
set $m12-notes {}
set $m13-notes {}
set $m13-nodes-on {0}
set $m13-nodes-transparency {0}
set $m13-nodes-as-disks {0}
set $m13-edges-on {0}
set $m13-edges-transparency {0}
set $m13-faces-on {1}
set $m13-use-normals {0}
set $m13-use-transparency {0}
set $m13-vectors-on {0}
set $m13-normalize-vectors {}
set $m13-has_vector_data {0}
set $m13-bidirectional {0}
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
set $m13-node_scale {0.0300}
set $m13-edge_scale {0.0150}
set $m13-vectors_scale {0.30}
set $m13-tensors_scale {0.30}
set $m13-scalars_scale {0.300}
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
set $m17-notes {}
set $m17-isoval {0.5000}
set $m17-isoval-min {0}
set $m17-isoval-max {1}
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
set $m19-threshold {}
set $m20-notes {}
set $m20-aniso_metric {}
set $m20-threshold {}
set $m21-notes {}
set $m22-notes {}
set $m23-notes {}
set $m23-port-index {0}
set $m24-notes {}
set $m24-port-index {1}
set $m25-notes {}
set $m25-isFixed {0}
set $m25-min {0}
set $m25-max {0.99999958276748657}
set $m25-makeSymmetric {0}
set $m26-notes {}
set $m26-tcl_status {Calling GenStandardColorMaps!}
set $m26-positionList {}
set $m26-nodeList {}
set $m26-width {1}
set $m26-height {1}
set $m26-mapType {3}
set $m26-minRes {12}
set $m26-resolution {256}
set $m26-realres {256}
set $m26-gamma {0}
set $m27-notes {}
set $m27-port-index {0}
set $m28-notes {}
set $m28-interpolation_basis {linear}
set $m28-map_source_to_single_dest {0}
set $m28-exhaustive_search {1}
set $m28-exhaustive_search_max_dist {-1}
set $m28-np {1}
set $m29-notes {}
set $m29-tcl_status {Calling GenStandardColorMaps!}
set $m29-positionList {}
set $m29-nodeList {}
set $m29-width {1}
set $m29-height {1}
set $m29-mapType {3}
set $m29-minRes {12}
set $m29-resolution {256}
set $m29-realres {256}
set $m29-gamma {0}
set $m30-notes {}
set $m30-isFixed {0}
set $m30-min {0}
set $m30-max {1}
set $m30-makeSymmetric {0}
set $m31-notes {}
set $m31-nodes-on {1}
set $m31-nodes-transparency {0}
set $m31-nodes-as-disks {0}
set $m31-edges-on {1}
set $m31-edges-transparency {0}
set $m31-faces-on {1}
set $m31-use-normals {0}
set $m31-use-transparency {0}
set $m31-vectors-on {0}
set $m31-normalize-vectors {}
set $m31-has_vector_data {0}
set $m31-bidirectional {0}
set $m31-tensors-on {0}
set $m31-has_tensor_data {0}
set $m31-scalars-on {0}
set $m31-scalars-transparency {0}
set $m31-has_scalar_data {0}
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
set $m31-active_tab {Nodes}
set $m31-node_scale {0.03}
set $m31-edge_scale {0.015}
set $m31-vectors_scale {0.30}
set $m31-tensors_scale {0.30}
set $m31-scalars_scale {0.30}
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
set $m34-port-index {0}
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
set $m44-ViewWindow_0-pos {z1_y0}
set $m44-ViewWindow_0-caxes {0}
set $m44-ViewWindow_0-raxes {1}
set $m44-ViewWindow_0-iaxes {}
set $m44-ViewWindow_0-have_collab_vis {0}
set $m44-ViewWindow_0-view-eyep-x {-5}
set $m44-ViewWindow_0-view-eyep-y {110.5}
set $m44-ViewWindow_0-view-eyep-z {1104.5088994161194}
set $m44-ViewWindow_0-view-lookat-x {-5}
set $m44-ViewWindow_0-view-lookat-y {110.5}
set $m44-ViewWindow_0-view-lookat-z {51}
set $m44-ViewWindow_0-view-up-x {0}
set $m44-ViewWindow_0-view-up-y {-1}
set $m44-ViewWindow_0-view-up-z {0}
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
set $m44-ViewWindow_0-global-clip {1}
set $m44-ViewWindow_0-global-cull {0}
set $m44-ViewWindow_0-global-dl {0}
set $m44-ViewWindow_0-global-type {Gouraud}
set $m44-ViewWindow_0-ortho-view {1}
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
set $m48-axis {}
set $m48-add {1}
set $m48-filename {/home/darbyb/sci/SCIRunData/anat.nrrd}
set $m49-notes {}
set $m49-port-index {0}
set $m50-notes {}
set $m50-label {unknown}
set $m50-type {Scalar}
set $m50-axis {axis0}
set $m50-add {0}
set $m50-filename {/home/darbyb/sci/SCIRunData/grads.txt}
set $m51-notes {}
set $m51-join-axis {0}
set $m51-incr-dim {0}
set $m51-dim {4}
set $m52-notes {}
set $m52-threshold {100}
set $m52-soft {0}
set $m52-scale {1}
set $m53-notes {}
set $m53-aniso_metric {tenAniso_FA}
set $m53-threshold {100}
set $m54-notes {}
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
set $m63-axis {}
set $m63-add {0}
set $m63-filename {}

set mods(NrrdReader1) $m0
set mods(NrrdChoose1) $m45
set mods(NrrdInfo1) $m62

### Original Data Stuff
set mods(UnuSlice1) $m1
set mods(UnuProject1) $m57


set mods(UnuJoin2) $m51
###
set mods(TendEpireg) $m2

set mods(NrrdReader-Gradient) $m50
set mods(NrrdReader-BMatrix) $m63

set mods(GenStandardColorMaps1)  $m58
set mods(RescaleColorMap2) $m61

set mods(Viewer) $m44

                                                                               
#######################################################
# Build up a simplistic standalone application.
#######################################################
wm deiconify .
# wm withdraw .

class BioTensorApp {

    method modname {} {
	return "BioTensorApp"
    }

    constructor {} {
	toplevel .standalone
	wm title .standalone "BioTensor"	 
	set win .standalone

	set notebook_width 300
	set notebook_height 315

	set viewer_width 640
	set viewer_height 512
    
	set process_width 370
	set process_height $viewer_height

	set vis_width [expr $notebook_width + 30]
	set vis_height $viewer_height

	set screen_width [winfo screenwidth .]
	set screen_height [winfo screenheight .]

        # Dummy variables
        set number_of_images 0
        set file_prefix "/scratch/darbyb/data/img"
        set threshold 270
        set flood_fill 1

        set ref_image 1
        set ref_image_state 0

        set ref_image1 ""
        set ref_image2 ""
        set registration1 ""
        set registration2 ""

        set dt1 ""
        set dt2 ""

        set error_module ""

        set current_nrrd ""
        set vol_1a ""
        set vol_1b ""

        set vol_2a ""
        set vol_2b ""
        set original_plane 2
        set original_slice 0

        set current_step "Data Acquisition"

        set proc_color "dark red"
        set next_color "darkseagreen4"
        set feedback_color "dodgerblue4"
        set error_color "red4"

        initialize_blocks

    }

    destructor {
	destroy $this
    }

    method initialize_blocks {} { 
	global mods

        # Blocking Data Section
        # block_connection $mods(NrrdChoose1) 0 $mods(UnuSlice1) 0 "purple"

        set data_blocked 1

        block_connection $mods(NrrdChoose1) 0 $mods(UnuJoin2) 1 "purple"
	block_connection $mods(NrrdChoose1) 0 $mods(TendEpireg) 0 "purple"
        # disableModule $mods(TendEpireg) disabled
        block_connection $mods(GenStandardColorMaps1) 0 $mods(RescaleColorMap2) 0 "purple"

        disableModule $mods(NrrdReader-BMatrix) disabled

    }

    method build_app {} {
	global mods

	# Embed the Viewer
	set eviewer [$mods(Viewer) ui_embedded]
	$eviewer setWindow $win.viewer

	set att_msg "Detach from Viewer"
	set det_msg " Attach to Viewer "


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

	init_Pframe $detachedPFr.f $det_msg 0
	init_Pframe $attachedPFr.f $att_msg 1

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
	
	init_Vframe $detachedVFr.f $det_msg 0
	init_Vframe $attachedVFr.f $att_msg 1



	### pack 3 frames
	pack $attachedPFr $win.viewer $attachedVFr -side left \
	     -anchor n -fill both -expand 1

	set total_width [expr [expr $process_width + $viewer_width] + $vis_width]
	set total_height $vis_height

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]
	set pos_y [expr [expr $screen_height / 2] - [expr $total_height / 2]]

	append geom $total_width x $viewer_height + $pos_x + $pos_y
	wm geometry .standalone $geom
	update	
    }


    method init_Pframe { m msg case } {
        global mods
        
	if { [winfo exists $m] } {
	    ### Menu
	    frame $m.main_menu -relief raised -borderwidth 3
	    pack $m.main_menu -fill x -anchor nw
	    
	    menubutton $m.main_menu.file -text "File" -underline 0 \
		-menu $m.main_menu.file.menu
	    
	    menu $m.main_menu.file.menu -tearoff false
	    
	    $m.main_menu.file.menu add command -label "Save Ctr+S" \
		-underline 0 -command "$this save_session" -state active
	    
	    $m.main_menu.file.menu add command -label "Load  Ctr+O" \
		-underline 0 -command "$this load_session" -state active
	    
	    $m.main_menu.file.menu add command -label "Quit   Ctr+Q" \
		-underline 0 -command "$this exit_app" -state active
	    
	    pack $m.main_menu.file -side left
	    
	    menubutton $m.main_menu.help -text "Help" -underline 0 \
		-menu $m.main_menu.help.menu
	    
	    menu $m.main_menu.help.menu -tearoff false
	    
	    $m.main_menu.help.menu add command -label "View Help" \
		-underline 0 -command "$this show_help" -state active
	    
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
                -height [expr $process_height - 170] \
                -tabpos n
	    pack $process.tnb -side top -anchor n -fill both -expand 1 

            set step_tab [$process.tnb add -label "Data" \
                 -command {puts "Data Acquisition"}]

            if {$case == 0} {
                set proc_tab1 $process.tnb
            } else {
                set proc_tab2 $process.tnb
            }

	    ### Data Acquisition
            iwidgets::tabnotebook $step_tab.tnb -height 200 \
                 -tabpos n 
            pack $step_tab.tnb -side top -anchor n \
                 -padx 3 -pady 3 -fill x 

            ### Nrrd
            set page [$step_tab.tnb add -label "Nrrd" -command "puts Nrrd"]

            message $page.instr -text "Please load a 4-D Nrrd containing the Diffusion Weighted Images." \
                -width 250
            pack $page.instr -side top -anchor nw -pady 5

            global $mods(NrrdReader1)-filename
            iwidgets::entryfield $page.file -labeltext "Nrrd File:" -labelpos w \
                -textvariable $mods(NrrdReader1)-filename \
                -command "$this execute_reader_and_set_tuple_axis"
            pack $page.file -side top -padx 3 -pady 6 -anchor n \
	        -fill x 

            button $page.load -text "Browse" \
                -command "$this load_nrrd" \
                -width 15
            pack $page.load -side top -anchor n -padx 3 -pady 6


            frame $page.b 
	    pack $page.b -side bottom -anchor se -pady 3 -padx 3 \
               -fill x 
	
            ### Dicom
            set page [$step_tab.tnb add -label "Dicom" -command "puts Dicom"]
            label $page.label -text "Not implemented yet."
	    pack $page.label -side top -anchor n

            ### Analyze
            set page [$step_tab.tnb add -label "Analyze" -command "puts Analyze"]
            label $page.label -text "Not implemented yet."
	    pack $page.label -side top -anchor n
            
            $step_tab.tnb view "Nrrd"

	    button $step_tab.ex -text "Next" \
                -command "$this execute_DataAcquisition" -width 8 \
                -background $next_color 
            pack $step_tab.ex -side bottom -anchor ne \
		-padx 5 -pady 5 


	    ### Registration
            set step_tab [$process.tnb add -label "Registration" \
                 -command {puts "Registration"}]

            
            global $mods(TendEpireg)-reference

            iwidgets::labeledframe $step_tab.refimg \
                 -labeltext "Reference Image" \
                 -labelpos nw
            pack $step_tab.refimg -side top -anchor n -padx 3 -pady 0

            set refimg [$step_tab.refimg childsite]

	    if {$case == 0} {
                 set ref_image1 $refimg
            } else {
                 set ref_image2 $refimg
            }
  
	    radiobutton $refimg.est -text "Implicit Reference: estimate distortion\nparameters from all images" \
                 -state disabled \
                 -variable $ref_image_state -value 0 \
                 -command "global mods; global  $mods(TendEpireg)-reference; set $mods(TendEpireg)-reference \"-1\"; $refimg.s.ref configure -state disabled; $refimg.s.label configure -state disabled"

            pack $refimg.est -side top -anchor nw -padx 2 -pady 0

            frame $refimg.s
            pack $refimg.s -side top -anchor nw -padx 2 -pady 0

            radiobutton $refimg.s.choose -text "Choose Reference Image:" \
                 -state disabled \
                 -variable $ref_image_state -value 1  \
                 -command "$refimg.s.ref configure -state normal; $refimg.s.label configure -state normal"

            label $refimg.s.label -textvariable $ref_image -state disabled
            pack $refimg.s.choose $refimg.s.label -side left -anchor n

            scale $refimg.s.ref -label "" \
                 -state disabled \
                 -variable $ref_image \
                 -from 1 -to 7 \
                 -showvalue false \
                 -length 150  -width 15 \
                 -sliderlength 15 \
                 -command "$this configure_reference_image" \
                 -orient horizontal
            pack $refimg.s.ref -side top -anchor ne -padx 0 -pady 0


	    global mods
	    global  $mods(TendEpireg)-reference
            set $mods(TendEpireg)-reference "-1"


            iwidgets::labeledframe $step_tab.blur \
                 -labeltext "Blur" \
                 -labelpos nw
            pack $step_tab.blur -side top -anchor n -padx 2 -pady 0 \
                 -fill x -expand 1

            set blur [$step_tab.blur childsite]

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


            iwidgets::labeledframe $step_tab.thresh \
                 -labeltext "Threshold" \
                 -labelpos nw
            pack $step_tab.thresh -side top -anchor n -padx 2 -pady 0 \
	         -fill x -expand 1

            set thresh [$step_tab.thresh childsite]
	  
	    global $mods(TendEpireg)-threshold
            global $mods(TendEpireg)-use-default-threshold
            radiobutton $thresh.auto -text "Automatically Determine Threshold             " \
                 -state disabled \
                 -variable $mods(TendEpireg)-use-default-threshold -value 1 
            pack $thresh.auto -side top -anchor nw -padx 2 -pady 0
            frame $thresh.choose
            pack $thresh.choose -side top -anchor nw -padx 0 -pady 0

            radiobutton $thresh.choose.button -text "Specify Threshold:" \
                 -state disabled \
                 -variable $mods(TendEpireg)-use-default-threshold -value 0 
            entry $thresh.choose.entry -width 10 \
                 -textvariable $mods(TendEpireg)-threshold \
                 -state disabled -foreground grey64
            pack $thresh.choose.button $thresh.choose.entry -side left -anchor n -padx 2 -pady 3

            $thresh.auto select


            checkbutton $step_tab.cc -variable $mods(TendEpireg)-cc_analysis \
                 -text "Connected Component Based Segmentation"\
                 -state disabled
            pack $step_tab.cc -side top -anchor nw -padx 8 -pady 1

            frame $step_tab.fit
            pack $step_tab.fit -side top -anchor nw -padx 8 -pady 1 \
               -fill x -expand 1

            global $mods(TendEpireg)-fitting
            label $step_tab.fit.l -text "Fitting: " -state disabled
            label $step_tab.fit.f -text "70" -state disabled
            label $step_tab.fit.p -text "%" -state disabled
            scale $step_tab.fit.s -label "" \
                 -state disabled \
                 -variable $mods(TendEpireg)-fitting \
                 -from .25 -to 1.0 \
                 -resolution 0.01 \
                 -length 80  -width 15 \
                 -sliderlength 15 \
                 -showvalue false \
                 -orient horizontal \
                 -command "$this configure_fitting_label $step_tab.fit.f"

            button $step_tab.fit.gradient -text "Load Gradients..." \
                 -command {puts "fix loading gradients file"} \
                 -state disabled

            pack $step_tab.fit.l $step_tab.fit.f $step_tab.fit.p \
                 $step_tab.fit.s  -side left \
                 -anchor nw -padx 0 -pady 0
            pack $step_tab.fit.gradient -side right -anchor ne \
                 -padx 2 -pady 0 -ipadx 4

            frame $step_tab.last
            pack $step_tab.last -side top -anchor n -fill x -expand 1 \
                 -padx 3 -pady 2

            iwidgets::optionmenu $step_tab.last.rf -labeltext "Resampling Filter" \
                -labelpos w -width 133 \
                -state disabled \
                -command "puts FIXME"
         
            pack $step_tab.last.rf -side left \
                 -anchor nw -padx 0 -pady 0

            $step_tab.last.rf insert end Catmull-Rom Linear "Windowed Sinc"

	    button $step_tab.last.ex -text "Next" -state disabled -width 8 \
	         -command "$this execute_registration" 
            pack $step_tab.last.ex -side right \
                 -anchor e -padx 2 -pady 0
              
	    ### Build DT
            set step_tab [$process.tnb add -label "Build DTs" \
                 -command {puts "DT"}]

	    frame $step_tab.b1
	    pack $step_tab.b1 -side top -anchor nw -fill x -expand 1

            button $step_tab.b1.ex -text "Next" -state disabled -width 10 \
                 -command "$this execute_dt"
            pack $step_tab.b1.ex -side top -anchor n \
                 -padx 5 -pady 5 


	    ### Progress
	    iwidgets::labeledframe $process.progress \
		-labelpos nw -labeltext "Progress" 
	    pack $process.progress -side bottom -anchor s -fill both
	    
	    set progress_section [$process.progress childsite]
	    iwidgets::feedback $progress_section.fb \
                -labeltext "$current_step..." \
		-labelpos nw \
		-steps 10 -barcolor $feedback_color \
                -barheight 20
		
	    pack $progress_section.fb -side top -padx 2 -pady 2 \
                -anchor nw -fill x
	    
	    if {$case == 0} {
	        set standalone_progress1 $progress_section.fb
                bind $standalone_progress1.lwchildsite.trough <Button> { app display_module_error }
                bind $standalone_progress1.lwchildsite.trough.bar <Button> { app display_module_error }

	        # Tooltip $standalone_progress1.lwchildsite.trough "Click progress bar when\nred to view errors"
	        # Tooltip $standalone_progress1.lwchildsite.trough.bar "Click progress bar when\nred to view errors"
	        # Tooltip $standalone_progress1.label "Indicates current step"
            } else {
	        set standalone_progress2 $progress_section.fb
                bind $standalone_progress2.lwchildsite.trough <Button> { app display_module_error }
                bind $standalone_progress2.lwchildsite.trough.bar <Button> { app display_module_error }

	        # Tooltip $standalone_progress2.lwchildsite.trough "Click progress bar when\nred to view errors"
	        # Tooltip $standalone_progress2.lwchildsite.trough.bar "Click progress bar when\nred to view errors"
	        # Tooltip $standalone_progress2.label "Indicates current step"

            }

            $process.tnb view "Data"

	    ### Attach/Detach button
            frame $m.d 
	    pack $m.d -side left -anchor e
            for {set i 0} {$i<25} {incr i} {
                button $m.d.cut$i -text " | " -borderwidth 0 \
                    -foreground "gray25" \
                    -activeforeground "gray25" \
                    -command "$this switch_P_frames" 
	        pack $m.d.cut$i -side top -anchor se -pady 0 -padx 0
                # Tooltip $m.d.cut$i "Click to $msg"
            }

	}
	    
    }

    method init_Vframe { m msg case} {
	global mods
	if { [winfo exists $m] } {
	    ### Visualization Frame
	    
	    iwidgets::labeledframe $m.vis \
		-labelpos n -labeltext "Visualization" 
	        # -background "LightSteelBlue3"
	    pack $m.vis -side right -anchor n -fill both -expand 1
	    
	    set vis [$m.vis childsite]

	    ### Tabs
	    iwidgets::tabnotebook $vis.tnb -width $notebook_width \
		-height 490 -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	    ### Data Vis Tab
	    set page [$vis.tnb add -label "Data Vis"]
	    iwidgets::scrolledlistbox $page.data -labeltext "Loaded Data" \
		-vscrollmode dynamic -hscrollmode dynamic \
		-selectmode single \
		-height 0.9i \
		-width $notebook_width \
		-labelpos nw -selectioncommand "$this data_selected"
	    
	    pack $page.data -padx 4 -pady 4 -anchor n
	    
	    if {$case == 0} {
		# detached case
		set data_listbox_Det $page.data
	    } else {
		# attached case
		set data_listbox_Att $page.data
	    }
	    
	    $page.data insert 0 {None}
	    
	    
	    ### Data Info
	    frame $page.f -relief groove -borderwidth 2
	    pack $page.f -side top -anchor n -fill x
	    
	    iwidgets::notebook $page.f.nb -width $notebook_width \
		-height $notebook_height
	    pack $page.f.nb -padx 4 -pady 4 -anchor n -fill both -expand 1

	    if {$case == 0} {
		# detached case
		set notebook_Det $page.f.nb
	    } else {
		# attached case
		set notebook_Att $page.f.nb
	    }
	    
	    ### Renderer Options Tab
	    create_viewer_tab $vis


	    ### Attach/Detach button
            frame $m.d 
	    pack $m.d -side left -anchor e
            for {set i 0} {$i<27} {incr i} {
                button $m.d.cut$i -text " | " -borderwidth 0 \
                    -foreground "gray25" \
                    -activeforeground "gray25" \
                    -command "$this switch_V_frames" 
	        pack $m.d.cut$i -side top -anchor se -pady 0 -padx 0
                # Tooltip $m.d.cut$i "Click to $msg"
            }
	}
    }

    method create_viewer_tab { vis } {
	global mods
	set page [$vis.tnb add -label "Global Options"]
	
	iwidgets::labeledframe $page.viewer_opts \
	    -labelpos nw -labeltext "Global Render Options"
	
	pack $page.viewer_opts -side top -anchor n -fill both -expand 1
	
	set view_opts [$page.viewer_opts childsite]
	
	frame $view_opts.eframe -relief flat
	pack $view_opts.eframe -side top -padx 4 -pady 4
	
	frame $view_opts.eframe.a -relief groove -borderwidth 2
	pack $view_opts.eframe.a -side left 
	
	
	checkbutton $view_opts.eframe.a.light -text "Lighting" \
	    -variable $mods(Viewer)-ViewWindow_0-global-light \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	checkbutton $view_opts.eframe.a.fog -text "Fog" \
	    -variable $mods(Viewer)-ViewWindow_0-global-fog \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	checkbutton $view_opts.eframe.a.bbox -text "BBox" \
	    -variable $mods(Viewer)-ViewWindow_0-global-debug \
	    -command "$mods(Viewer)-ViewWindow_0-c redraw"
	
	pack $view_opts.eframe.a.light $view_opts.eframe.a.fog \
	    $view_opts.eframe.a.bbox  \
	    -side top -anchor w -padx 4 -pady 4
	
	
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
	
	menu $view_opts.buttons.v1.views.def.m
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
	
	menu $view_opts.buttons.v1.views.def.m.posx
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
	
	menu $view_opts.buttons.v1.views.def.m.posy
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
	
	menu $view_opts.buttons.v1.views.def.m.posz
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
	
	menu $view_opts.buttons.v1.views.def.m.negx
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
	
	menu $view_opts.buttons.v1.views.def.m.negy
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
	
	menu $view_opts.buttons.v1.views.def.m.negz
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

            # configure reference state
            global mods
            global $mods(TendEpireg)-reference
            if {[set $mods(TendEpireg)-reference] == "-1"} {
              # re-disable reference image scale
	      $ref_image1.s.label configure -state disabled
	      $ref_image1.s.ref configure -state disabled
	      $ref_image2.s.label configure -state disabled
	      $ref_image2.s.ref configure -state disabled
            } else {
              # enable reference image scale
	      $ref_image1.s.label configure -state normal
	      $ref_image1.s.ref configure -state normal
	      $ref_image2.s.label configure -state normal
	      $ref_image2.s.ref configure -state normal
            }
	} else {
	    wm withdraw $detachedPFr
	    pack $attachedPFr -anchor n -side left -before $win.viewer \
	       -fill both -expand 1
	    set new_width [expr $c_width + $process_width]
            append geom $new_width x $c_height + [expr $x - $process_width] + $y
	    wm geometry $win $geom
	    set IsPAttached 1

            # configure reference state
            global mods
            global $mods(TendEpireg)-reference
            if {[set $mods(TendEpireg)-reference] == "-1"} {
              # re-disable reference image scale
	      $ref_image1.s.label configure -state disabled
	      $ref_image1.s.ref configure -state disabled
	      $ref_image2.s.label configure -state disabled
	      $ref_image2.s.ref configure -state disabled
            } else {
              # enable reference image scale
	      $ref_image1.s.label configure -state normal
	      $ref_image1.s.ref configure -state normal
	      $ref_image2.s.label configure -state normal
	      $ref_image2.s.ref configure -state normal
           }
	}
	update
    }

    method switch_V_frames {} {
	set c_width [winfo width $win]
	set c_height [winfo height $win]

      	set x [winfo x $win]
	set y [expr [winfo y $win] - 20]

	if { $IsVAttached } {


	    if {[$data_listbox_Att curselection] != ""} {
		$data_listbox_Det selection set [$data_listbox_Att curselection] [$data_listbox_Att curselection]
	    }
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
	    if {[$data_listbox_Det curselection] != ""} {
	    $data_listbox_Att selection set [$data_listbox_Det curselection] [$data_listbox_Det curselection]
	    }

	    wm withdraw $detachedVFr
	    pack $attachedVFr -anchor n -side left -after $win.viewer \
	       -fill both -expand 1
	    set new_width [expr $c_width + $vis_width]
            append geom $new_width x $c_height
	    wm geometry $win $geom
	    set IsVAttached 1
	}
	update
    }

    method add_data { name } {
	
	# add the data to the listbox
	if {[array size data] == 1} then {
	    # first data set should replace "none" in listbox and be selected
	    $data_listbox_Att delete 0
	    $data_listbox_Att insert 0 $name

	    $data_listbox_Det delete 0
	    $data_listbox_Det insert 0 $name

	    if {$IsVAttached} {
		$data_listbox_Att selection clear 0 end
		$data_listbox_Att selection set 0 0
		$data_listbox_Att see 0
	    } else {
		$data_listbox_Det selection clear 0 end
		$data_listbox_Det selection set 0 0
		$data_listbox_Det see 0
	    }
	} else {
	    # otherwise add to bottom and select
	    $data_listbox_Att insert end $name
	    $data_listbox_Det insert end $name

	    if {$IsVAttached} {
		$data_listbox_Att selection clear 0 end
		$data_listbox_Att selection set end end
		$data_listbox_Att see end
	    } else {
		$data_listbox_Det selection clear 0 end
		$data_listbox_Det selection set end end
		$data_listbox_Det see end
	    }
	}
    }

    method data_selected {} {
	global mods

	set current_data ""
	if {$IsVAttached} {
	    set current_data [$data_listbox_Att getcurselection]
	} else {
	    set current_data [$data_listbox_Det getcurselection]
	}
	if {[info exists data($current_data)] == 1} {

            # bring data info page forward
	    $notebook_Att view $current_data
	    $notebook_Det view $current_data
	} 
   }

   method load_nrrd {} {
	global mods
        $mods(NrrdReader1) make_file_open_box

	tkwait window .ui$mods(NrrdReader1)-fb

	update idletasks

        execute_reader_and_set_tuple_axis
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
	    puts $fileid "Save\n"
	    
	    close $fileid
	}
    }
    
    method load_session {} {
	global mods
	set types {
	    {{App Settings} {.set} }
	    {{Other} { * }}
	}

	set file [tk_getOpenFile -filetypes $types]
	if {$file != ""} {
		source $file

	}	
	
    }
        
    method exit_app {} {
	netedit quit
    }

    method show_help {} {
	puts "NEED TO IMPLEMENT SHOW HELP"
    }

    method get_file_name { filename } {
	set end [string length $filename]
	set start [string last "/" $filename]
	set start [expr 1 + $start]
	
	return [string range $filename $start $end]	
    }

    method build_original_data_info_page { w which } {
	set page [$w add -label $which]

	iwidgets::scrolledframe $page.sf \
	    -width $notebook_width \
            -height $notebook_height \
	    -labeltext "Data: $which" \
	    -vscrollmode static \
	    -hscrollmode none \
	    -background Grey

	pack $page.sf -anchor n

        set p [$page.sf childsite]

        global mods

    }

    method update_progress { which state } {

	global mods
        if {$current_step == "Data Acquisition"} {
	if {$which == $mods(NrrdReader1)} {
           if {$state == "JustStarted 1123"} {
     	      after 1 "$standalone_progress1 reset"
     	      after 1 "$standalone_progress2 reset"
           } elseif {$state == "Executing"} {
     	      after 1 "$standalone_progress1 step"
     	      after 1 "$standalone_progress2 step"

           } elseif {$state == "NeedData"} {

           } elseif {$state == "Completed"} {
              set remaining [$standalone_progress1 cget -steps]
	      after 1 "$standalone_progress1 step $remaining"
	      after 1 "$standalone_progress2 step $remaining"
              fill_in_data_pages
	   } 
	}
        }
    }


    method indicate_error { which msg_state } {
	if {$msg_state == "Error"} {
           if {$error_module == ""} {
              set error_module $which
	      # turn progress graph red
              $standalone_progress1 configure -barcolor $error_color
              $standalone_progress1 configure -labeltext "Error"

              $standalone_progress2 configure -barcolor $error_color
              $standalone_progress2 configure -labeltext "Error"
           }
	}
       if {$msg_state == "Reset" || $msg_state == "Remark" || \
           $msg_state == "Warning"} {
           if {$which == $error_module} {
	      set error_module ""
              $standalone_progress1 configure -barcolor $feedback_color
              $standalone_progress1 configure -labeltext "$current_step..."

              $standalone_progress2 configure -barcolor $feedback_color
              $standalone_progress2 configure -labeltext "$current_step..."
            }
       }

    }

    method block_connection { modA portA modB portB color} {
	set connection $modA
	append connection "_p$portA"
	append connection "_to_$modB"
	append connection "_p$portB"

	block_pipe $connection $modA $portA $modB $portB 
    }

    method execute_reader_and_set_tuple_axis {} {
       global mods 

        global $mods(NrrdReader1)-filename
        if {[set $mods(NrrdReader1)-filename] != ""} {
	        set current_nrrd [set $mods(NrrdReader1)-filename]

	        # set tuple axis to 0 always
		global $mods(NrrdReader1)-axis
	        set $mods(NrrdReader1)-axis 0

        	$mods(NrrdReader1)-c needexecute

                # Add data to vis list box
                global $mods(NrrdReader1)-filename
                set data(Original) [set $mods(NrrdReader1)-filename]
 
                set title "Variance of Original Diffusion Weighted Images"
                add_data $title


                # Build original data vis pages
		set page [$notebook_Att add -label $title]

		iwidgets::scrolledframe $page.sf \
		    -width $notebook_width \
        	    -height $notebook_height \
		    -labeltext "Data: $title" \
		    -vscrollmode static \
		    -hscrollmode none \
		    -background Grey

		pack $page.sf -anchor n

		set page [$notebook_Det add -label $title]

		iwidgets::scrolledframe $page.sf \
		    -width $notebook_width \
        	    -height $notebook_height \
		    -labeltext "Data: $title" \
		    -vscrollmode static \
		    -hscrollmode none \
		    -background Grey

		pack $page.sf -anchor n

                # bring new data info page forward
                $notebook_Att view $title
                $notebook_Det view $title

                # unblock section to visualize original data
 	        # unblock_data_pipes

                # execute section to visualize original data
	        # $mods(ShowField1)-c needexecute

                if {$current_step == "Data Acquisition"} {
                   activate_registration

                # re-disable reference image scale
	        $ref_image1.s.label configure -state disabled
	        $ref_image1.s.ref configure -state disabled
	        $ref_image2.s.label configure -state disabled
	        $ref_image2.s.ref configure -state disabled

                # configure ref image scale
                global $mods(NrrdInfo1)-size0
                $ref_image1.s.ref configure -from 1 -to [expr [set $mods(NrrdInfo1)-size0] + 1]
                $ref_image2.s.ref configure -from 1 -to [expr [set $mods(NrrdInfo1)-size0] + 1]
                }
                	
        } else {
            set answer [tk_messageBox -message \
                 "Please select a Nrrd by entering a file into the entry box or selecting the Browse button." -type ok -icon info]
        }
    }

    method fill_in_data_pages {} {
	global mods
        global $mods(UnuSlice1)-position
	   set f1 [$notebook_Att.cs.page0.cs.sf childsite]
	   set f2 [$notebook_Det.cs.page0.cs.sf childsite]
   
           message $f1.instr -width 250 \
               -text "Select a slice in the Z direction to view the variance."
           pack $f1.instr -side top -anchor nw -padx 3 -pady 3

           ### Slice Slider 
	   scale $f1.slice -label "Slice:" \
               -variable $mods(UnuSlice1)-position \
               -from 0 -to 34 \
               -showvalue true \
               -orient horizontal \
               -length 250

           pack $f1.slice -side top -anchor n -padx 3 -pady 3

	   scale $f2.slice -label "Slice:" \
               -variable $mods(UnuSlice1)-position \
               -from 0 -to 34 \
               -showvalue true \
               -orient horizontal \
               -length 5

           pack $f2.slice -side top -anchor n -padx 3 -pady 3

   	   bind $f1.slice <ButtonRelease> "$mods(UnuSlice1)-c needexecute"
   	   bind $f2.slice <ButtonRelease> "$mods(UnuSlice1)-c needexecute"

    }

    method execute_DataAcquisition {} {
        if {$current_step == "Registration"} {
            # view registration tab
            $proc_tab1 view "Registration"
            $proc_tab2 view "Registration"
            
        } else {
            set answer [tk_messageBox -message \
                 "Please complete the Data Acquisition step before Registration." -type ok -icon info]
        }

    }

    method unblock_data_pipes {} {
        global mods

	if {$data_blocked} {
          # block_connection $mods(NrrdChoose1) 0 $mods(UnuSlice1) 0 "purple"
          set data_blocked 0
       }
    }

    method activate_registration { } {
        global mods
	foreach w [winfo children $proc_tab1] {
	    activate_widget $w
        }

	foreach w [winfo children $proc_tab2] {
	    activate_widget $w
        }

        set current_step "Registration"
        $standalone_progress1 configure -labeltext "$current_step..."
        $standalone_progress2 configure -labeltext "$current_step..."

        # re-disable reference image scale
        $ref_image1.s.label configure -state disabled
	$ref_image1.s.ref configure -state disabled
	$ref_image2.s.label configure -state disabled
	$ref_image2.s.ref configure -state disabled

        # configure ref image scale
        global $mods(NrrdInfo1)-size0
        $ref_image1.s.ref configure -from 1 -to [expr [set $mods(NrrdInfo1)-size0] + 1]
        $ref_image2.s.ref configure -from 1 -to [expr [set $mods(NrrdInfo1)-size0] + 1]


        puts "unblock registration connections"
    }


    method execute_registration {} {
	global mods
 
        # execute modules

        if {$current_step == "Registration"} {

           activate_dt 

           puts "NEED TO BRING DT TAB FORWARD"

        }

    }


    method activate_dt { } {
        puts "change data registration color and borderwidth back"
	puts "hightlight vis section???"

	foreach w [winfo children $dt1] {
	    activate_widget $w
        }

	foreach w [winfo children $dt2] {
	    activate_widget $w
        }

        set current_step "Building Diffusion Tensors"
        $standalone_progress1 configure -labeltext "$current_step..."
        $standalone_progress2 configure -labeltext "$current_step..."

        puts "unblock dt connections"
    }


    method execute_dt {} {

    }

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
           }
        }

        foreach widg [winfo children $w] {
	     activate_widget $widg
        }
    }

    method configure_fitting_label { w val } {
	$w configure -text "[expr round([expr $val * 100])]"
    }

    method configure_reference_image { val } {
       if {$ref_image_state == 1} {
  	  global mods
          global $mods(TendEpireg)-reference
	  set $mods(TendEpireg)-reference [expr $val - 1]
       }
    }

    method display_module_error {} {
	set result [$error_module displayLog]
    }

    method indicate_dynamic_compile { which mode } {

	if {$mode == "start"} {
           $standalone_progress1 configure -labeltext "Compiling..."
           # Tooltip $standalone_progress1.label "Dynamically Compiling Algorithms.\nPlease see SCIRun Developer's\nGuide for more information"
           $standalone_progress2 configure -labeltext "Compiling..."
           # Tooltip $standalone_progress2.label "Dynamically Compiling Algorithms.\nPlease see SCIRun Developer's\nGuide for more information"
        } else {
           $standalone_progress1 configure -labeltext "$current_step..."
           # Tooltip $standalone_progress1.label "Indicates current step"
           $standalone_progress2 configure -labeltext "$current_step..."
           # Tooltip $standalone_progress2.label "Indicates current step"
        }
   }





    variable eviewer

    variable win

    variable data

    variable proc_tab1
    variable proc_tab2

    variable data_pages1
    variable data_pages2

    variable data_blocked
    variable current_nrrd
    public variable vol_1a
    public variable vol_1b
    public variable vol_2a
    public variable vol_2b
    variable original_plane
    variable original_slice

    # pointers to steps
    variable ref_image1
    variable ref_image2
    variable registration1
    variable registration2
    variable ref_image
    variable ref_image_state

    variable dt1
    variable dt2

    variable data_listbox_Att
    variable data_listbox_Det

    variable notebook_Att
    variable notebook_Det
    variable notebook_width
    variable notebook_height

    variable standalone_progress1
    variable standalone_progress2

    variable IsPAttached
    variable detachedPFr
    variable attachedPFr

    variable IsVAttached
    variable detachedVFr
    variable attachedVFr

    variable process_width
    variable process_height

    variable viewer_width
    variable viewer_height

    variable vis_width
    variable vis_height

    variable screen_width
    variable screen_height

    variable error_module

    variable current_step

    variable proc_color
    variable next_color
    variable feedback_color
    variable error_color

    variable steps1
    variable steps2

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


