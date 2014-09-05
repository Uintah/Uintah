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

setProgressText "Loading BioFEM Modules..."


#######################################################################
# Check environment variables.  Ask user for input if not set:
init_DATADIR_and_DATASET
############# NET ##############

::netedit dontschedule

global userName
set userName "dmw"

global runDate
set runDate " Tue  Jan 1 2002"

global runTime
set runTime " 14:25:13"

set m0 [addModuleAtPosition "SCIRun" "DataIO" "FieldReader" 9 8]
set m1 [addModuleAtPosition "BioPSE" "Forward" "SetupFEMatrix" 27 194]
set m2 [addModuleAtPosition "BioPSE" "Forward" "ApplyFEMCurrentSource" 45 288]
set m3 [addModuleAtPosition "SCIRun" "Math" "SolveMatrix" 27 352]
set m4 [addModuleAtPosition "SCIRun" "FieldsData" "ManageFieldData" 9 417]
set m5 [addModuleAtPosition "SCIRun" "FieldsData" "Gradient" 640 497]
set m6 [addModuleAtPosition "BioPSE" "Visualization" "ShowDipoles" 411 209]
set m7 [addModuleAtPosition "SCIRun" "Render" "Viewer" 393 822]
set m8 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 233 562]
set m9 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 233 625]
set m10 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 9 738]
set m11 [addModuleAtPosition "SCIRun" "DataIO" "FieldReader" 233 400]
set m12 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 215 503]
set m13 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 215 721]
set m14 [addModuleAtPosition "SCIRun" "FieldsCreate" "SampleField" 658 560]
set m15 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 622 745]
set m16 [addModuleAtPosition "SCIRun" "Visualization" "StreamLines" 640 621]
set m17 [addModuleAtPosition "SCIRun" "FieldsData" "DirectInterpolate" 622 682]
set m18 [addModuleAtPosition "SCIRun" "FieldsCreate" "FieldBoundary" 447 543]
set m19 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 447 607]
set m20 [addModuleAtPosition "SCIRun" "DataIO" "FieldReader" 300 24]
set m22 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 160 822]

setProgressText "Creating BioFEM Connections..."
addConnection $m0 0 $m1 0
addConnection $m0 0 $m2 0
addConnection $m1 0 $m3 0
addConnection $m2 0 $m3 1
addConnection $m3 0 $m4 1
addConnection $m0 0 $m4 0
addConnection $m4 0 $m5 0
addConnection $m6 0 $m2 1
addConnection $m8 0 $m9 0
addConnection $m9 0 $m22 1 
addConnection $m4 0 $m10 0
addConnection $m11 0 $m12 1
addConnection $m4 0 $m12 0
addConnection $m12 0 $m13 0
addConnection $m9 0 $m13 1
addConnection $m5 0 $m16 0
addConnection $m14 0 $m16 1
addConnection $m4 0 $m14 0
addConnection $m16 0 $m17 1
addConnection $m4 0 $m17 0
addConnection $m17 0 $m15 0
addConnection $m9 0 $m15 1
addConnection $m10 0 $m22 0
addConnection $m22 0 $m7 0
addConnection $m13 0 $m7 1
addConnection $m6 1 $m7 2
addConnection $m12 0 $m9 1
addConnection $m18 0 $m19 0
addConnection $m19 0 $m7 3
addConnection $m14 1 $m7 4
addConnection $m15 0 $m7 5
addConnection $m0 0 $m18 0
addConnection $m20 0 $m6 0

if {[file exists $DATADIR/$DATASET/$DATASET-mesh.tvt.fld]} {
    set $m0-filename $DATADIR/$DATASET/$DATASET-mesh.tvt.fld
} else {
    set $m0-filename ""
}

set $m3-target_error {0.000001}
set $m3-orig_error {1}
set $m3-current_error {9.84662e-07}
set $m3-iteration {86}
set $m3-maxiter {500}
set $m3-use_previous_soln {1}
set $m3-emit_partial {0}

set $m6-widgetSizeGui_ [expr 0.05 * ${global-scale}]
set $m6-scaleModeGui_ {normalize}
set $m6-showLastVecGui_ {1}
set $m6-showLinesGui_ {0}

#$m7 initialize_ui
set $m7-ViewWindow_0-view-eyep-x ${view-eyep-x}
set $m7-ViewWindow_0-view-eyep-y ${view-eyep-y}
set $m7-ViewWindow_0-view-eyep-z ${view-eyep-z}
set $m7-ViewWindow_0-view-lookat-x ${view-lookat-x}
set $m7-ViewWindow_0-view-lookat-y ${view-lookat-y}
set $m7-ViewWindow_0-view-lookat-z ${view-lookat-z}
set $m7-ViewWindow_0-view-up-x ${view-up-x}
set $m7-ViewWindow_0-view-up-y ${view-up-y}
set $m7-ViewWindow_0-view-up-z ${view-up-z}
set $m7-ViewWindow_0-view-fov ${view-fov}
set $m7-ViewWindow_0-specular-scale {0.4}

set $m8-mapName {Rainbow}
set $m8-resolution {256}
set $m8-realres {256}

set $m10-extract-from-new-field {1}
set $m10-update_type {on release}
set $m10-active_tab {NOISE}

if {[file exists $DATADIR/$DATASET/$DATASET-electrodes.pcd.fld]} {
    set $m11-filename $DATADIR/$DATASET/$DATASET-electrodes.pcd.fld
} else {
    set $m11-filename ""
}

set $m12-exhaustive_search {1}

set $m13-nodes-on {1}
set $m13-edges-on {0}
set $m13-faces-on {0}
set $m13-text-on {0}
set $m13-text-color-r {1.0}
set $m13-text-color-g {1.0}
set $m13-text-color-b {1.0}
set $m13-text-fontsize {1}
set $m13-text-precision {3}
set $m13-text-render_locations {0}
set $m13-text-show-data {1}
set $m13-text-show-nodes {0}
set $m13-text-show-edges {0}
set $m13-text-show-faces {0}
set $m13-text-show-cells {0}
set $m13-node_display_type {Spheres}
set $m13-node_scale [expr 0.03 * ${global-scale}]
set $m13-resolution {7}

set $m14-maxseeds {50}
set $m14-numseeds {35}

set $m15-nodes-on {0}
set $m15-edges-on {1}
set $m15-faces-on {0}
set $m15-edge_display_type {Cylinders}
set $m15-node_scale [expr 0.01 * ${global-scale}]
set $m15-edge_scale [expr 0.01 * ${global-scale}]
set $m15-edge-resolution {8}

set $m16-stepsize [expr 0.004 * ${global-scale}]
set $m16-tolerance [expr 0.004 * ${global-scale}]
set $m16-maxsteps {250}
set $m16-method {5}

set $m17-exhaustive_search {0}

set $m19-def-color-r {0.5}
set $m19-def-color-g {0.5}
set $m19-def-color-b {0.5}
set $m19-def-color-a {0.85}
set $m19-nodes-on {0}
set $m19-edges-on {1}
set $m19-faces-on {0}
set $m19-edges-transparency {1}
set $m19-edge_display_type {Lines}
set $m19-edge_scale {1.0}

if {[file exists $DATADIR/$DATASET/$DATASET-dipole.pcv.fld]} {
    set $m20-filename $DATADIR/$DATASET/$DATASET-dipole.pcv.fld
} else {
    set $m20-filename ""
}

set $m22-nodes-on {0}
set $m22-edges-on {0}
set $m22-faces-on {1}
set $m22-use-normals {1}

::netedit scheduleok


# global array indexed by module name to keep track of modules
global mods

set mods(Viewer) $m7

set mods(FieldReader-conductivities) $m0
set mods(FieldReader-electrodes) $m11
set mods(FieldReader-probe) $m20

set mods(Isosurface) $m10
set mods(ShowField-Isosurface) $m22

set mods(StreamLines) $m16
set mods(StreamLines-rake) $m14
set mods(StreamLines-Gradient) $m5
set mods(ShowField-StreamLines) $m15

set mods(ShowField-Electrodes) $m13

set mods(GenStandardColorMaps) $m8

set mods(ShowDipole) $m6

set mods(SolveMatrix) $m3

#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

setProgressText "Creating BioFEM GUI..."

set auto_index(::PowerAppBase) "source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/PowerAppBase.app"

class BioFEMApp {
    inherit ::PowerAppBase
    
    method appname {} {
	return "BioFEM"
    }
    
    constructor {} {
	toplevel .standalone
	wm title .standalone "BioFEM"	 
	set win .standalone
	
	set notebook_width 290
	set notebook_height 600
	
	set viewer_width 640
	set viewer_height 670
	
	set vis_width [expr $notebook_width + 60]
	set vis_height $viewer_height

        set initialized 0


        set i_width 300
        set i_height 20
        set stripes 10

        set vis_frame_tab1 ""
        set vis_frame_tab2 ""
	set c_left_tab ""

        set vis_tab1 ""
        set vis_tab2 ""
     
        set error_module ""

        set isosurface_tab1 ""
        set isosurface_tab2 ""

        set streamlines_tab1 ""
        set streamlines_tab2 ""

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

    }
    

    destructor {
	destroy $this
    }

    
    method build_app {} {
	global mods
	
	# Embed the Viewer
	set eviewer [$mods(Viewer) ui_embedded]
	$eviewer setWindow $win.viewer $viewer_width $viewer_height
	set_dataset 0

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
	
	init_Vframe $detachedVFr.f 1
	init_Vframe $attachedVFr.f 2

	# call back to re-configure isosurface slider
	global $mods(Isosurface)-isoval-max
	trace variable $mods(Isosurface)-isoval-max w "$this set_minmax_callback"

	### pack 3 frames
 	pack $attachedVFr -side right \
 	    -anchor n -fill both -expand 0

 	pack $win.viewer -side right \
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

        set initialized 1

	global PowerAppSession
	if {[info exists PowerAppSession] && [set PowerAppSession] != ""} { 
	    set saveFile $PowerAppSession
	    wm title .standalone "BioFEM - [getFileName $saveFile]"
	    $this load_session_data
	}
    }

    method set_dataset { andexec } {
	global mods
	global DATADIR
	global DATASET

	source $DATADIR/$DATASET/$DATASET.settings

	#Fix up global scale.
	global $mods(ShowDipole)-widgetSizeGui_
	global $mods(ShowField-Electrodes)-node_scale
	global $mods(ShowField-StreamLines)-node_scale
	global $mods(ShowField-StreamLines)-edge_scale
	global $mods(StreamLines)-stepsize
	global $mods(StreamLines)-tolerance
	set $mods(ShowDipole)-widgetSizeGui_ [expr 0.05 * ${global-scale}]
	set $mods(ShowField-Electrodes)-node_scale [expr 0.03 * ${global-scale}]
	set $mods(ShowField-StreamLines)-node_scale [expr 0.01 * ${global-scale}]
	set $mods(ShowField-StreamLines)-edge_scale [expr 0.01 * ${global-scale}]
	set $mods(StreamLines)-stepsize [expr 0.004 * ${global-scale}]
	set $mods(StreamLines)-tolerance [expr 0.004 * ${global-scale}]

	global $mods(FieldReader-conductivities)-filename
	set $mods(FieldReader-conductivities)-filename $DATADIR/$DATASET/$DATASET-mesh.tvt.fld
	global $mods(FieldReader-electrodes)-filename
	set $mods(FieldReader-electrodes)-filename $DATADIR/$DATASET/$DATASET-electrodes.pcd.fld
	global $mods(FieldReader-probe)-filename
	set $mods(FieldReader-probe)-filename $DATADIR/$DATASET/$DATASET-dipole.pcv.fld

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


    method update_local_filenames { junk0 junk1 junk2 } {
	global mods
	global $mods(FieldReader-conductivities)-filename
	global $mods(FieldReader-electrodes)-filename
	global $mods(FieldReader-probe)-filename
	global filenameconductivities filenameelectrodes filenameprobe

	set tmp [set $mods(FieldReader-conductivities)-filename]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set filenameconductivities [string range $tmp $pos end]
	} else {
	    set filenameconductivities $tmp
	}

	set tmp [set $mods(FieldReader-electrodes)-filename]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set filenameelectrodes [string range $tmp $pos end]
	} else {
	    set filenameelectrodes $tmp
	}

	set tmp [set $mods(FieldReader-probe)-filename]
	set pos [expr [string last "/" $tmp] + 1]
	if {$pos != -1} {
	    set filenameprobe [string range $tmp $pos end]
	} else {
	    set filenameprobe $tmp
	}
    }

    method init_data_selection_frame { f } {
        global mods

	$this update_local_filenames junk junk junk
	global $mods(FieldReader-conductivities)-filename
	trace variable $mods(FieldReader-conductivities)-filename w "$this update_local_filenames"
	global $mods(FieldReader-electrodes)-filename
	trace variable $mods(FieldReader-electrodes)-filename w "$this update_local_filenames"
	global $mods(FieldReader-probe)-filename
	trace variable $mods(FieldReader-probe)-filename w "$this update_local_filenames"

	iwidgets::labeledframe $f.dataset \
	    -labelpos n -labeltext "DATASET" 
	pack $f.dataset -side top -anchor w -fill x
	    
	set dataset [$f.dataset childsite]

	radiobutton $dataset.brain-eg -text "Head Model (70K nodes, 396K elements)" -variable DATASET -value brain-eg -command "$this set_dataset 1"
	radiobutton $dataset.cyl3 -text "Cylinder Phantom (2.7K nodes, 13K elements)" -variable DATASET -value cyl3 -command "$this set_dataset 1"
	radiobutton $dataset.sphere -text "Sphere Phantom (1K nodes, 6K elements)" -variable DATASET -value sphere -command "$this set_dataset 1"
	radiobutton $dataset.utahtorso-lowres -text "Utah Torso Lowres (8K nodes, 51K elements)" -variable DATASET -value utahtorso-lowres -command "$this set_dataset 1"
	radiobutton $dataset.utahtorso -text "Utah Torso (169K nodes, 1083K elements)" -variable DATASET -value utahtorso -command "$this set_dataset 1"

	pack $dataset.brain-eg $dataset.cyl3 $dataset.sphere $dataset.utahtorso-lowres $dataset.utahtorso -anchor w -side top

	frame $f.datadir
	label $f.datadir.l -text "Data Directory:" -width 14 -anchor w
	entry $f.datadir.e -textvar DATADIR -width 120 -relief flat
	pack $f.datadir.l $f.datadir.e -side left
	pack $f.datadir -padx 5 -anchor w

	frame $f.cond
	label $f.cond.l -text "Conductivity File:" -width 14 -anchor w
	label $f.cond.e -textvar filenameconductivities -width 28 -anchor w
	button $f.cond.b -text Browse -command "$mods(FieldReader-conductivities) initialize_ui"
	pack $f.cond.l $f.cond.e $f.cond.b -side left
	pack $f.cond -padx 5 -anchor w


	frame $f.elec
	label $f.elec.l -text "Electrodes File:" -width 14 -anchor w
	label $f.elec.e -textvar filenameelectrodes -width 28 -anchor w
	button $f.elec.b -text Browse -command "$mods(FieldReader-electrodes) initialize_ui"
	pack $f.elec.l $f.elec.e $f.elec.b -side left
	pack $f.elec -padx 5 -anchor w


	frame $f.probe
	label $f.probe.l -text "Probe File:" -width 14 -anchor w
	label $f.probe.e -textvar filenameprobe -width 28 -anchor w
	button $f.probe.b -text Browse -command "$mods(FieldReader-probe) initialize_ui"
	pack $f.probe.l $f.probe.e $f.probe.b -side left
	pack $f.probe -padx 5 -anchor w

        global $mods(SolveMatrix)-target_error
	set err [set $mods(SolveMatrix)-target_error]

	iwidgets::labeledframe $f.graph -labelpos n -labeltext "Convergence" 
	pack $f.graph -side bottom -anchor w -fill x

	set g [$f.graph childsite]

	blt::graph $g.graph -height 200 -plotbackground gray99
	$g.graph yaxis configure -logscale true -title "Error (RMS)"  -min [expr $err/10] -max 1 -loose true
	$g.graph xaxis configure -title "Iteration" \
		-loose true
	bind $g.graph <ButtonPress-1> "$mods(SolveMatrix) select_error $g.graph %x %y"
	bind $g.graph <Button1-Motion> "$mods(SolveMatrix) move_error $g.graph %x %y"
	bind $g.graph <ButtonRelease-1> "$mods(SolveMatrix) deselect_error $g.graph %x %y"
	set iter 1
	$g.graph element create "Current Target" -linewidth 0
	$g.graph element configure "Current Target" -data "0 $err" \
		-symbol diamond
	pack $g.graph -fill x
        $mods(SolveMatrix) add_graph $g.graph
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
		-height [expr $vis_height - 160] -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	    set vis_frame_tab$case $vis.tnb
	    set data [$vis.tnb add -label "Data Selection" -command "$this change_vis_frame 0"]

	    init_data_selection_frame $data
	    

	    set page [$vis.tnb add -label "Vis Options" -command "$this change_vis_frame 1"]


	    ### Isosurface
	    iwidgets::labeledframe $page.isoframe -labelpos nw \
		-labeltext "Isopotential Surface"

	    set iso [$page.isoframe childsite]
	    set isosurface_tab$case $iso
	    build_isosurface_tab $iso
	    
            pack $page.isoframe -padx 4 -pady 4 -fill x

	    
	    ### StreamLines
	    iwidgets::labeledframe $page.slframe -labelpos nw \
		-labeltext "Electric Field Lines"

	    set sl [$page.slframe childsite]
	    set streamlines_tab$case $sl	    
	    build_streamlines_tab $sl
	    
            pack $page.slframe -padx 4 -pady 4 -fill x


	    ### Electrodes
	    iwidgets::labeledframe $page.elecframe -labelpos nw \
		-labeltext "Electrodes"

	    set elec [$page.elecframe childsite]
	    
	    build_electrodes_tab $elec
	    
            pack $page.elecframe -padx 4 -pady 4 -fill x
	    

	    ### ColorMaps
	    iwidgets::labeledframe $page.colorframe -labelpos nw \
		-labeltext "Color Map for Potentials"

	    set color [$page.colorframe childsite]
	    
	    build_colormap_tab $color
	    
            pack $page.colorframe -padx 4 -pady 4 -fill x

	    
	    ### Renderer Options Tab
	    create_viewer_tab $vis

            $vis.tnb view "Vis Options"
	    
	    
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

	    set data_ex_button$case $vis.last.ex


            ### Indicator
	    frame $vis.indicator -relief sunken -borderwidth 2
            pack $vis.indicator -side bottom -anchor s -padx 3 -pady 5
	    
	    canvas $vis.indicator.canvas -bg "white" -width $i_width \
	        -height $i_height 
	    pack $vis.indicator.canvas -side top -anchor n -padx 3 -pady 3
	    
            bind $vis.indicator <Button> {app display_module_error} 
	    
            label $vis.indicatorL -text "Press Execute to Load Data..."
            pack $vis.indicatorL -side bottom -anchor sw -padx 5 -pady 3
	    
	    
	    if {$case == 1} {
		set indicator0 $vis.indicator.canvas
		set indicatorL0 $vis.indicatorL
	    } else {
		set indicator1 $vis.indicator.canvas
		set indicatorL1 $vis.indicatorL
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
	    pack $attachedVFr -anchor n -side right -before $win.viewer \
	       -fill both -expand 0
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
	    wm title .standalone "BioFEM - [getFileName $saveFile]" 

	    set fileid [open $saveFile w]
	    
	    # Save out data information 
	    puts $fileid "# BioFEM Session\n"
	    puts $fileid "set app_version 1.0"

	    save_module_variables $fileid
	    # ShowDipoles uses the position of the input dipole
	    # regardless of what was saved out. By setting 
	    # num-dipoles to 0, instead of 1, the
	    # module will disregard the position values that cause
	    # a saved session to get degenerate cylinders and hang.
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
	
	wm title .standalone "BioFEM - [getFileName $saveFile]"

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
	tk_messageBox -message "Please refer to the online BioFEM Tutorial\nhttp://software.sci.utah.edu/doc/User/BioFEMTutorial" -type ok -icon info -parent .standalone
    }
    
    method show_about {} {
	tk_messageBox -message "BioFEM is a SCIRun PowerApp that computes the electric field in a volume produced by a set of dipoles. BioFEM computes a solution to the bioelectric field forward problem. BioFEM also computes voltage values at electrode positions, which can be compared with values recorded via ECG or EKG." -type ok -icon info -parent .standalone
    }
    

    method indicate_dynamic_compile { which mode } {
	global mods

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
	
	if {$which == $mods(ShowField-StreamLines) && $state == "JustStarted"} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-StreamLines) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Isosurface) && $state == "JustStarted"} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Isosurface) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Electrodes) && $state == "JustStarted"} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Electrodes) && $state == "Completed"} {
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

	global $mods(StreamLines-rake)-force-rake-reset
	set $mods(StreamLines-rake)-force-rake-reset 1

	global $mods(ShowDipole)-force-field-reset
	set $mods(ShowDipole)-force-field-reset 1

	netedit scheduleall
    }
    

    method toggle_streamlines {} {
	global mods
	global $mods(ShowField-StreamLines)-edges-on
	if { [set $mods(ShowField-StreamLines)-edges-on] } {
	    disableModule $mods(StreamLines) 0
	    disableModule $mods(StreamLines-rake) 0
	    disableModule $mods(StreamLines-Gradient) 0
	    set "$eviewer-StreamLines rake (5)" 1
	    $eviewer-c redraw
	    $streamlines_tab1.isoval.s configure -state normal
	    $streamlines_tab2.isoval.s configure -state normal
	    $streamlines_tab1.isoval.l configure -state normal
	    $streamlines_tab2.isoval.l configure -state normal
	    $streamlines_tab1.isoval.val configure -state normal
	    $streamlines_tab2.isoval.val configure -state normal
	    $streamlines_tab1.fast configure -state normal
	    $streamlines_tab2.fast configure -state normal
	    $streamlines_tab1.adapt configure -state normal
	    $streamlines_tab2.adapt configure -state normal
	    bind $streamlines_tab1.isoval.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_tab2.isoval.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_tab1.isoval.val <Return> \
		"$mods(StreamLines-rake)-c needexecute"
	    bind $streamlines_tab2.isoval.val <Return> \
		"$mods(StreamLines-rake)-c needexecute"
	} else {
	    disableModule $mods(StreamLines) 1
	    disableModule $mods(StreamLines-rake) 1
	    disableModule $mods(StreamLines-Gradient) 1
	    set "$eviewer-StreamLines rake (5)" 0
	    $eviewer-c redraw
	    $streamlines_tab1.isoval.s configure -state disabled
	    $streamlines_tab2.isoval.s configure -state disabled
	    $streamlines_tab1.isoval.l configure -state disabled
	    $streamlines_tab2.isoval.l configure -state disabled
	    $streamlines_tab1.isoval.val configure -state disabled
	    $streamlines_tab2.isoval.val configure -state disabled
	    $streamlines_tab1.fast configure -state disabled
	    $streamlines_tab2.fast configure -state disabled
	    $streamlines_tab1.adapt configure -state disabled
	    $streamlines_tab2.adapt configure -state disabled
	    bind $streamlines_tab1.isoval.s <ButtonRelease> ""
	    bind $streamlines_tab2.isoval.s <ButtonRelease> ""
	    bind $streamlines_tab1.isoval.val <Return> ""
	    bind $streamlines_tab2.isoval.val <Return> ""
	}
	$mods(ShowField-StreamLines)-c toggle_display_edges
    }


    method build_streamlines_tab { f } {
	global mods
	global $mods(ShowField-StreamLines)-edges-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Electric Field Lines" \
		-variable $mods(ShowField-StreamLines)-edges-on \
		-command "$this toggle_streamlines"
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    # Isoval
	    frame $f.isoval
	    pack $f.isoval -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.isoval.l -text "Number of Field Lines:"
	    scale $f.isoval.s -from 1 -to 200 \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution 1 \
		-variable $mods(StreamLines-rake)-maxseeds \
		-showvalue false \
		-orient horizontal
	    
	    bind $f.isoval.s <ButtonRelease> \
		"$mods(StreamLines-rake)-c needexecute"
	    
	    entry $f.isoval.val -width 3 -relief flat \
		-textvariable $mods(StreamLines-rake)-maxseeds
	    
	    bind $f.isoval.val <Return> "$mods(StreamLines-rake)-c needexecute"

	    pack $f.isoval.l $f.isoval.s $f.isoval.val \
		-side left -anchor n -padx 3      
	    
	    radiobutton $f.fast -text "Fast Tracking" \
		-variable $mods(StreamLines)-method -value 5 \
		-command "$mods(StreamLines-rake)-c needexecute"
	    radiobutton $f.adapt -text "Adaptive Tracking" \
		-variable $mods(StreamLines)-method -value 4 \
		-command "$mods(StreamLines-rake)-c needexecute"

	    pack $f.fast $f.adapt -side top -anchor w -padx 20
	}
    }


    method build_electrodes_tab { f } {
	global mods
	global $mods(ShowField-Electrodes)-nodes-on
	global $mods(ShowField-Electrodes)-text-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Electrodes" \
		-variable $mods(ShowField-Electrodes)-nodes-on \
		-command "$mods(ShowField-Electrodes)-c toggle_display_nodes"
	    pack $f.show -side top -anchor nw -padx 3 -pady 3

	    checkbutton $f.text -text "Print Potentials at Electrodes" \
		-variable $mods(ShowField-Electrodes)-text-on \
		-command "$mods(ShowField-Electrodes)-c toggle_display_text"

	    pack $f.text -side top -anchor nw -padx 3 -pady 3
	}
    }


    method build_isosurface_tab { f } {
	global mods
	global $mods(ShowField-Isosurface)-faces-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Isopotential Surface" \
		-variable $mods(ShowField-Isosurface)-faces-on \
		-command "$mods(ShowField-Isosurface)-c toggle_display_faces"
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    # Isoval
	    frame $f.isoval
	    pack $f.isoval -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.isoval.l -text "Electric Potential Value:"
	    scale $f.isoval.s -from -2.0 -to 2.0 \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution 0.0001 \
		-variable $mods(Isosurface)-isoval \
		-showvalue false \
		-orient horizontal \
                -command "$mods(Isosurface) updateSliderEntry $mods(Isosurface)-isoval $mods(Isosurface)-isoval-typed"

	    bind $f.isoval.s <ButtonRelease> \
		"$mods(Isosurface)-c needexecute"

	    entry $f.isoval.val -width 5 -relief flat \
		-textvariable $mods(Isosurface)-isoval

	    bind $f.isoval.val <Return> "$mods(Isosurface)-c needexecute"

	    pack $f.isoval.l $f.isoval.s $f.isoval.val \
		-side left -anchor nw -padx 3

            frame $f.buttons
            pack $f.buttons -side top -anchor w

	    checkbutton $f.buttons.normals -text "Smooth Faces" \
		    -variable $mods(ShowField-Isosurface)-use-normals \
		    -command "$mods(ShowField-Isosurface)-c rerender_faces"

	    pack $f.buttons.normals -side left -anchor n -padx 20

	    checkbutton $f.buttons.update -text "Continuous Updates" \
		    -variable $mods(Isosurface)-update_type \
                    -offvalue "on release" -onvalue "Auto"

	    pack $f.buttons.update -side left -anchor n -padx 20
	}
    }	 

    method set_minmax_callback {varname varele varop} {
	global mods
 	global $mods(Isosurface)-isoval-min $mods(Isosurface)-isoval-max
 	set min [set $mods(Isosurface)-isoval-min]
 	set max [set $mods(Isosurface)-isoval-max]

	set w $isosurface_tab1.isoval.s
 	if [ expr [winfo exists $w] ] {
 	    $w configure -from $min -to $max
 	    $w configure -resolution [expr ($max - $min)/10000.]
 	}

	set w $isosurface_tab2.isoval.s
 	if [ expr [winfo exists $w] ] {
 	    $w configure -from $min -to $max
 	    $w configure -resolution [expr ($max - $min)/10000.]
 	}
    }
	
    method build_colormap_tab { f } {
	global mods
	if {![winfo exists $f.show]} {
	    
	    set isocolor $f
	    frame $isocolor.select
	    pack $isocolor.select -side top -anchor nw -padx 3 -pady 3
	    
	    set maps $f
	    global $mods(GenStandardColorMaps)-mapName
	    
	    # Gray
	    frame $maps.gray
	    pack $maps.gray -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.gray.b -text "Gray" \
		-variable $mods(GenStandardColorMaps)-mapName \
		-value Gray \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.gray.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.gray.f -relief sunken -borderwidth 2
	    pack $maps.gray.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.gray.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.gray.f.canvas -anchor e \
		-fill both -expand 1
	    
	    draw_colormap Gray $maps.gray.f.canvas
	    
	    # Rainbow
	    frame $maps.rainbow2
	    pack $maps.rainbow2 -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.rainbow2.b -text "Rainbow" \
		-variable $mods(GenStandardColorMaps)-mapName \
		-value Rainbow \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.rainbow2.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.rainbow2.f -relief sunken -borderwidth 2
	    pack $maps.rainbow2.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.rainbow2.f.canvas -bg "#ffffff" \
		-height $colormap_height -width $colormap_width
	    pack $maps.rainbow2.f.canvas -anchor e
	    
	    draw_colormap Rainbow $maps.rainbow2.f.canvas
	    
	    # Darkhue
	    frame $maps.darkhue
	    pack $maps.darkhue -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.darkhue.b -text "Darkhue" \
		-variable $mods(GenStandardColorMaps)-mapName \
		-value Darkhue \
		-command "$mods(GenStandardColorMaps)-c needexecute"
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
		-variable $mods(GenStandardColorMaps)-mapName \
		-value Blackbody \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.blackbody.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.blackbody.f -relief sunken -borderwidth 2 
	    pack $maps.blackbody.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.blackbody.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.blackbody.f.canvas -anchor e
	    
	    draw_colormap Blackbody $maps.blackbody.f.canvas
	    
	    # Don
	    frame $maps.don
	    pack $maps.don -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.don.b -text "Don" \
		-variable $mods(GenStandardColorMaps)-mapName \
		-value Don \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.don.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.don.f -relief sunken -borderwidth 2 
	    pack $maps.don.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.don.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.don.f.canvas -anchor e
	    
	    draw_colormap Don $maps.don.f.canvas
	    
	    # Blue-to-Red
	    frame $maps.bpseismic
	    pack $maps.bpseismic -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.bpseismic.b -text "Blue-to-Red" \
		-variable $mods(GenStandardColorMaps)-mapName \
		-value "BP Seismic" \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "Blue-to-Red" $maps.bpseismic.f.canvas
	}
    }
    
    
    
    method toggle_show_isosurface {} {
       global mods
       global $mods(ShowField-Isosurface)-faces-on
 
	if {[set $mods(ShowField-Isosurface)-faces-on] == 1} {
	    foreach w [winfo children $isosurface_tab1] {
		enable_widget $w
	    }
	    foreach w [winfo children $isosurface_tab2] {
		enable_widget $w
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
	
	$mods(ShowField-Isosurface)-c toggle_display_faces
    }



    method change_vis_tab { which } {
	# change vis tab for attached/detached

        if {$initialized != 0} {
	    if {$which == "Isosurface"} {
		# Isosurface
		$vis_tab1 view "Isosurface"
		$vis_tab2 view "Isosurface"
	    } elseif {$which == "StreamLines"} {
		# StreamLines
		$vis_tab1 view "StreamLines"
		$vis_tab2 view "StreamLines"
	    }
	}
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
    
    # Visualiztion frame tabnotebook
    variable vis_frame_tab1
    variable vis_frame_tab2
    variable c_left_tab

    # Vis tabs notebook
    variable vis_tab1
    variable vis_tab2

    variable isosurface_tab1
    variable isosurface_tab2

    variable streamlines_tab1
    variable streamlines_tab2

    # Application placing and size
    variable notebook_width
    variable notebook_height
}

BioFEMApp app

setProgressText "Displaying BioFEM GUI..."

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
