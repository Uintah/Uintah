# SCI Network 1.0
#
# The contents of this file are subject to the University of Utah Public
# License (the "License"); you may not use this file except in compliance
# with the License.
# 
# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations under
# the License.
# 
# The Original Source Code is SCIRun, released March 12, 2001.
# 
# The Original Source Code was developed by the University of Utah.
# Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
# University of Utah. All Rights Reserved.

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

if { $results == "failed" } {

    ::netedit scheduleok
    return 

} else {

    set DATADIR [lindex $results 0]
    set DATASET [lindex $results 1]
}

source $DATADIR/$DATASET/$DATASET.settings

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

set $m0-filename $DATADIR/$DATASET/$DATASET-mesh.tvt.fld

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

set $m8-mapType {3}
set $m8-minRes {12}
set $m8-resolution {255}
set $m8-realres {128}

set $m10-extract-from-new-field {1}
set $m10-update_type {on release}
set $m10-active_tab {NOISE}

set $m11-filename $DATADIR/$DATASET/$DATASET-electrodes.pcd.fld

set $m13-nodes-on {1}
set $m13-edges-on {0}
set $m13-faces-on {0}
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
set $m15-resolution {8}

set $m16-stepsize [expr 0.004 * ${global-scale}]
set $m16-tolerance [expr 0.004 * ${global-scale}]
set $m16-maxsteps {250}
set $m16-method {5}

set $m19-def-color-r {0.3}
set $m19-def-color-g {0.3}
set $m19-def-color-b {0.3}
set $m19-def-color-a {0.85}
set $m19-nodes-on {0}
set $m19-edges-on {1}
set $m19-faces-on {0}
set $m19-edges-transparency {1}
set $m19-edge_display_type {Lines}
set $m19-edge_scale {1.0}

set $m20-filename $DATADIR/$DATASET/$DATASET-dipole.pcv.fld

set $m22-nodes-on {0}
set $m22-edges-on {0}
set $m22-faces-on {1}

::netedit scheduleok


# global array indexed by module name to keep track of modules
global mods

# set mods(NrrdReader1) $m0
# set mods(DicomToNrrd1) $m97
# set mods(AnalyzeToNrrd1) $m96
# set mods(ChooseNrrd1) $m45
# set mods(NrrdInfo1) $m62
# 
# ### Original Data Stuff
# set mods(UnuSlice1) $m1
# set mods(UnuProject1) $m57
# set mods(ShowField-Orig) $m8
# set mods(ChooseNrrd-ToProcess) $m113
# 
# ### Registered
# set mods(ShowField-Reg) $m9
# set mods(UnuSlice2) $m5
# set mods(UnuJoin) $m47
# set mods(TendEpireg) $m2
# set mods(ChooseNrrd2) $m49
# set mods(ChooseNrrd-ToReg) $m105
# set mods(ChooseNrrd-ToSmooth) $m104
# 
# set mods(NrrdReader-Gradient) $m50
# 
# set mods(NrrdReader-BMatrix) $m63
# 
# ### T2 Reference Image
# set mods(NrrdReader-T2) $m48
# set mods(DicomToNrrd-T2) $m99
# set mods(AnalyzeToNrrd-T2) $m100
# set mods(ChooseNrrd-T2) $m98
# 
# 
# set mods(GenStandardColorMaps1)  $m58
# set mods(RescaleColorMap2) $m61
# 
# ### Build DT
# set mods(TendEstim) $m3
# set mods(UnuResample-XY) $m103
# set mods(UnuResample-Z) $m106
# set mods(ChooseNrrd-DT) $m114
# 
# ### Planes
# set mods(ChooseField-ColorPlanes) $m27
# set mods(GenStandardColorMaps-ColorPlanes) $m29
# set mods(RescaleColorMap-ColorPlanes) $m30
# 
# ### Isosurface
# set mods(ShowField-Isosurface) $m13
# set mods(ChooseField-Isoval) $m23
# set mods(ChooseField-Isosurface) $m24
# set mods(GenStandardColorMaps-Isosurface) $m26
# set mods(RescaleColorMap-Isosurface) $m25
# set mods(Isosurface) $m17
# 
# # Planes
# set mods(SamplePlane-X) $m67
# set mods(SamplePlane-Y) $m68
# set mods(SamplePlane-Z) $m69
# 
# set mods(QuadToTri-X) $m73
# set mods(QuadToTri-Y) $m74
# set mods(QuadToTri-Z) $m75
# 
# set mods(ChooseField-X) $m33
# set mods(ChooseField-Y) $m82
# set mods(ChooseField-Z) $m83
# 
# set mods(ShowField-X) $m31
# set mods(ShowField-Y) $m86
# set mods(ShowField-Z) $m87
# 

### Viewer
set mods(Viewer) $m7

set mods(FieldReader1) $m0
set mods(FieldReader2) $m11

set mods(Isosurface) $m10
set mods(ShowField-Isosurface) $m22

set mods(StreamLines) $m16
set mods(StreamLines-rake) $m14
set mods(ShowField-StreamLines) $m15

set mods(ShowField-Electrodes) $m13

set mods(GenStandardColorMaps) $m8

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


#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

class ForwardFEMApp {
    
    method modname {} {
	return "ForwardFEMApp"
    }
    
    constructor {} {
	toplevel .standalone
	wm title .standalone "ForwardFEM"	 
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

        set indicator1 ""
        set indicator2 ""
        set indicatorL1 ""
        set indicatorL2 ""
        set indicate 0
        set cycle 0
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
        set size_y 0
        set size_z 0

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

        set streamlines_tab1 ""
        set streamlines_tab2 ""

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

        # colormaps
        set colormap_width 150
        set colormap_height 15
        set colormap_res 64

        set indicatorID 0

	### Define Tooltips
	##########################
	# General
	set tips(Indicator) "Indicates progress of\napplication. Click when\nred to view errors"

	# Data Acquisition Tab
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

        set initialized 1

    }


    method init_Pframe { m case } {
        global mods
        
	if { [winfo exists $m] } {
	    ### Menu
	    frame $m.main_menu -relief raised -borderwidth 3
	    pack $m.main_menu -fill x -anchor nw


	    menubutton $m.main_menu.file -text "File" -underline 0 \
		-menu $m.main_menu.file.menu
	    
	    menu $m.main_menu.file.menu -tearoff false

	    $m.main_menu.file.menu add command -label "Load       Ctr+O" \
		-underline 1 -command "$this load_session" -state active
	    
	    $m.main_menu.file.menu add command -label "Save      Ctr+S" \
		-underline 0 -command "$this save_session" -state active
	    
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

	    $m.main_menu.help.menu add command -label "About ForwardFEM" \
		-underline 0 -command "$this show_about" -state active
	    
	    pack $m.main_menu.help -side left
	    
	    tk_menuBar $m.main_menu $win.main_menu.file $win.main_menu.help
	    

	    ### Processing Steps
	    #####################
	    iwidgets::labeledframe $m.p \
		-labelpos n -labeltext "Data Selection" 
	    pack $m.p -side left -fill both -anchor n -expand 1
	    
	    set process [$m.p childsite]

	    # Execute and Next buttons
            frame $process.last
            pack $process.last -side bottom -anchor ne \
		-padx 5 -pady 5
	    
            button $process.last.ex -text "Execute" \
		-background $execute_color \
		-activebackground $execute_color \
		-width 8 \
		-command "$this execute_Data"
	    Tooltip $process.last.ex $tips(Execute-DataAcquisition)

	    button $process.last.ne -text "Next" \
                -command "$this change_processing_tab Registration" -width 8 \
                -activebackground $next_color \
                -background grey75 -state disabled 
	    Tooltip $process.last.ne $tips(Next-DataAcquisition)

            pack $process.last.ne $process.last.ex -side right -anchor ne \
		-padx 2 -pady 0

	    if {$case == 0} {
		set data_next_button1 $process.last.ne
		set data_ex_button1 $process.last.ex
	    } else {
		set data_next_button2 $process.last.ne
		set data_ex_button2 $process.last.ex
	    }

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
	    
            construct_indicator $process.indicator.canvas
	    
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
	    
	    set page [$vis.tnb add -label "Data Vis" -command "$this change_vis_frame 0"]


	    ### Isosurface
	    iwidgets::labeledframe $page.isoframe -labelpos nw \
		-labeltext "IsoSurface"

	    set iso [$page.isoframe childsite]
	    
	    build_isosurface_tab $iso
	    
            pack $page.isoframe -padx 4 -pady 4 -fill x

	    
	    ### StreamLines
	    iwidgets::labeledframe $page.slframe -labelpos nw \
		-labeltext "StreamLines"

	    set sl [$page.slframe childsite]
	    
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
		-labeltext "Color Map"

	    set color [$page.colorframe childsite]
	    
	    build_colormap_tab $color
	    
            pack $page.colorframe -padx 4 -pady 4 -fill x

	    
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
	set page [$vis.tnb add -label "Global Options" -command "$this change_vis_frame 1"]
	
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
	    puts $fileid "# ForwardFEM Session\n"
	    puts $fileid "set version 1.0"

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
	    if {![array exists $var] && $var != "this"} {
		puts $fileid "set $var \{[set $var]\}"
	    }
	}
    }
    
    
    method save_global_variables { fileid } {
	puts $fileid "\n# Global Variables\n"
	
	#foreach g [info globals] {
	# puts $fileid "set $g \{[set $g]\}"
	#}
	
	# Save out my globals by hand because otherwise they conflict with
	# the module variables
	global data_mode
	puts $fileid "set data_mode $data_mode"
	
	### planes variables that must be globals (all checkbuttons)
	global show_planes
	puts $fileid "set show_planes \{$show_planes\}"
	global show_plane_x
	puts $fileid "set show_plane_x $show_plane_x"
	global show_plane_y
	puts $fileid "set show_plane_y $show_plane_y"
	global show_plane_z
	puts $fileid "set show_plane_z $show_plane_z"
	global plane_x
	puts $fileid "set plane_x $plane_x"
	global plane_y
	puts $fileid "set plane_y $plane_y"
	global plane_z
	puts $fileid "set plane_z $plane_z"
	
	### registration globals
	global ref_image
	puts $fileid "set ref_image $ref_image"
	global ref_image_state
	puts $fileid "set ref_image_state $ref_image_state"
	
	global clip_to_isosurface
	puts $fileid "set clip_to_isosurface $clip_to_isosurface"
	
	global clip_to_isosurface_color
	puts $fileid "set clip_to_isosurface_color $clip_to_isosurface_color"
	global clip_to_isosurface_color-r
	puts $fileid "set clip_to_isosurface_color-r $clip_to_isosurface_color-r"
	global clip_to_isosurface_color-g
	puts $fileid "set clip_to_isosurface_color-g $clip_to_isosurface_color-g"
	global clip_to_isosurface_color-b
	puts $fileid "set clip_to_isosurface_color-b $clip_to_isosurface_color-b"
	
	global bmatrix
	puts $fileid "set bmatrix $bmatrix"
	
	### DT Globals
	global xy_radius
	puts $fileid "set xy_radius $xy_radius"
	global z_radius 
	puts $fileid "set z_radius $z_radius"
	
	### isosurface variables
	global clip_by_planes
	puts $fileid "set clip_by_planes $clip_by_planes"
	
	global do_registration 
	puts $fileid "set do_registration $do_registration"
	
	global do_smoothing
	puts $fileid "set do_smoothing $do_smoothing"
	
	global isosurface_color
	puts $fileid "set isosurface_color $isosurface_color"
	global isosurface_color-r
	puts $fileid "set isosurface_color-r $isosurface_color-r"
	global isosurface_color-g
	puts $fileid "set isosurface_color-g $isosurface_color-g"
	global isosurface_color-b
	puts $fileid "set isosurface_color-b $isosurface_color-b"
	
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
	    
	    # configure attach/detach
	    
	    # configure all tabs by calling all configure functions
	    
	    # activate proper step tabs
	    
            # fix next buttons
	    
	    # execute?
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
	tk_messageBox -message "Please refer to the online ForwardFEM Tutorial\nhttp://software.sci.utah.edu/doc/User/ForwardFEMTutorial" -type ok -icon info -parent .standalone
    }
    
    method show_about {} {
	tk_messageBox -message "ForwardFEM About Box" -type ok -icon info -parent .standalone
    }
    
    method display_module_error {} {
        if {$error_module != ""} {
	    set result [$error_module displayLog]
        }
    }
    
    method indicate_dynamic_compile { which mode } {
	if {$mode == "start"} {
	    change_indicator_labels "Dynamically Compiling Code..."
        } else {
	    if {$dt_completed} {
		change_indicator_labels "Visualization..."
	    } elseif {$reg_completed} {
		change_indicator_labels "Building Diffusion Tensors..."
	    } elseif {$data_completed} {
		change_indicator_labels "Registration..."
	    } else {
		change_indicator_labels "Data Acquisition..."
	    }
	}
    }
    
    
    method update_progress { which state } {
	global mods
	global $mods(ShowField-Isosurface)-faces-on
	global show_plane_x show_plane_y show_plane_z
	
	return
	
	if {$which == $mods(FieldReader1) && $state == "NeedData"} {
	    change_indicator_labels "Data Acquisition..."
	    change_indicate_val 1
	} elseif {$which == $mods(FieldReader1) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(FieldReader2) && $state == "NeedData"} {
	    change_indicator_labels "Data Acquisition..."
	    change_indicate_val 1
	} elseif {$which == $mods(FieldReader2) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(ShowField-Orig) && $state == "Completed"} {
	    after 100
	    # Bring images into view
	    $mods(Viewer)-ViewWindow_0-c autoview
	    global $mods(Viewer)-ViewWindow_0-pos 
	    set $mods(Viewer)-ViewWindow_0-pos "z1_y0"
	    $mods(Viewer)-ViewWindow_0-c Views
	} elseif {$which == $mods(ShowField-Reg) && $state == "Completed"} {
	    after 100
	    # Bring images into view
	    $mods(Viewer)-ViewWindow_0-c autoview
	    global $mods(Viewer)-ViewWindow_0-pos 
	    set $mods(Viewer)-ViewWindow_0-pos "z1_y0"
	    $mods(Viewer)-ViewWindow_0-c Views
        } elseif {$which == $mods(TendEstim) && $state == "NeedData"} {
	    change_indicator_labels "Building Diffusion Tensors..."
	    change_indicate_val 1
	} elseif {$which == $mods(TendEstim) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(NrrdInfo1) && $state == "Completed"} {
	    global $mods(NrrdInfo1)-size1
	    
            global data_mode
	    if {[info exists $mods(NrrdInfo1)-size1]} {
		global $mods(NrrdInfo1)-size0
		global $mods(NrrdInfo1)-size1
		global $mods(NrrdInfo1)-size2
		global $mods(NrrdInfo1)-size3
		
		set volumes [set $mods(NrrdInfo1)-size0]
		set size_x [expr [set $mods(NrrdInfo1)-size1] - 1]
		set size_y [expr [set $mods(NrrdInfo1)-size2] - 1]
		set size_z [expr [set $mods(NrrdInfo1)-size3] - 1]
		
		configure_sample_planes
		
		if {$data_mode == "DWI"} {
		    # new data has been loaded, build/configure
		    # the variance slider
		    fill_in_data_pages
		    
		    # reconfigure registration reference image slider
		    $ref_image1.s.ref configure -from 1 -to $volumes
		    $ref_image2.s.ref configure -from 1 -to $volumes
		} else {
		    set data_completed 1
		    set reg_completed 1
		    set dt_completed 1
		    activate_vis
		}
	    }
        } elseif {$which == $mods(SamplePlane-X) && $state == "NeedData" && $show_plane_x == 1} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-X) && $state == "Completed" && $show_plane_x == 1} {
	    if {$dt_completed} {
		activate_vis
	    }
	    change_indicate_val 0
	} elseif {$which == $mods(SamplePlane-Y) && $state == "NeedData" && $show_plane_y == 1} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Y) && $state == "Completed" && $show_plane_y == 1} {
	    change_indicate_val 0
	} elseif {$which == $mods(SamplePlane-Z) && $state == "NeedData" && $show_plane_z == 1} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Z) && $state == "Completed" && $show_plane_z == 1} {
	    change_indicate_val 0
	} elseif {$which == $mods(Isosurface) && $state == "NeedData" && [set $mods(ShowField-Isosurface)-faces-on] == 1} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Isosurface) && $state == "Completed" && [set $mods(ShowField-Isosurface)-faces-on] == 1} {
	    change_indicate_val 0
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
	set plane_x [expr $size_x/2]
	set plane_y [expr $size_y/2]
	set plane_z [expr $size_z/2]
	
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
	set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $plane_x) && (y $clip_y $plane_y) && (z$clip_z $plane_z)"
        
	
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
	    activate_registration

	    activate_dt
	
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

    method execute_Data {} {
	global mods 
	global data_mode
	
	
	$mods(FieldReader1)-c needexecute
	#$mods(FieldReader2)-c needexecute
	set data_completed 1	
	#activate_registration

	# enable Next button
	$data_next_button1 configure -state normal \
	    -foreground black -background $next_color
	$data_next_button2 configure -state normal \
	    -foreground black -background $next_color
    }
    
    method toggle_streamlines {} {
	global mods
	global $mods(ShowField-StreamLines)-edges-on
	if { [set $mods(ShowField-StreamLines)-edges-on] } {
	    disableModule $mods(StreamLines-rake) 0
	    set "$eviewer-StreamLines rake (5)" 1
	    $eviewer-c redraw
	} else {
	    disableModule $mods(StreamLines-rake) 1
	    set "$eviewer-StreamLines rake (5)" 0
	    $eviewer-c redraw
	}
	$mods(ShowField-StreamLines)-c toggle_display_edges
    }

    method build_streamlines_tab { f } {
	global mods
	global $mods(ShowField-StreamLines)-edges-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show StreamLines" \
		-variable $mods(ShowField-StreamLines)-edges-on \
		-command "$this toggle_streamlines"
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    # Isoval
	    frame $f.isoval
	    pack $f.isoval -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.isoval.l -text "Seeds:"
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
	    
	    radiobutton $f.fast -text "Fast" \
		-variable $mods(StreamLines)-method -value 5 \
		-command "$mods(StreamLines-rake)-c needexecute"
	    radiobutton $f.adapt -text "Adaptive" \
		-variable $mods(StreamLines)-method -value 4 \
		-command "$mods(StreamLines-rake)-c needexecute"

	    pack $f.fast $f.adapt -side top -anchor w -padx 20
	}
    }


    method build_electrodes_tab { f } {
	global mods
	global $mods(ShowField-Electrodes)-nodes-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show Electrodes" \
		-variable $mods(ShowField-Electrodes)-nodes-on \
		-command "$mods(ShowField-Electrodes)-c toggle_display_nodes"
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	}
    }


    method build_isosurface_tab { f } {
	global mods
	global $mods(ShowField-Isosurface)-faces-on

	if {![winfo exists $f.show]} {
	    checkbutton $f.show -text "Show IsoSurface" \
		-variable $mods(ShowField-Isosurface)-faces-on \
		-command "$mods(ShowField-Isosurface)-c toggle_display_faces"
	    pack $f.show -side top -anchor nw -padx 3 -pady 3
	    
	    # Isoval
	    frame $f.isoval
	    pack $f.isoval -side top -anchor nw -padx 3 -pady 3
	    
	    label $f.isoval.l -text "Isoval:"
	    scale $f.isoval.s -from -2.0 -to 2.0 \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution 0.0001 \
		-variable $mods(Isosurface)-isoval \
		-showvalue false \
		-orient horizontal

	    bind $f.isoval.s <ButtonRelease> \
		"$mods(Isosurface)-c needexecute"
	    
	    entry $f.isoval.val -width 5 -relief flat \
		-textvariable $mods(Isosurface)-isoval

	    bind $f.isoval.val <Return> "$mods(Isosurface)-c needexecute"

	    pack $f.isoval.l $f.isoval.s $f.isoval.val \
		-side left -anchor nw -padx 3      
	}
    }	    

    method build_colormap_tab { f } {
	global mods
	if {![winfo exists $f.show]} {
	    global isosurface_color

	    set isocolor $f
	    frame $isocolor.select
	    pack $isocolor.select -side top -anchor nw -padx 3 -pady 3
	    
	    set maps $f
	    global $mods(GenStandardColorMaps)-mapType
	    
	    # Gray
	    frame $maps.gray
	    pack $maps.gray -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.gray.b -text "Gray" \
		-variable $mods(GenStandardColorMaps)-mapType \
		-value 0 \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.gray.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.gray.f -relief sunken -borderwidth 2
	    pack $maps.gray.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.gray.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.gray.f.canvas -anchor e \
		-fill both -expand 1
	    
	    draw_colormap Gray $maps.gray.f.canvas
	    
	    # Reverse Rainbow
	    frame $maps.rainbow1
	    pack $maps.rainbow1 -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.rainbow1.b -text "Rainbow1" \
		-variable $mods(GenStandardColorMaps)-mapType \
		-value 3 \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.rainbow1.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.rainbow1.f -relief sunken -borderwidth 2
	    pack $maps.rainbow1.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.rainbow1.f.canvas -bg "#ffffff" \
		-height $colormap_height -width $colormap_width
	    pack $maps.rainbow1.f.canvas -anchor e
	    
	    draw_colormap Rainbow1 $maps.rainbow1.f.canvas

	    # Rainbow
	    frame $maps.rainbow2
	    pack $maps.rainbow2 -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.rainbow2.b -text "Rainbow2" \
		-variable $mods(GenStandardColorMaps)-mapType \
		-value 2 \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.rainbow2.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.rainbow2.f -relief sunken -borderwidth 2
	    pack $maps.rainbow2.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.rainbow2.f.canvas -bg "#ffffff" \
		-height $colormap_height -width $colormap_width
	    pack $maps.rainbow2.f.canvas -anchor e
	    
	    draw_colormap Rainbow2 $maps.rainbow2.f.canvas
	    
	    # Darkhue
	    frame $maps.darkhue
	    pack $maps.darkhue -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.darkhue.b -text "Darkhue" \
		-variable $mods(GenStandardColorMaps)-mapType \
		-value 5 \
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
		-variable $mods(GenStandardColorMaps)-mapType \
		-value 7 \
		-command "$mods(GenStandardColorMaps)-c needexecute"
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
		-variable $mods(GenStandardColorMaps)-mapType \
		-value 17 \
		-command "$mods(GenStandardColorMaps)-c needexecute"
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "BP Seismic" $maps.bpseismic.f.canvas
	}
    }
    
    
    method select_isosurface_color { w } {
	global mods
       	global $mods(ChooseField-Isosurface)-port-index
	
	set which [$w.color get]
	
        if {$which == "Principle Eigenvector"} {
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(RescaleColorMap-Isosurface) 1
	    set $mods(ChooseField-Isosurface)-port-index 3
        } elseif {$which == "Fractional Anisotropy"} {
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	    
	    disableModule $mods(RescaleColorMap-Isosurface) 0
	    set $mods(ChooseField-Isosurface)-port-index 0
        } elseif {$which == "Linear Anisotropy"} {
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	   
	    disableModule $mods(RescaleColorMap-Isosurface) 0
	    set $mods(ChooseField-Isosurface)-port-index 1
        } elseif {$which == "Planar Anisotropy"} {
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled	    
	    disableModule $mods(RescaleColorMap-Isosurface) 0
	    set $mods(ChooseField-Isosurface)-port-index 2
        } else {
	    # constant color
	    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state normal
	    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state normal	   
	    disableModule $mods(RescaleColorMap-Isosurface) 1
        }

	$isosurface_tab1.isocolor.childsite.select.color select $which
	$isosurface_tab2.isocolor.childsite.select.color select $which
	
        # execute 
        $mods(ChooseField-Isosurface)-c needexecute
    }
    

    method draw_colormap { which canvas } {
	set color ""
	if {$which == "Gray"} {
	    set color { "Gray" { { 0 0 0 } { 255 255 255 } } }
	} elseif {$which == "Rainbow1"} {
	    set color { "Rainbow1" {	
		{ 0 0 255} { 0 102 255}
		{ 0 204 255} { 0 255 204}
		{ 0 255 102} { 0 255 0}
		{ 102 255 0} { 204 255 0}
		{ 255 234 0} { 255 204 0}
		{ 255 102 0} { 255 0 0}}}
	} elseif {$which == "Rainbow2"} {
	    set color { "Rainbow2" {	
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
    
    method setColor {col color mod} {
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

            $mods(Isosurface)-c needexecute
	}
    }

   
    
    method select_color_planes_color { w } {
        global mods
	global $mods(ChooseField-ColorPlanes)-port-index
	
        set which [$w.color get]
	
        if {$which == "Principle Eigenvector"} {
	    set plane_type "Principle Eigenvector"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(RescaleColorMap-ColorPlanes) 1
	    set $mods(ChooseField-ColorPlanes)-port-index 3
        } elseif {$which == "Fractional Anisotropy"} {
	    set plane_type "Fractional Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(RescaleColorMap-ColorPlanes) 0
	    set $mods(ChooseField-ColorPlanes)-port-index 0
        } elseif {$which == "Linear Anisotropy"} {
	    set plane_type "Linear Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(RescaleColorMap-ColorPlanes) 0
	    set $mods(ChooseField-ColorPlanes)-port-index 1
        } elseif {$which == "Planar Anisotropy"} {
	    set plane_type "Planar Anisotropy"
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state disabled
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state disabled
	    disableModule $mods(RescaleColorMap-ColorPlanes) 0
	    set $mods(ChooseField-ColorPlanes)-port-index 2
        } else {
	    set plane_type "Constant"
	    # specified color
            $planes_tab1.color.childsite.select.colorFrame.set_color configure -state normal
            $planes_tab2.color.childsite.select.colorFrame.set_color configure -state normal
	    disableModule $mods(RescaleColorMap-ColorPlanes) 1
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

        # 1
        set plane(-X) "on"
        global $clip-normal-x-1
        set $clip-normal-x-1 "-1.0"
        global $clip-normal-d-1 
        set $clip-normal-d-1 [expr -$plane_x + $plane_inc]
        global $clip-visible-1
        set $clip-visible-1 1

        # 2
        set plane(+X) "off"
        global $clip-normal-x-2
        set $clip-normal-x-2 1.0
        global $clip-normal-d-2 
        set $clip-normal-d-2 [expr $plane_x + $plane_inc]

        # 3
        set plane(-Y) "on"
        global $clip-normal-y-3
        set $clip-normal-y-3 "-1.0"
        global $clip-normal-d-3 
        set $clip-normal-d-3 [expr -$plane_y + $plane_inc]
        global $clip-visible-3
        set $clip-visible-3 1

        # 4
        set plane(+Y) "off"
        global $clip-normal-y-4
        set $clip-normal-y-4 1.0
        global $clip-normal-d-4 
        set $clip-normal-d-4 [expr $plane_y + $plane_inc]

        # 5
        set plane(-Z) "on"
        global $clip-normal-z-5
        set $clip-normal-z-5 "-1.0"
        global $clip-normal-d-5 
        set $clip-normal-d-5 [expr -$plane_z + $plane_inc]
        global $clip-visible-5
        set $clip-visible-5 1

        # 6
        set plane(+Z) "off"
        global $clip-normal-z-6
        set $clip-normal-z-6 1.0
        global $clip-normal-d-6 
        set $clip-normal-d-6 [expr $plane_z + $plane_inc]

        $mods(Viewer)-ViewWindow_0-c redraw
    }

    method toggle_clip_by_planes { w } {
	global mods
        global clip_by_planes
        global $mods(Viewer)-ViewWindow_0-global-clip
        if {$clip_by_planes == 0} {
	    set $mods(Viewer)-ViewWindow_0-global-clip 0
	    $isosurface_tab1.clip.flipx configure -state disabled
	    $isosurface_tab2.clip.flipx configure -state disabled
	    
	    $isosurface_tab1.clip.flipy configure -state disabled
	    $isosurface_tab2.clip.flipy configure -state disabled
	    
	    $isosurface_tab1.clip.flipz configure -state disabled
	    $isosurface_tab2.clip.flipz configure -state disabled
        } else {
	    set $mods(Viewer)-ViewWindow_0-global-clip 1

	    $isosurface_tab1.clip.flipx configure -state normal
	    $isosurface_tab2.clip.flipx configure -state normal
	    
	    $isosurface_tab1.clip.flipy configure -state normal
	    $isosurface_tab2.clip.flipy configure -state normal
	    
	    $isosurface_tab1.clip.flipz configure -state normal
	    $isosurface_tab2.clip.flipz configure -state normal
        }

        $mods(Viewer)-ViewWindow_0-c redraw
    }

    method flip_x_clipping_plane {} {
        global mods
        global show_plane_x
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
           set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $plane_x) && (y $clip_y $plane_y) && (z$clip_z $plane_z)"

           $mods(Viewer)-ViewWindow_0-c redraw
        }
    }

    method flip_y_clipping_plane {} {
        global mods
        global show_plane_y
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
           set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $plane_x) && (y $clip_y $plane_y) && (z$clip_z $plane_z)"

           $mods(Viewer)-ViewWindow_0-c redraw
        }
    }

    method flip_z_clipping_plane {} {
        global mods
        global show_plane_z
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
           set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $plane_x) && (y $clip_y $plane_y) && (z$clip_z $plane_z)"

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
        disableModule $mods(QuadToTri-X) 0
        disableModule $mods(QuadToTri-Y) 0
        disableModule $mods(QuadToTri-Z) 0
 
        set $mods(ChooseField-X)-port-index 1
        set $mods(ChooseField-Y)-port-index 1
        set $mods(ChooseField-Z)-port-index 1
       } else {
	# disable Unstructure modules and change ChooseField port to 0
        disableModule $mods(QuadToTri-X) 1
        disableModule $mods(QuadToTri-Y) 1
        disableModule $mods(QuadToTri-Z) 1
 
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
 
       if {$size_x != 0} {
          # set the sample plane position to be the normalized value
          set result [expr [expr $plane_x / [expr $size_x / 2.0] ] - 1.0]
          set $mods(SamplePlane-X)-pos $result

          # set the glabal clipping planes values
          set clip $mods(Viewer)-ViewWindow_0-clip
          global $clip-normal-d-1
          global $clip-normal-d-2
          set $clip-normal-d-1 [expr -$plane_x  + $plane_inc]
          set $clip-normal-d-2 [expr $plane_x  + $plane_inc]

          # configure ClipByFunction
          global $mods(ClipByFunction-Seeds)-clipfunction
          global $mods(Isosurface)-isoval
          set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $plane_x) && (y $clip_y $plane_y) && (z$clip_z $plane_z)"

          $mods(SamplePlane-X)-c needexecute
          $mods(Viewer)-ViewWindow_0-c redraw
       }
    }

    method update_plane_y {} {
       global mods plane_x plane_y plane_z
       global $mods(SamplePlane-Y)-pos
 
       if {$size_y != 0} {
          # set the sample plane position to be the normalized value
          set result [expr [expr $plane_y / [expr $size_y / 2.0] ] - 1.0]
          set $mods(SamplePlane-Y)-pos $result

          # set the glabal clipping planes values
          set clip $mods(Viewer)-ViewWindow_0-clip
          global $clip-normal-d-3
          global $clip-normal-d-4
          set $clip-normal-d-3 [expr -$plane_y  + $plane_inc]
          set $clip-normal-d-4 [expr $plane_y  + $plane_inc]

          # configure ClipByFunction
          global $mods(Isosurface)-isoval
          global $mods(ClipByFunction-Seeds)-clipfunction
          set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $plane_x) && (y $clip_y $plane_y) && (z$clip_z $plane_z)"

          $mods(SamplePlane-Y)-c needexecute
          $mods(Viewer)-ViewWindow_0-c redraw
       }
    }

    method update_plane_z {} {
       global mods plane_x plane_y plane_z
       global $mods(SamplePlane-Z)-pos
 
       if {$size_z != 0} {
          # set the sample plane position to be the normalized value
          set result [expr [expr $plane_z / [expr $size_z / 2.0] ] - 1.0]
          set $mods(SamplePlane-Z)-pos $result

          # set the glabal clipping planes values
          set clip $mods(Viewer)-ViewWindow_0-clip
          global $clip-normal-d-5
          global $clip-normal-d-6
          set $clip-normal-d-5 [expr -$plane_z  + $plane_inc]
          set $clip-normal-d-6 [expr $plane_z  + $plane_inc]

          # configure ClipByFunction
          global $mods(Isosurface)-isoval
          global $mods(ClipByFunction-Seeds)-clipfunction
          set $mods(ClipByFunction-Seeds)-clipfunction "(v > [set $mods(Isosurface)-isoval]) && (x $clip_x $plane_x) && (y $clip_y $plane_y) && (z$clip_z $plane_z)"

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

    method toggle_show_isosurface {} {
       global mods
       global $mods(ShowField-Isosurface)-faces-on
 
	if {[set $mods(ShowField-Isosurface)-faces-on] == 1} {
	    foreach w [winfo children $isosurface_tab1] {
		activate_widget $w
	    }
	    foreach w [winfo children $isosurface_tab2] {
		activate_widget $w
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


    method fill_in_data_pages {} {
	global mods
        global $mods(UnuSlice1)-position
	   set f1 $variance_tab1
	   set f2 $variance_tab2

        global $mods(NrrdInfo1)-size3
        set num_slices [expr [set $mods(NrrdInfo1)-size3] - 1]
           if {![winfo exists $f1.instr]} {
              # detached
              checkbutton $f1.orig -text "View Variance of Original Data" \
                  -variable $mods(ShowField-Orig)-faces-on \
                  -command {
                     global mods
                     $mods(ShowField-Orig)-c toggle_display_faces
                   }

              checkbutton $f1.reg -text "View Variance of Registered Data" \
                  -variable $mods(ShowField-Reg)-faces-on \
                  -state disabled \
                  -command {
                     global mods
                     $mods(ShowField-Reg)-c toggle_display_faces
                   }

              pack $f1.orig $f1.reg -side top -anchor nw -padx 3 -pady 3

              message $f1.instr -width [expr $notebook_width - 60] \
                  -text "Select a slice in the Z direction to view the variance."
              pack $f1.instr -side top -anchor n -padx 3 -pady 3

              ### Slice Slider 
	      scale $f1.slice -label "Slice:" \
                  -variable $mods(UnuSlice1)-position \
                  -from 0 -to $num_slices \
                  -showvalue true \
                  -orient horizontal \
                  -command "$this change_variance_slice" \
                  -length [expr $notebook_width - 60]

              pack $f1.slice -side top -anchor n -padx 3 -pady 3

      	      bind $f1.slice <ButtonRelease> "app update_variance_slice"


              # attached
              checkbutton $f2.orig -text "View Variance of Original Data" \
                  -variable $mods(ShowField-Orig)-faces-on \
                  -command {
                     global mods
                     $mods(ShowField-Orig)-c toggle_display_faces
                   }

              checkbutton $f2.reg -text "View Variance of Registered Data" \
                  -variable $mods(ShowField-Reg)-faces-on \
                  -state disabled \
                  -command {
                     global mods
                     $mods(ShowField-Reg)-c toggle_display_faces
                   }

              pack $f2.orig $f2.reg -side top -anchor nw -padx 3 -pady 3

              message $f2.instr -width [expr $notebook_width - 60] \
                  -text "Select a slice in the Z direction to view the variance."
              pack $f2.instr -side top -anchor n -padx 3 -pady 3
 
	      scale $f2.slice -label "Slice:" \
                  -variable $mods(UnuSlice1)-position \
                  -from 0 -to $num_slices \
                  -showvalue true \
                  -orient horizontal \
                  -command "$this change_variance_slice" \
                  -length [expr $notebook_width - 60]                  

              pack $f2.slice -side top -anchor n -padx 3 -pady 3


   	      bind $f2.slice <ButtonRelease> "app update_variance_slice"
          } else {
              # configure ref image scale
              $f1.slice configure -from 1 -to $num_slices
              $f2.slice configure -from 1 -to $num_slices

              update
          }

    }

    method update_variance_slice {} {
      global mods
      $mods(UnuSlice1)-c needexecute

      if {$reg_completed} {
        $mods(UnuSlice2)-c needexecute
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
	} else {
	    # unblock modules
	    disableModule $mods(TendEstim) 0
	    disableModule $mods(ChooseNrrd-DT) 0
	    
	    # unblock modules
	    disableModule $mods(TendEstim) 0
	    disableModule $mods(ChooseNrrd-DT) 0
	    
	    # execute
	    $mods(ChooseNrrd-ToSmooth)-c needexecute

	    set dt_completed 1
	    
	    view_Vis
	}
    }


    method change_variance_slice { val } {
       global mods
       global $mods(UnuSlice2)-position
       set $mods(UnuSlice2)-position $val
    }


    method view_Vis {} {
        if {$dt_completed} {
            # view planes tab
            $vis_tab1 view "Planes"
            $vis_tab2 view "Planes"
        } else {
            set answer [tk_messageBox -message \
                 "Please finish constructing the Diffusion Tensors." -type ok -icon info -parent .standalone]
        }
    }


    method activate_registration { } {
        global mods
	foreach w [winfo children $reg_tab1] {
	    activate_widget $w
        }

	foreach w [winfo children $reg_tab2] {
	    activate_widget $w
        }

	# configure Registrations next button
	if {$reg_completed} {
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
	
        toggle_reference_image_state
	toggle_registration_threshold

        # configure ref image scale
        global $mods(NrrdInfo1)-size0
        $ref_image1.s.ref configure -from 1 -to [expr [set $mods(NrrdInfo1)-size0] + 1]
        $ref_image2.s.ref configure -from 1 -to [expr [set $mods(NrrdInfo1)-size0] + 1]


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


    method activate_vis {} {
       global mods

       if {![winfo exists $planes_tab1.show]} {
          # build vis tabs
          build_planes_tab $planes_tab1
          build_planes_tab $planes_tab2

          build_isosurface_tab $isosurface_tab1
          build_isosurface_tab $isosurface_tab2

          # turn off variances
          global $mods(ShowField-Orig)-faces-on
          global $mods(ShowField-Reg)-faces-on
          set $mods(ShowField-Orig)-faces-on 0
          set $mods(ShowField-Reg)-faces-on 0
          $mods(ShowField-Orig)-c toggle_display_faces
 	  $mods(ShowField-Reg)-c toggle_display_faces

          $mods(Viewer)-ViewWindow_0-c autoview
          global $mods(Viewer)-ViewWindow_0-pos
          set $mods(Viewer)-ViewWindow_0-pos "z1_y0"
          $mods(Viewer)-ViewWindow_0-c Views

          uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (8)\}" 0
          uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (7)\}" 0
          uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-Probe Selection Widget (11)\}" 0
          uplevel \#0 set "\{$mods(Viewer)-ViewWindow_0-StreamLines rake (12)\}" 0

          $mods(Viewer)-ViewWindow_0-c redraw

          # setup global clipping planes
	  initialize_clip_info

          change_indicator_labels "Visualization..."

          # bring planes tab forward
          view_Vis
      } else {
	  puts "FIX ME: Configure tabs???"
      }
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
		# Data Vis
		$vis_frame_tab1 view "Data Vis"
		$vis_frame_tab2 view "Data Vis"
	    } else {
 		$vis_frame_tab1 view "Global Options"
 		$vis_frame_tab2 view "Global Options"
	    }
	}
    }
    

    method change_processing_tab { which } {
	global mods
	global do_registration

	change_indicate_val 0
	if {$initialized} {
	    if {$which == "Data"} {
		# Data Acquisition step
		$proc_tab1 view "Data"
		$proc_tab2 view "Data"
		change_indicator_labels "Data Acquisition..."
	    } elseif {$which == "Registration"} {
		# Registration step
		if {$data_completed} {
		    $proc_tab1 view "Registration"
		    $proc_tab2 view "Registration"
		    change_indicator_labels "Registration..."
		} 
	    } elseif {$which == "Build DTs"} {
		if {!$do_registration} {
		    set reg_completed 1
		    disableModule $mods(ChooseNrrd-ToReg) 0
		    disableModule $mods(RescaleColorMap2) 0
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
	    }
	    
	    set indicator 0
	}
    }
	
	
    method configure_isosurface_tabs {} {
	global mods
	global $mods(ShowField-Isosurface)-faces-on

	if {$initialized != 0} {
	    if {[set $mods(ShowField-Isosurface)-faces-on] == 1} {
		# configure color button
		if {$plane_type == "Constant"} {
		    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state normal
		    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state normal
		} else {
		    $isosurface_tab1.isocolor.childsite.select.colorFrame.set_color configure -state disabled
		    $isosurface_tab2.isocolor.childsite.select.colorFrame.set_color configure -state disabled
		}
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
		$isosurface_tab1.clip.flipx configure -state disabled
		$isosurface_tab2.clip.flipx configure -state disabled

		$isosurface_tab1.clip.flipy configure -state disabled
		$isosurface_tab2.clip.flipy configure -state disabled

		$isosurface_tab1.clip.flipz configure -state disabled
		$isosurface_tab2.clip.flipz configure -state disabled
            }
	}
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
       Tooltip $canvas $tips(Indicator)
   }
    
    
    method change_indicate_val { v } {
	if {$indicate != 3 || $error_module == ""} {
	    # only change an error state if it has been cleared (error_module empty)
	    # it will be changed by the indicate_error method when fixed
	    set indicate $v
	    change_indicator
	}
    }
    
    method change_indicator_labels { msg } {
	$indicatorL1 configure -text $msg
	$indicatorL2 configure -text $msg
    }
    
    
    # Tooltips array
    variable tips

    # Embedded Viewer
    variable eviewer

    # Standalone
    variable win

    # Data size variables
    variable volumes
    variable size_x
    variable size_y
    variable size_z

    # Flag to indicate whether entire gui has been built
    variable initialized

    # State
    variable data_completed
    variable reg_completed
    variable dt_completed

    
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

    variable streamlines_tab1
    variable streamlines_tab2

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

    # colormaps
    variable colormap_width
    variable colormap_height
    variable colormap_res

}

ForwardFEMApp app

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
