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
set $m15-edge-resolution {8}

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

global data_mode
set data_mode "DWI"


#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

class BioFEMApp {
    
    method modname {} {
	return "BioFEMApp"
    }
    
    constructor {} {
	toplevel .standalone
	wm title .standalone "BioFEM"	 
	set win .standalone
	
	set notebook_width 350
	set notebook_height 600
	
	set viewer_width 640
	set viewer_height 670
	
	set vis_width [expr $notebook_width + 40]
	set vis_height $viewer_height

	set screen_width [winfo screenwidth .]
	set screen_height [winfo screenheight .]

        set initialized 0

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
     
	set volumes 0
        set size_x 0
        set size_y 0
        set size_z 0

        set error_module ""

        set isosurface_tab1 ""
        set isosurface_tab2 ""

        set streamlines_tab1 ""
        set streamlines_tab2 ""

	set data_next_button1 ""
	set data_next_button2 ""
	set data_ex_button1 ""
	set data_ex_button2 ""

        set proc_color "dark red"
	set next_color "#cdc858"
	set execute_color "#5377b5"
        set feedback_color "dodgerblue4"
        set error_color "red4"

        # colormaps
        set colormap_width 100
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

	# Viewer Options Tab

    }
    

    destructor {
	destroy $this
    }

    
    method build_app {} {
	global mods
	
	# Embed the Viewer
	set eviewer [$mods(Viewer) ui_embedded]
	$eviewer setWindow $win.viewer $viewer_width $viewer_height
	
	### Menu
	frame $win.main_menu -relief raised -borderwidth 3
	pack $win.main_menu -fill x -anchor nw


	menubutton $win.main_menu.file -text "File" -underline 0 \
	    -menu $win.main_menu.file.menu
	
	menu $win.main_menu.file.menu -tearoff false

	$win.main_menu.file.menu add command -label "Load       Ctr+O" \
	    -underline 1 -command "$this load_session" -state active
	
	$win.main_menu.file.menu add command -label "Save      Ctr+S" \
	    -underline 0 -command "$this save_session" -state active
	
	$win.main_menu.file.menu add command -label "Quit        Ctr+Q" \
	    -underline 0 -command "$this exit_app" -state active
	
	pack $win.main_menu.file -side left

	
	global tooltipsOn
	menubutton $win.main_menu.help -text "Help" -underline 0 \
	    -menu $win.main_menu.help.menu
	
	menu $win.main_menu.help.menu -tearoff false

	$win.main_menu.help.menu add check -label "Show Tooltips" \
	    -variable tooltipsOn \
	    -underline 0 -state active

	$win.main_menu.help.menu add command -label "Help Contents" \
	    -underline 0 -command "$this show_help" -state active

	$win.main_menu.help.menu add command -label "About BioFEM" \
	    -underline 0 -command "$this show_about" -state active
	
	pack $win.main_menu.help -side left
	
	tk_menuBar $win.main_menu $win.main_menu.file $win.main_menu.help

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
	pack $win.viewer $attachedVFr -side left \
	    -anchor n -fill both -expand 1

	set total_width [expr $viewer_width + $vis_width]

	set total_height $viewer_height

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]
	set pos_y [expr [expr $screen_height / 2] - [expr $total_height / 2]]

	append geom $total_width x $total_height + $pos_x + $pos_y
	wm geometry .standalone $geom
	update	

        set initialized 1

    }

    method set_dataset {} {
	global mods
	global DATADIR
	global DATASET

	puts $DATASET

	source $DATADIR/$DATASET/$DATASET.settings

	#Fix up global scale.
	global global_scale
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
	$this execute_Data
    }


    method init_data_selection_frame { f } {
        global mods

	frame $f.datadir
	label $f.datadir.l -text "DATADIR ="
	entry $f.datadir.e -textvar DATADIR -width 120 -relief flat
	pack $f.datadir.l $f.datadir.e -side left -anchor nw
	pack $f.datadir -side top -anchor w -pady 10

	iwidgets::labeledframe $f.dataset \
	    -labelpos n -labeltext "DATASET" 
	pack $f.dataset -side top -anchor w -fill x
	    
	set dataset [$f.dataset childsite]

	radiobutton $dataset.brain-eg -text "Brain EG" -variable DATASET -value brain-eg -command "$this set_dataset"
	radiobutton $dataset.cyl3 -text "Cyl3" -variable DATASET -value cyl3 -command "$this set_dataset"
	radiobutton $dataset.sphere -text "Sphere" -variable DATASET -value sphere -command "$this set_dataset"
	radiobutton $dataset.utahtorso-lowres -text "Utah Torso Lowres" -variable DATASET -value utahtorso-lowres -command "$this set_dataset"
	radiobutton $dataset.utahtorso -text "Utah Torso" -variable DATASET -value utahtorso -command "$this set_dataset"

	pack $dataset.brain-eg $dataset.cyl3 $dataset.sphere $dataset.utahtorso-lowres $dataset.utahtorso -anchor w -side top

	frame $f.cond
	label $f.cond.l -text "Conductivity File:"
	entry $f.cond.e -textvar $mods(FieldReader-conductivities)-filename -width 100
	button $f.cond.b -text Browse -command "$mods(FieldReader-conductivities) ui"
	pack $f.cond.l $f.cond.e $f.cond.b
	pack $f.cond


	frame $f.elec
	label $f.elec.l -text "Electrode File:"
	entry $f.elec.e -textvar $mods(FieldReader-electrodes)-filename -width 100
	button $f.elec.b -text Browse -command "$mods(FieldReader-electrodes) ui"
	pack $f.elec.l $f.elec.e $f.elec.b
	pack $f.elec


	frame $f.probe
	label $f.probe.l -text "Probe File:"
	entry $f.probe.e -textvar $mods(FieldReader-probe)-filename -width 100
	button $f.probe.b -text Browse -command "$mods(FieldReader-probe) ui"
	pack $f.probe.l $f.probe.e $f.probe.b
	pack $f.probe
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
		-height [expr $vis_height - 160] -tabpos n
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            if {$case == 0} {
		set vis_frame_tab1 $vis.tnb
            } else {
		set vis_frame_tab2 $vis.tnb	    
            }


	    set data [$vis.tnb add -label "Data Selection" -command "$this change_vis_frame 0"]

	    init_data_selection_frame $data
	    


	    set page [$vis.tnb add -label "Vis Options" -command "$this change_vis_frame 1"]


	    ### Isosurface
	    iwidgets::labeledframe $page.isoframe -labelpos nw \
		-labeltext "IsoSurface"

	    set iso [$page.isoframe childsite]
	    
	    build_isosurface_tab $iso
	    
            pack $page.isoframe -padx 4 -pady 4 -fill x

            if {$case == 0} {
		set isosurface_tab1 $iso
            } else {
		set isosurface_tab2 $iso
            }
	    
	    ### StreamLines
	    iwidgets::labeledframe $page.slframe -labelpos nw \
		-labeltext "StreamLines"

	    set sl [$page.slframe childsite]
	    
	    build_streamlines_tab $sl
	    
            pack $page.slframe -padx 4 -pady 4 -fill x

            if {$case == 0} {
		set streamlines_tab1 $sl
            } else {
		set streamlines_tab2 $sl
            }

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
	    
            label $vis.indicatorL -text "Data Acquisition..."
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
		    Tooltip $m.d.cut$i $tips(VDetachedMsg)
		} else {
		    Tooltip $m.d.cut$i $tips(VAttachedMsg)
		}
            }
	}
    }
    

    method create_viewer_tab { vis } {
	global mods
	set page [$vis.tnb add -label "Viewer Options" -command "$this change_vis_frame 2"]
	
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
	
	$vis.tnb view "Data Selection"
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
	    puts $fileid "# BioFEM Session\n"
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
	tk_messageBox -message "Please refer to the online BioFEM Tutorial\nhttp://software.sci.utah.edu/doc/User/BioFEMTutorial" -type ok -icon info -parent .standalone
    }
    
    method show_about {} {
	tk_messageBox -message "BioFEM About Box" -type ok -icon info -parent .standalone
    }
    
    method display_module_error {} {
        if {$error_module != ""} {
	    set result [$error_module displayLog]
        }
    }
    
    method indicate_dynamic_compile { which mode } {
	if {$mode == "start"} {
	    change_indicator_labels "Dynamically Compiling Code..."
	}
    }
    
    
    method update_progress { which state } {
	global mods
	global $mods(ShowField-Isosurface)-faces-on
	
	return
	
	if {$which == $mods(FieldReader-conductivities) && $state == "NeedData"} {
	    change_indicator_labels "Data Acquisition..."
	    change_indicate_val 1
	} elseif {$which == $mods(FieldReader-conductivities) && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {$which == $mods(FieldReader-electrodes) && $state == "NeedData"} {
	    change_indicator_labels "Data Acquisition..."
	    change_indicate_val 1
	} elseif {$which == $mods(FieldReader-electrodes) && $state == "Completed"} {
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
	    }
	} elseif {$which == $mods(Isosurface) && $state == "NeedData" && [set $mods(ShowField-Isosurface)-faces-on] == 1} {
	    change_indicator_labels "Visualization..."
	    change_indicate_val 1
	} elseif {$which == $mods(ShowField-Isosurface) && $state == "Completed" && [set $mods(ShowField-Isosurface)-faces-on] == 1} {
	    change_indicate_val 0
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
		change_indicator_labels "Data Acquisition..."
		change_indicate_val 0
	    }
	}
    }
	
	
    method execute_Data {} {
	global mods 
	global data_mode
	
	
	$mods(FieldReader-conductivities)-c needexecute
	$mods(FieldReader-electrodes)-c needexecute
	$mods(FieldReader-probe)-c needexecute
    }
    

    method toggle_streamlines {} {
	global mods
	global $mods(ShowField-StreamLines)-edges-on
	if { [set $mods(ShowField-StreamLines)-edges-on] } {
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


    method activate_vis {} {
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
	    } elseif {$which == 1} {
		# Data Vis
		$vis_frame_tab1 view "Vis Options"
		$vis_frame_tab2 view "Vis Options"
	    } else {
 		$vis_frame_tab1 view "Viewer Options"
 		$vis_frame_tab2 view "Viewer Options"
	    }
	}
    }
    

    method change_processing_tab { which } {
	global mods

	change_indicate_val 0
	if {$initialized} {
	    if {$which == "Data"} {
		# Data Acquisition step
		$proc_tab1 view "Data"
		$proc_tab2 view "Data"
		change_indicator_labels "Data Acquisition..."
	    }
	    set indicator 0
	}
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

    # Data tabs
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

    variable isosurface_tab1
    variable isosurface_tab2

    variable streamlines_tab1
    variable streamlines_tab2

    # Application placing and size
    variable notebook_width
    variable notebook_height

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

    # colormaps
    variable colormap_width
    variable colormap_height
    variable colormap_res

}

BioFEMApp app

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
