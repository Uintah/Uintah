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

itk::usual Linkedpane {
    keep -background -cursor -sashcursor
}

setProgressText "Loading BioImage Modules, Please Wait..."

#######################################################################
# Check environment variables.  Ask user for input if not set:
# Attempt to get environment variables:
set DATADIR [netedit getenv SCIRUN_DATA]
set DATASET [netedit getenv SCIRUN_DATASET]
#######################################################################

############# NET ##############
::netedit dontschedule
set bbox {0 0 3100 3100}

set m1 [addModuleAtPosition "SCIRun" "Render" "Viewer" 17 2900]


# This is a hack.  For some reason it takes a long time to add the first
# ShowField module.  The load ui uses ShowField and I want to cut down
# on the instantiation time.  So I'll just instantiate a dummy one that
# will never get used.
set m2 [addModuleAtPosition "SCIRun" "Visualization" "ShowField" 3000 3000]
global mods
set mods(Viewer) $m1
set mods(ViewImage) ""
set mods(EditTransferFunc) ""

# Tooltips
global tips

global new_label
set new_label "Unknown"

global eye
set eye 0

# show planes
global show_plane_x
global show_plane_y
global show_plane_z
global show_guidelines
set show_plane_x 0
set show_plane_y 0
set show_plane_z 0
set show_guidelines 1
global planes_mapType
set planes_mapType 0

# volume rendering
global show_vol_ren
set show_vol_ren 1

global show_iso
set show_iso 0
global iso_mapType
set iso_mapType 2


setProgressText "Loading BioImage Application, Please Wait..."

#######################################################
# Build up a simplistic standalone application.
#######################################################
wm withdraw .

set auto_index(::PowerAppBase) "source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/PowerAppBase.app"

class BioImageApp {
    inherit ::PowerAppBase
        
    constructor {} {
	global mods
	toplevel .standalone
	wm title .standalone "BioImage"	 
	set win .standalone

	# Set window sizes
	set i_width 260

	set viewer_width 436
	set viewer_height 540
	
	set notebook_width 305
	set notebook_height [expr $viewer_height - 50]
	
	set process_width 325
	set process_height $viewer_height
	
	set vis_width [expr $notebook_width + 25]
	set vis_height $viewer_height

	set num_filters 0

	set loading_ui 0

	set vis_frame_tab1 ""
	set vis_frame_tab2 ""

	set history1 ""
	set history2 ""

	set dimension 3

	set current 0
	set scolor $execute_color

	# filter indexes
	set filter_type 0
	set modules 1
	set input 2
	set output 3
	set prev_index 4
	set next_index 5
	set choose_port 6
	set which_row 7
	set visibility 8


	set load_choose_input 5
	set load_nrrd 0
	set load_dicom 1
	set load_analyze 2
	set load_field 3
	set load_choose_vis 6
        set load_info 24

	set grid_rows 0

	set label_width 31

	set 0_samples 2
	set 1_samples 2
	set 2_samples 2
        set sizex 0
        set sizey 0
        set sizez 0

        set has_autoviewed 0
        set has_executed 0
	set iso_slider1 ""
	set iso_slider2 ""
        set data_dir ""
        set 2D_fixed 0

	### Define Tooltips
	##########################
	global tips

    }
    

    destructor {
	destroy $this
    }
    
    
    method appname {} {
	return "BioImage"
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
	    change_indicator_labels "Visualizing..."
	}
    }
    
    ##########################
    ### indicate_error
    ##########################
    # This method should change the indicator and labels to
    # the error state.  This should be done using the change_indicate_val
    # and change_indicator_labels methods. We catch errors from
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
		puts "FIX ME implement indicate_error"
		change_indicator_labels "Visualizing..."
		change_indicate_val 0
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

	if {[string first "NodeGradient" $which] != -1 && $state == "JustStarted"} {
	    change_indicator_labels "Volume Rendering..."
	    change_indicate_val 1
	} elseif {[string first "NodeGradient" $which] != -1 && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {[string first "Isosurface" $which] != -1 && $state == "NeedData"} {
	    change_indicate_val 1
	    change_indicator_labels "Isosurfacing..."
	} elseif {[string first "Isosurface" $which] != -1 && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {[string first "NrrdTextureBuilder" $which] != -1 && $state == "JustStarted"} {
	    change_indicator_labels "Volume Rendering..."
	    change_indicate_val 1
	} elseif {[string first "NrrdTextureBuilder" $which] != -1 && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {[string first "EditTransferFunc2" $which] != -1 && $state == "JustStarted"} {
	    change_indicator_labels "Volume Rendering..."
	    change_indicate_val 1
	} elseif {[string first "EditTransferFunc2" $which] != -1 && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {[string first "VolumeVisualizer" $which] != -1 && $state == "Completed"} {
	    if {$has_autoviewed == 0} {
		set has_autoviewed 1
		after 100 "$mods(Viewer)-ViewWindow_0-c autoview"
	    }
        } elseif {[string first "ViewImage" $which] != -1 && $state == "Completed"} {
            if {$2D_fixed == 0} {
                # simulate a click in each window and set them to the correct views
		global mods

		# force initial draw in correct modes
		global $mods(ViewImage)-axial-viewport0-axis
		global $mods(ViewImage)-sagittal-viewport0-axis
		global $mods(ViewImage)-coronal-viewport0-axis

		set $mods(ViewImage)-axial-viewport0-axis 2
		set $mods(ViewImage)-sagittal-viewport0-axis 0
		set $mods(ViewImage)-coronal-viewport0-axis 1

		global $mods(ViewImage)-nrrd1-flip_y
		set $mods(ViewImage)-nrrd1-flip_y 1

		global $mods(ViewImage)-nrrd1-flip_z
		set $mods(ViewImage)-nrrd1-flip_z 1
		
                $mods(ViewImage)-c rebind .standalone.viewers.topbot.pane0.childsite.lr.pane1.childsite.axial
                $mods(ViewImage)-c rebind .standalone.viewers.topbot.pane1.childsite.lr.pane0.childsite.sagittal
                $mods(ViewImage)-c rebind .standalone.viewers.topbot.pane1.childsite.lr.pane1.childsite.coronal


                set 2D_fixed 1
	    }
	} elseif {[string first "Teem_NrrdData_NrrdInfo_1" $which] != -1 && $state == "Completed"} {
	    # update slice sliders
	    global $which-size0 $which-size1 $which-size2

	    set sizex [expr [set $which-size0] - 1]
	    set sizey [expr [set $which-size1] - 1]
	    set sizez [expr [set $which-size2] - 1]

  	    .standalone.viewers.topbot.pane0.childsite.lr.pane1.childsite.modes.slice.s configure -from 0 -to $sizez
  	    .standalone.viewers.topbot.pane1.childsite.lr.pane0.childsite.modes.slice.s configure -from 0 -to $sizex
 	    .standalone.viewers.topbot.pane1.childsite.lr.pane1.childsite.modes.slice.s configure -from 0 -to $sizey

 	    # set slice to be middle slice
 	    global $mods(ViewImage)-axial-viewport0-slice
 	    global $mods(ViewImage)-sagittal-viewport0-slice
 	    global $mods(ViewImage)-coronal-viewport0-slice

 	    set $mods(ViewImage)-axial-viewport0-slice [expr $sizez/2]
 	    set $mods(ViewImage)-sagittal-viewport0-slice [expr $sizex/2]
 	    set $mods(ViewImage)-coronal-viewport0-slice [expr $sizey/2]
	} elseif {[string first "Teem_NrrdData_NrrdInfo_0" $which] != -1 && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Loading Volume..."
	} elseif {[string first "Teem_NrrdData_NrrdInfo_0" $which] != -1 && $state == "Completed"} {
	    change_indicate_val 2
	    set NrrdInfo [lindex [lindex $filters(0) $modules] $load_info]
 	    global $NrrdInfo-dimension
 	    set dimension [set $NrrdInfo-dimension]

 	    global $NrrdInfo-size1
	    
 	    if {[info exists $NrrdInfo-size1]} {
 		global $NrrdInfo-size0
 		global $NrrdInfo-size1
		
 		set 0_samples [set $NrrdInfo-size0]
 		set 1_samples [set $NrrdInfo-size1]

		# configure samples info
 		if {$dimension == 3} {
 		    global $NrrdInfo-size2
 		    set 2_samples [set $NrrdInfo-size2]
 		    $history1.f0.childsite.ui.samples configure -text \
 			"Original Samples: ($0_samples, $1_samples, $2_samples)"
 		    $history2.f0.childsite.ui.samples configure -text \
 			"Original Samples: ($0_samples, $1_samples, $2_samples)"
 		} elseif {$dimension == 2} {
 		    $history1.f0.childsite.ui.samples configure -text \
 			"Original Samples: ($0_samples, $1_samples)"
 		    $history2.f0.childsite.ui.samples configure -text \
 			"Original Samples: ($0_samples, $1_samples)"
 		} else {
 		    puts "ERROR: Only 2D and 3D data supported."
 		    return
 		}
	    }	
	} elseif {[string first "UnuResample" $which 0] != -1 && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Resampling Volume..."
	} elseif {[string first "UnuResample" $which 0] != -1 && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {[string first "UnuCrop" $which 0] != -1 && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Cropping Volume..."
	} elseif {[string first "UnuCrop" $which 0] != -1 && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {[string first "UnuHeq" $which 0] != -1 && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Performing Histogram Equilization..."
	} elseif {[string first "UnuHeq" $which 0] != -1 && $state == "Completed"} {
	    change_indicate_val 2
	} elseif {[string first "UnuCmedian" $which 0] != -1 && $state == "JustStarted"} {
	    change_indicate_val 1
	    change_indicator_labels "Performing Median/Mode Filtering..."
	} elseif {[string first "UnuCmedian" $which 0] != -1 && $state == "Completed"} {
	    change_indicate_val 2
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
			puts "FIX ME change_indicate_val - labels"
			change_indicator_labels "Visualizing..."
		    }
		} elseif {$executing_modules < 0} {
		    # something wasn't caught, reset
		    set executing_modules 0
		    set indicate 2
		    change_indicator

		    if {$loading} {
			set loading 0
			puts "FIX ME change_indicate_val - labels"
			change_indicator_labels "Visualizing..."
		    }

		}
	    }
	}
    }
    
    method change_indicator_labels { msg } {
	$indicatorL1 configure -text $msg
	$indicatorL2 configure -text $msg
    }

    method keypress {w} {
	global mods
      
	$mods(ViewImage)-c keypress $w 60 a 175035491
    }



    ############################
    ### build_app
    ############################
    # Build the processing and visualization frames and pack along with viewer
    method build_app {d} {
	set data_dir $d
	global mods

	# Embed the Viewers

	# add a viewer and tabs to each
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

	init_Pframe $detachedPFr.f 0
	init_Pframe $attachedPFr.f 1

	change_current 0

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
	pack $attachedPFr -side left -anchor n

	pack $win.viewers -side left -anchor n -fill both -expand 1

	pack $attachedVFr -side left -anchor n 

	set total_width [expr $process_width + $viewer_width + $vis_width]

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
	    wm title .standalone "BioImage - [getFileName $saveFile]"
	    $this load_session
	} 
    }

    method build_viewers {viewer viewimage} {
	global mods
	set mods(ViewImage) $viewimage

	set w $win.viewers
	iwidgets::panedwindow $w.topbot -orient horizontal -thickness 0 \
	    -sashwidth 5000 -sashindent 0 -sashborderwidth 2 -sashheight 6 \
	    -sashcursor sb_v_double_arrow -width $viewer_width -height $viewer_height
	pack $w.topbot -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
	
	$w.topbot add top -margin 0 -minimum 0
	$w.topbot add bottom  -margin 0 -minimum 0

	set top [$w.topbot childsite top]
	set bot [$w.topbot childsite bottom]
	
	Linkedpane $top.lr -orient vertical -thickness 0 \
	    -sashheight 5000 -sashwidth 6 -sashindent 0 -sashborderwidth 2 \
	    -sashcursor sb_h_double_arrow

	$top.lr add left -margin 3 -minimum 0
	$top.lr add right -margin 3 -minimum 0
	set topl [$top.lr childsite left]
	set topr [$top.lr childsite right]

	Linkedpane $bot.lr  -orient vertical -thickness 0 \
	    -sashheight 5000 -sashwidth 6 -sashindent 0 -sashborderwidth 2 \
	    -sashcursor sb_h_double_arrow

	$bot.lr set_link $top.lr
	$top.lr set_link $bot.lr

	$bot.lr add left -margin 3 -minimum 0
	$bot.lr add right -margin 3 -minimum 0
	set botl [$bot.lr childsite left]
	set botr [$bot.lr childsite right]

	pack $top.lr -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
	pack $bot.lr -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0

	$viewimage control_panel $w.cp
	$viewimage add_nrrd_tab $w 1
	
	# modes for axial
	frame $topr.modes
	pack $topr.modes -side bottom -padx 0 -pady 0

	frame $topr.modes.slice
	pack $topr.modes.slice -side top -pady 0 -anchor nw

	global $mods(ViewImage)-axial-viewport0-mode
	radiobutton $topr.modes.slice.b -text "Slice Mode" \
	    -variable  $mods(ViewImage)-axial-viewport0-mode -value 0 \
	    -command "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane0.childsite.lr.pane1.childsite.axial"
	pack $topr.modes.slice.b -side left -padx 0 -anchor nw

	global $mods(ViewImage)-axial-viewport0-slice
 	scale $topr.modes.slice.s \
 	    -from 0 -to 20 \
 	    -orient horizontal -showvalue false \
 	    -length 110 \
	    -variable $mods(ViewImage)-axial-viewport0-slice \
	    -command "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane0.childsite.lr.pane1.childsite.axial"
	

	label $topr.modes.slice.l -textvariable $mods(ViewImage)-axial-viewport0-slice
 	pack $topr.modes.slice.s $topr.modes.slice.l -side left -anchor n -padx 0

	radiobutton $topr.modes.mip -text "MIP Mode" \
	    -variable $mods(ViewImage)-axial-viewport0-mode -value 1 \
	    -command "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane0.childsite.lr.pane1.childsite.axial"
	pack $topr.modes.mip -side top -padx 0 -pady 0 -anchor nw

	# modes for sagittal
	frame $botl.modes
	pack $botl.modes -side bottom -padx 0 -pady 0

	frame $botl.modes.slice
	pack $botl.modes.slice -side top -pady 0 -anchor nw

	global $mods(ViewImage)-sagittal-viewport0-mode
	radiobutton $botl.modes.slice.b -text "Slice Mode" \
	    -variable $mods(ViewImage)-sagittal-viewport0-mode -value 0 \
	    -command "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane1.childsite.lr.pane0.childsite.sagittal"
	pack $botl.modes.slice.b -side left -padx 0 -anchor nw


	global $mods(ViewImage)-sagittal-viewport0-slice
 	scale $botl.modes.slice.s \
 	    -from 0 -to 254 \
 	    -orient horizontal -showvalue false \
 	    -length 110 \
	    -variable $mods(ViewImage)-sagittal-viewport0-slice \
	    -command  "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane1.childsite.lr.pane0.childsite.sagittal"
	
	label $botl.modes.slice.l -textvariable $mods(ViewImage)-sagittal-viewport0-slice
 	pack $botl.modes.slice.s $botl.modes.slice.l -side left -anchor n -padx 0

	radiobutton $botl.modes.mip -text "MIP Mode" \
	    -variable $mods(ViewImage)-sagittal-viewport0-mode -value 1 \
	    -command "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane1.childsite.lr.pane0.childsite.sagittal"
	pack $botl.modes.mip -side top -padx 0 -pady 0 -anchor nw

	# modes for coronal
	frame $botr.modes
	pack $botr.modes -side bottom -padx 0 -pady 0


	frame $botr.modes.slice
	pack $botr.modes.slice -side top -pady 0 -anchor nw

	global $mods(ViewImage)-coronal-viewport0-mode
	radiobutton $botr.modes.slice.b -text "Slice Mode" \
	    -variable $mods(ViewImage)-coronal-viewport0-mode -value 0 \
	    -command "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane1.childsite.lr.pane1.childsite.coronal"
	pack $botr.modes.slice.b -side left -padx 0 -anchor nw

	global $mods(ViewImage)-coronal-viewport0-slice
 	scale $botr.modes.slice.s \
 	    -from 0 -to 254 \
 	    -orient horizontal -showvalue false \
 	    -length 110 \
	    -variable $mods(ViewImage)-coronal-viewport0-slice \
	    -command "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane1.childsite.lr.pane1.childsite.coronal"

	label $botr.modes.slice.l -textvariable $mods(ViewImage)-coronal-viewport0-slice
 	pack $botr.modes.slice.s $botr.modes.slice.l -side left -anchor n -padx 0

	radiobutton $botr.modes.mip -text "MIP Mode" \
	    -variable $mods(ViewImage)-coronal-viewport0-mode -value 1 \
	    -command "$mods(ViewImage)-c rebind .standalone.viewers.topbot.pane1.childsite.lr.pane1.childsite.coronal"
	pack $botr.modes.mip -side top -padx 0 -pady 0 -anchor nw


	# embed viewer in top left
	global mods
 	set eviewer [$mods(Viewer) ui_embedded]

 	$eviewer setWindow $topl [expr $viewer_width/2] \
 	    [expr $viewer_height/2] \

 	pack $topl -side top -anchor n \
 	    -expand 1 -fill both -padx 0 -pady 0

	# add 3 slice windows
	pack [$viewimage gl_frame $topr "Axial" $w] -side top -padx 0 -ipadx 0 -pady 0 -ipady 0
	pack [$viewimage gl_frame $botl "Sagittal" $w] -side top -padx 0 -ipadx 0 -pady 0 -ipady 0
	pack [$viewimage gl_frame $botr "Coronal" $w] -side top -padx 0 -pady 0 -ipadx 0 -ipady 0
    }



    #############################
    ### init_Pframe
    #############################
    # Initialize the processing frame on the left. For this app
    # that includes the Load Volume, Restriation, and Build Tensors steps.
    # This method will call the base class build_menu method and sets 
    # the variables that point to the tabs and tabnotebooks.
    method init_Pframe { m case } {
        global mods
	global tips
        
	if { [winfo exists $m] } {

	    build_menu $m


	    frame $m.p -borderwidth 2 -relief groove
	    pack $m.p -side left -fill both -anchor nw -expand yes

	    ### Filter Menu
	    frame $m.p.filters
	    pack $m.p.filters -side top -expand yes -fill x

	    set filter $m.p.filters
	    button $filter.resamp -text "Resample" \
		-background $scolor \
		-activebackground "#6c90ce" \
		-command "$this add_Resample"
	    Tooltip $filter.resamp "Resample using UnuResample"

	    button $filter.crop -text "Crop" \
		-background $scolor \
		-activebackground "#6c90ce" \
		-command "$this add_Crop"
	    Tooltip $filter.crop "Crop the image"

	    button $filter.cmedian -text "Cmedian" \
		-background $scolor \
		-activebackground "#6c90ce" \
		-command "$this add_Cmedian"
	    Tooltip $filter.crop "Cmedian"

	    button $filter.histo -text "Histogram" \
		-background $scolor \
		-activebackground "#6c90ce" \
		-command "$this add_Histo"
	    Tooltip $filter.histo "Perform Histogram Equilization\nusing UnuHeq"

	    button $filter.delete -text "Delete" \
		-command "$this filter_Delete" \
		-background "#c1300c" \
		-activebackground "#d73e18"
	    Tooltip $filter.delete "Delete the highlighted filter"

	    button $filter.update -text "Update" \
		-command "$this execute_current" \
		-background "#09ac24" \
		-activebackground "#23c43d"
	    Tooltip $filter.update "Update the View"

	    pack $filter.resamp $filter.crop $filter.histo $filter.cmedian $filter.delete $filter.update \
		-side left -padx 1 -expand yes -fill x

	    iwidgets::scrolledframe $m.p.sf -width [expr $process_width - 20] \
		-height [expr $process_height - 150] -labeltext "History"
	    pack $m.p.sf -side top -anchor nw
	    set history [$m.p.sf childsite]

	    # Add Load UI
	    $this add_Load $history $case
	    
	    set grid_rows 1
	    set num_filters 1	 	 

	    if {$case == 0} {
		set history1 $history
	    } else {
		set history2 $history
	    }
	    
            ### Indicator
	    frame $m.p.indicator -relief sunken -borderwidth 2
            pack $m.p.indicator -side bottom -anchor s -padx 3 -pady 5
	    
	    canvas $m.p.indicator.canvas -bg "white" -width $i_width \
	        -height $i_height 
	    pack $m.p.indicator.canvas -side top -anchor n -padx 3 -pady 3
	    
            bind $m.p.indicator <Button> {app display_module_error} 
	    
            label $m.p.indicatorL -text "Press Execute to Load Volume..."
            pack $m.p.indicatorL -side bottom -anchor sw -padx 5 -pady 3
	    
            if {$case == 0} {
		set indicator1 $m.p.indicator.canvas
		set indicatorL1 $m.p.indicatorL
            } else {
		set indicator2 $m.p.indicator.canvas
		set indicatorL2 $m.p.indicatorL
            }
	    Tooltip $m.p.indicatorL $tips(IndicatorLabel)
	    
            construct_indicator $m.p.indicator.canvas
	    
	    
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

    
    # Method to create Load/Vis modules, and build
    # Load UI.  Variable m gives the path and case
    # indicates whether it is being built for the attached
    # or detached frame.  The modules only need to be created
    # once, so when case equals 0, create modules and ui, for case 1
    # just create ui.
    method add_Load {history case} {

	# if first time in this method (case == 0)
	# create the modules and connections
	if {$case == 0} {
	    global mods
	    
	    # create load modules and inner connections
	    set m1 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 10 10]
	    set m2 [addModuleAtPosition "Teem" "DataIO" "DicomNrrdReader" 28 70]
	    set m3 [addModuleAtPosition "Teem" "DataIO" "AnalyzeNrrdReader" 46 128]
	    set m4 [addModuleAtPosition "SCIRun" "DataIO" "FieldReader" 65 186]
	    set m5 [addModuleAtPosition "Teem" "DataIO" "FieldToNrrd" 65 245]
	    set m6 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 324]
	    set m25 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 10 400]
	    
	    set c1 [addConnection $m4 0 $m5 0]
	    set c2 [addConnection $m1 0 $m6 0]
	    set c3 [addConnection $m2 0 $m6 1]
	    set c4 [addConnection $m3 0 $m6 2]
	    set c5 [addConnection $m5 2 $m6 3]
            set c6 [addConnection $m6 0 $m25 0]
	    
	    # Disable other load modules (Dicom, Analyze, Field)
	    disableModule $m2 1
	    disableModule $m3 1
	    disableModule $m4 1
	    
	    # create vis modules and inner connections
	    set m7 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 1900]
	    set m8 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuQuantize" 10 2191]
	    set m9 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuQuantize" 517 2215]
	    set m10 [addModuleAtPosition "Teem" "UnuAtoM" "UnuJoin" 218 2278]
	    set m11 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuQuantize" 218 2198]
	    set m12 [addModuleAtPosition "SCIRun" "Visualization" "NrrdTextureBuilder" 182 2674]
	    set m13 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuProject" 447 2138]
	    set m14 [addModuleAtPosition "SCIRun" "Visualization" "EditTransferFunc2" 375 2675]
	    set m15 [addModuleAtPosition "SCIRun" "Visualization" "VolumeVisualizer" 224 2760]
	    set m16 [addModuleAtPosition "Teem" "DataIO" "NrrdToField" 182 1937]
	    set m17 [addModuleAtPosition "SCIRun" "FieldsData" "NodeGradient" 182 1997]
	    set m18 [addModuleAtPosition "Teem" "DataIO" "FieldToNrrd" 182 2056]
	    set m19 [addModuleAtPosition "Teem" "UnuAtoM" "UnuHeq" 392 2473]
	    set m20 [addModuleAtPosition "Teem" "UnuAtoM" "UnuGamma" 392 2535]
	    set m21 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuQuantize" 392 2597]
	    set m22 [addModuleAtPosition "Teem" "UnuAtoM" "UnuJhisto" 410 2286]
	    set m23 [addModuleAtPosition "Teem" "UnuAtoM" "Unu2op" 392 2348]
	    set m24 [addModuleAtPosition "Teem" "UnuAtoM" "Unu1op" 392 2409]
            set m26 [addModuleAtPosition "SCIRun" "Render" "ViewImage" 704 2057]
	    set m27 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 741 1977]
            set m28 [addModuleAtPosition "SCIRun" "Visualization" "Isosurface" 472 2052]
            set m29 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 490 1973]
	    set m30 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 369 1889]

	    set mods(EditTransferFunc) $m14
	    
	    set c1 [addConnection $m17 0 $m18 0]
	    set c2 [addConnection $m12 0 $m15 0]
	    set c3 [addConnection $m18 2 $m13 0]
	    set c4 [addConnection $m18 2 $m11 0]
	    set c5 [addConnection $m16 0 $m17 0]
	    set c6 [addConnection $m24 0 $m19 0]
	    set c7 [addConnection $m23 0 $m24 0]
	    set c8 [addConnection $m20 0 $m21 0]
	    set c9 [addConnection $m19 0 $m20 0]
	    set c10 [addConnection $m10 0 $m12 0]
	    set c11 [addConnection $m13 0 $m9 0]
	    set c12 [addConnection $m11 0 $m10 0]
	    set c13 [addConnection $m22 0 $m23 1]
	    set c14 [addConnection $m8 0 $m10 1]
	    set c15 [addConnection $m9 0 $m12 1]
	    set c16 [addConnection $m21 0 $m14 1]
	    set c17 [addConnection $m14 0 $m15 2]
	    set c18 [addConnection $m13 0 $m22 2]
	    set c19 [addConnection $m7 0 $m26 0]
	    set c20 [addConnection $m27 0 $m26 2]
	    set c26 [addConnection $m16 0 $m28 0]
	    set c27 [addConnection $m29 0 $m28 1]
	    set c28 [addConnection $m28 1 $mods(Viewer) 1]
	    set c29 [addConnection $m7 0 $m30 0]

            global Disabled
	    set Disabled($c28) 0
	    disableModule $m28 1

	    # connect load to vis
	    set c21 [addConnection $m6 0 $m7 0]
	    set c22 [addConnection $m7 0 $m8 0]
	    set c23 [addConnection $m7 0 $m16 2]
	    set c24 [addConnection $m7 0 $m22 1]

	    # connect vis to Viewer
	    set c25 [addConnection $m15 0 $mods(Viewer) 0]

	    # set some ui parameters
	    global $m1-filename
	    set $m1-filename $data_dir/volume/CThead.nhdr
	    #set $m1-filename "/home/darbyb/work/data/TR0600-TE020.nhdr"

	    global $m8-nbits
	    set $m8-nbits {8}
	    global $m8-useinputmin
	    set $m8-useinputmin 1
	    global $m8-useinputmax
	    set $m8-useinputmax 1

	    global $m9-nbits
	    set $m9-nbits {8}
	    global $m9-useinputmin
	    set $m9-useinputmin 1
	    global $m9-useinputmax
	    set $m9-useinputmax 1

	    global $m11-nbits
	    set $m11-nbits {8}
	    global $m11-useinputmin
	    set $m11-useinputmin 1
	    global $m11-useinputmax
	    set $m11-useinputmax 1

	    global $m13-measure
	    set $m13-measure {9}

	    global $m14-faux
	    global $m14-num-entries
	    global $m14-name-0 $m14-name-1 $m14-name-2 $m14-name-3
	    global $m14-0-color-r $m14-0-color-g $m14-0-color-b $m14-0-color-a
	    global $m14-1-color-r $m14-1-color-g $m14-1-color-b $m14-1-color-a
#	    global $m14-2-color-r $m14-2-color-g $m14-2-color-b $m14-2-color-a
#	    global $m14-3-color-r $m14-3-color-g $m14-3-color-b $m14-3-color-a
	    global $m14-state-0 $m14-state-1 $m14-state-2 $m14-state-3
	    global $m14-marker
	    set $m14-faux {1}
	    set $m14-histo {0.5}
	    set $m14-name-0 {Generic}
	    set $m14-0-color-r {1.0}
	    set $m14-0-color-g {1.0}
	    set $m14-0-color-b {0.7}
	    set $m14-0-color-a {0.709999978542}
	    set $m14-state-0 {r 0 0.123047 0.117188 0.208985 0.199219 0.25}
	    set $m14-name-1 {Generic}
	    set $m14-1-color-r {0.5}
	    set $m14-1-color-g {0.0}
	    set $m14-1-color-b {0.0}
	    set $m14-1-color-a {1.0}
	    set $m14-state-1 {r 0 0.398438 0.109375 0.185547 0.152344 0.25}
	    set $m14-marker {end}


	    global $m15-sw_raster $m15-alpha_scale
	    global $m15-shading $m15-ambient
	    global $m15-diffuse $m15-specular
	    global $m15-shine
            global $m15-adaptive
	    set $m15-sw_raster {1}
	    set $m15-alpha_scale {-0.554}
	    set $m15-shading {1}
	    set $m15-ambient {0.5}
	    set $m15-diffuse {0.5}
	    set $m15-specular {0.388}
	    set $m15-shine {24}
            set $m15-adaptive {1}

	    global $m19-bins
	    global $m19-sbins
	    set $m19-bins {3000}
	    set $m19-sbins {1}

	    global $m20-gamma
	    set $m20-gamma {0.5}

	    global $m21-nbits
	    set $m21-nbits {8}
	    global $m21-useinputmin
	    set $m21-useinputmin 1
	    global $m21-useinputmax
	    set $m21-useinputmax 1

	    global $m22-bins
	    global $m22-type
	    global $m22-mins $m22-maxs
	    set $m22-bins {512 256}
	    set $m22-mins {nan nan}
	    set $m22-maxs {nan nan}
	    set $m22-type {nrrdTypeFloat}

	    global $m23-operator
	    set $m23-operator {+}

	    global $m24-operator
	    set $m24-operator {log}

            global $m27-mapType planes_mapType
	    set $m27-mapType $planes_mapType

	    global $m28-isoval
	    set $m28-isoval 710

	    global $m28-isoval-min
	    global $m28-isoval-max
	    trace variable $m28-isoval-min w "$this update_iso_slider_min"
	    trace variable $m28-isoval-max w "$this update_iso_slider_max"

	    # create filter index in the form of the list:
	    # filter_type modules input output prev_index next_index choose_port which_row visibility 
	    set mod_list [list $m1 $m2 $m3 $m4 $m5 $m6 $m7 $m8 $m9 $m10 $m11 $m12 $m13 $m14 $m15 $m16 $m17 $m18 $m19 $m20 $m21 $m22 $m23 $m24 $m25 $m26 $m27 $m28 $m29 $m30]
	    set filters(0) [list load $mod_list [list $m6] [list $m6 0] start end 0 0 1]

            $this build_viewers $m25 $m26
	}
	
	$this add_Load_UI $history 0 0
    }
    
    method add_Load_UI {history row which} {
	global mods

	### Load Data UI
	set ChooseNrrd [lindex [lindex $filters($which) $modules] $load_choose_vis] 
	global eye
 	radiobutton $history.eye$which -text "" \
 	    -variable eye -value $which \
	    -command "$this change_eye $which"
	
 	grid config $history.eye$which -column 0 -row 0 -sticky "nw"
	
 	iwidgets::labeledframe $history.f$which \
 	    -labeltext "Load Data" \
 	    -labelpos nw 

 	grid config $history.f$which -column 1 -row 0 -sticky "nw"
 	set data [$history.f$which childsite]
	
 	frame $data.expand
 	pack $data.expand -side top -anchor nw
	
 	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
 	set show [image create photo -file ${image_dir}/play-icon-small.ppm]
 	button $data.expand.b -image $show \
 	    -anchor nw \
 	    -command "$this change_visibility $which" \
 	    -relief flat
 	label $data.expand.l -text "Data - Unknown" -width $label_width \
 	    -anchor nw
 	pack $data.expand.b $data.expand.l -side left -anchor nw 
	
 	bind $data.expand.l <ButtonPress-1> "$this change_current $which"
 	bind $data.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
 	frame $data.ui
 	pack $data.ui -side top -anchor nw -expand yes -fill x

 	label $data.ui.samples -text "Original Samples: unknown"
 	pack $data.ui.samples -side top -anchor nw -pady 3
 	bind $data.ui.samples <ButtonPress-1> "$this change_current $which"

	# Build data tabs
	iwidgets::tabnotebook $data.ui.tnb \
	    -width [expr $process_width - 115] -height 75 \
	    -tabpos n 
	pack $data.ui.tnb -side top -anchor nw \
	    -padx 0 -pady 3
	
	# Make pointers to modules 
	set NrrdReader  [lindex [lindex $filters($which) $modules] $load_nrrd]
	set DicomNrrdReader  [lindex [lindex $filters($which) $modules] $load_dicom]
	set AnalzyeNrrdReader  [lindex [lindex $filters($which) $modules] $load_analyze]
	set FieldReader  [lindex [lindex $filters($which) $modules] $load_field]

	# Nrrd
	set page [$data.ui.tnb add -label "Nrrd" \
		      -command "$this configure_readers Nrrd"]       

	global [set NrrdReader]-filename
	frame $page.file
	pack $page.file -side top -anchor nw -padx 3 -pady 0 -fill x

	label $page.file.l -text "Nrrd File:" 
	entry $page.file.e -textvariable [set NrrdReader]-filename 
	pack $page.file.l $page.file.e -side left -padx 3 -pady 0 -anchor nw \
	    -fill x 
	bind $page.file.l <ButtonPress-1> "$this change_current $which"
	bind $page.file.e <ButtonPress-1> "$this change_current $which"
	bind $page.file.e <Return> "$this execute_Data"
	
	button $page.load -text "Browse" \
	    -command "$NrrdReader initialize_ui" \
	    -width 12
	pack $page.load -side top -anchor n -padx 3 -pady 1
	
	
	### Dicom
	set page [$data.ui.tnb add -label "Dicom" \
		      -command "$this configure_readers Dicom"]
	
	button $page.load -text "Dicom Loader" \
	    -command "puts \"Fix opening Dicom UI\""
	
	pack $page.load -side top -anchor n \
	    -padx 3 -pady 10 -ipadx 2 -ipady 2
	
	### Analyze
	set page [$data.ui.tnb add -label "Analyze" \
		      -command "$this configure_readers Analyze"]
	
	button $page.load -text "Analyze Loader" \
	    -command "puts \"Fix opening Analyze UI\""
	
	pack $page.load -side top -anchor n \
	    -padx 3 -pady 10 -ipadx 2 -ipady 2
	
	### Field
	set page [$data.ui.tnb add -label "Field" \
		      -command "$this configure_readers Field"]
	
	global [set FieldReader]-filename
	frame $page.file
	pack $page.file -side top -anchor nw -padx 3 -pady 0 -fill x

	label $page.file.l -text "Field File:" 
	entry $page.file.e -textvariable [set FieldReader]-filename 
	pack $page.file.l $page.file.e -side left -padx 3 -pady 0 -anchor nw \
	    -fill x 
	bind $page.file.l <ButtonPress-1> "$this change_current $which"
	bind $page.file.e <ButtonPress-1> "$this change_current $which"
	bind $page.file.e <Return> "$this execute_Data"

	button $page.load -text "Browse" \
	    -command "$FieldReader initialize_ui" \
	    -width 12
	pack $page.load -side top -anchor n -padx 3 -pady 1
	
	# Set default view to be Nrrd
	$data.ui.tnb view "Nrrd"
    }

    ##############################
    ### configure_readers
    ##############################
    # Keeps the readers in sync.  Every time a different
    # data tab is selected (Nrrd, Dicom, Analyze) the other
    # readers must be disabled to avoid errors.
    method configure_readers { which } {
	set ChooseNrrd  [lindex [lindex $filters(0) $modules] $load_choose_input]
	set NrrdReader  [lindex [lindex $filters(0) $modules] $load_nrrd]
	set DicomNrrdReader  [lindex [lindex $filters(0) $modules] $load_dicom]
	set AnalyzeNrrdReader  [lindex [lindex $filters(0) $modules] $load_analyze]
	set FieldReader  [lindex [lindex $filters(0) $modules] $load_field]

        global [set ChooseNrrd]-port-index

	if {$which == "Nrrd"} {
	    set [set ChooseNrrd]-port-index 0
	    disableModule $NrrdReader 0
	    disableModule $DicomNrrdReader 1
	    disableModule $AnalyzeNrrdReader 1
	    disableModule $FieldReader 1
# 	    if {$initialized != 0} {
# 		$data_tab1 view "Nrrd"
# 		$data_tab2 view "Nrrd"
# 		set c_data_tab "Nrrd"
# 	    }
        } elseif {$which == "Dicom"} {
	    set [set ChooseNrrd]-port-index 1
	    disableModule $NrrdReader 1
	    disableModule $DicomNrrdReader 0
	    disableModule $AnalyzeNrrdReader 1
	    disableModule $FieldReader 1
#             if {$initialized != 0} {
# 		$data_tab1 view "Dicom"
# 		$data_tab2 view "Dicom"
# 		set c_data_tab "Dicom"
# 	    }
        } elseif {$which == "Analyze"} {
	    # Analyze
	    set [set ChooseNrrd]-port-index 2
	    disableModule $NrrdReader 1
	    disableModule $DicomNrrdReader 1
	    disableModule $AnalyzeNrrdReader 0
	    disableModule $FieldReader 1
# 	    if {$initialized != 0} {
# 		$data_tab1 view "Analyze"
# 		$data_tab2 view "Analyze"
# 		set c_data_tab "Analyze"
# 	    }
        } elseif {$which == "Field"} {
	    # Field
	    set [set ChooseNrrd]-port-index 3
	    disableModule $NrrdReader 1
	    disableModule $DicomNrrdReader 1
	    disableModule $AnalyzeNrrdReader 1
	    disableModule $FieldReader 0
# 	    if {$initialized != 0} {
# 		$data_tab1 view "Field"
# 		$data_tab2 view "Field"
# 		set c_data_tab "Field"
# 	    }
	} elseif {$which == "all"} {
	    if {[set [set ChooseNrrd]-port-index] == 0} {
		# nrrd
		disableModule $NrrdReader 0
		disableModule $DicomNrrdReader 1
		disableModule $AnalyzeNrrdReader 1
		disableModule $FieldReader 1
	    } elseif {[set [set ChooseNrrd]-port-index] == 1} {
		# dicom
		disableModule $NrrdReader) 1
		disableModule $DicomNrrdReader) 0
		disableModule $AnalyzeNrrdReader) 1
		disableModule $FieldReader) 1
	    } elseif {[set [set ChooseNrrd]-port-index] == 2} {
		# analyze
		disableModule $NrrdReader 1
		disableModule $DicomNrrdReader 1
		disableModule $AnalyzeNrrdReader 0
		disableModule $FieldReader 1
	    } else {
		# field
		disableModule $NrrdReader 1
		disableModule $DicomNrrdReader 1
		disableModule $AnalyzeNrrdReader 1
		disableModule $FieldReader 0
	    }
	}
    }
    
    

    #############################
    ### init_Vframe
    #############################
    # Initialize the visualization frame on the right. For this app
    # that includes the Vis Options and Viewer Options tabs.  
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

            if {$case == 0} {
               set vis_frame_tab1 $vis.tnb
            } else {
               set vis_frame_tab2 $vis.tnb	    
            }

	    set page [$vis.tnb add -label "Vis Options" \
			  -command "$this change_vis_frame \"Vis Options\""]
	    
	    ### Vis Options Tab
	    set v $page
            iwidgets::tabnotebook $v.tnb -width [expr $notebook_width - 20] \
		-height [expr $vis_height - 35] -tabpos n \
                -equaltabs false
	    pack $v.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            if {$case == 0} {
               #set vis_frame_tab1 $vis.tnb
            } else {
               #set vis_frame_tab2 $vis.tnb	    
            }

	    set page [$v.tnb add -label "Planes" \
			  -command "puts \"FIX ME: change vis type tabs\""]


	    global show_plane_x show_plane_y show_plane_z
	    checkbutton $page.x -text "Show Plane X (Sagittal)" \
		-variable show_plane_x \
		-command "$this toggle_show_plane_x"

	    checkbutton $page.y -text "Show Plane Y (Coronal)" \
		-variable show_plane_y \
		-command "$this toggle_show_plane_y"

	    checkbutton $page.z -text "Show Plane Z (Axial)" \
		-variable show_plane_z \
		-command "$this toggle_show_plane_z"

	    pack $page.x $page.y $page.z -side top -anchor nw \
		-padx 4 -pady 4

	    checkbutton $page.lines -text "Show Guidelines" \
		-variable show_guidelines \
		-command "$this toggle_show_guidelines" -state disabled
            pack $page.lines -side top -anchor nw -padx 4 -pady 7

	    global planes_color
	    iwidgets::labeledframe $page.isocolor \
		-labeltext "Color Planes Using" \
		-labelpos nw 
	    pack $page.isocolor -side top -anchor nw -padx 3 -pady 5
	    
	    set isocolor [$page.isocolor childsite]
	    
	    iwidgets::labeledframe $isocolor.maps \
		-labeltext "Color Maps" \
		-labelpos nw 
	    pack $isocolor.maps -side top -anchor n -padx 3 -pady 0 -fill x
	    
	    set maps [$isocolor.maps childsite]

	    global planes_mapType
	    
	    # Gray
	    frame $maps.gray
	    pack $maps.gray -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.gray.b -text "Gray" \
		-variable planes_mapType \
		-value 0 \
		-command "$this update_planes_color_by"
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
		-variable planes_mapType \
		-value 2 \
		-command "$this update_planes_color_by"
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
		-variable planes_mapType \
		-value 5 \
		-command "$this update_planes_color_by"
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
		-variable planes_mapType \
		-value 7 \
		-command "$this update_planes_color_by"
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
		-variable planes_mapType \
		-value 17 \
		-command "$this update_planes_color_by"
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "Blue-to-Red" $maps.bpseismic.f.canvas


            #######
	    set page [$v.tnb add -label "Isosurface" \
			  -command "puts \"FIX ME: change vis type tabs\""]

            global show_iso
	    checkbutton $page.toggle -text "Show Isosurface" \
		-variable show_iso \
		-command "$this toggle_show_iso"
            pack $page.toggle -side top -anchor nw -padx 3 -pady 3


	    # Isoval
            set Isosurface [lindex [lindex $filters(0) $modules] 27]
            global [set Isosurface]-isoval

	    frame $page.isoval
	    pack $page.isoval -side top -anchor nw -padx 3 -pady 3
	    
	    label $page.isoval.l -text "Isoval:" 
	    scale $page.isoval.s -from 0.0 -to 1000 \
		-length 100 -width 15 \
		-sliderlength 15 \
		-resolution 0.0001 \
                -variable [set Isosurface]-isoval \
		-showvalue false \
		-orient horizontal

	    if {$case == 0} {
		set iso_slider1 $page.isoval.s
	    } else {
		set iso_slider2 $page.isoval.s
	    }
	    
            bind $page.isoval.s <ButtonRelease> "global show_iso; if{$show_iso == 1} {[set Isosurface]-c needexecute}"
	    
            label $page.isoval.val -textvariable [set Isosurface]-isoval 
	    
	    pack $page.isoval.l $page.isoval.s $page.isoval.val -side left -anchor nw -padx 3     

	    global iso_color
	    iwidgets::labeledframe $page.isocolor \
		-labeltext "Color Isosurface Using" \
		-labelpos nw 
	    pack $page.isocolor -side top -anchor nw -padx 3 -pady 5
	    
	    set isocolor [$page.isocolor childsite]
	    
	    iwidgets::labeledframe $isocolor.maps \
		-labeltext "Color Maps" \
		-labelpos nw 
	    pack $isocolor.maps -side top -anchor n -padx 3 -pady 0 -fill x
	    
	    set maps [$isocolor.maps childsite]

	    global iso_mapType
	    
	    # Gray
	    frame $maps.gray
	    pack $maps.gray -side top -anchor nw -padx 3 -pady 1 \
		-fill x -expand 1
	    radiobutton $maps.gray.b -text "Gray" \
		-variable iso_mapType \
		-value 0 \
		-command "$this update_iso_color_by"
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
		-variable iso_mapType \
		-value 2 \
		-command "$this update_iso_color_by"
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
		-variable iso_mapType \
		-value 5 \
		-command "$this update_iso_color_by"
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
		-variable iso_mapType \
		-value 7 \
		-command "$this update_iso_color_by"
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
		-variable iso_mapType \
		-value 17 \
		-command "$this update_iso_color_by"
	    pack $maps.bpseismic.b -side left -anchor nw -padx 3 -pady 0
	    
	    frame $maps.bpseismic.f -relief sunken -borderwidth 2
	    pack $maps.bpseismic.f -padx 2 -pady 0 -side right -anchor e
	    canvas $maps.bpseismic.f.canvas -bg "#ffffff" -height $colormap_height -width $colormap_width
	    pack $maps.bpseismic.f.canvas -anchor e
	    
	    draw_colormap "Blue-to-Red" $maps.bpseismic.f.canvas



            #######
	    set page [$v.tnb add -label "Volume Rendering" \
			  -command "puts \"FIX ME: change vis type tabs\""]


            global show_volume_ren
	    checkbutton $page.toggle -text "Show Volume Rendering" \
		-variable show_vol_ren \
		-command "$this toggle_show_vol_ren"
            pack $page.toggle -side top -anchor nw -padx 3 -pady 3


            button $page.vol -text "Edit Transfer Function" \
                -command "$mods(EditTransferFunc) initialize_ui"
            pack $page.vol -side top -anchor n -padx 3 -pady 3
            
            set VolumeVisualizer [lindex [lindex $filters(0) $modules] 14]
            set n "[set VolumeVisualizer]-c needexecute"
            set s "[set VolumeVisualizer] state"

            global [set VolumeVisualizer]-render_style

            frame $page.fmode
            pack $page.fmode -padx 2 -pady 2 -fill x
            label $page.fmode.mode -text "Mode"
	    radiobutton $page.fmode.modeo -text "Over Operator" -relief flat \
		    -variable [set VolumeVisualizer]-render_style -value 0 \
    	  	    -anchor w -command $n
   	    radiobutton $page.fmode.modem -text "MIP" -relief flat \
		    -variable [set VolumeVisualizer]-render_style -value 1 \
		    -anchor w -command $n
   	    pack $page.fmode.mode $page.fmode.modeo $page.fmode.modem \
                -side left -fill x -padx 4 -pady 4

            frame $page.fres
            pack $page.fres -padx 2 -pady 2 -fill x
            label $page.fres.res -text "Resolution (bits)"
	    radiobutton $page.fres.b0 -text 8 -variable [set VolumeVisualizer]-blend_res -value 8 \
	        -command $n
    	    radiobutton $page.fres.b1 -text 16 -variable [set VolumeVisualizer]-blend_res -value 16 \
	        -command $n
 	    radiobutton $page.fres.b2 -text 32 -variable [set VolumeVisualizer]-blend_res -value 32 \
	        -command $n
	    pack $page.fres.res $page.fres.b0 $page.fres.b1 $page.fres.b2 \
                -side left -fill x -padx 4 -pady 4

        #-----------------------------------------------------------
        # Shading
        #-----------------------------------------------------------
	checkbutton $page.shading -text "Shading" -relief flat \
            -variable [set VolumeVisualizer]-shading -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"
        pack $page.shading -side top -fill x -padx 4

        #-----------------------------------------------------------
        # Light
        #-----------------------------------------------------------
 	frame $page.f5
 	pack $page.f5 -padx 2 -pady 2 -fill x
 	label $page.f5.light -text "Attach Light to"
 	radiobutton $page.f5.light0 -text "Light 0" -relief flat \
            -variable [set VolumeVisualizer]-light -value 0 \
            -anchor w -command $n
 	radiobutton $page.f5.light1 -text "Light 1" -relief flat \
            -variable [set VolumeVisualizer]-light -value 1 \
            -anchor w -command $n
        pack $page.f5.light $page.f5.light0 $page.f5.light1 \
            -side left -fill x -padx 4

#         #-----------------------------------------------------------
#         # Material
#         #-----------------------------------------------------------
# 	frame $page.f6 -relief groove -borderwidth 2
# 	pack $page.f6 -padx 2 -pady 2 -fill x
#  	label $page.f6.material -text "Material"
# 	global [set VolumeVisualizer]-ambient
# 	scale $page.f6.ambient -variable [set VolumeVisualizer]-ambient \
#             -from 0.0 -to 1.0 -label "Ambient" \
#             -showvalue true -resolution 0.001 \
#             -orient horizontal
# 	global [set VolumeVisualizer]-diffuse
# 	scale $page.f6.diffuse -variable [set VolumeVisualizer]-diffuse \
# 		-from 0.0 -to 1.0 -label "Diffuse" \
# 		-showvalue true -resolution 0.001 \
# 		-orient horizontal
# 	global [set VolumeVisualizer]-specular
# 	scale $page.f6.specular -variable [set VolumeVisualizer]-specular \
# 		-from 0.0 -to 1.0 -label "Specular" \
# 		-showvalue true -resolution 0.001 \
# 		-orient horizontal
# 	global [set VolumeVisualizer]-shine
# 	scale $page.f6.shine -variable [set VolumeVisualizer]-shine \
# 		-from 1.0 -to 128.0 -label "Shine" \
# 		-showvalue true -resolution 1.0 \
# 		-orient horizontal
#         pack $page.f6.material $page.f6.ambient $page.f6.diffuse \
#             $page.f6.specular $page.f6.shine \
#             -side top -fill x -padx 4

        #-----------------------------------------------------------
        # Sampling
        #-----------------------------------------------------------
        frame $page.sampling -relief groove -borderwidth 2
        pack $page.sampling -padx 2 -pady 2 -fill x
        label $page.sampling.l -text "Sampling"

	scale $page.sampling.srate_hi -variable [set VolumeVisualizer]-sampling_rate_hi \
            -from 0.5 -to 10.0 -label "Sampling Rate" \
            -showvalue true -resolution 0.1 \
            -orient horizontal \

	scale $page.sampling.srate_lo -variable [set VolumeVisualizer]-sampling_rate_lo \
            -from 0.1 -to 5.0 -label "Interactive Sampling Rate" \
            -showvalue true -resolution 0.1 \
            -orient horizontal \

	pack $page.sampling.l $page.sampling.srate_hi \
            $page.sampling.srate_lo -side top -fill x -padx 4 -pady 2
        
        #-----------------------------------------------------------
        # Transfer Function
        #-----------------------------------------------------------
        frame $page.tf -relief groove -borderwidth 2
        pack $page.tf -padx 2 -pady 2 -fill x
        label $page.tf.l -text "Transfer Function"

	scale $page.tf.stransp -variable [set VolumeVisualizer]-alpha_scale \
		-from -1.0 -to 1.0 -label "Global Opacity" \
		-showvalue true -resolution 0.001 \
		-orient horizontal 

	pack $page.tf.l $page.tf.stransp \
            -side top -fill x -padx 4 -pady 2

#        bind $page.f6.ambient <ButtonRelease> $n
#        bind $page.f6.diffuse <ButtonRelease> $n
#        bind $page.f6.specular <ButtonRelease> $n
#        bind $page.f6.shine <ButtonRelease> $n

	bind $page.sampling.srate_hi <ButtonRelease> $n
	bind $page.sampling.srate_lo <ButtonRelease> $n

	bind $page.tf.stransp <ButtonRelease> $n
	



            $v.tnb select "Planes"





	    ### Renderer Options Tab
	    create_viewer_tab $vis


	    ### Attach/Detach button
            frame $m.d 
	    pack $m.d -side left -anchor e
            for {set i 0} {$i<42} {incr i} {
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
	    pack $attachedPFr -anchor n -side left -before $win.viewers
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
	    pack $attachedVFr -anchor n -side left -after $win.viewers 
	    set new_width [expr $c_width + $vis_width]
            append geom $new_width x $c_height
	    wm geometry $win $geom
	    set IsVAttached 1
	}
    }


    ############################
    ### change_vis_frame
    ############################
    # Method called when Visualization tabs are changed from
    # the standard options to the global viewer controls
    method change_vis_frame { which } {
	# change tabs for attached and detached
        if {$initialized != 0} {
	    if {$which == "Vis Options"} {
		# Vis Options
		$vis_frame_tab1 view "Vis Options"
		$vis_frame_tab2 view "Vis Options"
		set c_vis_tab "Vis Options"
	    } else {
 		$vis_frame_tab1 view "Viewer Options"
 		$vis_frame_tab2 view "Viewer Options"
		set c_vis_tab "Viewer Options"
	    }
	}
    }

    method execute_Data {} {
	# execute the appropriate reader
	
	set ChooseNrrd  [lindex [lindex $filters(0) $modules] $load_choose_input]
        global [set ChooseNrrd]-port-index
        set port [set [set ChooseNrrd]-port-index]
        set mod ""
        if {$port == 0} {
	    # Nrrd
            set mod [lindex [lindex $filters(0) $modules] $load_nrrd]
	} elseif {$port == 1} {
	    # Dicom
            set mod [lindex [lindex $filters(0) $modules] $load_dicom]
	} elseif {$port == 2} {
	    # Analyze
            set mod [lindex [lindex $filters(0) $modules] $load_analyze]
	} else {
	    #Field
            set mod [lindex [lindex $filters(0) $modules] $load_field]
	}

	$mod-c needexecute
        set has_executed 1
    }


    method add_Resample {} {
	global mods

	# Figure out what choose port to use
	set choose [$this determine_choose_port]

	# add modules
	set m1 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuResample" 100 [expr 10 * $num_filters + 500] ]
	
	# add connection to Choose module and new module
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis]
	set output_mod [lindex [lindex $filters($current) $output] 0]
	set output_port [lindex [lindex $filters($current) $output] 1]
	addConnection $output_mod $output_port $m1 0
	addConnection $m1 0 $ChooseNrrd $choose

	set row $grid_rows
	# if inserting, disconnect current to current's next and connect current
	# to new and new to current's next
	set insert 0
	if {[lindex $filters($current) $next_index] != "end"} {
            set insert 1
	    set n [lindex $filters($current) $next_index] 
	    set next_mod [lindex [lindex $filters($n) $input] 0]
	    set next_port [lindex [lindex $filters($n) $input] 1]

	    set current_mod [lindex [lindex $filters($current) $output] 0]
	    set current_port [lindex [lindex $filters($current) $output] 1]

	    destroyConnection "$current_mod $current_port $next_mod $next_port"
	    addConnection $m1 0 $next_mod $next_port
	    
	    set row [expr [lindex $filters($current) $which_row] + 1]
	    
	    $this move_down_filters $row
	}

        # add to filters array
        set filters($num_filters) [list resample [list $m1] [list $m1 0] [list $m1 0] $current [lindex $filters($current) $next_index] $choose $row 1]

	# Make current frame regular
	set p f$current
	$history1.$p configure -background grey75 -foreground black -borderwidth 2
	$history2.$p configure -background grey75 -foreground black -borderwidth 2

        $this add_Resample_UI $history1 $row $num_filters
        $this add_Resample_UI $history2 $row $num_filters

        if {!$insert} {
	    $attachedPFr.f.p.sf justify bottom
	    $detachedPFr.f.p.sf justify bottom
	}

	# Update choose port if
        global eye
        set eye $num_filters
        $this change_eye $num_filters

	# update vars
	set filters($current) [lreplace $filters($current) $next_index $next_index $num_filters]

        change_current $num_filters

	set num_filters [expr $num_filters + 1]
	set grid_rows [expr $grid_rows + 1]
    }


    method add_Crop {} {
	global mods

	# Figure out what choose port to use
	set choose [$this determine_choose_port]

	# add modules
	set m1 [addModuleAtPosition "Teem" "UnuAtoM" "UnuCrop" 100 [expr 10 * $num_filters + 500] ]
	
	# add connection to Choose module and new module
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis]
	set output_mod [lindex [lindex $filters($current) $output] 0]
	set output_port [lindex [lindex $filters($current) $output] 1]
	addConnection $output_mod $output_port $m1 0
	addConnection $m1 0 $ChooseNrrd $choose

	set row $grid_rows
	# if inserting, disconnect current to current's next and connect current
	# to new and new to current's next
	set insert 0
	if {[lindex $filters($current) $next_index] != "end"} {
            set insert 1
	    set n [lindex $filters($current) $next_index] 
	    set next_mod [lindex [lindex $filters($n) $input] 0]
	    set next_port [lindex [lindex $filters($n) $input] 1]

	    set current_mod [lindex [lindex $filters($current) $output] 0]
	    set current_port [lindex [lindex $filters($current) $output] 1]

	    destroyConnection "$current_mod $current_port $next_mod $next_port"
	    addConnection $m1 0 $next_mod $next_port
	    
	    set row [expr [lindex $filters($current) $which_row] + 1]
	    
	    $this move_down_filters $row
	}

        # add to filters array
        set filters($num_filters) [list crop [list $m1] [list $m1 0] [list $m1 0] $current [lindex $filters($current) $next_index] $choose $row 1]

	# Make current frame regular
	set p f$current
	$history1.$p configure -background grey75 -foreground black -borderwidth 2
	$history2.$p configure -background grey75 -foreground black -borderwidth 2

        $this add_Crop_UI $history1 $row $num_filters
        $this add_Crop_UI $history2 $row $num_filters

        if {!$insert} {
	    $attachedPFr.f.p.sf justify bottom
	    $detachedPFr.f.p.sf justify bottom
	}

	# Update choose port if
        global eye
        set eye $num_filters
        $this change_eye $num_filters

	# update vars
	set filters($current) [lreplace $filters($current) $next_index $next_index $num_filters]

        change_current $num_filters

	set num_filters [expr $num_filters + 1]
	set grid_rows [expr $grid_rows + 1]
    }
    
    method add_Cmedian {} {
	global mods

	# Figure out what choose port to use
	set choose [$this determine_choose_port]

	# add modules
	set m1 [addModuleAtPosition "Teem" "UnuAtoM" "UnuCmedian" 100 [expr 10 * $num_filters + 500] ]
	
	# add connection to Choose module and new module
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis]
	set output_mod [lindex [lindex $filters($current) $output] 0]
	set output_port [lindex [lindex $filters($current) $output] 1]
	addConnection $output_mod $output_port $m1 0
	addConnection $m1 0 $ChooseNrrd $choose

	set row $grid_rows
	# if inserting, disconnect current to current's next and connect current
	# to new and new to current's next
	set insert 0
	if {[lindex $filters($current) $next_index] != "end"} {
            set insert 1
	    set n [lindex $filters($current) $next_index] 
	    set next_mod [lindex [lindex $filters($n) $input] 0]
	    set next_port [lindex [lindex $filters($n) $input] 1]

	    set current_mod [lindex [lindex $filters($current) $output] 0]
	    set current_port [lindex [lindex $filters($current) $output] 1]

	    destroyConnection "$current_mod $current_port $next_mod $next_port"
	    addConnection $m1 0 $next_mod $next_port
	    
	    set row [expr [lindex $filters($current) $which_row] + 1]
	    
	    $this move_down_filters $row
	}

        # add to filters array
        set filters($num_filters) [list cmedian [list $m1] [list $m1 0] [list $m1 0] $current [lindex $filters($current) $next_index] $choose $row 1]

	# Make current frame regular
	set p f$current
	$history1.$p configure -background grey75 -foreground black -borderwidth 2
	$history2.$p configure -background grey75 -foreground black -borderwidth 2

        $this add_Cmedian_UI $history1 $row $num_filters
        $this add_Cmedian_UI $history2 $row $num_filters

        if {!$insert} {
	    $attachedPFr.f.p.sf justify bottom
	    $detachedPFr.f.p.sf justify bottom
	}

	# Update choose port if
        global eye
        set eye $num_filters
        $this change_eye $num_filters

	# update vars
	set filters($current) [lreplace $filters($current) $next_index $next_index $num_filters]

        change_current $num_filters

	set num_filters [expr $num_filters + 1]
	set grid_rows [expr $grid_rows + 1]
    }

    method add_Histo {} {
	global mods

	# Figure out what choose port to use
	set choose [$this determine_choose_port]

	# add modules
	set m1 [addModuleAtPosition "Teem" "UnuAtoM" "UnuHeq" 100 [expr 10 * $num_filters + 500] ]
	
	# add connection to Choose module and new module
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis]
	set output_mod [lindex [lindex $filters($current) $output] 0]
	set output_port [lindex [lindex $filters($current) $output] 1]
	addConnection $output_mod $output_port $m1 0
	addConnection $m1 0 $ChooseNrrd $choose

	set row $grid_rows
	# if inserting, disconnect current to current's next and connect current
	# to new and new to current's next
	set insert 0
	if {[lindex $filters($current) $next_index] != "end"} {
            set insert 1
	    set n [lindex $filters($current) $next_index] 
	    set next_mod [lindex [lindex $filters($n) $input] 0]
	    set next_port [lindex [lindex $filters($n) $input] 1]

	    set current_mod [lindex [lindex $filters($current) $output] 0]
	    set current_port [lindex [lindex $filters($current) $output] 1]

	    destroyConnection "$current_mod $current_port $next_mod $next_port"
	    addConnection $m1 0 $next_mod $next_port
	    
	    set row [expr [lindex $filters($current) $which_row] + 1]
	    
	    $this move_down_filters $row
	}

        # add to filters array
        set filters($num_filters) [list histo [list $m1] [list $m1 0] [list $m1 0] $current [lindex $filters($current) $next_index] $choose $row 1]

	# Make current frame regular
	set p f$current
	$history1.$p configure -background grey75 -foreground black -borderwidth 2
	$history2.$p configure -background grey75 -foreground black -borderwidth 2

        $this add_Histo_UI $history1 $row $num_filters
        $this add_Histo_UI $history2 $row $num_filters

        if {!$insert} {
	    $attachedPFr.f.p.sf justify bottom
	    $detachedPFr.f.p.sf justify bottom
	}

	# Update choose port if
        global eye
        set eye $num_filters
        $this change_eye $num_filters

	# update vars
	set filters($current) [lreplace $filters($current) $next_index $next_index $num_filters]

        change_current $num_filters

	set num_filters [expr $num_filters + 1]
	set grid_rows [expr $grid_rows + 1]
    }

    method print_filters {} {
	parray filters
    }

    method add_Resample_UI {history row which} {

	# Add eye radiobutton
        global eye
	radiobutton $history.eye$which -text "" \
	    -variable eye -value $which \
	    -command "$this change_eye $which"

	grid config $history.eye$which -column 0 -row $row -sticky "nw"

	iwidgets::labeledframe $history.f$which \
	    -labeltext "Resample" \
	    -labelpos nw -foreground white \
	    -borderwidth 2 -background $scolor
	grid config $history.f$which -column 1 -row $row -sticky "nw"

	bind $history.f$which <ButtonPress-1> "$this change_current $which"

	set w [$history.f$which childsite]

	frame $w.expand
	pack $w.expand -side top -anchor nw

	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
	set show [image create photo -file ${image_dir}/play-icon-small.ppm]
	button $w.expand.b -image $show \
	    -anchor nw \
	    -command "$this change_visibility $which" \
	    -relief flat
	label $w.expand.l -text "Resample - Unknown" -width $label_width
	pack $w.expand.b $w.expand.l -side left -anchor nw 

	bind $w.expand.l <ButtonPress-1> "$this change_current $which"
	bind $w.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
	frame $w.ui
	pack $w.ui -side top -expand yes -fill both

        set UnuResample [lindex [lindex $filters($which) $modules] 0]
	for {set i 0} {$i < $dimension} {incr i} {
	    global [set UnuResample]-resampAxis$i
	    if {!$loading_ui} {
               set [set UnuResample]-resampAxis$i "x1"
            }
	    make_entry $w.ui.$i "Axis $i:" $UnuResample-resampAxis$i $which
	    pack $w.ui.$i -side top -anchor nw -expand yes -fill x
	}

        global [set UnuResample]-sigma [set UnuResample]-extent
        make_entry $w.ui.sigma "Gaussian Sigma:" [set UnuResample]-sigma $which
        make_entry $w.ui.extent "Gaussian Extent:" [set UnuResample]-extent $which
	
	pack $w.ui.sigma $w.ui.extent -side top -anchor nw -fill x 
	
 	iwidgets::optionmenu $w.ui.kernel -labeltext "Filter Type:" \
 	    -labelpos w \
            -command "$this change_kernel $w.ui.kernel $which"
 	pack $w.ui.kernel -side top -anchor nw 

	bind $w.ui.kernel <ButtonPress-1> "$this change_current $which"
	
 	$w.ui.kernel insert end Box Tent "Cubic (Catmull-Rom)" \
 	    "Cubic (B-spline)" Quartic Gaussian
	
 	$w.ui.kernel select Gaussian
    }



    method add_Crop_UI {history row which} {

	# Add eye radiobutton
        global eye
	radiobutton $history.eye$which -text "" \
	    -variable eye -value $which \
	    -command "$this change_eye $which"

	grid config $history.eye$which -column 0 -row $row -sticky "nw"

	iwidgets::labeledframe $history.f$which \
	    -labeltext "Crop" \
	    -labelpos nw -foreground white \
	    -borderwidth 2 -background $scolor
	grid config $history.f$which -column 1 -row $row -sticky "nw"

	bind $history.f$which <ButtonPress-1> "$this change_current $which"

	set w [$history.f$which childsite]

	frame $w.expand
	pack $w.expand -side top -anchor nw

	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
	set show [image create photo -file ${image_dir}/play-icon-small.ppm]
	button $w.expand.b -image $show \
	    -anchor nw \
	    -command "$this change_visibility $which" \
	    -relief flat
	label $w.expand.l -text "Crop - Unknown" -width $label_width \
	    -anchor nw
	pack $w.expand.b $w.expand.l -side left -anchor nw 

	bind $w.expand.l <ButtonPress-1> "$this change_current $which"
	bind $w.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
	frame $w.ui
	pack $w.ui -side top -anchor nw -expand yes -fill x
	
	set UnuCrop [lindex [lindex $filters($which) $modules] 0]
	global [set UnuCrop]-num-axes
	set [set UnuCrop]-num-axes $dimension        

        global [set UnuCrop]-reset_data
        set [set UnuCrop]-reset_data 0

	for {set i 0} {$i < $dimension} {incr i} {
	    global [set UnuCrop]-minAxis$i
	    global [set UnuCrop]-maxAxis$i
            if {!$loading_ui} {
	        set [set UnuCrop]-minAxis$i 0
		set [set UnuCrop]-maxAxis$i "M"
	    }

	    frame $w.ui.$i
	    pack $w.ui.$i -side top -anchor nw -expand yes -fill x
	    label $w.ui.$i.minl -text "Min Axis $i:"
	    entry $w.ui.$i.minv -textvariable [set UnuCrop]-minAxis$i \
		-width 6
	    label $w.ui.$i.maxl -text "Max Axis $i:"
	    entry $w.ui.$i.maxv -textvariable [set UnuCrop]-maxAxis$i \
		-width 6
	    pack $w.ui.$i.minl $w.ui.$i.minv $w.ui.$i.maxl $w.ui.$i.maxv -side left -anchor nw \
		-expand yes -fill x
	    bind $w.ui.$i.minl <ButtonPress-1> "$this change_current $which"
	    bind $w.ui.$i.minv <ButtonPress-1> "$this change_current $which"
	    bind $w.ui.$i.maxl <ButtonPress-1> "$this change_current $which"
	    bind $w.ui.$i.maxv <ButtonPress-1> "$this change_current $which"
	}
    }	


    method add_Cmedian_UI {history row which} {

	# Add eye radiobutton
	global eye
	radiobutton $history.eye$which -text "" \
	    -variable eye -value $which \
	    -command "$this change_eye $which"

	grid config $history.eye$which -column 0 -row $row -sticky "nw"

	iwidgets::labeledframe $history.f$which \
	    -labeltext "Cmedian" \
	    -labelpos nw -foreground white \
	    -borderwidth 2 -background $scolor
	grid config $history.f$which -column 1 -row $row -sticky "nw"

	bind $history.f$which <ButtonPress-1> "$this change_current $which"

	set w [$history.f$which childsite]

	frame $w.expand
	pack $w.expand -side top -anchor nw

	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
	set show [image create photo -file ${image_dir}/play-icon-small.ppm]
	button $w.expand.b -image $show \
	    -anchor nw \
	    -command "$this change_visibility $which" \
	    -relief flat
	label $w.expand.l -text "Cmedian - Unknown" -width $label_width \
	    -anchor nw
	pack $w.expand.b $w.expand.l -side left -anchor nw 

	bind $w.expand.l <ButtonPress-1> "$this change_current $which"
	bind $w.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
	frame $w.ui
	pack $w.ui -side top -anchor nw -expand yes -fill x
	
        set UnuCmedian [lindex [lindex $filters($which) $modules] 0]
	global [set UnuCmedian]-radius
	frame $w.ui.radius
	pack $w.ui.radius -side top -anchor nw -expand yes -fill x
	label $w.ui.radius.l -text "Radius:"
	entry $w.ui.radius.v -textvariable [set UnuCmedian]-radius \
	    -width 6
	pack $w.ui.radius.l $w.ui.radius.v -side left -anchor nw \
	    -expand yes -fill x
	bind $w.ui.radius.l <ButtonPress-1> "$this change_current $which"
	bind $w.ui.radius.v <ButtonPress-1> "$this change_current $which"
    }

    method add_Histo_UI {history row which} {
	
	# Add eye radiobutton
	global eye
	radiobutton $history.eye$which -text "" \
	    -variable eye -value $which \
	    -command "$this change_eye $which"

	grid config $history.eye$which -column 0 -row $row -sticky "nw"

	iwidgets::labeledframe $history.f$which \
	    -labeltext "Histogram" \
	    -labelpos nw -foreground white \
	    -borderwidth 2 -background $scolor
	grid config $history.f$which -column 1 -row $row -sticky "nw"

	bind $history.f$which <ButtonPress-1> "$this change_current $which"

	set w [$history.f$which childsite]

	frame $w.expand
	pack $w.expand -side top -anchor nw

	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
	set show [image create photo -file ${image_dir}/play-icon-small.ppm]
	button $w.expand.b -image $show \
	    -anchor nw \
	    -command "$this change_visibility $which" \
	    -relief flat
	label $w.expand.l -text "Histogram - Unknown" -width $label_width \
	    -anchor nw
	pack $w.expand.b $w.expand.l -side left -anchor nw 

	bind $w.expand.l <ButtonPress-1> "$this change_current $which"
	bind $w.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
	frame $w.ui
	pack $w.ui -side top -anchor nw -expand yes -fill x
	
        set UnuHeq  [lindex [lindex $filters($which) $modules] 0]
	global [set UnuHeq]-bins
	global [set UnuHeq]-amount
      
        if {!$loading_ui} {
	    set [set UnuHeq]-bins 3000
   	    set [set UnuHeq]-amount 1.0
        }

	scale $w.ui.amount -label "Amount" \
	    -from 0.0 -to 1.0 \
	    -variable [set UnuHeq]-amount \
	    -showvalue true \
	    -orient horizontal \
	    -resolution 0.01
	pack $w.ui.amount -side top -anchor nw -expand yes -fill x
	
	bind $w.ui.amount <ButtonPress-1> "$this change_current $which"
    }

    method determine_choose_port {} {
	global Subnet
  	set choose 0
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis]

        foreach conn $Subnet([set ChooseNrrd]_connections) { ;# all module connections
  	    if {[lindex $conn 2] == $ChooseNrrd} {
  		set choose [expr $choose + 1]
  	    }
  	}
  	return $choose

    }


    method filter_Delete {} {
	global mods

	# Do not remove Load (0)
	if {$current == 0} {
	    tk_messageBox -message "Cannot delete Load step." -type ok -icon info -parent .standalone
	    return
	}

	set current_row [lindex $filters($current) $which_row]

	# remove ui
	grid remove $history1.f$current 
	grid remove $history2.f$current 

	grid remove $history1.eye$current
	grid remove $history2.eye$current
	
	set next [lindex $filters($current) $next_index]
	set current_choose [lindex $filters($current) $choose_port]
	
	if {$next != "end"} {
	    move_up_filters [lindex $filters($current) $which_row]
	}

        # delete filter modules
        set l [llength [lindex $filters($current) $modules]]
        for {set i 0} {$i < $l} {incr i} {
            moduleDestroy [lindex [lindex $filters($current) $modules] $i]
        }

        # update choose ports of other filters
        set port [lindex $filters($current) $choose_port]
        $this update_choose_ports $port

	set prev_mod [lindex [lindex $filters([lindex $filters($current) $prev_index]) $output] 0]
	set prev_port [lindex [lindex $filters([lindex $filters($current) $prev_index]) $output] 1]
	set current_mod [lindex [lindex $filters($current) $output] 0]
	set current_port [lindex [lindex $filters($current) $output] 1]

	# add connection from previous to next
	if {$next != "end"} {
	    set next_mod [lindex [lindex $filters([lindex $filters($current) $next_index]) $output] 0]
	    set next_port [lindex [lindex $filters([lindex $filters($current) $next_index]) $output] 1]
	    addConnection $prev_mod $prev_port $next_mod $next_port
	}    

	# set which_row to be -1
	set filters($current) [lreplace $filters($current) $which_row $which_row -1]

	# update prev's next
	set p [lindex $filters($current) $prev_index]
	set n [lindex $filters($current) $next_index]
	set filters($p) [lreplace $filters($p) $next_index $next_index $n]

	# update next's prev (only if not end)
	if {$next != "end"} {
	    set filters($n) [lreplace $filters($n) $prev_index $prev_index $p]
	} 

	# determine next filter to be currently selected
	# by iterating over all valid filters and choosing
	# the one on the previous row
        set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis]  
	global $ChooseNrrd-port-index

	set next_row [expr $current_row + 1]
	set next_filter 0

	if {[string equal $next "end"]} {
	    set next_row [expr $current_row - 1]
	} else {
	    set next_row [expr $next_row - 1]
	}

 	for {set i 0} {$i < $num_filters} {incr i} {
 	    # check if it hasn't been deleted
 	    set r [lindex $filters($i) $which_row]
 	    if {$r == $next_row} {
 		set next_filter $i
 	    }
 	}
	set next_choose [lindex $filters($next_filter) $choose_port]

	set $ChooseNrrd--port-index $next_choose

        global eye
        set eye $next_filter
	$this change_eye $next_filter

	$this change_current $next_filter

	set grid_rows [expr $grid_rows - 1]
    }

    method execute_current {} {
	set mod [lindex [lindex $filters($current) $input] 0]
	$mod-c needexecute

        set has_executed 1
    }

    method move_down_filters {row} {
	# Since we are inserting, we need to forget the grid rows
	# below us and move them down a row
	set re_pack [list]
	for {set i 1} {$i < $num_filters} {incr i} {
	    if {[info exists filters($i)]} {
		set tmp_row [lindex $filters($i) $which_row]
		if {$tmp_row != -1 && ($tmp_row > $row || $tmp_row == $row)} {
		    grid forget $history1.f$i
		    grid forget $history1.eye$i
		    grid forget $history2.f$i
		    grid forget $history2.eye$i
		    
		    set filters($i) [lreplace $filters($i) $which_row $which_row [expr $tmp_row + 1] ]		    
		    lappend re_pack $i
		}
	    }
	}
	# need to re configure them after they have all been removed
	for {set i 0} {$i < [llength $re_pack]} {incr i} {
	    set index [lindex $re_pack $i]
	    set new_row [lindex $filters($index) $which_row]
	    grid config $history1.f$index -row $new_row -column 1 -sticky "nw"
	    grid config $history1.eye$index -row $new_row -column 0 -sticky "nw"
	    grid config $history2.f$index -row $new_row -column 1 -sticky "nw"
	    grid config $history2.eye$index -row $new_row -column 0 -sticky "nw"
	}
    }

    method move_up_filters {row} {
	# Since we are deleting, we need to forget the grid rows
	# below us and move them up a row
	set re_pack [list]
	for {set i 1} {$i < $num_filters} {incr i} {
	    if {[info exists filters($i)]} {
		set tmp_row [lindex $filters($i) $which_row]
		if {$tmp_row != -1 && $tmp_row > $row } {
		    grid forget $history1.f$i
		    grid forget $history1.eye$i
		    grid forget $history2.f$i
		    grid forget $history2.eye$i
		    
		    set filters($i) [lreplace $filters($i) $which_row $which_row [expr $tmp_row - 1] ]		    
		    lappend re_pack $i
		}
	    }
	}
	# need to re configure them after they have all been removed
	for {set i 0} {$i < [llength $re_pack]} {incr i} {
	    set index [lindex $re_pack $i]
	    set new_row [lindex $filters($index) $which_row]
	    grid config $history1.f$index -row $new_row -column 1 -sticky "nw"
	    grid config $history1.eye$index -row $new_row -column 0 -sticky "nw"
	    grid config $history2.f$index -row $new_row -column 1 -sticky "nw"
	    grid config $history2.eye$index -row $new_row -column 0 -sticky "nw"
	}
    }

    # Decremement any choose ports that are greater than the current port
    method update_choose_ports {port} {
	for {set i 1} {$i < $num_filters} { incr i} {
	    set tmp_row [lindex $filters($i) $which_row]
	    set tmp_port [lindex $filters($i) $choose_port]
		if {$tmp_row != -1 && $tmp_port > $port } {  
                    set filters($i) [lreplace $filters($i) $choose_port $choose_port [expr $tmp_port - 1] ]
                }
	}
    }

    method change_eye {which} {
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis] 
        set port [lindex $filters($which) $choose_port]
        global [set ChooseNrrd]-port-index
        set [set ChooseNrrd]-port-index $port
    }

    method change_current {which} {
	global mods

	# fix old one
	set p f$current
	$history1.$p configure -background grey75 -foreground black -borderwidth 2
	$history2.$p configure -background grey75 -foreground black -borderwidth 2
	
	set current $which
	set p f$current
	$history1.$p configure -background $scolor -foreground white -borderwidth 2
	$history2.$p configure -background $scolor -foreground white -borderwidth 2
    }

    method change_label {x y which} {

	if {![winfo exists .standalone.change_label]} {
	    # bring up ui to type name
	    global new_label
	    set old_label [$history1.f$which.childsite.expand.l cget -text]
	    set new_label $old_label
	    
	    toplevel .standalone.change_label
	    wm minsize .standalone.change_label 150 50
	    set x [expr $x + 10]
	    wm geometry .standalone.change_label "+$x+$y"
	    
	    label .standalone.change_label.l -text "Please enter a label for this filter."
	    pack .standalone.change_label.l -side top -anchor nw -padx 4 -pady 4
	    
	    frame .standalone.change_label.info
	    pack .standalone.change_label.info -side top -anchor nw
	    
	    label .standalone.change_label.info.l -text "Label:"
	    entry .standalone.change_label.info.e -textvariable new_label \
		-selectbackground $scolor
	    pack .standalone.change_label.info.l .standalone.change_label.info.e -side left -anchor nw \
		-padx 4 -pady 4
	    bind .standalone.change_label.info.e <Return> "destroy .standalone.change_label"

	    .standalone.change_label.info.e selection to end
	    
	    frame .standalone.change_label.buttons
	    pack .standalone.change_label.buttons -side top -anchor n
	    
	    button .standalone.change_label.buttons.apply -text "Apply" \
		-command "destroy .standalone.change_label"
	    button .standalone.change_label.buttons.cancel -text "Cancel" \
		-command "global new_label; set new_label CaNceL; destroy .standalone.change_label"
	    pack .standalone.change_label.buttons.apply .standalone.change_label.buttons.cancel -side left \
		-padx 4 -pady 4 -anchor n
	    
	    tkwait window .standalone.change_label
	    
	    if {$new_label != "CaNceL" && $new_label != $old_label} {
		# change label
		$history1.f$which.childsite.expand.l configure -text $new_label
		$history2.f$which.childsite.expand.l configure -text $new_label
	    }
	} else {
	    SciRaise .standalone.change_label
	}
    }

    method change_visibility {num} {
	set visible [lindex $filters($num) $visibility]
	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
	set show [image create photo -file ${image_dir}/expand-icon-small.ppm]
	set hide [image create photo -file ${image_dir}/play-icon-small.ppm]
	if {$visible == 1} {
	    # hide

	    $history1.f$num.childsite.expand.b configure -image $show
	    $history2.f$num.childsite.expand.b configure -image $show
	    
	    pack forget $history1.f$num.childsite.ui 
	    pack forget $history2.f$num.childsite.ui 

	    set filters($num) [lreplace $filters($num) $visibility $visibility 0]
	} else {
	    # show

	    $history1.f$num.childsite.expand.b configure -image $hide
	    $history2.f$num.childsite.expand.b configure -image $hide

	    pack $history1.f$num.childsite.ui -side top -expand yes -fill both
	    pack $history2.f$num.childsite.ui -side top -expand yes -fill both

	    set filters($num) [lreplace $filters($num) $visibility $visibility 1]
	}
	
    }


    ##################################
    #### change_kernel
    ##################################
    # Update the resampling kernel variable and
    # update the other attached/detached optionmenu
    method change_kernel { w num} {
	set UnuResample [lindex [lindex $filters($num) $modules] 0]

        global [set UnuResample]-filtertype
	
	set which [$w get]

	if {$which == "Box"} {
	    set [set UnuResample]-filtertype box
	} elseif {$which == "Tent"} {
	    set [set UnuResample]-filtertype tent
	} elseif {$which == "Cubic (Catmull-Rom)"} {
	    set [set UnuResample]-filtertype cubicCR
	} elseif {$which == "Cubic (B-spline)"} {
	    set [set UnuResample]-filtertype cubicBS
	} elseif {$which == "Quartic"} {
	    set [set UnuResample]-filtertype quartic
	} elseif {$which == "Gaussian"} {
	    set [set UnuResample]-filtertype gaussian
	}

	# update attach/detach one
        $history1.f$num.childsite.ui.kernel select $which
	$history2.f$num.childsite.ui.kernel select $which

    }

    method update_kernel { num } {
	set UnuResample [lindex [lindex $filters($num) $modules] 0]
puts [set UnuResample]
        global [set UnuResample]-filtertype

        set f [set [set UnuResample]-filtertype]
        set t "Box"
  
        if {$f == "box"} {
	    set t "Box"
	} elseif {$f == "tent"} {
	    set t "Tent"
	} elseif {$f == "cubicCR"} {
	    set t "Cubic (Catmull-Rom)"
	} elseif {$f == "cubicBS"} {
	    set t "Cubic (B-spline)"
	} elseif {$f == "quartic"} {
	    set t "Quartic"
	} else {
	    set t "Gaussian"
	}

        $history1.f$num.childsite.ui.kernel select $t
        $history2.f$num.childsite.ui.kernel select $t
    }

    method make_entry {w text v num} {
        frame $w
        label $w.l -text "$text" 
        pack $w.l -side left
        global $v
        entry $w.e -textvariable $v 
        pack $w.e -side right

	bind $w.l <ButtonPress-1> "$this change_current $num"
	bind $w.e <ButtonPress-1> "$this change_current $num"
    }

    ##############################
    ### save_session
    ##############################
    # To be filled in by child class. It should save out a session
    # for the specific app.
    method save_session {} {

        tk_messageBox -message "Save not implemented yet." -type ok -icon info -parent .standalone
        return	

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
	    wm title .standalone "BioImage - [getFileName $saveFile]" 

	    # write out regular network
	    writeNetwork $saveFile

	    set fileid [open $saveFile {WRONLY APPEND}]
	    
	    # Save out data information 
	    puts $fileid "\n# BioImage Session\n"
	    puts $fileid "set app_version 1.0"

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
	    if {$var != "this" && $var != "filters"} {
		puts $fileid "set $var \{[set $var]\}"
	    }
	}
	
	# print out arrays
	for {set i 0} {$i < $num_filters} {incr i} {
	    if {[info exists filters($i)]} {
		puts $fileid "set filters($i) \{[set filters($i)]\}"
	    }
	}

        # save globals
        global eye
        puts $fileid "global eye"
        puts $fileid "set eye \{[set eye]\}"

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

    ###########################
    ### load_session
    ###########################
    # Load a saved session of BioTensor.  After sourcing the file,
    # reset some of the state (attached, indicate) and configure
    # the tabs and guis. This method also sets the loading to be 
    # true so that when executing, the progress labels don't get
    # all messed up.
    method load_session {} {

        tk_messageBox -message "Load not implemented yet." -type ok -icon info -parent .standalone
        return	

	set types {
	    {{App Settings} {.ses} }
	    {{Other} { * }}
	}
	
	if {$saveFile == "" } {
	    set saveFile [tk_getOpenFile -filetypes $types]
	}

	if {$saveFile != ""} {
	    # Clear all modules

	    ClearCanvas 0

	    #destroy 2D viewer windows
	    destroy $win.viewers.topbot
	    destroy $win.viewers.cp

	    puts "FIX ME: figure out how to wait until Network Changed"
#	    global NetworkChanged
#	    tkwait variable NetworkChanged
	    # This is a hack.  Unless I kill some time, the new modules
            # try to instantiate before the old ones are done clearing.
	    for {set i 0} {$i < 900000} {incr i} {
                set b 5
            }

	    # configure title
	    wm title .standalone "BioImage - [getFileName $saveFile]" 

	    # remove all UIs
	    for {set i 0} {$i < $num_filters} {incr i} {
		if {[info exists filters($i)]} {
		    set tmp_row [lindex $filters($i) $which_row]
		    if {$tmp_row != -1 } {
			destroy $history1.f$i
			destroy $history1.eye$i
			destroy $history2.f$i
			destroy $history2.eye$i
		    }
		}
	    }

	    # justify scroll region
	    $attachedPFr.f.p.sf justify top
	    $detachedPFr.f.p.sf justify top
	    
	    # load new net

  	    foreach g [info globals] {
  		global $g
  	    }

	    global mods

            update

            # source at the global level for module settings
	    set saveFile2 "$saveFile.net"
            uplevel \#0 source \{$saveFile2\}

            # source in class scope for class variables
	    source $saveFile

	    puts "FIX ME: module settings not being set properly"

	    $this build_viewers $mods(Viewer) $mods(ViewImage)

            set loading_ui 1
            set last_valid 0
		
	    # iterate over filters array and create UIs
	    for {set i 0} {$i < $num_filters} {incr i} {
		# only build ui for those with a row
		# value not -1
		set status [lindex $filters($i) $which_row]
                set p f$i
		if {$status != -1} {
		    set t [lindex $filters($i) $filter_type]
		    set last_valid $i
		    if {[string equal $t "load"]} {
			$this add_Load_UI $history1 $status $i
			$this add_Load_UI $history2 $status $i
		    } elseif {[string equal $t "resample"]} {
			$this add_Resample_UI $history1 $status $i
			$this add_Resample_UI $history2 $status $i
			$this update_kernel $i
		    } elseif {[string equal $t "crop"]} {
			$this add_Crop_UI $history1 $status $i
			$this add_Crop_UI $history2 $status $i
		    } elseif {[string equal $t "cmedian"]} {
			$this add_Cmedian_UI $history1 $status $i
			$this add_Cmedian_UI $history2 $status $i
		    } elseif {[string equal $t "histo"]} {
			$this add_Histo_UI $history1 $status $i
			$this add_Histo_UI $history2 $status $i
		    } else {
			puts "Error: Unknown filter type - $t"
		    }
		    $history1.$p configure -background grey75 -foreground black -borderwidth 2
		    $history2.$p configure -background grey75 -foreground black -borderwidth 2
		}
	    }

            set loading_ui 0

            puts "FIX ME: highlight last valid filter and change eye"
            $this change_current $current
            global eye
            $this change_eye $eye

 	    # set a few variables that need to be reset
 	    set indicate 0
 	    set cycle 0
 	    set IsPAttached 1
 	    set IsVAttached 1
 	    set executing_modules 0

 	    $indicatorL1 configure -text "Press Execute to run to save point..."
 	    $indicatorL2 configure -text "Press Execute to run to save point..."
	}	
    }

    #########################
    ### toggle_show_plane_n
    ##########################
    # Methods to turn on/off planes
    method toggle_show_plane_x {} {
	global mods show_plane_x
        puts "FIX ME: implement toggle_show_plane_x"
        tk_messageBox -message "Toggling planes not implemented yet" -type ok -icon info -parent .standalone
        set show_plane_x 0
        return
    }

    method toggle_show_plane_y {} {
	global mods show_plane_y
        puts "FIX ME: implement toggle_show_plane_y"
        tk_messageBox -message "Toggling planes not implemented yet" -type ok -icon info -parent .standalone
        set show_plane_y 0
        return
    }

    method toggle_show_plane_z {} {
	global mods show_plane_z
        puts "FIX ME: implement toggle_show_plane_z"
        tk_messageBox -message "Toggling planes not implemented yet" -type ok -icon info -parent .standalone
        set show_plane_z 0
        return
    }

    method toggle_show_guidelines {} {
         global mods show_guidelines
         puts "FIX ME: implement toggle_show_guidelines"
         global $mods(ViewImage)-show_guidelines
         set $mods(ViewImage)-show_guidelines $show_guidelines
  }

    method update_planes_color_by {} {
        global planes_mapType
        set GenStandard [lindex [lindex $filters(0) $modules] 26]
        global [set GenStandard]-mapType

        set [set GenStandard]-mapType $planes_mapType
        if {$has_executed == 1} {
	    [set GenStandard]-c needexecute
	}
    }

    method update_iso_color_by {} {
        global iso_mapType
        set GenStandard [lindex [lindex $filters(0) $modules] 28]
        global [set GenStandard]-mapType

        set [set GenStandard]-mapType $iso_mapType

        if {$has_executed == 1} {
	    [set GenStandard]-c needexecute
	}
    }

    method toggle_show_vol_ren {} {
	global mods show_vol_ren
        puts "FIX ME: implement toggle_show_vol_ren"
    }

    method toggle_show_iso {} {
	global mods show_iso mods
        puts "FIX ME: implement toggle_show_iso"

        set Isosurface [lindex [lindex $filters(0) $modules] 27]
        #set conn "[set Isosurface] 1 $mods(Viewer) 1"

        #disableConnection $conn
        if {$show_iso == 1} {
	    disableModule [set Isosurface] 0
            [set Isosurface]-c needexecute
	} else {
	    disableModule [set Isosurface] 1
	}
    }

    ################################
    ### update_iso_slider_min/max
    ################################
    # Method called when the isosurface min/max changes to 
    # reconfigure the slider
    method update_iso_slider_min {varname varele varop} {
        set Isosurface [lindex [lindex $filters(0) $modules] 27]
        global [set Isosurface]-isoval-min
        global [set Isosurface]-isoval-max

        set min [set [set Isosurface]-isoval-min]
        set max [set [set Isosurface]-isoval-max]
	
	$iso_slider1 configure -from $min \
	    -resolution [expr ($max - $min)/10000.]
	$iso_slider2 configure -from $min \
	    -resolution [expr ($max - $min)/10000.]
    }

    method update_iso_slider_max {varname varele varop} {
        set Isosurface [lindex [lindex $filters(0) $modules] 27]
        global [set Isosurface]-isoval-min
        global [set Isosurface]-isoval-max

        set min [set [set Isosurface]-isoval-min]
        set max [set [set Isosurface]-isoval-max]
	
	$iso_slider1 configure -to $max \
	    -resolution [expr ($max - $min)/10000.]
	$iso_slider2 configure -to $max \
	    -resolution [expr ($max - $min)/10000.]
    }


    # Application placing and size
    variable notebook_width
    variable notebook_height


    # Data Selection
    variable vis_frame_tab1
    variable vis_frame_tab2


    variable filters
    variable num_filters
    variable loading_ui

    variable history1
    variable history2

    variable dimension

    variable current
    variable scolor

    # filter indexes
    variable modules
    variable input
    variable output
    variable prev_index
    variable next_index
    variable choose_port
    # set to -1 when deleted
    variable which_row  
    variable visibility
    variable filter_type

    variable load_choose_input
    variable load_nrrd
    variable load_dicom
    variable load_analyze
    variable load_field
    variable load_choose_vis
    variable load_info

    variable grid_rows

    variable label_width

    variable 0_samples
    variable 1_samples
    variable 2_samples
    variable sizex
    variable sizey
    variable sizez

    variable has_autoviewed
    variable has_executed

    variable iso_slider1
    variable iso_slider2
    variable data_dir
    variable 2D_fixed
}


setProgressText "Building BioImage Window..."

BioImageApp app
app build_app $DATADIR

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

bind all <Control-n> {
    wm deiconify .
}


