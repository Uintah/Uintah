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
# Attempt to get environment variables:
set DATADIR [netedit getenv SCIRUN_DATA]
set DATASET [netedit getenv SCIRUN_DATASET]
#######################################################################

############# NET ##############
::netedit dontschedule
set bbox {0 0 3100 3100}

set m1 [addModuleAtPosition "SCIRun" "Render" "Viewer" 145 2629]

global mods
set mods(Viewer) $m1
setGlobal "$mods(Viewer)-ViewWindow_0-Slice0 (1)" 1
setGlobal "$mods(Viewer)-ViewWindow_0-Slice1 (1)" 1
setGlobal "$mods(Viewer)-ViewWindow_0-Slice2 (1)" 1
setGlobal "$mods(Viewer)-ViewWindow_0-MIP Slice0 (1)" 0
setGlobal "$mods(Viewer)-ViewWindow_0-MIP Slice1 (1)" 0
setGlobal "$mods(Viewer)-ViewWindow_0-MIP Slice2 (1)" 0



set mods(ViewSlices) ""
set mods(EditColorMap2D) ""

setGlobal new_label "Unknown"
setGlobal eye 0

# volume orientations
setGlobal top "S"
setGlobal front "A"
setGlobal side "L"

setGlobal show_guidelines 1
setGlobal planes_mapType 0

global slice_frame
set slice_frame(axial) ""
set slice_frame(coronal) ""
set slice_frame(sagittal) ""
set slice_frame(volume) ""
set slice_frame(axial_color) \#1A66FF ;#blue
set slice_frame(coronal_color) \#7FFF1A ;#green
set slice_frame(sagittal_color)  \#CC3366 ;#red

# volume rendering
setGlobal show_vol_ren 0
setGlobal link_winlevel 1
setGlobal vol_width 0
setGlobal vol_level 0

#global images
set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
set expandimg [image create photo -file ${image_dir}/expand-icon-small.ppm]
set play_img [image create photo -file ${image_dir}/play-icon-small.ppm]
set close_img [image create photo -file ${image_dir}/powerapp-close.ppm]
set insertimg [image create photo -file ${image_dir}/powerapp-insertbar.ppm]
set orientimg [image create photo -file ${image_dir}/OrientationsCube.ppm]

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
	wm withdraw .standalone
	wm title .standalone "BioImage"	 
	set win .standalone

	# Create insert menu
	menu .standalone.insertmenu -tearoff false -disabledforeground white

	# Set window sizes
	set i_width 260

	set viewer_width 436
	set viewer_height 620 
	
	set notebook_width 260
	set notebook_height [expr $viewer_height - 50]
	
	set process_width 300
	set process_height $viewer_height
	
	set vis_width [expr $notebook_width + 30]
	set vis_height $viewer_height

	set num_filters 0

	set loading_ui 0

	set vis_frame_tab1 ""
	set vis_frame_tab2 ""

	set history0 ""
	set history1 ""

	set dimension 3

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
	set filter_label 9


	set load_choose_input 5
	set load_nrrd 0
	set load_dicom 1
	set load_analyze 2
	set load_field 3
	set load_choose_vis 6
        set load_info 24

	set grid_rows 0

	set label_width 25

	set 0_samples 2
	set 1_samples 2
	set 2_samples 2

        set has_autoviewed 0
        set has_executed 0
        set data_dir ""
        set 2D_fixed 0
	set ViewSlices_executed_on_error 0
        set current_crop -1

        set needs_update 1

	set axial-size 0
	set sagittal-size 0
	set coronal-size 0


	set cur_data_tab "Nrrd"
	set c_vis_tab "Planes"

	set execute_choose 0

	set last_filter_changed 0

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

    ############################
    ### show_help
    ############################
    # Show the help menu
    method show_help {} {
	set splashImageFile [file join [netedit getenv SCIRUN_SRCDIR] Packages Teem Dataflow GUI splash-bioimage.ppm]

	showProgress 1 none 1

	global tutorial_link
	set tutorial_link "http://software.sci.utah.edu/doc/User/Tutorials/BioImage/bioimage.html"
	set help_font "-Adobe-Helvetica-normal-R-Normal-*-12-120-75-*"

	if {![winfo exists .splash.frame.m1]} {
	    label .splash.frame.m1 -text "Please refer to the online BioTensor Tutorial" \
		-font $help_font
	    
	    entry .splash.frame.m2 -relief flat -textvariable tutorial_link \
		-state disabled -width 55 -font $help_font
	    pack .splash.frame.m1 .splash.frame.m2 -before .splash.frame.ok -anchor n \
		-pady 2	    
	} else {
	    SciRaise .splash
	}
	update idletasks
    }

    
    method ok_box { args } {
	return [tk_messageBox -type ok -icon info \
		    -parent .standalone -message [join $args \n]]
    }

    method okcancel_box { args } {
	return [tk_messageBox -type okcancel -icon info \
		    -parent .standalone -message [join $args \n]]
    }
    

    ##########################
    ### show_about
    ##########################
    # Show about box
    method show_about {} {
	ok_box "BioImage is a SCIRun PowerApp for visualizing regular, three dimensional scalar volumes such as CT and MRI data. In addition to 2D and 3D visualization tools, BioImage provides a number of dynamic filters, such as re-sampling and cropping. Dynamic filters are used to emphasize the features of a dataset most important to the user. Dynamic filters can be applied at any time and in any order and can be easily undone.\n\nBioImage offers 2D visualization of three standard planes: axial, sagittal, and coronal. The 2D views allow the user to quickly investigate a volume slice by slice or via a maximum intensity projection. The 2D views provide zooming and translation capabilities. Width and level parameters can be used isolate a range of values for display.\n\nBioImage provides a powerful volume rendering tool that allows the user to interactively visualize an area of interest by assigning a color to a range of values in the volume. "
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
	    if {$executing_modules == 0} {
		if {$grid_rows == 1} {
		    global show_vol_ren
		    if {$show_vol_ren == 1} {
			change_indicator_labels "Done Volume Rendering"
		    } else {
			change_indicator_labels "Done Loading Volume"
		    }		
		} else {
		    change_indicator_labels "Done Updating Pipeline and Visualizing"
		}
	    } else {
		if {$grid_rows == 1} {
		    global show_vol_ren
		    if {$show_vol_ren == 1} {
			change_indicator_labels "Volume Rendering..."
		    } else {
			change_indicator_labels "Loading Volume..."
		    }
		} else {
		    change_indicator_labels "Updating Pipeline and Visualizing..."
		}
	    }
	}
    }
    
    ##########################
    ### indicate_error
    ##########################
    # This method should change the indicator and labels to
    # the error state.  This should be done using the change_indicate_val
    # and change_indicator_labels methods. We catch errors from
    method indicate_error { which msg_state } {

        # disregard UnuCrop errors, hopefully they are due to 
	# upstream crops changing bounds
	if {[string first "UnuCrop" $which] != -1 && \
		($msg_state == "Warning" || $msg_state == "Error")} {
	    if {![winfo exists .standalone.cropwarn]} {
		toplevel .standalone.cropwarn
		wm minsize .standalone.cropwarn 150 50
		wm title .standalone.cropwarn "Reset Crop Bounds"
  	        set pos_x [expr $screen_width / 2]
	        set pos_y [expr $screen_height / 2]
		wm geometry .standalone.cropwarn "+$pos_x+$pos_y"

		label .standalone.cropwarn.warn -text "W A R N I N G" \
		    -foreground "#830101"
		label .standalone.cropwarn.message \
		    -text "One or more of your cropping values was out of\nrange. This could be due to recent changes to\nan upstream crop filter's settings affecting a\ndownstream crop filter. The downstream crop\nvalues were reset to the new bounding box and the\ncrop widget was turned off." 
		    
		button .standalone.cropwarn.button -text " Ok " \
		    -command "wm withdraw .standalone.cropwarn" 
		pack .standalone.cropwarn.warn \
		    .standalone.cropwarn.message \
		    .standalone.cropwarn.button \
		    -side top -anchor n -pady 2 -padx 2
	    } else {
		SciRaise .standalone.cropwarn
	    }
	    after 700 "$this stop_crop"
	    return
	}         

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
		if {$grid_rows == 1} {
		    global show_vol_ren
		    if {$show_vol_ren == 1} {
			change_indicator_labels "Volume Rendering..."
		    } else {
			change_indicator_labels "Loading Volume..."
		    }
		} else {
		    change_indicator_labels "Updating Pipeline and Visualizing..."
		}
		change_indicate_val 0
		if {[string first "ViewSlices" $which] != -1} {
		    set ViewSlices_executed_on_error 0
		}
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
	set ChooseNrrd [lindex [lindex $filters(0) $modules] 5]
	set completed [string equal $state Completed]
	set juststarted [string equal $state JustStarted]

 	if {[string first $ChooseNrrd $which] != -1 && $completed} {
	    if {$execute_choose == 1} {
		set ChooseNrrd2 [lindex [lindex $filters(0) $modules] 35]
		set execute_choose 0
	    }
	} elseif { [string first "NrrdSetupTexture_0" $which] != -1 && \
		       $juststarted} {
	    change_indicator_labels "NrrdSetup Volume Rendering..."
	    change_indicate_val 1
	} elseif {[string first "NrrdSetupTexture" $which] != -1 && \
		      $completed} {
	    change_indicate_val 2
	} elseif {[string first "NrrdTextureBuilder" $which] != -1 && \
		      $juststarted} {
	    change_indicator_labels "Volume Rendering..."
	    change_indicate_val 1
	} elseif {[string first "NrrdTextureBuilder" $which] != -1 && \
		      $completed} {
	    change_indicate_val 2
	} elseif {[string first "VolumeVisualizer" $which] != -1 && \
		      $juststarted} {
	    change_indicator_labels "Volume Rendering..."
	    change_indicate_val 1
        } elseif {[string first "VolumeVisualizer" $which] != -1 && \
		      $completed} {
	    change_indicate_val 2
	    change_indicator_labels "Done Volume Rendering"
        } elseif {[string first "ViewSlices" $which] != -1 && \
		      $juststarted} {
	    change_indicate_val 1
	} elseif {[string first "ViewSlices" $which] != -1 && \
		      $completed} {
            if {$2D_fixed == 0} {
		global mods
		set ViewSlices $mods(ViewSlices)
		
		setGlobal $ViewSlices-sagittal-viewport0-axis 0
		setGlobal $ViewSlices-coronal-viewport0-axis 1
		setGlobal $ViewSlices-axial-viewport0-axis 2

		global $ViewSlices-clut_ww $ViewSlices-clut_wl
		set command "$this change_window_width_and_level 0"
		trace variable $ViewSlices-clut_ww w $command
		trace variable $ViewSlices-clut_wl w $command

		global vol_width vol_level
		set command "$this change_volume_window_width_and_level 0"
		trace variable vol_width w $command
		trace variable vol_level w $command

		global $ViewSlices-min $ViewSlices-max
		set command "$this update_window_level_scales"
		trace variable $ViewSlices-min w $command
		trace variable $ViewSlices-max w $command

		global $ViewSlices-background_threshold
		set command "$ViewSlices-c background_thresh"
		trace variable $ViewSlices-background_threshold w $command

		$this update_window_level_scales

		upvar \#0 $ViewSlices-min min $ViewSlices-max max
                set ww [expr abs($max-$min)]
                set wl [expr ($min+$max)/2.0]

		setGlobal $ViewSlices-clut_ww $ww
		setGlobal $ViewSlices-clut_wl $wl
		setGlobal vol_width $ww
		setGlobal vol_level $wl
                setGlobal $ViewSlices-background_threshold $min
				
		global slice_frame
		foreach axis {axial sagittal coronal} {
		    $ViewSlices-c rebind $slice_frame($axis).bd.$axis
		}

		$ViewSlices-c setclut

                set 2D_fixed 1
	    } 
	    change_indicate_val 2
	} elseif {[string first "Teem_NrrdData_NrrdInfo_1" $which] != -1 && \
		      $completed} {
	    set axis_num 0
	    global slice_frame
	    foreach axis "sagittal coronal axial" {
		# get Nrrd Dimensions from NrrdInfo Module
		upvar \#0 $which-size$axis_num nrrd_size
		if {![info exists nrrd_size]} return
		set size [expr $nrrd_size - 1]
		set sliderf $slice_frame($axis).modes.slider 

		$sliderf.slice.s configure -from 0 -to $size
		$sliderf.slab.s configure -from 0 -to $size

		set $axis-size $size

		upvar \#0 $mods(ViewSlices)-$axis-viewport0-slice slice
		upvar \#0 $mods(ViewSlices)-$axis-viewport0-slab_min slab_min
		upvar \#0 $mods(ViewSlices)-$axis-viewport0-slab_max slab_max

		if {!$loading} {
		    # set slice to be middle slice
		    set slice [expr $size/2]		;# 50%
		    set slab_min [expr $size/4]		;# 25%
		    set slab_max [expr $size*3/4]	;# 75%
		}
		incr axis_num
		$mods(ViewSlices)-c rebind $slice_frame($axis).bd.$axis
	    }
	    upvar \#0 $which-size$axis_num nrrd_size
	    set maxtime [expr $nrrd_size-1]
	    setGlobal $mods(NrrdSelectTime_2)-selectable_min 0
	    setGlobal $mods(NrrdSelectTime_2)-selectable_max $maxtime
	    setGlobal $mods(NrrdSelectTime_2)-range_max $maxtime

	    foreach page $mods(NrrdSelectTime_2_pages) {
		$mods(NrrdSelectTime_2) update_range $page
	    }

	    $mods(ViewSlices)-c redrawall
	} elseif {[string first "Teem_NrrdData_NrrdInfo_0" $which] != -1 && \
		      $juststarted} {
	    change_indicate_val 1
	    change_indicator_labels "Loading Volume..."
	} elseif { $which == "Teem_NrrdData_NrrdInfo_0" && $completed } {
	    change_indicate_val 2
	    set NrrdInfo $which
	    upvar \#0 $NrrdInfo-dimension dim
	    upvar \#0 $NrrdInfo-size0 size0
	    upvar \#0 $NrrdInfo-size1 size1
	    upvar \#0 $NrrdInfo-size2 size2
 	    if { [info exists size1] } {
		# configure samples info
 		if {$dim != 3 && $dim != 4} {
                    ok_box "BioImage only supports 3D data." \
			"Please load in a 3D dataset."
		    return;
		}
		if { $dim == 3 } {
		    setGlobal $mods(ChooseNrrd_6)-port-index 1
		} elseif { $dim == 4 } {
		    setGlobal $mods(ChooseNrrd_6)-port-index 0
		}
		disableModule $mods(ChooseNrrd_6) 0
		toggle_show_vol_ren

		$mods(ChooseNrrd_6)-c needexecute
		
		set text "Original Samples: ($size0, $size1, $size2)"
		$history0.0.f0.childsite.ui.samples configure -text $text
		$history1.0.f0.childsite.ui.samples configure -text $text
	    }	
        } elseif {[string first "UnuResample" $which 0] != -1 && \
		      $juststarted} {
	    change_indicate_val 1
	    change_indicator_labels "Resampling Volume..."
	} elseif {[string first "UnuResample" $which 0] != -1 && \
		      $completed} {
	    change_indicate_val 2
	    change_indicator_labels "Done Resampling Volume"
	} elseif {[string first "UnuCrop" $which 0] != -1 && \
		      $juststarted} {
	    change_indicate_val 1
	    change_indicator_labels "Cropping Volume..."
	} elseif {[string first "UnuCrop" $which 0] != -1 && \
		      $completed} {
	    change_indicate_val 2
	    change_indicator_labels "Done Cropping Volume"
	} elseif {[string first "UnuHeq" $which 0] != -1 && \
		      $which != "Teem_UnuAtoM_UnuHeq_0" && \
		      $juststarted} {
	    change_indicate_val 1
	    change_indicator_labels "Performing Histogram Equilization..."
	} elseif {[string first "UnuHeq" $which 0] != -1 && \
		      $which != "Teem_UnuAtoM_UnuHeq_0" && \
		      $completed} {
	    change_indicate_val 2
	    change_indicator_labels "Done Performing Histogram Equilization"
	} elseif {[string first "UnuCmedian" $which 0] != -1 && \
		      $juststarted} {
	    change_indicate_val 1
	    change_indicator_labels "Performing Median Filtering..."
	} elseif {[string first "UnuCmedian" $which 0] != -1 && \
		      $completed} {
	    change_indicate_val 2
	    change_indicator_labels "Done Performing Median Filtering"
	} elseif {[string first "ScalarFieldStats" $which] != -1 && \
		      $juststarted} {
	    change_indicate_val 1
	    change_indicator_labels "Building Histogram..."
	} elseif {[string first "ScalarFieldStats" $which] != -1 && \
		      $completed} {
	    change_indicate_val 2
	    change_indicator_labels "Done Building Histogram"
	} elseif {[string first "ChooseNrrd_6" $which] != -1 && \
		      $completed} {
	}

    }
    
    method change_indicate_val { v } {
#	puts "change_indicate_val $v: [info level [expr [info level]-1]]"
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
			if {$grid_rows == 1} {
			    global show_vol_ren
			    if {$show_vol_ren == 1} {
				change_indicator_labels "Volume Rendering..."
			    } else {
				change_indicator_labels "Loading Volume..."
			    }
			} else {
			    change_indicator_labels \
				"Updating Pipeline and Visualizing..."
			}
		    }


		    # something wasn't caught, reset
		    set executing_modules 0
		    set indicate 2
		    change_indicator

		    if {$loading} {
			set loading 0
			if {$grid_rows == 1} {
			    global show_vol_ren
			    if {$show_vol_ren == 1} {
				change_indicator_labels "Volume Rendering..."
			    } else {
				change_indicator_labels "Loading Volume..."
			    }			    
			} else {
			    change_indicator_labels \
				"Done Updating Pipeline and Visualizing"
			}		     
		    }

		}
	    }
	}
    }
    
    method change_indicator_labels { msg } {
	$indicatorL0 configure -text $msg
	$indicatorL1 configure -text $msg
    }



    ############################
    ### build_app
    ############################
    # Build the processing and visualization frames and pack along with viewer
    method build_app {d} {
	# make sure there is a slash on the end
	if {[string first "/" $d] != -1 && [string index $d end] != "/"} {
	    set d "$d/"
	} elseif {[string first "\\" $d] != -1 && \
		      [string index $d end] != "\\"} {
	    set d "$d\\"
	}

	set data_dir $d
	global mods
	incrProgress 5
	# Embed the Viewers

	# add a viewer and tabs to each
	frame $win.viewers
	incrProgress 5
	### Processing Part
	#########################
	### Create Detached Processing Part
	toplevel $win.detachedP
	frame $win.detachedP.f -relief flat
	pack $win.detachedP.f -side left -anchor n -fill both -expand 1
	
	wm title $win.detachedP "Processing Pane"
	
	wm sizefrom $win.detachedP user
	wm positionfrom $win.detachedP user
	incrProgress 5	
	wm withdraw $win.detachedP

	incrProgress 5
	### Create Attached Processing Part
	frame $win.attachedP 
	frame $win.attachedP.f -relief flat 
	pack $win.attachedP.f -side top -anchor n -fill both -expand 1

	set IsPAttached 1

	### set frame data members
	set detachedPFr $win.detachedP
	set attachedPFr $win.attachedP

	incrProgress 5
	init_Pframe $detachedPFr.f 0
	incrProgress 5
	init_Pframe $attachedPFr.f 1

	#change_current 0

	### create detached width and heigh
	append geomP $process_width x $process_height
	wm geometry $detachedPFr $geomP

	### Vis Part
	#####################
	### Create a Detached Vis Part
	toplevel $win.detachedV
	frame $win.detachedV.f -relief flat
	pack $win.detachedV.f -side left -anchor n -fill both -expand 1

	wm title $win.detachedV "Visualization Settings Pane"

	wm sizefrom $win.detachedV user
	wm positionfrom $win.detachedV user
	
	wm withdraw $win.detachedV

	incrProgress 10
	### Create Attached Vis Part
	frame $win.attachedV
	frame $win.attachedV.f -relief flat
	pack $win.attachedV.f -side left -anchor n -fill both

	set IsVAttached 1

	incrProgress 10
	### set frame data members
	set detachedVFr $win.detachedV
	set attachedVFr $win.attachedV
	
	init_Vframe $detachedVFr.f 1
	init_Vframe $attachedVFr.f 2

	incrProgress 5
	### pack 3 frames in proper order so that viewer
	# is the last to be packed and will be the one to resize
	pack $attachedVFr -side right -anchor n -fill y

 	incrProgress 5
	pack $attachedPFr -side left -anchor n -fill y

	incrProgress 5

	pack $win.viewers -side left -anchor n -fill both -expand 1

	set total_width [expr $process_width + $viewer_width + $vis_width]

	set total_height $viewer_height

	set pos_x [expr [expr $screen_width / 2] - [expr $total_width / 2]]
	set pos_y [expr [expr $screen_height / 2] - [expr $total_height / 2]]

	append geom $total_width x $total_height + $pos_x + $pos_y
	wm geometry .standalone $geom
	incrProgress 5
	update	

        set initialized 1
	global PowerAppSession
	if {[info exists PowerAppSession] && $PowerAppSession != ""} { 
	    set saveFile $PowerAppSession
	    wm title .standalone "BioImage - [getFileName $saveFile]"
	    $this load_session_data
	} 
	incrProgress 10
	wm deiconify .standalone
    }

    method build_viewers {viewer ViewSlices} {
	set w $win.viewers
	
	global mods

	iwidgets::panedwindow $w.topbot -orient horizontal -thickness 0 \
	    -sashwidth 5000 -sashindent 0 -sashborderwidth 2 -sashheight 6 \
	    -sashcursor sb_v_double_arrow \
	    -width $viewer_width -height $viewer_height
	pack $w.topbot -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
#	Tooltip $w.topbot "Click and drag to resize"
	
	$w.topbot add top -margin 3 -minimum 0
	$w.topbot add bottom  -margin 0 -minimum 0

	set bot [$w.topbot childsite top]
	set top [$w.topbot childsite bottom]

	$w.topbot fraction 62 38
	iwidgets::panedwindow $top.lmr -orient vertical -thickness 0 \
	    -sashheight 5000 -sashwidth 6 -sashindent 0 -sashborderwidth 2 \
	    -sashcursor sb_h_double_arrow
#	Tooltip $top.lmr "Click and drag to resize"

	$top.lmr add left -margin 3 -minimum 0
	$top.lmr add middle -margin 3 -minimum 0
	$top.lmr add right -margin 3 -minimum 0
	set topl [$top.lmr childsite left]
	set topm [$top.lmr childsite middle]
	set topr [$top.lmr childsite right]

	pack $top.lmr -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0

	$ViewSlices control_panel $w.cp
	$ViewSlices add_nrrd_tab $w 1
	global slice_frame
	set slice_frame(3d) $bot
	set slice_frame(sagittal) $topl
	set slice_frame(coronal) $topm
	set slice_frame(axial) $topr

	foreach axis "sagittal coronal axial" {
	    global $mods(ViewSlices)-$axis-viewport0-mode
	    create_2d_frame $slice_frame($axis) $axis
	    frame $slice_frame($axis).bd -bd 1 \
		-background $slice_frame(${axis}_color)
	    pack $slice_frame($axis).bd -expand 1 -fill both \
		-side top -padx 0 -ipadx 0 -pady 0 -ipady 0
	    $ViewSlices gl_frame $slice_frame($axis).bd.$axis
	    bind $slice_frame($axis).bd.$axis <Shift-ButtonRelease-1> \
		"$ViewSlices-c release %W %b %s %X %Y"
	    bind $slice_frame($axis).bd.$axis <ButtonRelease-1> \
		"$ViewSlices-c release %W %b %s %X %Y
                 $this change_window_width_and_level 1"

	    pack $slice_frame($axis).bd.$axis -expand 1 -fill both \
		-side top -padx 0 -ipadx 0 -pady 0 -ipady 0
	}

	setGlobal $mods(ViewSlices)-sagittal-viewport0-axis 0
	setGlobal $mods(ViewSlices)-coronal-viewport0-axis 1
	setGlobal $mods(ViewSlices)-axial-viewport0-axis 2
	


	# embed viewer in top left
	global mods
 	set eviewer [$mods(Viewer) ui_embedded]

 	$eviewer setWindow $slice_frame(3d) [expr $viewer_width/2] \
 	    [expr $viewer_height/2] \

 	pack $slice_frame(3d) -side top -anchor n \
 	    -expand 1 -fill both -padx 4 -pady 0

    }

    method create_2d_frame { window axis } {
	# Modes for $axis
	frame $window.modes
	pack $window.modes -side bottom -padx 0 -pady 0 -expand 0 -fill x
	
	frame $window.modes.buttons
	frame $window.modes.slider
	pack $window.modes.buttons $window.modes.slider \
	    -side top -pady 0 -anchor n -expand yes -fill x

	global mods slice_frame
	
	# Radiobuttons
	radiobutton $window.modes.buttons.slice -text "Slice" \
	    -variable $mods(ViewSlices)-$axis-viewport0-mode -value 0 \
	    -command "$this update_ViewSlices_mode $axis"
	Tooltip $window.modes.buttons.slice "Select to view in\nsingle slice mode.\nAdjust slider to\nchange current\nslice"
	radiobutton $window.modes.buttons.slab -text "Slab" \
	    -variable $mods(ViewSlices)-$axis-viewport0-mode -value 1 \
	    -command "$this update_ViewSlices_mode $axis"
	Tooltip $window.modes.buttons.slab "Select to view a\nmaximum intensity\nprojection of a slab\nof slices"
	radiobutton $window.modes.buttons.mip -text "MIP" \
	    -variable $mods(ViewSlices)-$axis-viewport0-mode -value 2 \
	    -command "$this update_ViewSlices_mode $axis"
	Tooltip $window.modes.buttons.mip "Select to view a\nmaximum intensity\nprojection of all\nslices"
	pack $window.modes.buttons.slice $window.modes.buttons.slab \
	    $window.modes.buttons.mip -side left -anchor n -padx 2 \
	    -expand yes -fill x
	
	# Initialize with slice scale visible
	frame $window.modes.slider.slice
	pack $window.modes.slider.slice -side top -anchor n -expand 1 -fill x

	# slice slider
	scale $window.modes.slider.slice.s \
	    -variable $mods(ViewSlices)-$axis-viewport0-slice \
	    -from 0 -to 20 -width 15 \
	    -showvalue false \
	    -orient horizontal \
	    -command "$mods(ViewSlices)-c rebind $slice_frame($axis).bd.$axis; \
                      $mods(ViewSlices)-c redrawall"

	# slice value label
	entry $window.modes.slider.slice.l \
	    -textvariable $mods(ViewSlices)-$axis-viewport0-slice \
	    -justify left -width 3
	bind $window.modes.slider.slice.l <Return>  "$mods(ViewSlices)-c rebind $slice_frame($axis).bd.$axis; $mods(ViewSlices)-c redrawall"

	
	pack $window.modes.slider.slice.l -anchor e -side right \
	    -padx 0 -pady 0 -expand 0

	pack $window.modes.slider.slice.s -anchor n -side left \
	    -padx 0 -pady 0 -expand 1 -fill x

	
	# Create range widget for slab mode
	frame $window.modes.slider.slab
	# min range value label
	entry $window.modes.slider.slab.min \
	    -textvariable $mods(ViewSlices)-$axis-viewport0-slab_min \
	    -justify right -width 3 
	bind $window.modes.slider.slab.min <Return> "$mods(ViewSlices)-c rebind $slice_frame($axis).bd.$axis; $mods(ViewSlices)-c redrawall" 
	# MIP slab range widget
	range $window.modes.slider.slab.s -from 0 -to 20 \
	    -orient horizontal -showvalue false \
	    -rangecolor "#830101" -width 15 \
	    -varmin $mods(ViewSlices)-$axis-viewport0-slab_min \
	    -varmax $mods(ViewSlices)-$axis-viewport0-slab_max \
	    -command "$mods(ViewSlices)-c rebind $slice_frame($axis).bd.$axis; \
                      $mods(ViewSlices)-c redrawall"
	Tooltip $window.modes.slider.slab.s "Click and drag the\nmin or max sliders\nto change the extent\nof the slab. Click\nand drage the red\nrange bar to change the\ncenter poisition of\nthe slab range"
	# max range value label
	entry $window.modes.slider.slab.max \
	    -textvariable $mods(ViewSlices)-$axis-viewport0-slab_max \
	    -justify left -width 3
	bind $window.modes.slider.slab.max <Return> "$mods(ViewSlices)-c rebind $slice_frame($axis).bd.$axis; $mods(ViewSlices)-c redrawall" 
	
	pack $window.modes.slider.slab.min -anchor w -side left \
	    -padx 0 -pady 0 -expand 0 

	pack $window.modes.slider.slab.max -anchor e -side right \
	    -padx 0 -pady 0 -expand 0 

	pack $window.modes.slider.slab.s \
	    -side left -anchor n -padx 0 -pady 0 -expand 1 -fill x

	# show/hide bar
	set img [image create photo -width 1 -height 1]
	button $window.modes.expand -height 4 -bd 2 \
	    -relief raised -image $img \
	    -cursor based_arrow_down \
	    -command "$this hide_control_panel $window.modes"
	Tooltip $window.modes.expand "Click to minimize/show the\nviewing mode controls"
	pack $window.modes.expand -side bottom -fill both
    }
    

    method show_control_panel { w } {
	pack forget $w.expand
	pack $w.buttons $w.slider -side top -pady 0 -anchor nw -expand yes -fill x
	pack $w.expand -side bottom -fill both

	$w.expand configure -command "$this hide_control_panel $w" \
	    -cursor based_arrow_down
    }

    method hide_control_panel { w } {
	pack forget $w.buttons $w.slider
	pack $w.expand -side bottom -fill both

	$w.expand configure -command "$this show_control_panel $w" \
	    -cursor based_arrow_up
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
	    pack $m.p.filters -side top -expand no -anchor n -pady 1

	    set filter $m.p.filters
	    button $filter.resamp -text "Resample" \
		-background $scolor -padx 3 \
		-activebackground "#6c90ce" \
		-command "$this add_Resample -1"
	    Tooltip $filter.resamp "Resample using UnuResample"

	    button $filter.crop -text "Crop" \
		-background $scolor -padx 3 \
		-activebackground "#6c90ce" \
		-command "$this add_Crop -1"
	    Tooltip $filter.crop "Crop the image"

	    button $filter.cmedian -text "Median Filtering" \
		-background $scolor -padx 3 \
		-activebackground "#6c90ce" \
		-command "$this add_Cmedian -1"
	    Tooltip $filter.cmedian "Perform median filtering"

	    button $filter.histo -text "Histogram" \
		-background $scolor -padx 3 \
		-activebackground "#6c90ce" \
		-command "$this add_Histo -1"
	    Tooltip $filter.histo \
		"Perform Histogram Equilization\nusing UnuHeq"

	    pack $filter.resamp $filter.crop $filter.histo $filter.cmedian \
		-side left -padx 2 -expand no

	    iwidgets::scrolledframe $m.p.sf -width [expr $process_width - 20] \
		-height [expr $process_height - 180] -labeltext "History"
	    pack $m.p.sf -side top -anchor nw -expand yes -fill both

	    set history [$m.p.sf childsite]

	    Tooltip $history \
		"Shows a history of steps\nin the dynamic pipeline"

	    # Add Load UI
	    $this add_Load $history $case
	    
	    set grid_rows 1
	    set num_filters 1	 	 

	    set history$case $history

	    button $m.p.update -text "U p d a t e" \
		-command "$this update_changes" \
		-background "#008b45" \
		-activebackground "#31a065"
	    Tooltip $m.p.update "Update any filter changes, also\nupdating the currenlty viewed data."

	    pack $m.p.update -side top -anchor s -padx 3 -pady 3 -ipadx 3 -ipady 2

	    
            ### Indicator
	    frame $m.p.indicator -relief sunken -borderwidth 2
            pack $m.p.indicator -side bottom -anchor s -padx 3 -pady 5
	    
	    canvas $m.p.indicator.canvas -bg "white" -width $i_width \
	        -height $i_height 
	    pack $m.p.indicator.canvas -side top -anchor n -padx 3 -pady 3
	    
            bind $m.p.indicator <Button> {app display_module_error} 
	    
            label $m.p.indicatorL -text "Press Update to Load Volume..."
            pack $m.p.indicatorL -side bottom -anchor sw -padx 5 -pady 3
	    
	    set indicator$case $m.p.indicator.canvas
	    set indicatorL$case $m.p.indicatorL

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


    method create_network {} {
	global mods
	set m1 [addModuleAtPosition "Teem" "DataIO" "NrrdReader" 10 10]
	set m2 [addModuleAtPosition "Teem" "DataIO" "DicomNrrdReader" 28 68]
	set m3 [addModuleAtPosition "Teem" "DataIO" "AnalyzeNrrdReader" 46 128]
	set m4 [addModuleAtPosition "SCIRun" "DataIO" "FieldReader" 91 184]
	set m5 [addModuleAtPosition "Teem" "Converters" "FieldToNrrd" 91 242]
	set m6 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 322]
	set m7 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuPermute" 28 402]
	set m8 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 482]
	set m9 [addModuleAtPosition "Teem" "UnuAtoM" "UnuFlip" 28 562]
	set m10 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 642]
	set m11 [addModuleAtPosition "Teem" "UnuAtoM" "UnuFlip" 28 722]
	set m12 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 802]
	set m13 [addModuleAtPosition "Teem" "UnuAtoM" "UnuFlip" 28 882]
	set m14 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 962]
	set m15 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 28 1042]
	set m16 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 174 1637]
	set m17 [addModuleAtPosition "Teem" "NrrdData" "NrrdInfo" 459 1897]
	set m18 [addModuleAtPosition "Teem" "NrrdData" "NrrdSetupTexture" 174 1895]
	set m19 [addModuleAtPosition "Teem" "UnuAtoM" "UnuJhisto" 410 1996]
	set m20 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuQuantize" 141 1985]
	set m21 [addModuleAtPosition "Teem" "UnuAtoM" "Unu2op" 392 2068]
	set m22 [addModuleAtPosition "Teem" "UnuAtoM" "Unu1op" 392 2129]
	set m23 [addModuleAtPosition "Teem" "UnuAtoM" "UnuHeq" 392 2191]
	set m24 [addModuleAtPosition "Teem" "UnuAtoM" "UnuGamma" 392 2253]
	set m25 [addModuleAtPosition "Teem" "UnuNtoZ" "UnuQuantize" 392 2315]
	set m26 [addModuleAtPosition "SCIRun" "Visualization" "NrrdTextureBuilder" 123 2228]
	set m27 [addModuleAtPosition "SCIRun" "Visualization" "GenStandardColorMaps" 141 2290]
	set m28 [addModuleAtPosition "SCIRun" "Render" "ViewSlices" 244 2548]
	set m29 [addModuleAtPosition "SCIRun" "Visualization" "EditColorMap2D" 374 2385]
	set m30 [addModuleAtPosition "SCIRun" "Visualization" "RescaleColorMap" 141 2357]
	set m31 [addModuleAtPosition "SCIRun" "Visualization" "VolumeVisualizer" 123 2457]
	set m32 [addModuleAtPosition "Teem" "NrrdData" "NrrdSelectTime" 141 2155]
	set m33 [addModuleAtPosition "Teem" "NrrdData" "NrrdSelectTime" 123 2074]
	set m34 [addModuleAtPosition "Teem" "NrrdData" "NrrdSelectTime" 0 1895]
	set m35 [addModuleAtPosition "Teem" "NrrdData" "ChooseNrrd" 10 1222]
	set m36 [addModuleAtPosition "Teem" "UnuAtoM" "UnuAxinsert" 28 1142]

	# Create the Connections between Modules
	set c1 [addConnection $m4 0 $m5 0]
	set c2 [addConnection $m28 1 $m29 0]

	set c3 [addConnection $m27 0 $m30 0]
	set c4 [addConnection $m26 0 $m31 0]
	set c6 [addConnection $m1 0 $m6 0]
	set c7 [addConnection $m6 0 $m8 0]
	set c8 [addConnection $m6 0 $m7 0]
	set c9 [addConnection $m8 0 $m10 0]
	set c10 [addConnection $m8 0 $m9 0]
	set c11 [addConnection $m10 0 $m12 0]
	set c12 [addConnection $m10 0 $m11 0]
	set c13 [addConnection $m12 0 $m14 0]
	set c14 [addConnection $m12 0 $m13 0]
	set c16 [addConnection $m14 0 $m15 0]
	set c17 [addConnection $m16 0 $m17 0]
	set c15 [addConnection $m35 0 $m16 0]
	set c18 [addConnection $m16 0 $m34 0]
	set c19 [addConnection $m16 0 $m18 0]
	set c20 [addConnection $m33 0 $m26 0]
	set c21 [addConnection $m34 0 $m28 0]
	set c22 [addConnection $m18 0 $m33 0]

	set c23 [addConnection $m18 1 $m20 0]
	set c24 [addConnection $m22 0 $m23 0]
	set c25 [addConnection $m21 0 $m22 0]
	set c26 [addConnection $m24 0 $m25 0]
	set c27 [addConnection $m23 0 $m24 0]
	set c28 [addConnection $m20 0 $m32 0]
	set c30 [addConnection $m30 0 $m31 1]
	set c31 [addConnection $m2 0 $m6 1]
	set c32 [addConnection $m16 0 $m19 1]
	set c33 [addConnection $m32 0 $m26 1]
	set c34 [addConnection $m33 1 $m32 1]
	set c35 [addConnection $m34 1 $m33 1]
	set c36 [addConnection $m9 0 $m10 1]
	set c37 [addConnection $m11 0 $m12 1]
	set c38 [addConnection $m13 0 $m14 1]
	set c39 [addConnection $m19 0 $m21 1]
	set c40 [addConnection $m7 0 $m8 1]
	set c41 [addConnection $m25 0 $m29 1]
	set c42 [addConnection $m29 0 $m31 2]
	set c43 [addConnection $m27 0 $m28 2]
	set c44 [addConnection $m3 0 $m6 2]
	set c45 [addConnection $m18 1 $m19 2]
	set c46 [addConnection $m5 2 $m6 3]
	set c47 [addConnection $m29 0 $m28 4]
	set c48 [addConnection $m33 0 $m28 5]
	set c49 [addConnection $m31 0 $mods(Viewer) 0]
	set c50 [addConnection $m28 0 $mods(Viewer) 1]
	set c51 [addConnection $m14 0 $m35 0]
	set c52 [addConnection $m14 0 $m36 0]
	set c53 [addConnection $m36 0 $m35 1]

	global ConnectionRoutes
	set ConnectionRoutes($c2) {274 2605 274 2614 579 2614 579 2378 386 2378 386 2385}
#	set ConnectionRoutes($c15) {22 963 22 1377 271 1277 271 31 468 31 468 1535 186 1535 186 1797}
	set ConnectionRoutes($c17) {186 1854 186 1872.0 471 1872.0 471 1897}
	set ConnectionRoutes($c18) {186 1854 186 1869.0 12 1869.0 12 1895}
	set ConnectionRoutes($c21) {12 1952 12 2533.0 256 2533.0 256 2548}
	set ConnectionRoutes($c22) {186 1952 186 1962.0 135 1962.0 135 2074}
	set ConnectionRoutes($c23) {204 1952 204 1972.0 153 1972.0 153 1985}
	set ConnectionRoutes($c28) {153 2042 153 2053.0 294 2053.0 294 2149 153 2149 153 2155}
	set ConnectionRoutes($c32) {186 1854 186 1887.0 440 1887.0 440 1996}
	set ConnectionRoutes($c34) {153 2131 153 2140.0 171 2140.0 171 2155}
	set ConnectionRoutes($c35) {30 1952 30 2064.0 153 2064.0 153 2074}
	set ConnectionRoutes($c42) {386 2442 386 2450.0 171 2450.0 171 2457}
	set ConnectionRoutes($c43) {153 2347 153 2423.0 292 2423.0 292 2548}
	set ConnectionRoutes($c45) {204 1952 204 1973.0 458 1973.0 458 1996}
	set ConnectionRoutes($c48) {135 2131 135 2140.0 346 2140.0 346 2548}

	set Notes($m18) {Gradient           }
	set Notes($m18-Position) {s}
	set Notes($m18-Color) {\#00ffff}

	set Notes($m20) {Quantized Gradient}
	set Notes($m20-Position) {none}
	set Notes($m20-Color) {\#00ffff}


	# set some ui parameters
	set filename [netedit getenv BIOIMAGE_FILENAME]
	if { [string length $filename] } {
	    setGlobal $m1-filename $filename
	} else {
	    setGlobal $m1-filename ${data_dir}volume/tooth.nhdr
	}

	setGlobal $m20-nbits {8}
	setGlobal $m20-useinputmin 1
	setGlobal $m20-useinputmax 1

	setGlobal $m18-valuesonly {0}
	setGlobal $m18-useinputmin {0}
	setGlobal $m18-useinputmax {0}

	# CHANGE THESE VARS FOR TRANSFER FUNCTION 
	setGlobal $m29-panx {0.0}
	setGlobal $m29-pany {0.0}
	setGlobal $m29-scale_factor {1.0}
	setGlobal $m29-faux {1}
	setGlobal $m29-histo {0.5}
	setGlobal $m29-name-0 {Triangle}
	setGlobal $m29-0-color-r {0.12221829371}
	setGlobal $m29-0-color-g {0.773248783139}
	setGlobal $m29-0-color-b {0.741646733309}
	setGlobal $m29-0-color-a {0.800000011921}
	setGlobal $m29-state-0 {t 0.670178 0.0621057 0.540499 \
				    0.495436 0.464177}
	setGlobal $m29-shadeType-0 {0}
	setGlobal $m29-on-0 {1}
	setGlobal $m29-name-1 {Rectangle}
	setGlobal $m29-1-color-r {0.0157082642279}
	setGlobal $m29-1-color-g {0.602349504633}
	setGlobal $m29-1-color-b {0.310323060825}
	setGlobal $m29-1-color-a {0.800000011921}
	setGlobal $m29-state-1 {r 0 0.222522 0.0544884 0.212415 \
				    0.318622 0.612325}
	setGlobal $m29-shadeType-1 {0}
	setGlobal $m29-on-1 {1}
	setGlobal $m29-marker {end}

	setGlobal $m31-alpha_scale {0.0}
	setGlobal $m31-shading {1}
	setGlobal $m31-ambient {0.5}
	setGlobal $m31-diffuse {0.5}
	setGlobal $m31-specular {0.388}
	setGlobal $m31-shine {24}
	setGlobal $m31-adaptive {1}
	global $m31-shading-button-state
	trace variable $m31-shading-button-state w \
	    "$this update_BioImage_shading_button_state"

	setGlobal $m23-bins {3000}
	setGlobal $m23-sbins {1}

	setGlobal $m24-gamma {0.5}

	setGlobal $m25-nbits {8}
	setGlobal $m25-useinputmin 1
	setGlobal $m25-useinputmax 1

	setGlobal $m19-bins {512 256}
	setGlobal $m19-mins {nan nan}
	setGlobal $m19-maxs {nan nan}
	setGlobal $m19-type {nrrdTypeFloat}

	setGlobal $m21-operator {+}

	setGlobal $m22-operator {log}

	set axes "minAxis0 minAxis1 minAxis2 maxAxis0 maxAxis1 maxAxis2"
	foreach axis $axes {
	    global $m28-crop_$axis
	    trace variable $m28-crop_$axis w "$this viewslices_crop_trace"
	}
	global $m28-geom_flushed
	trace variable $m28-geom_flushed w "$this maybe_autoview"

	global planes_mapType
	setGlobal $m27-mapType $planes_mapType
	setGlobal $m27-width 441
	setGlobal $m27-height 40
	setGlobal $m27-positionList {{0 0} {441 0}}
	setGlobal $m27-nodeList {514 1055}

	setGlobal $m9-axis 0

	setGlobal $m11-axis 1

	setGlobal $m13-axis 2

	setGlobal $m30-isFixed 1
	setGlobal $m30-min 0
	setGlobal $m30-max 0

	setGlobal $m34-playmode loop
	setGlobal $m36-axis M

	# disable other load modules
	disableModule $m2 1
	disableModule $m3 1
	disableModule $m4 1



	# disable the volume rendering
#	disableModule $m31 1
#	disableModule $m18 1
#	disableModule $m19 1
#	disableModule $m29 1
#	disableModule $m26 1
#	disableModule $m20 1
#	disableModule $m32 1
#	disableModule $m33 1

	# disable flip/permute modules
	disableModule $m7 1
	disableModule $m9 1
	disableModule $m11 1
	disableModule $m13 1

	set mods(ViewSlices) $m28
	set mods(EditColorMap2D) $m29
	set mods(NrrdSelectTime_0) $m32
	set mods(NrrdSelectTime_1) $m33
	set mods(NrrdSelectTime_2) $m34
	set mods(ChooseNrrd_6) $m35

	disableModule $m35 1


	set mod_list [list $m1 $m2 $m3 $m4 $m5 $m6 $m16 0 $m20 0 $m18 \
			  $m26 0 $m29 $m31 0 0 0 $m23 $m24 $m25 $m19 \
			  $m21 $m22 $m15 $m28 $m27 $m17 $m7 $m9 $m11 \
			  $m13 $m10 $m12 $m14 $m8 $m30]

	set filters(0) [list load $mod_list [list $m6] [list $m35 0] \
			    start end 0 0 1 "Data - Unknown"]

	toggle_show_vol_ren
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
	    create_network
	    global mods
	    $this build_viewers $mods(Viewer) $mods(ViewSlices)
	}
	set f [add_Load_UI $history 0 0]
	$this add_insert_bar $f 0
    }
    
    method add_Load_UI {history row which} {
	global mods

	frame $history.$which
	grid config $history.$which -column 0 -row $row -pady 0 -sticky news

	### Load Data UI
	set ChooseNrrd [lindex [lindex $filters($which) $modules] $load_choose_vis] 
 	iwidgets::labeledframe $history.$which.f$which \
 	    -labeltext "Load Data" \
 	    -labelpos nw 

 	grid config $history.$which.f$which -column 0 -row 0 -sticky news

 	set data [$history.$which.f$which childsite]
	
 	frame $data.expand 
 	pack $data.expand -side top -anchor nw

	global expandimg
 	button $data.expand.b -image $expandimg -anchor nw -relief flat \
 	    -command "$this change_visibility $which"
 	    
	Tooltip $data.expand.b "Click to minimize/show\nthe Load UI"
 	label $data.expand.l -text "Data - Unknown" -anchor nw \
	    -width [expr $label_width+2]
 	    
	Tooltip $data.expand.l "Right click to edit label."

 	pack $data.expand.b $data.expand.l -side left -anchor nw 
	
 	bind $data.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
 	frame $data.ui
 	pack $data.ui -side top -anchor nw -expand yes -fill x

 	label $data.ui.samples -text "Original Samples: unknown"
 	pack $data.ui.samples -side top -anchor nw -pady 3

	# Build data tabs
	iwidgets::tabnotebook $data.ui.tnb -width [expr $process_width-86] \
	    -height 75 -tabpos n -equaltabs false -backdrop gray
	pack $data.ui.tnb -side top -anchor nw -padx 0 -pady 3
	Tooltip $data.ui.tnb \
	    "Load 3D volume in Nrrd,\nDicom, Analyze, or Field format."
	
	# Make pointers to modules 
	set NrrdReader  [lindex [lindex $filters($which) $modules] $load_nrrd]
	set DicomNrrdReader \
	    [lindex [lindex $filters($which) $modules] $load_dicom]
	set AnalzyeNrrdReader \
	    [lindex [lindex $filters($which) $modules] $load_analyze]
	set FieldReader \
	    [lindex [lindex $filters($which) $modules] $load_field]

	# Nrrd
	set page [$data.ui.tnb add -label "Generic" \
		      -command "$this set_cur_data_tab Nrrd
                                $this configure_readers Nrrd"]       

	global $NrrdReader-filename
	frame $page.file
	pack $page.file -side top -anchor nw -padx 3 -pady 0 -fill x

	label $page.file.l -text ".vol/.vff/.nrrd file:" 
	entry $page.file.e -textvariable $NrrdReader-filename 
	Tooltip $page.file.e "Currently loaded data set"
	pack $page.file.l $page.file.e -side left -padx 3 -pady 0 -anchor nw \
	    -fill x 

	bind $page.file.e <Return> "$this update_changes"

	trace variable $NrrdReader-filename w "$this enable_update $which"
	
	button $page.load -text "Browse" \
	    -command "$this open_nrrd_reader_ui $which" \
	    -width 12
	Tooltip $page.load "Use a file browser to\nselect a Nrrd data set"
	pack $page.load -side top -anchor n -padx 3 -pady 1
	
	
	### Dicom
	set page [$data.ui.tnb add -label "Dicom" \
		      -command "$this set_cur_data_tab Dicom
                                $this configure_readers Dicom"]
	
	button $page.load -text "Dicom Loader" \
	    -command "$this enable_update $which
                      $this dicom_ui"
	Tooltip $page.load "Open Dicom Load user interface"
	
	pack $page.load -side top -anchor n \
	    -padx 3 -pady 10 -ipadx 2 -ipady 2
	
	### Analyze
	set page [$data.ui.tnb add -label "Analyze" \
		      -command "$this set_cur_data_tab Analyze
                                $this configure_readers Analyze"]
	
	button $page.load -text "Analyze Loader" \
	    -command "$this enable_update $which
                      $this analyze_ui"
	Tooltip $page.load "Open Dicom Load user interface"
	
	pack $page.load -side top -anchor n \
	    -padx 3 -pady 10 -ipadx 2 -ipady 2
	
	### Field
	set page [$data.ui.tnb add -label "Field" \
		      -command "$this configure_readers Field"]
	
	global $FieldReader-filename
	frame $page.file
	pack $page.file -side top -anchor nw -padx 3 -pady 0 -fill x

	label $page.file.l -text "Field File:" 
	entry $page.file.e -textvariable $FieldReader-filename 
	pack $page.file.l $page.file.e -side left -padx 3 -pady 0 -anchor nw \
	    -fill x 

	bind $page.file.e <Return> "$this update_changes"

	button $page.load -text "Browse" \
	    -command "$this open_field_reader_ui $which" \
	    -width 12
	Tooltip $page.load "Use a file browser to\nselect a Nrrd data set"

        trace variable $FieldReader-filename w "$this enable_update $which"
	pack $page.load -side top -anchor n -padx 3 -pady 1
	
	# Set default view to be Nrrd
	$data.ui.tnb view "Generic"

	frame $data.ui.f
	pack $data.ui.f
	
	set w $data.ui.f

	set orient_text "Options include Superior (S) or Inferior (I),\nAnterior (A) or Posterior (P), and Left (L) or Right (R).\nTo update the orientations, press the cube image."

	# Orientations button
	global orientimg
	button $w.orient -image $orientimg -anchor nw \
	    -command "$this update_orientations"
	TooltipMultiline $w.orient \
	    "Edit the entries to indicate the various orientations.\n" \
	    $orient_text
	
	grid config $w.orient -row 0 -rowspan 4  \
	    -column 1 -columnspan 3 -sticky "n"
	set orient_text "Indicates the current orientation.\n$orient_text"

	# Top entry
	global top
	entry $w.tentry -textvariable top -width 3
	Tooltip $w.tentry $orient_text
	grid config $w.tentry -row 0 -column 0 -sticky "e"

	# Front entry
	global front
	entry $w.fentry -textvariable front -width 3
	Tooltip $w.fentry $orient_text
	grid config $w.fentry -row 4 -column 2 -sticky "nw"
	
	# Side entry
	global side
	entry $w.sentry -textvariable side -width 3
	Tooltip $w.sentry $orient_text
	grid config $w.sentry -row 1 -column 4 -sticky "n"

        trace variable top w "$this enable_update $which"
        trace variable front w "$this enable_update $which"
        trace variable side w "$this enable_update $which"

        # reset button
	button $data.ui.reset -text "Reset" -command "$this reset_orientations"
	Tooltip $data.ui.reset "Reset the orientation labels to defaults."
	pack $data.ui.reset -side right -anchor se -padx 4 -pady 4

	return $history.$which
    }

    method open_nrrd_reader_ui {i} {
	# disable execute button and change behavior of execute command
	set m [lindex [lindex $filters($i) $modules] 0]
	
	$m initialize_ui

	.ui$m.f7.execute configure -state disabled

	upvar \#0 .ui$m data	
	set data(-command) "wm withdraw .ui$m"
    }

    method open_field_reader_ui {i} {
	# disable execute button and change behavior of execute command
	set m [lindex [lindex $filters($i) $modules] 3]

	$m initialize_ui

	.ui$m.f7.execute configure -state disabled

	upvar \#0 .ui$m data
	set data(-command) "wm withdraw .ui$m"
    }

    method dicom_ui { } {
	set m [lindex [lindex $filters(0) $modules] 1]
	$m initialize_ui

	if {[winfo exists .ui$m]} {
	    # disable execute button 
	    .ui$m.buttonPanel.btnBox.execute configure -state disabled
	}

	global $m-dir $m-num-files
	trace variable $m-dir w "$this enable_update 0"
	trace variable $m-num-files w "$this enable_update 0"
    }

    method analyze_ui { } {
	set m [lindex [lindex $filters(0) $modules] 2]
	$m initialize_ui
	if {[winfo exists .ui$m]} {
	    # disable execute button 
	    .ui$m.buttonPanel.btnBox.execute configure -state disabled
	}
	global $m-file $m-num-files
	trace variable $m-file w "$this enable_update 0"
	trace variable $m-num-files w "$this enable_update 0"
    }


    ### update/reset_orientations
    #################################################
    method reset_orientations {} {
	global top front side

        # disable flip and permute modules and change choose ports
	set UnuFlip1 [lindex [lindex $filters(0) $modules] 29]
	set Choose1 [lindex [lindex $filters(0) $modules] 32]
	global $Choose1-port-index
        disableModule $UnuFlip1 1
	set $Choose1-port-index 0

	set UnuFlip2 [lindex [lindex $filters(0) $modules] 30]
	set Choose2 [lindex [lindex $filters(0) $modules] 33]
	global $Choose2-port-index
        disableModule $UnuFlip2 1
	set $Choose2-port-index 0

	set UnuFlip3 [lindex [lindex $filters(0) $modules] 31]
	set Choose3 [lindex [lindex $filters(0) $modules] 34]
	global $Choose3-port-index
        disableModule $UnuFlip3 1
	set $Choose3-port-index 0

	set UnuPermute [lindex [lindex $filters(0) $modules] 28]
	set Choose4 [lindex [lindex $filters(0) $modules] 35]
	global Choose4-port-index
        disableModule $UnuPermute 1
	set $Choose4-port-index 0

	set top "S"
	set front "A"
	set side "L"

	# Re-execute
	if {$has_executed} {
	    set m [lindex [lindex $filters(0) $modules] 5]
	    $m-c needexecute
	}
    }

    method update_orientations {} {
	global top front side
	setGlobal top [string toupper $top]
	setGlobal front [string toupper $front]
	setGlobal side [string toupper $side]
	set errror [txt "Orientations must all be different." \
			"and consist of a Superior (S) or Inferior (I),"\
			"an Anterior (A) or Posterior (P)," \
			"and a Left (L) or Right (R)."]
		  
	# check that they are all different
	if {[string eq $top $front] || [string eq $top $side] || \
		[string eq $front $side]} {
	    ok_box $error
	    return
	}

	# check that they are all either S,I,A,P,L, or R
	foreach s "$top $front $side" { 
	    if {$s != "S" && $s != "I" && $s != "A" && $s != "P" && \
		    $s != "L" && $s != "R" } {
		ok_box $error
		return
	    }
	}

	if {!$loading} {
	    # reset any downstream crop and resample params and issue warning
	    set asked 0
	    for {set i 1} {$i < $num_filters} {incr i} {
		set type [lindex $filters($i) $filter_type]
		set row  [lindex $filters($i) $which_row]
		if { $row == -1 } continue

		# give the user a chance to opt out of setting the orientation
		if { !$asked && ($type == "crop" || $type == "resample") } {
		    set asked 1
		    if { [okcancel_box \
			      "Downstream crop and resample filters will be " \
			      "reset. Do you want to proceed with changing" \
			      "the orientation?"] == "cancel"} return
		}
		
		if { $type == "crop" } {
		    set UnuCrop [lindex [lindex $filters($i) $modules] 0]
		    foreach num "0 1 2" {
			setGlobal $UnuCrop-minAxis${num} 0
			setGlobal $UnuCrop-maxAxis${num} M
		    }
		} elseif { $type == "resample" } {
		    set UnuResample [lindex [lindex $filters($i) $modules] 0]
		    setGlobal $UnuResample-resampAxis0 "x1"
		    setGlobal $UnuResample-resampAxis1 "x1"
		    setGlobal $UnuResample-resampAxis2 "x1"
		}
	    }
	}
	
	# Permute into order where top   = S/I
	#                          front = A/P
	#                          side    L/R
	set new_side 0
	set new_front 1
	set new_top 2
	
	set c_side "L"
	set c_front "A"
	set c_top "I"
	
	
	set need_permute 0
	# Check side variable which corresponds to axis 0
	if {$side == "L"} {
	    set new_side 0
	    set c_side "L"
	} elseif {$side == "R"} {
	    set new_side 0
	    set c_side "R"
	} elseif {$side == "A"} {
	    set new_side 1
	    set need_permute 1
	    set c_side "A"
	} elseif {$side == "P"} {
	    set new_side 1
	    set need_permute 1
	    set c_side "P"
	} elseif {$side == "S"} {
	    set new_side 2
	    set need_permute 1
	    set c_side "S"
	} else {
	    set new_side 2
	    set need_permute 1
	    set c_side "I"
	}

	# Check front variable which corresponds to axis 1
	if {$front == "A"} {
	    set new_front 1
	    set c_front "A"
	} elseif {$front == "P"} {
	    set new_front 1
	    set c_front "P"
	} elseif {$front == "L"} {
	    set new_front 0
	    set need_permute 1
	    set c_front "L"
	} elseif {$front == "R"} {
	    set new_front 0
	    set need_permute 1
	    set c_front "R"
	} elseif {$front == "S"} {
	    set new_front 2
	    set need_permute 1
	    set c_front "S"
	} else {
	    set new_front 2
	    set need_permute 1
	    set c_front "I"
	}

	# Check top variable which is axis 2
	if {$top == "S"} {
	    set new_top 2
	    set c_top "S"
	} elseif {$top == "I"} {
	    set new_top 2
	    set c_top "I"
	} elseif {$top == "L"} { 
	    set new_top 0
	    set need_permute 1
	    set c_top "L"
	} elseif {$top == "R"} {
	    set new_top 0
	    set need_permute 1
	    set c_top "R"
	} elseif {$top == "A"} {
	    set new_top 1
	    set need_permute 1
	    set c_top "A"
	} else {
	    set new_top 1
	    set need_permute 1
	    set c_top "P"
	}

	# only use permute if needed to avoid copying data
	set UnuPermute [lindex [lindex $filters(0) $modules] 28]
	set Choose [lindex [lindex $filters(0) $modules] 35]

	if {$need_permute == 1} {
	    setGlobal $Choose-port-index 1
	    disableModule $UnuPermute 0
	    setGlobal $UnuPermute-axis0 $new_side
	    setGlobal $UnuPermute-axis1 $new_front
	    setGlobal $UnuPermute-axis2 $new_top
	} else {
	    setGlobal $Choose-port-index 0
	    disableModule $UnuPermute 1
	}

	set flip_0 0
	set flip_1 0
	set flip_2 0

	# only flip axes if needed
	if {$c_side != "L"} {
	    # need to flip axis 0
	    $this flip0 1
	    set flip_0 1
	} else {
	    $this flip0 0
	}

	if {$c_front != "A"} {
	    # need to flip axis 1
	    $this flip1 1
	    set flip_1 1
	} else {
	    $this flip1 0
	}

	if {$c_top != "S"} {
	    # need to flip axis 2
	    $this flip2 1
	    set flip_2 1
	} else {
	    $this flip2 0
	}

	# Re-execute
	if {!$loading && $has_executed} {
	    if {$need_permute == 1} {
		$UnuPermute-c needexecute
	    } elseif {$flip_0 == 1} {
		set UnuFlip [lindex [lindex $filters(0) $modules] 29]
		$UnuFlip-c needexecute
	    } elseif {$flip_1 == 1} {
		set UnuFlip [lindex [lindex $filters(0) $modules] 30]
		$UnuFlip-c needexecute
	    } elseif {$flip_2 == 1} {
		set UnuFlip [lindex [lindex $filters(0) $modules] 31]
		$UnuFlip-c needexecute
	    } else {
		set m [lindex [lindex $filters(0) $modules] 5]
		$m-c needexecute
	    }
	}
    }

    method flip0 { toflip } {
	set UnuFlip [lindex [lindex $filters(0) $modules] 29]
	set Choose [lindex [lindex $filters(0) $modules] 32]
	global $Choose-port-index
	
	if {$toflip == 1} {
	    disableModule $UnuFlip 0
	    set $Choose-port-index 1
	} else {
	    disableModule $UnuFlip 1
	    set $Choose-port-index 0
	}
    }
    
    method flip1 { toflip } {
	set UnuFlip [lindex [lindex $filters(0) $modules] 30]
	set Choose [lindex [lindex $filters(0) $modules] 33]
	global $Choose-port-index
	
	if {$toflip == 1} {
	    disableModule $UnuFlip 0
	    set $Choose-port-index 1
	} else {
	    disableModule $UnuFlip 1
	    set $Choose-port-index 0
	}
    }
    
    method flip2 { toflip } {
	set UnuFlip [lindex [lindex $filters(0) $modules] 31]
	set Choose [lindex [lindex $filters(0) $modules] 34]
	global $Choose-port-index
	
	if {$toflip == 1} {
	    disableModule $UnuFlip 0
	    set $Choose-port-index 1
	} else {
	    disableModule $UnuFlip 1
	    set $Choose-port-index 0
	}
    }
    
    ##############################
    ### configure_readers
    ##############################
    # Keeps the readers in sync.  Every time a different
    # data tab is selected (Nrrd, Dicom, Analyze) the other
    # readers must be disabled to avoid errors.
    method configure_readers { which } {
	set load_mods [lindex $filters(0) $modules]
	set ChooseNrrd [lindex $load_mods $load_choose_input]
	set NrrdReader [lindex $load_mods $load_nrrd]
	set DicomNrrdReader [lindex $load_mods $load_dicom]
	set AnalyzeNrrdReader [lindex $load_mods $load_analyze]
	set FieldReader [lindex $load_mods $load_field]
	if {$which == "Nrrd"} {
	    setGlobal $ChooseNrrd-port-index 0
	    disableModule $NrrdReader 0
	    disableModule $DicomNrrdReader 1
	    disableModule $AnalyzeNrrdReader 1
	    disableModule $FieldReader 1
        } elseif {$which == "Dicom"} {
	    setGlobal $ChooseNrrd-port-index 1
	    disableModule $NrrdReader 1
	    disableModule $DicomNrrdReader 0
	    disableModule $AnalyzeNrrdReader 1
	    disableModule $FieldReader 1
        } elseif {$which == "Analyze"} {
	    setGlobal $ChooseNrrd-port-index 2
	    disableModule $NrrdReader 1
	    disableModule $DicomNrrdReader 1
	    disableModule $AnalyzeNrrdReader 0
	    disableModule $FieldReader 1
        } elseif {$which == "Field"} {
	    setGlobal $ChooseNrrd-port-index 3
	    disableModule $NrrdReader 1
	    disableModule $DicomNrrdReader 1
	    disableModule $AnalyzeNrrdReader 1
	    disableModule $FieldReader 0
	}
    }
    
    #############################
    ### init_Vframe
    #############################
    # Initialize the visualization frame on the right. For this app
    # that includes the Planes, Volume Rendering, and 3D Options tabs.  
    method init_Vframe { m case} {
	global mods
	global tips
	if { [winfo exists $m] } {
	    ### Visualization Frame
	    iwidgets::labeledframe $m.vis \
		-labelpos n -labeltext "Visualization Settings" 
	    pack $m.vis -side right -anchor ne -fill both -expand yes
	    
	    set vis [$m.vis childsite]
	    
	    ### Tabs
	    iwidgets::tabnotebook $vis.tnb -width $notebook_width \
		-height [expr $vis_height - 25] -tabpos n \
                -equaltabs false  -backdrop gray
	    pack $vis.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

            set vis_frame_tab$case $vis.tnb


	    set command "$this change_vis_frame Planes"
	    set page [$vis.tnb add -label "Planes" -command $command]

            frame $page.planes 
            pack $page.planes -side top -anchor nw -expand no -fill x

	    checkbutton $page.planes.xp -text "Show Sagittal Plane" \
		-variable "$mods(Viewer)-ViewWindow_0-Slice0 (1)" \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"
            Tooltip $page.planes.xp "Turn Sagittal plane on/off"

  	    checkbutton $page.planes.xm -text "Show Sagittal MIP" \
  		-variable "$mods(Viewer)-ViewWindow_0-MIP Slice0 (1)" \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"
            Tooltip $page.planes.xm "Turn Sagittal MIP on/off"


	    checkbutton $page.planes.yp -text "Show Coronal Plane" \
		-variable "$mods(Viewer)-ViewWindow_0-Slice1 (1)" \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"
            Tooltip $page.planes.yp "Turn Sagittal plane on/off"

  	    checkbutton $page.planes.ym -text "Show Coronal MIP" \
  		-variable "$mods(Viewer)-ViewWindow_0-MIP Slice1 (1)" \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"
            Tooltip $page.planes.ym "Turn Sagittal MIP on/off"


	    checkbutton $page.planes.zp -text "Show Axial Plane" \
		-variable "$mods(Viewer)-ViewWindow_0-Slice2 (1)" \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"
            Tooltip $page.planes.zp "Turn Sagittal plane on/off"

  	    checkbutton $page.planes.zm -text "Show Axial MIP" \
  		-variable "$mods(Viewer)-ViewWindow_0-MIP Slice2 (1)" \
		-command "$mods(Viewer)-ViewWindow_0-c redraw"
            Tooltip $page.planes.zm "Turn Sagittal MIP on/off"

            grid configure $page.planes.xp -row 0 -column 0 -sticky "w"
            grid configure $page.planes.xm -row 0 -column 1 -sticky "w"
            grid configure $page.planes.yp -row 1 -column 0 -sticky "w"
            grid configure $page.planes.ym -row 1 -column 1 -sticky "w"
            grid configure $page.planes.zp -row 2 -column 0 -sticky "w"
            grid configure $page.planes.zm -row 2 -column 1 -sticky "w"

            # display window and level
            iwidgets::labeledframe $page.winlevel \
                 -labeltext "Window/Level Controls" -labelpos nw
            pack $page.winlevel -side top -anchor nw -expand no -fill x \
                -pady 3

            set winlevel [$page.winlevel childsite]
	    set wwf $winlevel.ww
	    set wlf $winlevel.wl
            frame $wwf
            frame $wlf
            pack $wwf $wlf -side top -anchor ne -pady 0
	    
            label $wwf.l -text "Window Width"
            scale $wwf.s \
                -variable $mods(ViewSlices)-clut_ww \
                -from 1 -to 9999 -length 120 -width 14 \
                -showvalue false -orient horizontal
            Tooltip $wwf.s "Control the window width of\nthe 2D viewers"
            entry $wwf.e -textvariable $mods(ViewSlices)-clut_ww -width 6
            pack $wwf.l $wwf.s $wwf.e -side left

            label $wlf.l -text "Window Level "
            scale $wlf.s \
                -variable $mods(ViewSlices)-clut_wl \
                -from 0 -to 9999 -length 120 -width 14 \
                -showvalue false -orient horizontal
            Tooltip $wlf.s "Control the window level of\nthe 2D viewers"

            entry $wlf.e -textvariable $mods(ViewSlices)-clut_wl -width 6

            pack $wlf.l $wlf.s $wlf.e -side left

	    set command "$this change_window_width_and_level 1"
            bind $wwf.e <Return> $command
            bind $wwf.s <ButtonRelease> $command
            bind $wlf.e <Return> $command
            bind $wlf.s <ButtonRelease> $command

            # Background threshold
            frame $page.thresh 
            pack $page.thresh -side top -anchor nw -expand no -fill x

            label $page.thresh.l -text "Background\nThreshold:"

            scale $page.thresh.s \
                -from 0 -to 100 \
 	        -orient horizontal -showvalue false \
 	        -length 140 -width 14 \
	        -variable $mods(ViewSlices)-background_threshold
            entry $page.thresh.l2 -textvariable $mods(ViewSlices)-background_threshold -width 6
            Tooltip $page.thresh.s "Clip out values less than\nspecified background threshold"

            pack $page.thresh.l -side left -anchor w
            pack $page.thresh.l2 $page.thresh.s -side right -anchor e -padx 2
            pack $page.thresh -side top -fill x -expand 0

            Tooltip $page.thresh.l "Change background threshold. Data\nvalues less than or equal to the threshold\nwill be transparent in planes."
            Tooltip $page.thresh.s "Change background threshold. Data\nvalues less than or equal to the threshold\nwill be transparent in planes."
            Tooltip $page.thresh.l2 "Change background threshold. Data\nvalues less than or equal to the threshold\nwill be transparent in planes."

	    # Fonts
	    frame $page.fonts -relief groove -borderwidth 2
	    pack $page.fonts -side top -fill x -expand 0 -padx 5 -pady 3

	    global $mods(ViewSlices)-show_text
	    checkbutton $page.fonts.fonttog -text "Show 2D Window Text" \
		-variable $mods(ViewSlices)-show_text \
		-command "$mods(ViewSlices)-c set_font_sizes"
           pack $page.fonts.fonttog -padx 2 -side top -anchor nw -expand 0
	    
            frame $page.fonts.font
            label $page.fonts.font.l -text "Text Size:"

            scale $page.fonts.font.s \
                -from 2 -to 30 -orient horizontal -showvalue 0 \
 	        -width 14 -length 100  -resolution 0.1 \
	        -variable $mods(ViewSlices)-font_size \
                -command "$mods(ViewSlices)-c set_font_sizes"
            entry $page.fonts.font.l2 -width 4 \
		-textvariable $mods(ViewSlices)-font_size
                
            bind $page.fonts.font.l2 <KeyPress> \
                "$mods(ViewSlices)-c set_font_sizes"
	    bind $page.thresh.s <Button1-Motion> \
                "$mods(ViewSlices)-c set_font_sizes"

            pack $page.fonts.font.l -side left -anchor w -padx 2
            pack $page.fonts.font.l2 $page.fonts.font.s -side right -anchor e -padx 2
            pack $page.fonts.font -side top -fill x -expand 0


            frame $page.fonts.fontc
            label $page.fonts.fontc.l -text "Text Color:"

	    button $page.fonts.fontc.c -width 4 -command \
	        "$mods(ViewSlices) raise_color $page.fonts.fontc.c $mods(ViewSlices)-color_font set_font_sizes" \
		-background white -activebackground white

            pack $page.fonts.fontc.l -side left -anchor w -padx 2
            pack $page.fonts.fontc.c -side right -anchor e -padx 2
            pack $page.fonts.fontc -side top -fill x -expand 0

           
	    checkbutton $page.lines -text "Show Guidelines" \
		-variable show_guidelines \
		-command "$this toggle_show_guidelines" 
            pack $page.lines -side top -anchor nw -padx 4 -pady 7
            Tooltip $page.lines "Toggle 2D Viewer guidelines"

	    checkbutton $page.2Dtext -text "Filter 2D Textures" \
		-variable $mods(ViewSlices)-texture_filter \
		-command "$mods(ViewSlices)-c texture_rebind" 
            pack $page.2Dtext -side top -anchor nw -padx 4 -pady 7
            Tooltip $page.2Dtext "Turn filtering 2D textures\non/off"

	    checkbutton $page.anatomical -text "Anatomical Coordinates" \
		-variable $mods(ViewSlices)-anatomical_coordinates \
		-command "$mods(ViewSlices)-c redrawall" 
            pack $page.anatomical -side top -anchor nw -padx 4 -pady 7

	    global planes_color
	    iwidgets::labeledframe $page.isocolor \
		-labeltext "Color Planes By" \
		-labelpos nw 
	    pack $page.isocolor -side top -anchor n -padx 3 -pady 0 -fill x
	    
	    set maps [$page.isocolor childsite]

	    global planes_mapType
	    foreach colormap { {gray 0} {rainbow 3} {darkhue 4} \
				   {blackbody 7} {blue-to-Red 17} } {
		set color [lindex $colormap 0]
		set value [lindex $colormap 1]
		set name [string totitle $color]
		set f $maps.$color
		frame $f
		pack $f -side top -anchor nw -padx 3 -pady 1 -fill x -expand 1
		radiobutton $f.b \
		    -text $name -variable planes_mapType -value $value \
		    -command "$this update_planes_color_by"
		Tooltip $f.b "Select color map for coloring planes"
		pack $f.b -side left -anchor nw -padx 3 -pady 0
		frame $f.f -relief sunken -borderwidth 2
		pack $f.f -padx 2 -pady 0 -side right -anchor e
		canvas $f.f.canvas -bg \#ffffff -height $colormap_height \
		    -width $colormap_width
		pack $f.f.canvas -anchor e -fill both -expand 1
		draw_colormap $name $f.f.canvas
	    }

	    button $page.clipping -text "Clipping Planes" \
		-command "$mods(Viewer)-ViewWindow_0 makeClipPopup"
	    pack $page.clipping -side top -anchor n

            #######
            set page [$vis.tnb add -label "Volume Rendering" \
			  -command "$this change_vis_frame {Volume Rendering}"]

            global show_volume_ren
	    checkbutton $page.toggle -text "Show Volume Rendering" \
		-variable show_vol_ren \
		-command "$this toggle_show_vol_ren"
            Tooltip $page.toggle "Turn volume rendering on/off"
            pack $page.toggle -side top -anchor nw -padx 3 -pady 3


            button $page.vol -text "Edit Transfer Function" \
		-command "$this open_transfer_function_editor"
	    TooltipMultiline $page.vol "Open up the interface\n" \
		"for editing the transfer function"
            pack $page.vol -side top -anchor n -padx 3 -pady 3
            
            set VolumeVisualizer [lindex [lindex $filters(0) $modules] 14]
            set n "$VolumeVisualizer-c needexecute"

            global $VolumeVisualizer-render_style

            frame $page.fmode
            pack $page.fmode -padx 2 -pady 2 -fill x
            label $page.fmode.mode -text "Mode"
	    radiobutton $page.fmode.modeo -text "Over Operator" -relief flat \
		    -variable $VolumeVisualizer-render_style -value 0 \
    	  	    -anchor w -command $n
   	    radiobutton $page.fmode.modem -text "MIP" -relief flat \
		    -variable $VolumeVisualizer-render_style -value 1 \
		    -anchor w -command $n
   	    pack $page.fmode.mode $page.fmode.modeo $page.fmode.modem \
                -side left -fill x -padx 4 -pady 4


	    #----------------------------------------------------------
	    # Disable Lighting
	    #----------------------------------------------------------
	    set NrrdSetupTexture [lindex [lindex $filters(0) $modules] 10]
	    global $NrrdSetupTexture-valuesonly
	    checkbutton $page.lighting \
		-text "Compute data for shaded volume rendering" \
		-relief flat -offvalue 1 \
		-variable $NrrdSetupTexture-valuesonly -onvalue 0 \
		-anchor w -command "$this toggle_compute_shading"
	    Tooltip $page.lighting \
		"Turn computing data for shaded volume\nrendering on/off."
	    pack $page.lighting -side top -fill x -padx 4

	    #-----------------------------------------------------------
	    # Shading
	    #-----------------------------------------------------------
	    checkbutton $page.shading -text "Show shaded volume rendering" \
		-relief flat -variable $VolumeVisualizer-shading \
		-onvalue 1 -offvalue 0 -anchor n -command "$n"
	    Tooltip $page.shading "If computed, turn use of shading on/off"
	    pack $page.shading -side top -fill x -padx 4


	    #-----------------------------------------------------------
	    # Sample Rates
	    #-----------------------------------------------------------
	    iwidgets::labeledframe $page.samplingrate \
		-labeltext "Sampling Rates" -labelpos nw
	    pack $page.samplingrate -side top -anchor nw -expand no -fill x
	    set sratehi [$page.samplingrate childsite]
	    
	    scale $sratehi.srate_hi -label "Final Rate" \
		-variable $VolumeVisualizer-sampling_rate_hi \
		-from 0.5 -to 20.0 \
		-showvalue true -resolution 0.1 \
		-orient horizontal -width 15 
	    
	    scale $sratehi.srate_lo -label "Interactive Rate" \
		-variable $VolumeVisualizer-sampling_rate_lo \
		-from 0.1 -to 20.0 \
		-showvalue true -resolution 0.1 \
		-orient horizontal -width 15 
	    pack $sratehi.srate_hi $sratehi.srate_lo \
		-side left -fill x -expand yes -padx 4
	    bind $sratehi.srate_hi <ButtonRelease> $n
	    bind $sratehi.srate_lo <ButtonRelease> $n
	    
	    
	    #-----------------------------------------------------------
	    # Global Opacity
	    #-----------------------------------------------------------
	    iwidgets::labeledframe $page.opacityframe \
		-labeltext "Global Opacity" -labelpos nw
	    pack $page.opacityframe -side top -anchor nw \
		-expand no -fill x
	    set oframe [$page.opacityframe childsite]
	    
	    scale $oframe.opacity \
		-variable $VolumeVisualizer-alpha_scale \
		-from -1.0 -to 1.0 -length 150 \
		-showvalue false -resolution 0.001 \
		-orient horizontal -width 15
	    entry $oframe.opacityl -relief flat \
		-textvariable $VolumeVisualizer-alpha_scale -width 4 \
		
	    pack $oframe.opacity -side left -fill x -expand yes -padx 4
	    pack $oframe.opacityl -side left -fill x -padx 4
	    bind $oframe.opacity <ButtonRelease> $n

	    #-----------------------------------------------------------
	    # Volume Rendering Window Level Controls
	    #-----------------------------------------------------------
	    global link_winlevel vol_width vol_level
	    
	    iwidgets::labeledframe $page.winlevel \
		-labeltext "Window/Level Controls" \
		-labelpos nw
	    pack $page.winlevel -side top -anchor nw -expand no -fill x
	    set winlevel [$page.winlevel childsite]
	    
	    checkbutton $winlevel.link -text "Link to Slice Window/Level" \
		-variable link_winlevel \
		-command "$this link_windowlevels 1"
	    TooltipMultiline $winlevel.link "Link the changes of the\n" \
		"window controls below to\nthe planes window controls"
	    pack $winlevel.link -side top -anchor nw -pady 1
	    
	    set wwf $winlevel.ww
	    set wlf $winlevel.wl

	    frame $wwf
	    frame $wlf
	    pack $wwf $wlf -side top -anchor ne -pady 0

	    label $wwf.l -text "Window Width"
	    scale $wwf.s -variable vol_width \
		-from 1 -to 9999 -length 120 -width 15 \
		-showvalue false -orient horizontal
	    Tooltip $wwf.s "Control the window width of\nthe volume rendering"
	    entry $wwf.e -textvariable vol_width -width 6
	    pack $wwf.l $wwf.s $wwf.e -side left

	    label $wlf.l -text "Window Level "
	    scale $wlf.s -variable vol_level \
		-from 0 -to 9999 -length 120 -width 15 \
		-showvalue false -orient horizontal
	    Tooltip $wlf.s "Control the window width of\nthe volume rendering"
	    entry $wlf.e -textvariable vol_level -width 6
	    pack $wlf.l $wlf.s $wlf.e -side left

	    set command "$this change_volume_window_width_and_level 1"
	    bind $wlf.s <ButtonRelease> $command
	    bind $wlf.e <Return> $command
	    bind $wwf.s <ButtonRelease> $command
	    bind $wwf.e <Return> $command

	    
	    #-----------------------------------------------------------
	    # Transfer Function Widgets
	    #-----------------------------------------------------------
	    
	    # Gradient threshold
	    set f $page.gthresh
	    frame $f
	    label $f.l -text "Gradient Threshold:"
	    set command "$mods(ViewSlices)-c gradient_thresh; 
                     $mods(ViewSlices)-c redrawall"
	    scale $f.s \
		-from 0.0 -to 1.0 -resolution 0.002 \
		-orient horizontal -showvalue false \
		-length 100 -width 14 \
		-variable $mods(ViewSlices)-gradient_threshold \
		-command $command
	    bind $f <Button1-Motion> $command
	    entry $f.l2 -width 6 \
		-textvariable $mods(ViewSlices)-gradient_threshold
	    pack $f.l -side left -anchor w
	    pack $f.l2 $f.s -side right -anchor e -padx 2
	    pack $f -side top -fill x -expand 0
	    
	    frame $page.buttons -bd 0
	    button $page.buttons.paint -text "Add Paint Layer" \
		-command "$mods(EditColorMap2D)-c addpaint"
	    button $page.buttons.undo -text "Undo Paint Stroke" \
		-command "$mods(ViewSlices)-c undo"
	    pack $page.buttons.paint $page.buttons.undo -side left \
		-fill x -padx 10 -pady 3 -expand 1
	    pack $page.buttons -side top -expand 0 -padx 0 -fill x -pady 3
	    
	    set f $page.applyColormap2D
	    frame $f -bd 0
	    checkbutton $f.button -text "Show Transfer Function in 2D" \
		-variable "$mods(ViewSlices)-show_colormap2" \
		-command "$mods(ViewSlices)-c needexecute"
	    pack $f.button -side left
	    pack $f -fill x -side top
	    
	    $mods(EditColorMap2D) label_widget_columns $page.widgets_label
	    pack $page.widgets_label -side top -fill x -padx 2
	    iwidgets::scrolledframe $page.widgets -hscrollmode none \
		-vscrollmode static
	    
	    pack $page.widgets -side top -fill both -expand yes -padx 2
	    $mods(EditColorMap2D) add_frame [$page.widgets childsite]
	    
	    
	    ### Renderer Options Tab
	    create_viewer_tab $vis "3D Options"
	    
	    ### Time tab
	    set command "$this change_vis_frame Time"
	    set page [$vis.tnb add -label "Time" -command $command]
	    $mods(NrrdSelectTime_2) build_ui $page
	    pack $page.vcr $page.playmode $page.min $page.cur $page.max \
		$page.inc -padx 5 -pady 5 -fill x -expand 0
	    $mods(NrrdSelectTime_2) update_range
	    lappend mods(NrrdSelectTime_2_pages) $page


	    $vis.tnb view "Planes"
	    
	    
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

    method open_transfer_function_editor {} {
	global mods
	$mods(EditColorMap2D) initialize_ui
	wm title .ui${mods(EditColorMap2D)} "Transfer Function Editor"
	pack forget .ui$mods(EditColorMap2D).buttonPanel.btnBox.highlight
    }

    method toggle_compute_shading {} {
        set NrrdSetupTexture [lindex [lindex $filters(0) $modules] 10]
	upvar \#0 $NrrdSetupTexture-valuesonly valuesonly
	set f f.vis.childsite.tnb.canvas.notebook.cs.page2.cs.shading 
	set state [expr $valuesonly?"disabled":"normal"]
	.standalone.detachedV.$f configure -state $state
	.standalone.attachedV.$f configure -state $state
        $NrrdSetupTexture-c needexecute
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
	    pack forget $win.viewers

	    if { $IsVAttached } {
		pack forget $attachedVFr
		pack $attachedVFr -side right -anchor n 
	    }

	    pack $attachedPFr -side left -anchor n

	    pack $win.viewers -side left -anchor n -fill both -expand 1

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
	    pack forget $win.viewers

	    pack $attachedVFr -anchor n -side right 

	    if { $IsPAttached } {
		pack forget $attachedPFr
		pack $attachedPFr -side left -anchor n 
	    }

	    pack $win.viewers -side left -anchor n -fill both -expand 1

	    set new_width [expr $c_width + $vis_width]
            append geom $new_width x $c_height
	    wm geometry $win $geom
	    set IsVAttached 1
	}
    }

    method set_cur_data_tab {which} {
	if {$initialized} {
	    set cur_data_tab $which
	}
    }


    ############################
    ### change_vis_frame
    ############################
    # Method called when Visualization tabs are changed from
    # the standard options to the global viewer controls
    method change_vis_frame { which } {
	if {!$initialized} return
	# change tabs for attached and detached
	$vis_frame_tab1 view $which
	$vis_frame_tab2 view $which
	set c_vis_tab $which
    }

    method add_insert_bar {f which} {
	set f $f.f
	frame $f
	set rb $f.eye_$which
 	radiobutton $rb -variable eye -value $which \
	    -command "$this change_eye"
	Tooltip $rb "Select to change current view\nof 3D and 2D windows"

	# Add a bar that when a user clicks, will bring
	# up the menu of filters to insert
	set img [image create photo -width 1 -height 1]	
  	button $f.insert -image $img -borderwidth 2 -relief raised \
	    -cursor plus -background "#6c90ce" \
	    -activebackground "#4c70ae" -height 3 -width 198
	pack $rb -side left -anchor w
	pack $f.insert -side left -fill x -expand 1
	grid config $f -column 0 -row 1
	bind $f.insert <ButtonPress-1> "app popup_insert_menu %X %Y $which"
  	bind $f.insert <ButtonPress-2> "app popup_insert_menu %X %Y $which"
  	bind $f.insert <ButtonPress-3> "app popup_insert_menu %X %Y $which"

        TooltipMultiline $f.insert "Click on this bar to insert any of the\n" \
	    "pre-processing filters at this location"
    }

    method popup_insert_menu {x y which} {
	set mouseX $x
	set mouseY $y
	set menu_id ".standalone.insertmenu"
	$this generate_insert_menu $menu_id $which
	tk_popup $menu_id $x $y
    }

    method generate_insert_menu {menu_id which} {
	set num_entries [$menu_id index end]
	if { $num_entries == "none" } { 
	    set num_entries 0
	}
	for {set c 0} {$c <= $num_entries} {incr c } {
	    $menu_id delete 0
	}
	
	$menu_id add command -label "Insert Resample" \
	    -command "$this add_Resample $which"
	$menu_id add command -label "Insert Crop" \
	    -command "$this add_Crop $which"
	$menu_id add command -label "Insert Histogram" \
	    -command "$this add_Histo $which"
	$menu_id add command -label "Insert Median Filtering" \
	    -command "$this add_Cmedian $which"
    }

    method execute_Data {} {
	# execute the appropriate reader
	# and verify valid loading file
	set valid_data 0
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_input]
	upvar \#0 $ChooseNrrd-port-index port
        if {$port == 0} {       ; # Nrrd
            set mod [lindex [lindex $filters(0) $modules] $load_nrrd]
	    upvar \#0 $mod-filename filename
	} elseif {$port == 1} { ; # Dicom
            set mod [lindex [lindex $filters(0) $modules] $load_dicom]
	    upvar \#0 $mod-entry-dir0 entry $mod-series-files files
            if { [info exists entry] && [info exists files] } {
		set filename [file join $entry [lindex $files 0]]
	    }
	} elseif {$port == 2} { ; # Analyze
            set mod [lindex [lindex $filters(0) $modules] $load_analyze]
	    upvar \#0 $mod-filenames0 filename
	} else {                ; # Field
            set mod [lindex [lindex $filters(0) $modules] $load_field]
	    upvar \#0 $mod-filename filename
	}

	if { ![info exists filename] || ![validFile $filename] } {
	    ok_box "Invalid filename specified.  Please select a" \
		"valid filename and click the Update button." 
	    return
	}

	# try to load a corresponding xff file into the EditColorMap2 module
	# currenlty, we only have nrrd demo data sets so this will only
	# work if we are reading in a nrrd and is one of our demo datasets
	# (i.e. tooth, CThead, engine)
	if {!$loading && $port == 0} {
	    global mods
	    set NrrdReader [lindex [lindex $filters(0) $modules] $load_nrrd]
	    upvar \#0 $mods(EditColorMap2D)-filename cm2filename
	    upvar \#0 $NrrdReader-filename nrrdfilename
	    set cmap2 [join [lrange [split $nrrdfilename .] 0 end-1] .].cmap2
	    if { [validFile $cmap2] } {
		set cm2filename $cmap2
		$mods(EditColorMap2D)-c load
	    }
	}
	set 2D_fixed 0
	
	# for some reason, the choosenrrds don't execute properly. 
	# Downstream ones execute before the upstream ones,
	# so a new dataset isn't propagated            
	set execute_choose 1
	
	$mod-c needexecute
	
	set has_executed 1
    } 

    method add_Resample { { which -1 } } {
	if { $which == -1 } {
	    set which [find_last_filter]
	}
	# add modules
	set m1 [addModule Teem UnuNtoZ UnuResample]
	# add connection to Choose module and new module
	set output_mod [lindex [lindex $filters($which) $output] 0]
	set output_port [lindex [lindex $filters($which) $output] 1]
	addConnection $output_mod $output_port $m1 0
        # add to filters array
	set choose [connect_filter_module_to_choose $m1]
	set next [lindex $filters($which) $next_index]
        set filters($num_filters) \
	    [list resample "$m1" "$m1 0" "$m1 0" \
		 $which $next $choose $grid_rows 1 "Resample - Unknown"]
	# update previous filter to expect us as next filter
	set filters($which) \
	    [lreplace $filters($which) $next_index $next_index $num_filters]
	# patch up connections if inserting
	insert_filter $num_filters
	# add the UI to the left pane
	create_filter_UI $num_filters
        $this enable_update $num_filters
        change_indicator_labels "Press Update to Resample Volume..."
	incr num_filters
    }


    method add_Crop { which } {
	if { $which == -1 } {
	    set which [find_last_filter]
	}

	# add modules
	set m1 [addModule Teem UnuAtoM UnuCrop]
	set m2 [addModule Teem NrrdData NrrdInfo]	
	# add connection to Choose module and new module
	set output_mod [lindex [lindex $filters($which) $output] 0]
	set output_port [lindex [lindex $filters($which) $output] 1]
	addConnection $output_mod $output_port $m1 0
	addConnection $output_mod $output_port $m2 0
        # add to filters array
	set choose [connect_filter_module_to_choose $m1]
	set next [lindex $filters($which) $next_index]
        set filters($num_filters) \
	    [list crop "$m1 $m2" "$m1 0 $m2 0" "$m1 0" \
		 $which $next $choose $grid_rows 1 "Crop - Unknown" \
		 "Crop - Unknown" [list 0 0 0] 0 [list 0 0 0 0 0 0]]
	# update previous filter to expect us as next filter
	set filters($which) \
	    [lreplace $filters($which) $next_index $next_index $num_filters]

	insert_filter $num_filters
	create_filter_UI $num_filters
	start_crop $num_filters

	incr num_filters

        change_indicator_labels "Press Update to Crop Volume..."
        $this disable_update
    }
    
    method add_Cmedian { which } {
	if { $which == -1 } {
	    set which [find_last_filter]
	}
       	# add modules
	set m1 [addModule Teem UnuAtoM UnuCmedian]
	set output_mod [lindex [lindex $filters($which) $output] 0]
	set output_port [lindex [lindex $filters($which) $output] 1]
	# add connections
	addConnection $output_mod $output_port $m1 0
        # add to filters array
	set next [lindex $filters($which) $next_index]
	set choose [connect_filter_module_to_choose $m1]
        set filters($num_filters) \
	    [list cmedian "$m1" "$m1 0" "$m1 0" $which $next \
		 $choose $grid_rows 1 "Median Filtering - Unknown"]
	# update previous filter to expect us as next filter
	set filters($which) \
	    [lreplace $filters($which) $next_index $next_index $num_filters]

	insert_filter $num_filters
	create_filter_UI $num_filters
        change_indicator_labels "Press Update to Perform Median Filtering..."
        $this enable_update $num_filters

	incr num_filters
    }

    method add_Histo {which} {
       	if { $which == -1} {
	    set which [find_last_filter]
	}
	# add modules
	set m1 [addModule Teem UnuAtoM UnuHeq]	
	set m2 [addModule Teem UnuNtoZ UnuQuantize]	
	set m3 [addModule Teem Converters NrrdToField]
	set m4 [addModule SCIRun FieldsOther ScalarFieldStats]
	# add connections
	set output_mod [lindex [lindex $filters($which) $output] 0]
	set output_port [lindex [lindex $filters($which) $output] 1]
	addConnection $output_mod $output_port $m3 2
	addConnection $m3 0 $m4 0
	addConnection $output_mod $output_port $m1 0
	addConnection $m1 0 $m2 0

	global mods
	upvar \#0 $mods(ViewSlices)-min vmin $mods(ViewSlices)-max vmax
        set min $vmin
        set max $vmax

        if {$min == -1 && $max == -1} {
	    # min/max haven't been set becuase it hasn't executed yet
	    set min 0
	    set max 255
	}

        setGlobal $m1-bins 3000
        setGlobal $m2-nbits 8
        setGlobal $m2-minf $min
        setGlobal $m2-maxf $max
        setGlobal $m2-useinputmin 1
        setGlobal $m2-useinputmax 1
        setGlobal $m4-setdata 1
        global $m4-args
        trace variable $m4-args w \
	    "$this update_histo_graph_callback $num_filters"
	# Create the filter array, must be set before maybe_insert_filter
	set choose [connect_filter_module_to_choose $m2]
	set next [lindex $filters($which) $next_index]
        set filters($num_filters) \
	    [list histo "$m1 $m3 $m2 $m4" "$m1 0 $m3 2" "$m2 0" \
		 $which $next $choose $grid_rows 1 "Histo-Unknown"]
	# update previous filter to expect us as next filter
	set filters($which) \
	    [lreplace $filters($which) $next_index $next_index $num_filters]
	
	insert_filter $num_filters
	create_filter_UI $num_filters
        change_indicator_labels \
	    "Press Update to Perform Histogram Equalization..."
	$this enable_update $num_filters
	incr num_filters

        # execute histogram part so that is visible to user
        if { $has_executed } {
	    $m4-c needexecute
	}

    }


    ############################
    ### update_histo_graph_callback
    ############################
    # Called when the ScalarFieldStats updates the graph
    # so we can update ours
    method update_histo_graph_callback {i varname varele varop} {

	global mods
        set ScalarFieldStats [lindex [lindex $filters($i) $modules] 3]
        global $ScalarFieldStats-min $ScalarFieldStats-max

	global $ScalarFieldStats-args
        global $ScalarFieldStats-nmin
        global $ScalarFieldStats-nmax

	set nmin [set $ScalarFieldStats-nmin]
	set nmax [set $ScalarFieldStats-nmax]
	set args [set $ScalarFieldStats-args]

	if {$args == "?"} {
	    return
	}
        
        # for some reason the other graph will only work if I set temp 
        # instead of using the $i value 
	set temp $i

 	set graph $history1.$i.f$i.childsite.ui.histo.childsite.graph

         if { ($nmax - $nmin) > 1000 || ($nmax - $nmin) < 1e-3 } {
             $graph axis configure y -logscale yes
         } else {
             $graph axis configure y -logscale no
         }

         set min [set $ScalarFieldStats-min]
         set max [set $ScalarFieldStats-max]
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

# 	## other window
  	 set graph $history0.$temp.f$temp.childsite.ui.histo.childsite.graph

          if { ($nmax - $nmin) > 1000 || ($nmax - $nmin) < 1e-3 } {
              $graph axis configure y -logscale yes
          } else {
              $graph axis configure y -logscale no
          }


          $graph configure -barwidth $frac
          $graph axis configure x -min $min -max $max -subdivisions 4 -loose 1

          for {set i 0} { $i < [llength $yvector] } {incr i} {
              set val [expr $min + $i*$frac*($max-$min)]
              lappend xvector $val
          }
        
           if { [$graph element exists "h"] == 1 } {
               $graph element delete "h"
           }

           $graph element create "h" -xdata $xvector -ydata $yvector

    }

    method update_window_level_scales { args } {
	global mods
	upvar \#0 $mods(ViewSlices)-min min $mods(ViewSlices)-max max
        set ww [expr abs($max - $min)]
	set rez [expr $ww/1000.0]
	set rez [expr ($rez>1.0)?1.0:$rez]
	# foreach detached and attached right frame
	foreach vfr "$attachedVFr $detachedVFr" {
	    set prefix $vfr.f.vis.childsite.tnb.canvas.notebook.cs 
	    $prefix.page1.cs.thresh.s configure \
		-from $min -to $max -resolution $rez
	    # foreach tab, (2D pane and volume rendering pane)
	    foreach page "page1 page2" {
		set f $prefix.$page.cs.winlevel.childsite
		# configure window width scale
		if [winfo exists $f.ww.s] {
		    $f.ww.s configure -from 1 -to $ww -resolution $rez
		}
		# configure window level scale
		if [winfo exists $f.wl.s] {
		    $f.wl.s configure -from $min -to $max -resolution $rez
		}
	    }
	}
    }

    method change_window_width_and_level { { execute 0 } args } {
	global mods
	$mods(ViewSlices)-c setclut ;# set windows to be dirty
	if {$execute} {
	    $mods(ViewSlices)-c background_thresh
	}
	$this link_windowlevels $execute
    }

    method change_volume_window_width_and_level { { execute 0 } args } {
	# Change UnuJhisto and NrrdSetupTexture values
	global mods vol_width vol_level link_winlevel
	
	set min [expr $vol_level-$vol_width/2.0]
	set max [expr $vol_level+$vol_width/2.0]
	
	set NrrdSetupTexture [lindex [lindex $filters(0) $modules] 10] 
	setGlobal $NrrdSetupTexture-minf $min
	setGlobal $NrrdSetupTexture-maxf $max
	
	set UnuJhisto [lindex [lindex $filters(0) $modules] 21]
	setGlobal $UnuJhisto-mins "$min nan"
	setGlobal $UnuJhisto-maxs "$max nan"
	
	set Rescale [lindex [lindex $filters(0) $modules] 36] 
	setGlobal $Rescale-min $min
	setGlobal $Rescale-max $max
	
	# if linked, change the ViewSlices window width and level
	if {$link_winlevel == 1} {
	    set link_winlevel 0
	    setGlobal $mods(ViewSlices)-clut_ww $vol_width
	    setGlobal $mods(ViewSlices)-clut_wl $vol_level
	    set link_winlevel 1
	}
	
	if { $execute } {
	    $this execute_vol_ren
	}
	    
    }

    # execute modules if volume rendering enabled    
    method execute_vol_ren {} {
	global mods show_vol_ren
	upvar #0 $mods(ViewSlices)-crop crop
	if { $crop } return

 	if {$show_vol_ren == 1} {
   	    set NrrdSetupTexture [lindex [lindex $filters(0) $modules] 10] 
   	    set Rescale [lindex [lindex $filters(0) $modules] 36] 
    	    $Rescale-c needexecute
 	    $NrrdSetupTexture-c needexecute
         }
     }

    method link_windowlevels { { execute 1 } } {
	global link_winlevel mods
	if {$link_winlevel == 1} {
	    # Set vol_width and vol_level to ViewSlices window width and level
	    upvar \#0 $mods(ViewSlices)-clut_ww ww $mods(ViewSlices)-clut_wl wl
	    set link_winlevel 0
            setGlobal vol_width $ww
            setGlobal vol_level $wl
	    set link_winlevel 1

            # execute the volume rendering if it's on
	    if { $execute } {
		$this execute_vol_ren
	    }
	} 
    }


     method update_BioImage_shading_button_state {varname varele varop} {
         set VolumeVisualizer [lindex [lindex $filters(0) $modules] 14]

         global $VolumeVisualizer-shading-button-state
         
         set path f.vis.childsite.tnb.canvas.notebook.cs.page2.cs.shading
         if {[set $VolumeVisualizer-shading-button-state]} {
 	     $attachedVFr.$path configure -fg "black"
 	     $detachedVFr.$path configure -fg "black"
 	 } else {
 	     $attachedVFr.$path configure -fg "darkgrey"
 	     $detachedVFr.$path configure -fg "darkgrey"
 	 }
     }

    method update_crop_roi { which } {
	if {"$which" == "end" || \
		$which >= $num_filters || \
		[lindex $filters($which) $which_row] == -1 || \
		[lindex $filters($which) $filter_type] != "crop"} {
	    stop_crop
	    return
	}
	set UnuCrop [lindex [lindex $filters($which) $modules] 0]
	upvar \#0 $UnuCrop-show_roi show
	puts "show crop roi $show"

	if {$show} {
	    start_crop $which
	} else {
	    stop_crop
	}
    }

    method stop_crop { } {
        global mods
	set ViewSlices $mods(ViewSlices)
	$ViewSlices-c stopcrop
	set current_crop -1
    }
    
    method start_crop { which args } {
        global mods eye
	if { $loading || [expr $eye+1] != $which || \
		 [lindex $filters($which) $filter_type] != "crop"} return

	set ViewSlices $mods(ViewSlices)
	set current_crop $which
	set UnuCrop [lindex [lindex $filters($which) $modules] 0]
	set reset 0
	foreach axis "minAxis0 minAxis1 minAxis2 maxAxis0 maxAxis1 maxAxis2" {
	    set num [string index $axis end]
	    upvar \#0 $UnuCrop-$axis cropval
	    setGlobal $ViewSlices-crop_$axis $cropval
	    if { $cropval == "M" } {
		set reset 1		
	    }
	}
	$ViewSlices-c startcrop $reset
    }


    method add_Resample_UI {history row which} {
	frame $history.$which
	grid config $history.$which -column 0 -row $row -pady 0 -sticky news

	iwidgets::labeledframe $history.$which.f$which \
	    -labeltext "Resample" \
	    -labelpos nw \
	    -borderwidth 2 
	grid config $history.$which.f$which -column 0 -row 0 -sticky news

	set w [$history.$which.f$which childsite]

	frame $w.expand
	pack $w.expand -side top -anchor nw 

	global expandimg close_img
	button $w.expand.b -image $expandimg \
	    -anchor nw \
	    -command "$this change_visibility $which" \
	    -relief flat
	Tooltip $w.expand.b "Click to minimize/show\nthe Resample UI"
	label $w.expand.l -text "Resample - Unknown" -width $label_width \
            -anchor nw
	Tooltip $w.expand.l "Right click to edit label"

 	button $w.expand.c -image $close_img \
 	    -anchor nw \
 	    -command "$this filter_Delete $which" \
 	    -relief flat

	TooltipMultiline $w.expand.c "Click to delete this filter from\n" \
	    "the pipeline. All settings\nfor this filter will be lost."

	pack $w.expand.b $w.expand.l $w.expand.c -side left -anchor nw

	bind $w.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
	frame $w.ui
	pack $w.ui -side top -expand yes -fill both

        set UnuResample [lindex [lindex $filters($which) $modules] 0]
	for {set i 0} {$i < $dimension} {incr i} {
	    global $UnuResample-resampAxis$i
	    if {!$loading_ui} {
               set $UnuResample-resampAxis$i "x1"
            }
            trace variable $UnuResample-resampAxis$i w \
		"$this enable_update $which"
	    make_entry $w.ui.$i "Axis $i:" $UnuResample-resampAxis$i $which
	    pack $w.ui.$i -side top -anchor nw -expand yes -fill x
	}

        # configure labels
        $w.ui.0.l configure -text "Sagittal" -width 10
        $w.ui.1.l configure -text "Coronal" -width 10
        $w.ui.2.l configure -text "Axial" -width 10

        if {!$loading} {
	    setGlobal $UnuResample-filtertype cubicBS
        }
        setGlobal $UnuResample-sigma 2
        setGlobal $UnuResample-extent 2

 	iwidgets::optionmenu $w.ui.kernel -labeltext "Filter Type:" \
 	    -labelpos w \
            -command "$this change_kernel $w.ui.kernel $which"
 	pack $w.ui.kernel -side top -anchor nw 

 	$w.ui.kernel insert end Box Tent "Cubic (Catmull-Rom)" \
 	    "Cubic (B-spline)" Quartic Gaussian
	
 	$w.ui.kernel select "Cubic (B-spline)"

	return $history.$which
    }



    method add_Crop_UI {history row which} {
	frame $history.$which
	grid config $history.$which -column 0 -row $row -pady 0 -sticky news

	iwidgets::labeledframe $history.$which.f$which -labeltext Crop \
	    -labelpos nw  -borderwidth 2 
	grid config $history.$which.f$which -column 0 -row 0 -sticky news

	set w [$history.$which.f$which childsite]

	frame $w.expand
	pack $w.expand -side top -anchor nw


	global expandimg close_img
	# Expand/Hide button
	button $w.expand.b -image $expandimg -anchor nw -relief flat \
	    -command "$this change_visibility $which"
	Tooltip $w.expand.b "Click to minimize/show\nthe Crop UI"
	# Filter Label
	set title [lindex $filters($which) $filter_label]
	label $w.expand.l -text $title -width $label_width  -anchor nw
	Tooltip $w.expand.l "Right click to edit label"
	bind $w.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	# Delete filter button
 	button $w.expand.c -image $close_img -anchor nw -relief flat \
 	    -command "$this filter_Delete $which"
	TooltipMultiline $w.expand.c "Click to delete this filter from\n" \
	    "the pipeline. All settings\nfor the filter will be lost."
	pack $w.expand.b $w.expand.l $w.expand.c -side left -anchor nw 

	frame $w.ui
	pack $w.ui -side top -anchor nw -expand yes -fill x
	
	set UnuCrop [lindex [lindex $filters($which) $modules] 0]
	setGlobal $UnuCrop-num-axes $dimension        
        setGlobal $UnuCrop-reset_data 0
        setGlobal $UnuCrop-digits_only 1
	setGlobal $UnuCrop-show_roi 1

	checkbutton $w.ui.roi -text "Show 2D Crop Region" -anchor w \
	    -variable $UnuCrop-show_roi -command "$this update_crop_roi $which"
	pack $w.ui.roi -side top -fill x -anchor w -pady 2 -expand 1 

	for {set i 0} {$i < $dimension} {incr i} {
	    upvar \#0 $UnuCrop-minAxis$i min $UnuCrop-maxAxis$i max

            trace variable min w "$this ui_crop_trace"
            trace variable max w "$this ui_crop_trace"

	    set f $w.ui.$i
	    frame $f
	    pack $f -side top -anchor nw -expand yes -fill x

	    label $f.minl
	    iwidgets::spinner $f.minv -textvariable $UnuCrop-minAxis$i \
	        -increment "$this change_crop $UnuCrop min $i 1" \
		-decrement "$this change_crop $UnuCrop min $i -1" \
		-validate "$this change_crop $UnuCrop min $i 0 %P" -width 4
	    label $f.maxl
	    iwidgets::spinner $f.maxv -textvariable $UnuCrop-maxAxis$i \
	        -increment "$this change_crop $UnuCrop max $i 1" \
		-decrement "$this change_crop $UnuCrop max $i -1" \
		-validate "$this change_crop $UnuCrop max $i 0 %P" -width 4

	    set r [expr $i+1]
            grid configure $f.minl -row $r -column 0 -sticky w -padx 2
            grid configure $f.minv -row $r -column 1 -sticky e -padx 2
            grid configure $f.maxl -row $r -column 2 -sticky w -padx 2
            grid configure $f.maxv -row $r -column 3 -sticky e -padx 2
	}

        # Configure labels
        $w.ui.0.minl configure -text "Right:" -width 7 -anchor w
        $w.ui.0.maxl configure -text "Left:" -width 7 -anchor w

        $w.ui.1.minl configure -text "Posterior:" -width 7 -anchor w
        $w.ui.1.maxl configure -text "Anterior:" -width 7 -anchor w

        $w.ui.2.minl configure -text "Inferior:" -width 7 -anchor w
        $w.ui.2.maxl configure -text "Superior:" -width 7 -anchor w

	return $history.$which
    }

    method change_crop { crop kind axis amount { newval "" } } {
	global mods
	set varname $crop-${kind}Axis$axis
	set oppvar $crop-[expr ("$kind"=="min")?"max":"min"]Axis$axis
		     
	upvar \#0 $varname var $oppvar opp $mods(ViewSlices)-dim$axis max
	if { ![string length $newval] } {
	    set newval [expr $var+$amount]
	}

	if { ($amount == 0 && ![string is integer $newval]) || \
		 ($kind == "min" && ($newval > $opp)) || \
		 ($kind == "max" && ($newval < $opp)) } {
	    return 0
	} elseif { $newval < 0 } {
	    setGlobal $varname 0
	    return 0
	} elseif { $newval >= $max } {
	    setGlobal $varname [expr $max-1]
	    return 0
	} elseif { $amount != 0 } {
	    setGlobal $varname $newval
	}
	return 1
    }
	
	
    method viewslices_crop_trace { varname args } {
	global mods

	if {$current_crop == -1} return
	if {[lindex $filters($current_crop) $filter_type] != "crop"} return
	
	set UnuCrop [lindex [lindex $filters($current_crop) $modules] 0]
	set axis [lindex [split $varname _] end]
	upvar \#0 $varname val
	# disables trace to not call crop_ui_trace
	set cache_crop $current_crop
	set current_crop -1
#	puts "setGlobal $UnuCrop-$axis $val"
	setGlobal $UnuCrop-$axis $val
	set current_crop $cache_crop
	enable_update $current_crop
    }

    method ui_crop_trace { varname args } {
	if {$current_crop == -1} return
	if {[lindex $filters($current_crop) $filter_type] != "crop"} return
	
	set UnuCrop [lindex [lindex $filters($current_crop) $modules] 0]
	set axis [lindex [split $varname -] end]
	global mods
	upvar \#0 $varname val
	# This aint tobacco were dealin with here, folks.
	set cache_crop $current_crop
	set current_crop -1
#	puts "setGlobal $mods(ViewSlices)-crop_$axis $val"
	setGlobal $mods(ViewSlices)-crop_$axis $val
	$mods(ViewSlices)-c updatecrop
	set current_crop $cache_crop
	enable_update $current_crop
    }


    method add_Cmedian_UI {history row which} {
	frame $history.$which
	grid config $history.$which -column 0 -row $row -pady 0 -sticky news

	iwidgets::labeledframe $history.$which.f$which \
	    -labeltext "Median Filtering" \
	    -labelpos nw \
	    -borderwidth 2 
	grid config $history.$which.f$which -column 0 -row 0 -sticky news

	set w [$history.$which.f$which childsite]

	frame $w.expand
	pack $w.expand -side top -anchor nw

	global expandimg close_img
	button $w.expand.b -image $expandimg \
	    -anchor nw \
	    -command "$this change_visibility $which" \
	    -relief flat
        Tooltip $w.expand.b "Click to minimize/show\nthe Median Filtering UI"
	label $w.expand.l -text "Median Filtering - Unknown" -width $label_width \
	    -anchor nw
	Tooltip $w.expand.l "Right click to edit label"

 	button $w.expand.c -image $close_img \
 	    -anchor nw \
 	    -command "$this filter_Delete $which" \
 	    -relief flat

	Tooltip $w.expand.c "Click to delete this filter from\nthe pipeline. All settings\nfor the filter will be lost."

	pack $w.expand.b $w.expand.l $w.expand.c -side left -anchor nw 

	bind $w.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
	frame $w.ui
	pack $w.ui -side top -anchor nw -expand yes -fill x
	
        set UnuCmedian [lindex [lindex $filters($which) $modules] 0]
	global $UnuCmedian-radius
        trace variable $UnuCmedian-radius w "$this enable_update $which"

	frame $w.ui.radius
	pack $w.ui.radius -side top -anchor nw -expand yes -fill x
	label $w.ui.radius.l -text "Radius:"
	entry $w.ui.radius.v -textvariable $UnuCmedian-radius \
	    -width 6

	pack $w.ui.radius.l $w.ui.radius.v -side left -anchor nw \
	    -expand yes -fill x

	return $history.$which
    }

    method add_Histo_UI {history row which} {
	frame $history.$which
       	grid config $history.$which -column 0 -row $row -pady 0 -sticky news
	set updatecmd "$this enable_update $which"

	iwidgets::labeledframe $history.$which.f$which \
	    -labeltext "Histogram" \
	    -labelpos nw \
	    -borderwidth 2 
	grid config $history.$which.f$which -column 0 -row 0 -sticky news

	set w [$history.$which.f$which childsite]

	frame $w.expand
	pack $w.expand -side top -anchor nw


	global expandimg close_img
	button $w.expand.b -image $expandimg \
	    -anchor nw \
	    -command "$this change_visibility $which" \
	    -relief flat
	label $w.expand.l -text "Histogram - Unknown" -width $label_width \
	    -anchor nw

 	button $w.expand.c -image $close_img \
 	    -anchor nw \
 	    -command "$this filter_Delete $which" \
 	    -relief flat

	Tooltip $w.expand.c "Click to delete this filter from\nthe pipeline. All settings\nfor the filter will be lost."

	pack $w.expand.b $w.expand.l $w.expand.c -side left -anchor nw 

	bind $w.expand.l <ButtonPress-3> "$this change_label %X %Y $which"
	
	frame $w.ui
	pack $w.ui -side top -anchor nw -expand yes -fill x
	

        set UnuHeq  [lindex [lindex $filters($which) $modules] 0]
        set UnuQuantize [lindex [lindex $filters($which) $modules] 2]
        set ScalarFieldStats [lindex [lindex $filters($which) $modules] 3]

        ### Histogram
        iwidgets::labeledframe $w.ui.histo \
            -labelpos nw -labeltext "Histogram"
 	pack $w.ui.histo -side top -fill x -anchor n -expand 1

 	set histo [$w.ui.histo childsite]

        global $ScalarFieldStats-min
	global $ScalarFieldStats-max
	global $ScalarFieldStats-nbuckets

	frame $histo.row1
	pack $histo.row1 -side top

	blt::barchart $histo.graph -title "" \
	    -height [expr [set $ScalarFieldStats-nbuckets]*3/5.0] \
	    -width 200 -plotbackground gray80
        pack $histo.graph

	global $UnuHeq-amount
        trace variable $UnuHeq-amount w $updatecmd
      
        if {!$loading_ui} {
   	    set $UnuHeq-amount 1.0
        }

	scale $w.ui.amount -label "Amount" \
	    -from 0.0 -to 1.0 \
	    -variable $UnuHeq-amount \
	    -showvalue true \
	    -orient horizontal \
	    -resolution 0.01
	pack $w.ui.amount -side top -anchor nw -expand yes -fill x

        global $UnuQuantize-minf $UnuQuantize-maxf
        global $UnuQuantize-useinputmin $UnuQuantize-useinputmax
        trace variable $UnuQuantize-minf w $updatecmd
        trace variable $UnuQuantize-maxf w $updatecmd
        trace variable $UnuQuantize-useinputmin w $updatecmd
        trace variable $UnuQuantize-useinputmax w $updatecmd

        frame $w.ui.min -relief groove -borderwidth 2
	pack $w.ui.min -side top -expand yes -fill x

        iwidgets::entryfield $w.ui.min.v -labeltext "Min:" \
	    -textvariable $UnuQuantize-minf
        pack $w.ui.min.v -side top -expand yes -fill x

        checkbutton $w.ui.min.useinputmin \
	    -text "Use lowest value of input nrrd as min" \
	    -variable $UnuQuantize-useinputmin
        pack $w.ui.min.useinputmin -side top -expand yes -fill x

	frame $w.ui.max -relief groove -borderwidth 2
	pack $w.ui.max -side top -expand yes -fill x

        iwidgets::entryfield $w.ui.max.v -labeltext "Max:" \
	    -textvariable $UnuQuantize-maxf
        pack $w.ui.max.v -side top -expand yes -fill x

        checkbutton $w.ui.max.useinputmax \
	    -text "Use highest value of input nrrd as max" \
	    -variable $UnuQuantize-useinputmax
        pack $w.ui.max.useinputmax -side top -expand yes -fill x
	
	return $history.$which
    }


    method connect_filter_module_to_choose { mod } {
	global Subnet
  	set choose 0
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis]
	# all module connections
        foreach conn $Subnet(${ChooseNrrd}_connections) { 
  	    if { [lindex $conn 2] == $ChooseNrrd } {
		incr choose
  	    }
  	}
	addConnection $mod 0 $ChooseNrrd $choose
  	return $choose
    }


    method filter_Delete {which} {
	# Do not remove Load (0)
	if {$which == 0} return
	set filter $filters($which)
	# remove ui
	grid remove $history0.$which
	grid remove $history1.$which
        # delete filter modules
	foreach mod [lindex $filter $modules] {
	    moduleDestroy $mod
	}
        # update choose ports of other filters
        $this update_choose_ports [lindex $filter $choose_port]
	# update the Previous and Next filters to be neighbors
	set prev [lindex $filter $prev_index]
	set next [lindex $filter $next_index]	
	set filters($prev) \
	    [lreplace $filters($prev) $next_index $next_index $next]
	if {$next != "end"} {
	    move_up_filters [lindex $filter $which_row]
	    set filters($next) \
		[lreplace $filters($next) $prev_index $prev_index $prev]
	    # patch connections from previous filter to next filter
	    foreach {pmod pport} [lindex $filters($prev) $output] {
		foreach {nmod nport} [lindex $filters($next) $input] {
		    addConnection $pmod $pport $nmod $nport
		}
	    }
	    $this enable_update $prev
	} else {
	    setGlobal eye $prev
	}
	# Delete Filter Information
	set filters($which) { {} {} {} {} -1 -1 -1 -1 0 {} }
	incr grid_rows -1
	change_eye 1
	arrange_filter_modules

    }

    method update_changes {} {
	global eye
	if { $last_filter_changed == [expr $eye+1] } {
	    set eye $last_filter_changed
	    change_eye 0
	}

	if { !$has_executed } {
	    $this set_viewer_position
	    $this execute_Data
	    $this disable_update
	    return
	}


	if {$grid_rows == 1} {
	    $this execute_Data
	} else {
	    # find first valid filter and execute that
	    for {set i 1} {$i < $num_filters} {incr i} {
		if {[info exists filters($i)]} {
		    set tmp_row [lindex $filters($i) $which_row]
		    if {$tmp_row != -1} {
			set mod [lindex [lindex $filters($i) $input] 0]
			break
		    }
		}
	    }
	    $mod-c needexecute
	}
	
	$this disable_update
    }

    method disable_update { args } {
	set needs_update 0
        # grey out update button
        $attachedPFr.f.p.update configure -background "grey75" -state disabled
        $detachedPFr.f.p.update configure -background "grey75" -state disabled
    }

    method enable_update { which args } {
	set needs_update 1
	set last_filter_changed $which
        # fix  update button
        $attachedPFr.f.p.update configure -background "#008b45" -state normal
        $detachedPFr.f.p.update configure -background "#008b45" -state normal
    }

    method move_down_filters {row} {
	# Since we are inserting, we need to forget the grid rows
	# below us and move them down a row
	set re_pack [list]
	for {set i 1} {$i < $num_filters} {incr i} {
	    if { [info exists filters($i)] } {
		set tmp_row [lindex $filters($i) $which_row]
		if { $tmp_row >= $row } {
		    grid forget $history0.$i
		    grid forget $history1.$i
		    
		    set filters($i) [lreplace $filters($i) $which_row $which_row [expr $tmp_row + 1] ]		    
		    lappend re_pack $i
		}
	    }
	}
	# need to re configure them after they have all been removed
	for {set i 0} {$i < [llength $re_pack]} {incr i} {
	    set index [lindex $re_pack $i]
	    set new_row [lindex $filters($index) $which_row]
            grid config $history0.$index -row $new_row -column 0
            grid config $history1.$index -row $new_row -column 0
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
		    grid forget $history0.$i
		    grid forget $history1.$i
		    set filters($i) [lreplace $filters($i) $which_row $which_row [expr $tmp_row - 1] ]		    
		    lappend re_pack $i
		}
	    }
	}
	# need to re configure them after they have all been removed
	for {set i 0} {$i < [llength $re_pack]} {incr i} {
	    set index [lindex $re_pack $i]
	    set new_row [lindex $filters($index) $which_row]
            grid config $history0.$index -row $new_row -column 0
            grid config $history1.$index -row $new_row -column 0
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

    method change_eye { {execute 1 } } {
	global eye
	set ChooseNrrd [lindex [lindex $filters(0) $modules] $load_choose_vis] 
	setGlobal $ChooseNrrd-port-index [lindex $filters($eye) $choose_port]
	if { $execute } {
	    $ChooseNrrd-c needexecute
	}
	update_crop_roi [lindex $filters($eye) $next_index]
    }

    method change_label {x y which} {
	if {![winfo exists .standalone.change_label]} {
	    # bring up ui to type name
	    global new_label
	    set old_label [$history0.$which.f$which.childsite.expand.l cget -text]
	    set new_label $old_label
	    
	    toplevel .standalone.change_label
	    wm minsize .standalone.change_label 150 50
            wm title .standalone.change_label "Change Label"
	    set x [expr $x + 10]
	    wm geometry .standalone.change_label "+$x+$y"
	    
	    label .standalone.change_label.l -text "Please enter a label for this filter."
	    pack .standalone.change_label.l -side top -anchor nw -padx 4 -pady 4
	    
	    frame .standalone.change_label.info
	    pack .standalone.change_label.info -side top -anchor nw
	    
	    label .standalone.change_label.info.l -text "Label:"
	    entry .standalone.change_label.info.e -textvariable new_label 
	    pack .standalone.change_label.info.l .standalone.change_label.info.e -side left -anchor nw \
		-padx 4 -pady 4
	    bind .standalone.change_label.info.e <Return> "destroy .standalone.change_label"

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
		$history0.$which.f$which.childsite.expand.l configure -text $new_label
		$history1.$which.f$which.childsite.expand.l configure -text $new_label
		set filters($which) [lreplace $filters($which) $filter_label $filter_label $new_label]
	    }
	} else {
	    SciRaise .standalone.change_label
	}
    }

    method change_visibility {num} {
	# v represents the new visibilty state
	set v [expr ![lindex $filters($num) $visibility]]
	# set the new visibility state in the filter
	set filters($num) [lreplace $filters($num) $visibility $visibility $v]

	global expandimg play_img
	set subf $num.f$num.childsite
	if {!$v} { ;# hide
	    $history0.$subf.expand.b configure -image $play_img
	    $history1.$subf.expand.b configure -image $play_img
	    pack forget $history0.$subf.ui 
	    pack forget $history1.$subf.ui 
	    set filters($num) \
		[lreplace $filters($num) $visibility $visibility 0]
	} else { ;# show
	    $history0.$subf.expand.b configure -image $expandimg
	    $history1.$subf.expand.b configure -image $expandimg
	    pack $history0.$subf.ui -side top -expand yes -fill both
	    pack $history1.$subf.ui -side top -expand yes -fill both
	}
    }


    ##################################
    #### change_kernel
    ##################################
    # Update the resampling kernel variable and
    # update the other attached/detached optionmenu
    method change_kernel { w which } {
	set UnuResample [lindex [lindex $filters($which) $modules] 0]
	set num [$w get]
	if {$num == "Box"} {
	    setGlobal $UnuResample-filtertype box
	} elseif {$num == "Tent"} {
	    setGlobal $UnuResample-filtertype tent
	} elseif {$num == "Cubic (Catmull-Rom)"} {
	    setGlobal $UnuResample-filtertype cubicCR
	} elseif {$num == "Cubic (B-spline)"} {
	    setGlobal $UnuResample-filtertype cubicBS
	} elseif {$num == "Quartic"} {
	    setGlobal $UnuResample-filtertype quartic
	} elseif {$num == "Gaussian"} {
	    setGlobal $UnuResample-filtertype gaussian
	}

        $this enable_update $which

	# update attach/detach one
        $history0.$which.f$which.childsite.ui.kernel select $num
	$history1.$which.f$which.childsite.ui.kernel select $num

    }

    method update_kernel { num } {
	set UnuResample [lindex [lindex $filters($num) $modules] 0]
        upvar \#0 $UnuResample-filtertype f
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

        $history0.$num.f$num.childsite.ui.kernel select $t
        $history1.$num.f$num.childsite.ui.kernel select $t
    }

    method make_entry {w text v num} {
        frame $w
        label $w.l -text "$text" 
        pack $w.l -side left
        global $v
        entry $w.e -textvariable $v 
        pack $w.e -side right
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

    ##############################
    ### save_session
    ##############################
    # To be filled in by child class. It should save out a session
    # for the specific app.
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
		puts $fileid "app set_saved_class_var $var \{[set $var]\}"
	    }
	}
	
	# print out arrays
	for {set i 0} {$i < $num_filters} {incr i} {
	    if {[info exists filters($i)]} {
		puts $fileid "app set_saved_class_var filters($i) \{[set filters($i)]\}"
	    }
	}

        # save globals
        global eye
        puts $fileid "global eye"
        puts $fileid "set eye \{[set eye]\}"

        global show_guidelines
        puts $fileid "global show_guidelines"
        puts $fileid "set show_guidelines \{[set show_guidelines]\}"

        global top
        puts $fileid "global top"
        puts $fileid "set top \{[set top]\}"

        global front
        puts $fileid "global front"
        puts $fileid "set front \{[set front]\}"

        global side
        puts $fileid "global side"
        puts $fileid "set side \{[set side]\}"

        global planes_mapType
        puts $fileid "global planes_mapType"
        puts $fileid "set planes_mapType \{[set planes_mapType]\}"

        global show_vol_ren
        puts $fileid "global show_vol_ren"
        puts $fileid "set show_vol_ren \{[set show_vol_ren]\}"

        global link_winlevel
        puts $fileid "global link_winlevel"
        puts $fileid "set link_winlevel \{[set link_winlevel]\}"

        global vol_width
        puts $fileid "global vol_width"
        puts $fileid "set vol_width \{[set vol_width]\}"

        global vol_level
        puts $fileid "global vol_level"
        puts $fileid "set vol_level \{[set vol_level]\}"

	puts $fileid "app set_saved_class_var loading 1"
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
       # Clear all modules
       ClearCanvas 0
       
       # configure title
       wm title .standalone "BioImage - [getFileName $saveFile]" 
       
       # remove all UIs from history
       for {set i 0} {$i < $num_filters} {incr i} {
	   if {[info exists filters($i)]} {
               set tmp_row [lindex $filters($i) $which_row]
               if {$tmp_row != -1 } {
		   destroy $history0.$i
		   destroy $history1.$i
	       }
           }
       }

       # justify scroll region
       $attachedPFr.f.p.sf justify top
       $detachedPFr.f.p.sf justify top

       #destroy 2D viewer windows and control panel
       destroy $win.viewers.cp
       destroy $win.viewers.topbot

       # load new net
       foreach g [info globals] {
	   global $g
       }

       update

       # source at the global level for module settings
       uplevel \#0 source \{$saveFile\}

       # save out ViewSlices and Viewer variables that will be overwritten
       # two modules are destroyed and recreated and reset their state
       
       global $mods(ViewSlices)-axial-viewport0-mode
       global $mods(ViewSlices)-sagittal-viewport0-mode
       global $mods(ViewSlices)-coronal-viewport0-mode
       global $mods(ViewSlices)-axial-viewport0-slice
       global $mods(ViewSlices)-sagittal-viewport0-slice
       global $mods(ViewSlices)-coronal-viewport0-slice
       global $mods(ViewSlices)-axial-viewport0-slab_min
       global $mods(ViewSlices)-sagittal-viewport0-slab_min
       global $mods(ViewSlices)-coronal-viewport0-slab_min
       global $mods(ViewSlices)-axial-viewport0-slab_max
       global $mods(ViewSlices)-sagittal-viewport0-slab_max
       global $mods(ViewSlices)-coronal-viewport0-slab_max
       global $mods(ViewSlices)-clut_ww
       global $mods(ViewSlices)-clut_wl
       global $mods(ViewSlices)-min $mods(ViewSlices)-max
       global $mods(ViewSlices)-axial-viewport0-axis
       global $mods(ViewSlices)-sagittal-viewport0-axis
       global $mods(ViewSlices)-coronal-viewport0-axis

       set axial_mode [set $mods(ViewSlices)-axial-viewport0-mode]
       set sagittal_mode [set $mods(ViewSlices)-sagittal-viewport0-mode]
       set coronal_mode [set $mods(ViewSlices)-coronal-viewport0-mode]
       set axial_slice [set $mods(ViewSlices)-axial-viewport0-slice]
       set sagittal_slice [set $mods(ViewSlices)-sagittal-viewport0-slice]
       set coronal_slice [set $mods(ViewSlices)-coronal-viewport0-slice]
       set axial_slab_min [set $mods(ViewSlices)-axial-viewport0-slab_min]
       set sagittal_slab_min [set $mods(ViewSlices)-sagittal-viewport0-slab_min]
       set coronal_slab_min [set $mods(ViewSlices)-coronal-viewport0-slab_min]
       set axial_slab_max [set $mods(ViewSlices)-axial-viewport0-slab_max]
       set sagittal_slab_max [set $mods(ViewSlices)-sagittal-viewport0-slab_max]
       set coronal_slab_max [set $mods(ViewSlices)-coronal-viewport0-slab_max]
       set ww [set $mods(ViewSlices)-clut_ww]
       set wl [set $mods(ViewSlices)-clut_wl]

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

       set eyepx [set $mods(Viewer)-ViewWindow_0-view-eyep-x] 
       set eyepy [set $mods(Viewer)-ViewWindow_0-view-eyep-y] 
       set eyepz [set $mods(Viewer)-ViewWindow_0-view-eyep-z] 
       set lookx [set $mods(Viewer)-ViewWindow_0-view-lookat-x]
       set looky [set $mods(Viewer)-ViewWindow_0-view-lookat-y]
       set lookz [set $mods(Viewer)-ViewWindow_0-view-lookat-z]
       set upx [set $mods(Viewer)-ViewWindow_0-view-up-x]
       set upy [set $mods(Viewer)-ViewWindow_0-view-up-y]
       set upz [set $mods(Viewer)-ViewWindow_0-view-up-z]
       set fov [set $mods(Viewer)-ViewWindow_0-view-fov]

       set loading_ui 1
       set last_valid 0
       set data_tab $cur_data_tab
		
       # iterate over filters array and create UIs
	    for {set i 0} {$i < $num_filters} {incr i} {
		# only build ui for those with a row
		# value not -1
		set status [lindex $filters($i) $which_row]
                set p $i.f$i
		if {$status != -1} {
		    set t [lindex $filters($i) $filter_type]
		    set v [lindex $filters($i) $visibility]
		    set l [lindex $filters($i) $filter_label]
		    set last_valid $i
		    
		    if {[string equal $t "load"]} {
			set f [add_Load_UI $history0 $status $i]
                        $this add_insert_bar $f $i
			set f [add_Load_UI $history1 $status $i]
                        $this add_insert_bar $f $i
			if {$v == 0} {
			    set filters($i) [lreplace $filters($i) $visibility $visibility 1]
			    $this change_visibility $i
			}
		    } elseif {[string equal $t "resample"]} {
			set f [add_Resample_UI $history0 $status $i]
                        $this add_insert_bar $f $i
			set f [add_Resample_UI $history1 $status $i]
                        $this add_insert_bar $f $i
			$this update_kernel $i
			if {$v == 0} {
			    set filters($i) [lreplace $filters($i) $visibility $visibility 1]
			    $this change_visibility $i
			}
		    } elseif {[string equal $t "crop"]} {
			set f [add_Crop_UI $history0 $status $i]
                        $this add_insert_bar $f $i
			set f [add_Crop_UI $history1 $status $i]
                        $this add_insert_bar $f $i
			if {$v == 0} {
			    set filters($i) [lreplace $filters($i) $visibility $visibility 1]
			    $this change_visibility $i
			}
		    } elseif {[string equal $t "cmedian"]} {
			set f [add_Cmedian_UI $history0 $status $i]
                        $this add_insert_bar $f $i
			set f [add_Cmedian_UI $history1 $status $i]
                        $this add_insert_bar $f $i
			if {$v == 0} {
			    set filters($i) [lreplace $filters($i) $visibility $visibility 1]
			    $this change_visibility $i
			}
		    } elseif {[string equal $t "histo"]} {
			set f [add_Histo_UI $history0 $status $i]
                        $this add_insert_bar $f $i
			set f [add_Histo_UI $history1 $status $i]
                        $this add_insert_bar $f $i
			if {$v == 0} {
			    set filters($i) [lreplace $filters($i) $visibility $visibility 1]
			    $this change_visibility $i
			}
		    } else {
			puts "Error: Unknown filter type - $t"
		    }

                    # fix label
		    $history0.$p.childsite.expand.l configure -text $l
		    $history1.$p.childsite.expand.l configure -text $l

		    $history0.$p configure -background grey75 -foreground black -borderwidth 2
		    $history1.$p configure -background grey75 -foreground black -borderwidth 2
            }
	}


        set loading_ui 0

 	# set a few variables that need to be reset
 	set indicate 0
 	set cycle 0
 	set IsPAttached 1
 	set IsVAttached 1
 	set executing_modules 0

 	$indicatorL0 configure -text "Press Update to run to save point..."
 	$indicatorL1 configure -text "Press Update to run to save point..."

        # update components using globals
        $this update_orientations
        $this update_planes_color_by
        $this change_volume_window_width_and_level 0
        $this toggle_show_guidelines

        # bring proper tabs forward
        set cur_data_tab $data_tab
        $attachedPFr.f.p.sf.lwchildsite.clipper.canvas.sfchildsite.0.f0.childsite.ui.tnb view $cur_data_tab
        $detachedPFr.f.p.sf.lwchildsite.clipper.canvas.sfchildsite.0.f0.childsite.ui.tnb view $cur_data_tab
        $this change_vis_frame $c_vis_tab

        # rebuild the viewer windows
        $this build_viewers $mods(Viewer) $mods(ViewSlices)

        # configure slice/mip sliders
        global slice_frame
        foreach axis "sagittal coronal axial" {
            $slice_frame($axis).modes.slider.slice.s configure -from 0 -to [set $axis-size]
            $slice_frame($axis).modes.slider.slab.s configure -from 0 -to [set $axis-size]
        }

        # reset saved ViewSlices variables
        set $mods(ViewSlices)-clut_ww $ww
        set $mods(ViewSlices)-clut_wl $wl

        set $mods(ViewSlices)-axial-viewport0-mode $axial_mode
        set $mods(ViewSlices)-axial-viewport0-slice $axial_slice
        set $mods(ViewSlices)-axial-viewport0-slab_min $axial_slab_min
        set $mods(ViewSlices)-axial-viewport0-slab_max $axial_slab_max
        set $mods(ViewSlices)-axial-viewport0-axis 2

        set $mods(ViewSlices)-sagittal-viewport0-mode $sagittal_mode
        set $mods(ViewSlices)-sagittal-viewport0-slice $sagittal_slice
        set $mods(ViewSlices)-sagittal-viewport0-slab_min $sagittal_slab_min
        set $mods(ViewSlices)-sagittal-viewport0-slab_max $sagittal_slab_max
        set $mods(ViewSlices)-sagittal-viewport0-axis 0

        set $mods(ViewSlices)-coronal-viewport0-mode $coronal_mode
        set $mods(ViewSlices)-coronal-viewport0-slice $coronal_slice
        set $mods(ViewSlices)-coronal-viewport0-slab_min $coronal_slab_min
        set $mods(ViewSlices)-coronal-viewport0-slab_max $coronal_slab_max
        set $mods(ViewSlices)-coronal-viewport0-axis 1

        # make calls to set up ViewSlices settings properly
        $this update_ViewSlices_mode axial
        $this update_ViewSlices_mode coronal
        $this update_ViewSlices_mode sagittal

        # set viewer settings
       set $mods(Viewer)-ViewWindow_0-view-eyep-x $eyepx
       set $mods(Viewer)-ViewWindow_0-view-eyep-y $eyepy
       set $mods(Viewer)-ViewWindow_0-view-eyep-z $eyepz
       set $mods(Viewer)-ViewWindow_0-view-lookat-x $lookx
       set $mods(Viewer)-ViewWindow_0-view-lookat-y $looky
       set $mods(Viewer)-ViewWindow_0-view-lookat-z $lookz
       set $mods(Viewer)-ViewWindow_0-view-up-x $upx
       set $mods(Viewer)-ViewWindow_0-view-up-y $upy
       set $mods(Viewer)-ViewWindow_0-view-up-z $upz
       set $mods(Viewer)-ViewWindow_0-view-fov $fov

       set has_autoviewed 1
       set 2D_fixed 1
    }	

    method toggle_show_guidelines {} {
	global mods show_guidelines
	foreach axis {axial sagittal coronal} {
	    setGlobal $mods(ViewSlices)-$axis-viewport0-show_guidelines \
		$show_guidelines	    
	}
    }

    method update_planes_color_by {} {
        global planes_mapType
        set GenStandard [lindex [lindex $filters(0) $modules] 26]
	setGlobal $GenStandard-mapType $planes_mapType
        if {!$loading && $has_executed} {
	    $GenStandard-c needexecute
	}
    }

    method toggle_show_vol_ren {} {
	global mods show_vol_ren 

	set VolumeVisualizer [lindex [lindex $filters(0) $modules] 14]
	set UnuQuantize [lindex [lindex $filters(0) $modules] 8]
	set UnuJhisto [lindex [lindex $filters(0) $modules] 21]
        set EditColorMap2D [lindex [lindex $filters(0) $modules] 13]
        set NrrdTextureBuilder [lindex [lindex $filters(0) $modules] 11]
        set NrrdSetupTexture [lindex [lindex $filters(0) $modules] 10]
	set Rescale [lindex [lindex $filters(0) $modules] 36] 

        if {$show_vol_ren == 1} {
	    disableModule $VolumeVisualizer 0
	    disableModule $NrrdSetupTexture 0
	    disableModule $UnuQuantize 0
	    disableModule $UnuJhisto 0
	    disableModule $EditColorMap2D 0
	    disableModule $NrrdTextureBuilder 0
	    disableModule $mods(NrrdSelectTime_0) 0
	    disableModule $mods(NrrdSelectTime_1) 0

            change_indicator_labels "Volume Rendering..."
    	    [set Rescale]-c needexecute
            $NrrdSetupTexture-c needexecute
        } else {
	    disableModule $VolumeVisualizer 1
	    disableModule $NrrdSetupTexture 1
	    disableModule $UnuQuantize 1
	    disableModule $UnuJhisto 1
	    disableModule $EditColorMap2D 1
	    disableModule $NrrdTextureBuilder 1
	    disableModule $mods(NrrdSelectTime_0) 1
	    disableModule $mods(NrrdSelectTime_1) 1

        }
    }

    method update_ViewSlices_mode { axis args } {
	global mods slice_frame
        upvar \#0 $mods(ViewSlices)-$axis-viewport0-mode mode

        set w $slice_frame($axis)
        # forget and repack appropriate widget
	if {$mode == 0} {
	    # Slice mode
            pack forget $w.modes.slider.slab
            pack $w.modes.slider.slice -side top -anchor n -expand 1 -fill x
	} elseif {$mode == 1} {
	    # Slab mode
    	    pack forget $w.modes.slider.slice
            pack $w.modes.slider.slab -side top -anchor n -expand 1 -fill x
	} else {
	    # Full MIP mode
            pack forget $w.modes.slider.slice
  	    pack forget $w.modes.slider.slab
	}
        $mods(ViewSlices)-c rebind $w.bd.$axis
        $mods(ViewSlices)-c redrawall
    }

    method set_saved_class_var {var val} {
	set $var $val
    }

    method scroll_history {p which} {
	if {[string first "$attachedPFr.f.p.sf.lwchildsite" $p] != -1} {
            set x [lindex [$attachedPFr.f.p.sf xview] 0]
            set y [lindex [$attachedPFr.f.p.sf yview] 0]
            if {$which == "up"} {
		$attachedPFr.f.p.sf yview moveto [expr $y - 0.05]
		$detachedPFr.f.p.sf yview moveto [expr $y - 0.05]
	    } elseif {$which == "down"} {
		$attachedPFr.f.p.sf yview moveto [expr $y + 0.05]
		$detachedPFr.f.p.sf yview moveto [expr $y + 0.05]
	    } elseif {$which == "right"} {
		$attachedPFr.f.p.sf xview moveto [expr $x + 0.05]
		$detachedPFr.f.p.sf xview moveto [expr $x + 0.05]
	    } else {
		$attachedPFr.f.p.sf xview moveto [expr $x - 0.05]
		$detachedPFr.f.p.sf xview moveto [expr $x - 0.05]
	    }

        } elseif {[string first "$detachedPFr.f.p.sf.lwchildsite" $p] != -1} {
            set x [lindex [$detachedPFr.f.p.sf xview] 0]
            set y [lindex [$detachedPFr.f.p.sf yview] 0]
            if {$which ==" up"} {                
		$attachedPFr.f.p.sf yview moveto [expr $y - 0.0.5]
		$detachedPFr.f.p.sf yview moveto [expr $y - 0.0.5]
	    } elseif {$which == "down"} {
		$attachedPFr.f.p.sf yview moveto [expr $y + 0.0.5]
		$detachedPFr.f.p.sf yview moveto [expr $y + 0.0.5]
	    } elseif {$which == "right"} {
		$attachedPFr.f.p.sf xview moveto [expr $x + 0.05]
		$detachedPFr.f.p.sf xview moveto [expr $x + 0.05]
	    } else {
		$attachedPFr.f.p.sf xview moveto [expr $x - 0.05]
		$detachedPFr.f.p.sf xview moveto [expr $x - 0.05]
	    }
        }
    }
    

    method maybe_autoview { varname args } {
	upvar \#0 $varname autoview
	if { $autoview } {
	    global mods
	    set var $mods(Viewer)-ViewWindow_0-total_frames
	    global $var
	    trace variable $var w "$this autoview"
	}
    }
    method autoview { varname args } {
	global mods
	$mods(Viewer)-ViewWindow_0-c autoview
	set var $mods(Viewer)-ViewWindow_0-total_frames
	global $var
	trace vdelete $var w "$this autoview"

    }

    method find_last_filter { } {
	set prev 0
	set cur [lindex $filters($prev) $next_index]
	while {$cur != "end"} {
	    set prev $cur
	    set cur [lindex $filters($cur) $next_index]
	}
	return $prev
    }

    method insert_filter { cur } {
	if { [lindex $filters($cur) $next_index] == "end"} return
	set p [lindex $filters($cur) $prev_index] 
	set n [lindex $filters($cur) $next_index] 
	# if inserting, disconnect current to current's next 
	# and connect current to new and new to current's next
	set cur_mod [lindex [lindex $filters($cur) $output] 0]
	set cur_port [lindex [lindex $filters($cur) $output] 1]
	set prev_mod [lindex [lindex $filters($p) $output] 0]
	set prev_port [lindex [lindex $filters($p) $output] 1]
	foreach {mod port} [lindex $filters($n) $input] {
	    destroyConnection "$prev_mod $prev_port $mod $port"
	    addConnection $cur_mod $cur_port $mod $port
	}
	# set the row for cur filter to be the row of the next filter
	set row [lindex $filters($n) $which_row]
	set filters($cur) [lreplace $filters($cur) $which_row $which_row $row]
	# move all filters after this one down a row
	$this move_down_filters $row
	# set the next filters previous index to point to current
	set filters($n) [lreplace $filters($n) $prev_index $prev_index $cur]
    }


    method create_filter_UI { cur } {
	arrange_filter_modules
	set row [lindex $filters($cur) $which_row]

	# Make current frame regular
	set type [string totitle [lindex $filters($cur) $filter_type]]
	foreach h "$history0 $history1" {
	    set f [add_${type}_UI $h $row $num_filters]
	    $this add_insert_bar $f $num_filters
	}
	incr grid_rows	
        if { [lindex $filters($cur) $next_index] == "end" } {
	    $attachedPFr.f.p.sf justify bottom
	    $detachedPFr.f.p.sf justify bottom
	}
    }

    method arrange_filter_modules {} {
	global Subnet
	set canvas $Subnet(Subnet0_canvas)
	set i 0
	set cur [lindex $filters(0) $next_index]
	while {$cur != "end"} {
	    set mods [lindex $filters($cur) $modules]
	    for {set m 0} {$m < [llength $mods]} {incr m} {
		set mod [lindex $mods $m]
		set bbox [$canvas bbox $mod]
		set x [expr 500 + $i*20 +($m%2)*200-[lindex $bbox 0]]
		set y [expr 100 + $i*160 +($m/2)*80-[lindex $bbox 1]]
		if {$x != 0 || $y != 0 } {
		    $canvas move $mod $x $y
		    drawConnections $Subnet(${mod}_connections)
		}
	    }
	    set cur [lindex $filters($cur) $next_index]
	    incr i
	}
    }
  
    method set_viewer_position {} {
 	global mods
 	set vw $mods(Viewer)-ViewWindow_0
    	setGlobal $vw-view-eyep-x {560.899236544}
	setGlobal $vw-view-eyep-y {356.239586973}
	setGlobal $vw-view-eyep-z {178.810334192}
	setGlobal $vw-view-lookat-x {51.5}
	setGlobal $vw-view-lookat-y {47.0}
	setGlobal $vw-view-lookat-z {80.5}
	setGlobal $vw-view-up-x {-0.181561715965}
	setGlobal $vw-view-up-y {0.0242295849764}
	setGlobal $vw-view-up-z {0.983081009128}
	setGlobal $vw-view-fov {20.0}
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

    variable history0
    variable history1

    variable dimension

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
    variable filter_label
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

    variable has_autoviewed
    variable has_executed

    variable data_dir
    variable 2D_fixed
    variable ViewSlices_executed_on_error
    variable last_filter_changed
    variable current_crop
    variable needs_update

    variable cur_data_tab
    variable c_vis_tab

    variable axial-size
    variable sagittal-size
    variable coronal-size

    variable execute_choose
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

bind all <ButtonPress-5>  "app scroll_history %W down"
bind all <ButtonPress-4>  "app scroll_history %W up"
bind all <KeyPress-Down>  "app scroll_history %W down"
bind all <KeyPress-Up>  "app scroll_history %W up"
bind all <KeyPress-Right>  "app scroll_history %W right"
bind all <KeyPress-Left>  "app scroll_history %W left"

