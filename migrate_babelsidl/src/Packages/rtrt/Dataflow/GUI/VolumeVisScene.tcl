#
#  RTRTViewer.tcl
#
#  Written by:
#   James Bigler
#   Department of Computer Science
#   University of Utah
#   April 2002
#
#  Copyright (C) 2002 SCI Group
#

itcl_class rtrt_Scenes_VolumeVisScene {
    inherit Module

    constructor {config} {
	set name VolumeVisScene
	set_defaults
    }
    
    method set_defaults {} {
	global $this-scene_type_gui
	set $this-scene_type_gui 6
	global $this-data_file_gui
#	set $this-data_file_gui "/local/csafe/raid1/bigler/data/NRRD/two.list"
	set $this-data_file_gui "/local/csafe/raid1/bigler/nasa/weather/qc_001500.bob.nhdr"
	global $this-do_phong_gui
	set $this-do_phong_gui 1
	global $this-ncolors_gui
	set $this-ncolors_gui 256
	global $this-t_inc_gui
	set $this-t_inc_gui 0.01
	global $this-spec_coeff_gui
	set $this-spec_coeff_gui 64 
	global $this-ambient_gui
	set $this-ambient_gui 0.5
	global $this-diffuse_gui 
	set $this-diffuse_gui 1.0
	global $this-specular_gui
	set $this-specular_gui 1.0
	global $this-val_gui
	set $this-val_gui 1.0
	global $this-override_data_min_gui
	set $this-override_data_min_gui 0
	global $this-override_data_max_gui
	set $this-override_data_max_gui 0
	global $this-data_min_in_gui
	set $this-data_min_in_gui 0
	global $this-data_max_in_gui
	set $this-data_max_in_gui 0
	global $this-frame_rate_gui
	set $this-frame_rate_gui 3
#	global $this-
#	set $this-
    }
    # returns 1 if the window is visible, 0 otherwise
    method isVisible {} {
	if {[winfo exists .ui[modname]]} {
	    return 1
	} else {
	    return 0
	}
    }
    # this makes a text box, to make it uneditable do entry ... -state disabled
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v
	bind $w.e <Return> $c
	pack $w.e -side right
    }

    # this is just a dummy function that does nothing
    method do_nothing {} {
    }

    # this is the main function which creates the initial window
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 50
	set n "$this-c needexecute "
	set nothing "$this do_nothing"

	frame $w.options
	pack $w.options  -side top -fill x -padx 2 -pady 2

	# add data file
	make_entry $w.options.data_file "Data file" $this-data_file_gui \
	    nothing
	pack $w.options.data_file -side top -fill x -padx 2 -pady 2

	# add 
#	make_entry $w.options. "" $this-nworkers nothing
#	pack $w.options. -side top -fill x -padx 2 -pady 2

	# add frame rate
	set ratechange "$this-c rate_change"
	make_entry $w.options.frame_rate "Frame Rate" $this-frame_rate_gui \
	    "$this-c rate_change"
	pack $w.options.frame_rate -side top -fill x -padx 2 -pady 2

	menubutton $w.options.scene_type -text "Scene Type" \
	    -menu $w.options.scene_type.list -relief groove
	pack $w.options.scene_type -side top -anchor w -padx 2 -pady 2
	
	menu $w.options.scene_type.list
	$w.options.scene_type.list add radiobutton \
	    -variable $this-scene_type_gui \
	    -command "$this do_nothing" \
	    -label 0 \
	    -value 0
	$w.options.scene_type.list add radiobutton \
	    -variable $this-scene_type_gui \
	    -command "$this do_nothing" \
	    -label 1 \
	    -value 1
	$w.options.scene_type.list add radiobutton \
	    -variable $this-scene_type_gui \
	    -command "$this do_nothing" \
	    -label 2 \
	    -value 2
	$w.options.scene_type.list add radiobutton \
	    -variable $this-scene_type_gui \
	    -command "$this do_nothing" \
	    -label 3 \
	    -value 3
	$w.options.scene_type.list add radiobutton \
	    -variable $this-scene_type_gui \
	    -command "$this do_nothing" \
	    -label 4 \
	    -value 4
	$w.options.scene_type.list add radiobutton \
	    -variable $this-scene_type_gui \
	    -command "$this do_nothing" \
	    -label 5 \
	    -value 5
	$w.options.scene_type.list add radiobutton \
	    -variable $this-scene_type_gui \
	    -command "$this do_nothing" \
	    -label Nrrd \
	    -value 6
	$w.options.scene_type.list add radiobutton \
	    -variable $this-scene_type_gui \
	    -command "$this do_nothing" \
	    -label "Nrrd list" \
	    -value 7

	# close button
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
    }
}	
	
    

    









