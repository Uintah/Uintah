#
#  RTRTViewer.tcl
#
#  Written by:
#   James Bigler
#   Department of Computer Science
#   University of Utah
#   May 2001
#
#  Copyright (C) 2001 SCI Group
#

itcl_class rtrt_Render_RTRTViewer {
    inherit Module

    constructor {config} {
	set name RTRTViewer
	set_defaults
    }
    
    method set_defaults {} {
	global $this-nworkers
	set $this-nworkers 1
	global $this-xres_gui
	set $this-xres_gui 600
	global $this-yres_gui
	set $this-yres_gui 400
	global $this-render_mode
	set $this-render_mode 0
	global $this-scene_opt_type
	set $this-scene_opt_type 1
	global $this-gridcellsize_gui
	set $this-gridcellsize_gui 4
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
	
	frame $w.options
	pack $w.options  -side top -fill x -padx 2 -pady 2

	# add the number of processors box
	make_entry $w.options.nworkers "Number of processors" $this-nworkers \
		"$this-c needexecute"
	pack $w.options.nworkers -side top -fill x -padx 2 -pady 2

	# add the xres box
	make_entry $w.options.xres "xres" $this-xres_gui \
		"$this-c needexecute"
	pack $w.options.xres -side top -fill x -padx 2 -pady 2

	# add the yres box
	make_entry $w.options.yres "yres" $this-yres_gui \
		"$this-c needexecute"
	pack $w.options.yres -side top -fill x -padx 2 -pady 2

	menubutton $w.options.renmode -text "Render Mode" \
	    -menu $w.options.renmode.list -relief groove
	pack $w.options.renmode -side top -anchor w -padx 2 -pady 2
	
	menu $w.options.renmode.list
	$w.options.renmode.list add radiobutton \
	    -variable $this-render_mode \
	    -command "$this do_nothing" \
	    -label Frames \
	    -value 0
	$w.options.renmode.list add radiobutton \
	    -variable $this-render_mode \
	    -command "$this do_nothing" \
	    -label Frameless \
	    -value 1

	menubutton $w.options.sceneopt -text "Scene Optimization Type" \
	    -menu $w.options.sceneopt.list -relief groove
	pack $w.options.sceneopt -side top -anchor w -padx 2 -pady 2
	
	menu $w.options.sceneopt.list
	$w.options.sceneopt.list add radiobutton \
	    -variable $this-scene_opt_type \
	    -command "$this do_nothing" \
	    -label None \
	    -value 0
	$w.options.sceneopt.list add radiobutton \
	    -variable $this-scene_opt_type \
	    -command "$this do_nothing" \
	    -label BVH1 \
	    -value 1
	$w.options.sceneopt.list add radiobutton \
	    -variable $this-scene_opt_type \
	    -command "$this do_nothing" \
	    -label BVH2 \
	    -value 2
	$w.options.sceneopt.list add radiobutton \
	    -variable $this-scene_opt_type \
	    -command "$this do_nothing" \
	    -label Grid \
	    -value 3
	
	# add the gridcellsize variable
	make_entry $w.options.gridcellsize "Grid cell size" \
	    $this-gridcellsize_gui "$this do_nothing"
	pack $w.options.gridcellsize -side top -fill x -padx 2 -pady 2

	frame $w.render
	pack $w.render  -side top -fill x -padx 2 -pady 2
	
	# Start the rendering
	button $w.render.start_rtrt -text "Start rtrt" \
		-command "$this-c start_rtrt"
	pack $w.render.start_rtrt -side top -expand yes -fill x
	
	# Stop the rendering
	button $w.render.stop_rtrt -text "Stop rtrt" \
		-command "$this-c stop_rtrt"
	pack $w.render.stop_rtrt -side top -expand yes -fill x
	
	# close button
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
    }
}	
	
    

    









