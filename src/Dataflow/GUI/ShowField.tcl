
itcl_class SCIRun_Visualization_ShowField {
    inherit Module
    constructor {config} {
	set name ShowField
	set_defaults
    }

    method set_defaults {} {
	global $this-show_nodes
	global $this-show_edges
	global $this-show_faces
	global $this-node_display_type
	global $this-scale
	set $this-node_display_type Spheres
	set $this-scale 0.03
	$this-c needexecute
    }

#    method raiseColor {col colRoot colMsg} {
#	 global $colRoot
#	 set window .ui[modname]
#	 if {[winfo exists $window.color]} {
#	     raise $window.color
#	     return;
#	 } else {
#	     toplevel $window.color
#	     makeColorPicker $window.color $colRoot \
#		     "$this setColor $col $colRoot $colMsg" \
#		     "destroy $window.color"
#	 }
#    }
#
#    method setColor {col colRoot colMsg} {
#	 global $colRoot
#	 global $colRoot-r
#	 global $colRoot-g
#	 global $colRoot-b
#	 set ir [expr int([set $colRoot-r] * 65535)]
#	 set ig [expr int([set $colRoot-g] * 65535)]
#	 set ib [expr int([set $colRoot-b] * 65535)]
#
#	 set window .ui[modname]
#	 $col config -background [format #%04x%04x%04x $ir $ig $ib]
#	 $this-c $colMsg
#    }
#
#    method addColorSelection {frame colRoot colMsg} {
#	 #add node color picking 
#	 global $colRoot
#	 global $colRoot-r
#	 global $colRoot-g
#	 global $colRoot-b
#	 set ir [expr int([set $colRoot-r] * 65535)]
#	 set ig [expr int([set $colRoot-g] * 65535)]
#	 set ib [expr int([set $colRoot-b] * 65535)]
#	 
#	 frame $frame.colorFrame
#	 frame $frame.colorFrame.col -relief ridge -borderwidth \
#		 4 -height 0.7c -width 0.7c \
#		 -background [format #%04x%04x%04x $ir $ig $ib]
#	 
#	 set cmmd "$this raiseColor $frame.colorFrame.col $colRoot $colMsg"
#	 button $frame.colorFrame.set_color -text "Set Color" -command $cmmd
#	 
#	 #pack the node color frame
#	 pack $frame.colorFrame.set_color $frame.colorFrame.col -side left 
#	 pack $frame.colorFrame -side bottom 
#
#    }
#
    method ui {} {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    raise $window
	    return;
	}
	toplevel $window
	
	#frame for all options to live
	frame $window.options
 
	# node frame holds ui related to vert display (left side)
	frame $window.options.disp -relief groove -borderwidth 2
	pack $window.options.disp -padx 2 -pady 2 -side left -fill y
	set n "$this-c needexecute"	

	label $window.options.disp.frame_title -text "Display Options"

	checkbutton $window.options.disp.show_nodes \
		-text "Display Nodes" \
		-command "$this-c toggle_display_nodes"
	$window.options.disp.show_nodes select

	global $this-node_display_type
	set b $this-node_display_type
	make_labeled_radio $window.options.disp.radio \
		"Node Display Type" "$this-c node_display_type" top \
		$this-node_display_type {Spheres Axes Points}

	if {$b == "Spheres"} {
	    $window.options.disp.radio.0 select
	}
	if {$b == "Axes"} {
	    $window.options.disp.radio.1 select
	}
	if {$b == "Points"} {
	    $window.options.disp.radio.1 select
	}

	scale $window.options.disp.slide -label NodeScale \
		-from 0.0018 -to 0.05 \
		-length 5c \
		-showvalue true \
		-orient horizontal -resolution 0.001 \
		-digits 8 -variable $this-scale -command "$this-c scale"

	checkbutton $window.options.disp.show_edges \
		-text "Show Edges" \
		-command "$this-c toggle_display_edges"
	$window.options.disp.show_edges select

	checkbutton $window.options.disp.show_faces \
		-text "Show Faces" \
		-command "$this-c toggle_display_faces"
	$window.options.disp.show_faces select

	#pack the node radio button
	pack $window.options.disp.frame_title $window.options.disp.show_nodes \
		$window.options.disp.radio $window.options.disp.slide \
		$window.options.disp.show_edges \
		$window.options.disp.show_faces \
		-side top -fill x -padx 2 -pady 2 -anchor w

	bind $window.options.disp.slide <ButtonRelease> "$this-c needexecute"
	#add bottom frame for execute and dismiss buttons
	frame $window.control -relief groove -borderwidth 2
	pack $window.options $window.control -padx 2 -pady 2 -side top

	button $window.control.execute -text Execute -command $n
	button $window.control.dismiss -text Dismiss -command "destroy $window"
	pack $window.control.execute $window.control.dismiss -side left 
    }
}


















