#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#


itcl_class SCIRun_Visualization_ShowField {
    inherit Module
    constructor {config} {
	set name ShowField
	set_defaults
    }

    method set_defaults {} {
	global $this-nodes-on
	global $this-edges-on
	global $this-faces-on
	global $this-node_display_type
	global $this-def-color-r
	global $this-def-color-g
	global $this-def-color-b
	global $this-scale
	set $this-node_display_type Spheres
	set $this-scale 0.03
	set $this-def-color-r 0.5
	set $this-def-color-g 0.5
	set $this-def-color-b 0.5
	set $this-nodes-on 1
	set $this-edges-on 1
	set $this-faces-on 1
    }

    method raiseColor {col color colMsg} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     raise $window.color
	     return;
	 } else {
	     toplevel $window.color
	     makeColorPicker $window.color $color \
		     "$this setColor $col $color $colMsg" \
		     "destroy $window.color"
	 }
    }

    method setColor {col color colMsg} {
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 set window .ui[modname]
	 $col config -background [format #%04x%04x%04x $ir $ig $ib]
	 $this-c $colMsg
    }

    method addColorSelection {frame color colMsg} {
	 #add node color picking 
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]
	 
	 frame $frame.colorFrame
	 frame $frame.colorFrame.col -relief ridge -borderwidth \
		 4 -height 0.7c -width 0.7c \
		 -background [format #%04x%04x%04x $ir $ig $ib]
	 
	 set cmmd "$this raiseColor $frame.colorFrame.col $color $colMsg"
	 button $frame.colorFrame.set_color \
		 -text "Default Color" -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.col -side left 
	 pack $frame.colorFrame -side bottom 

    }

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
		-command "$this-c toggle_display_nodes" \
		-variable $this-nodes-on
	#$window.options.disp.show_nodes select

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


	checkbutton $window.options.disp.show_edges \
		-text "Show Edges" \
		-command "$this-c toggle_display_edges" \
		-variable $this-edges-on
	#$window.options.disp.show_edges select

	checkbutton $window.options.disp.show_faces \
		-text "Show Faces" \
		-command "$this-c toggle_display_faces" \
		-variable $this-faces-on
	#$window.options.disp.show_faces select

	#pack the node radio button
	pack $window.options.disp.frame_title $window.options.disp.show_nodes \
		$window.options.disp.radio \
		$window.options.disp.show_edges \
		$window.options.disp.show_faces \
		-side top -fill x -padx 2 -pady 2 -anchor w

	
	expscale $window.options.disp.slide -label NodeScale \
		-orient horizontal \
		-variable $this-scale -command "$this-c scale"

	addColorSelection $window.options.disp $this-def-color \
		"default_color_change"

	bind $window.options.disp.slide.scale <ButtonRelease> "$this-c needexecute"
	#add bottom frame for execute and dismiss buttons
	frame $window.control -relief groove -borderwidth 2
	pack $window.options $window.control -padx 2 -pady 2 -side top

	button $window.control.execute -text Execute -command $n
	button $window.control.dismiss -text Dismiss -command "destroy $window"
	pack $window.control.execute $window.control.dismiss -side left 
    }
}


















