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
	global $this-node_scale
	global $this-edge_scale
	set $this-node_display_type Spheres
	set $this-edge_display_type Lines
	set $this-node_scale 0.03
	set $this-edge_scale 0.03
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
		 4 -height 0.8c -width 1.0c \
		 -background [format #%04x%04x%04x $ir $ig $ib]
	 
	 set cmmd "$this raiseColor $frame.colorFrame.col $color $colMsg"
	 button $frame.colorFrame.set_color \
		 -text "Default Color" -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.col -side left
	 pack $frame.colorFrame -side left

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
	frame $window.options.disp -borderwidth 2
	frame $window.options.disp.node -relief groove -borderwidth 2
	frame $window.options.disp.edge -relief groove -borderwidth 2
	frame $window.options.disp.face -relief groove -borderwidth 2

	pack $window.options.disp -padx 2 -pady 2 -side left -fill y

	set n "$this-c needexecute"	

	label $window.options.disp.frame_title -text "Display Options"

	checkbutton $window.options.disp.node.show_nodes \
		-text "Show Nodes" \
		-command "$this-c toggle_display_nodes" \
		-variable $this-nodes-on
	#$window.options.disp.show_nodes select

	global $this-node_display_type
	set b $this-node_display_type
	make_labeled_radio $window.options.disp.node.radio \
		"Node Display Type" "$this-c node_display_type" top \
		$this-node_display_type {Spheres Axes Points}

	if {$b == "Spheres"} {
	    $window.options.disp.node.radio.0 select
	}
	if {$b == "Axes"} {
	    $window.options.disp.node.radio.1 select
	}
	if {$b == "Points"} {
	    $window.options.disp.node.radio.2 select
	}


	checkbutton $window.options.disp.edge.show_edges \
		-text "Show Edges" \
		-command "$this-c toggle_display_edges" \
		-variable $this-edges-on

	global $this-edge_display_type
	set e $this-edge_display_type
	make_labeled_radio $window.options.disp.edge.radio \
		"Edge Display Type" "$this-c edge_display_type" top \
		$this-edge_display_type {Lines Cylinders}

	if {$e == "Lines"} {
	    $window.options.disp.node.radio.0 select
	}
	if {$e == "Cylinders"} {
	    $window.options.disp.node.radio.1 select
	}

	#$window.options.disp.show_edges select

	checkbutton $window.options.disp.face.show_faces \
		-text "Show Faces" \
		-command "$this-c toggle_display_faces" \
		-variable $this-faces-on
	#$window.options.disp.show_faces select

	#pack the node radio button
	pack $window.options.disp.frame_title -side top
	pack $window.options.disp.node $window.options.disp.edge \
		$window.options.disp.face -side left -fill y \
		-padx 3 -pady 2 -anchor w

	pack $window.options.disp.node.show_nodes \
		$window.options.disp.node.radio -fill y -anchor w
	
	expscale $window.options.disp.node.slide -label NodeScale \
		-orient horizontal \
		-variable $this-node_scale -command "$this-c node_scale"
	frame $window.options.disp.edge.space -height 0.68c 
	pack $window.options.disp.edge.show_edges \
		$window.options.disp.edge.radio \
		$window.options.disp.edge.space \
		-side top -fill y -anchor w

	expscale $window.options.disp.edge.slide -label CylinderScale \
		-orient horizontal \
		-variable $this-edge_scale -command "$this-c edge_scale"

	pack $window.options.disp.face.show_faces \
		-side top -fill y -anchor w


	bind $window.options.disp.node.slide.scale <ButtonRelease> \
		"$this-c needexecute"
	bind $window.options.disp.edge.slide.scale <ButtonRelease> \
		"$this-c needexecute"
	#add bottom frame for execute and dismiss buttons
	frame $window.control -relief groove -borderwidth 2
	frame $window.def_col -borderwidth 2

	addColorSelection $window.def_col $this-def-color \
		"default_color_change"

	pack $window.options $window.def_col $window.control \
		-padx 2 -pady 2 -side top

	button $window.control.execute -text Execute -command $n
	button $window.control.dismiss -text Dismiss -command "destroy $window"
	pack $window.control.execute $window.control.dismiss -side left 
    }
}


















