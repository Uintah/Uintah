#
#  GridVisualizer.tcl
#
#  Written by:
#   James Bigler
#   Department of Computer Science
#   University of Utah
#   Aug. 2000
#
#  Copyright (C) 2000 SCI Group
#

itcl_class Uintah_Visualization_GridVisualizer {
    inherit Module
    constructor {config} {
	set name GridVisualizer
	set_defaults
    }
    
    method set_defaults {} {
	#the node colors
	global $this-level1_grid_color
	set $this-level1_grid_color red
	global $this-level2_grid_color
	set $this-level2_grid_color green
	global $this-level3_grid_color
	set $this-level3_grid_color yellow
	global $this-level4_grid_color
	set $this-level4_grid_color magenta
	global $this-level5_grid_color
	set $this-level5_grid_color cyan
	global $this-level6_grid_color
	set $this-level6_grid_color blue
	
	#the node colors
	global $this-level1_node_color
	set $this-level1_node_color red
	global $this-level2_node_color
	set $this-level2_node_color green
	global $this-level3_node_color
	set $this-level3_node_color yellow
	global $this-level4_node_color
	set $this-level4_node_color magenta
	global $this-level5_node_color
	set $this-level5_node_color cyan
	global $this-level6_node_color
	set $this-level6_node_color blue
	
	#plane selection
	global $this-plane_on
	set $this-plane_on 0
	global $this-node_select_on
	set $this-node_select_on 0
	$this-c needexecute

	#sphere stuff
	global $this-raduis
	global $this-polygons
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 20
	set n "$this-c needexecute "

	# set up the stuff for the grid colors
	frame $w.gridcolor
	pack $w.gridcolor -side top -fill x -padx 2 -pady 2

	make_labeled_radio $w.gridcolor.l1grid "Level 1 grid color: " $n \
		top $this-level1_grid_color \
		{red green blue yellow cyan magenta}
	pack $w.gridcolor.l1grid -side left

	make_labeled_radio $w.gridcolor.l2grid "Level 2 grid color: " $n \
		top $this-level2_grid_color \
		{red green blue yellow cyan magenta}
	pack $w.gridcolor.l2grid -side left

	make_labeled_radio $w.gridcolor.l3grid "Level 3 grid color: " $n \
		top $this-level3_grid_color \
		{red green blue yellow cyan magenta}
	pack $w.gridcolor.l3grid -side left

	make_labeled_radio $w.gridcolor.l4grid "Level 4 grid color: " $n \
		top $this-level4_grid_color \
		{red green blue yellow cyan magenta}
	pack $w.gridcolor.l4grid -side left

	make_labeled_radio $w.gridcolor.l5grid "Level 5 grid color: " $n \
		top $this-level5_grid_color \
		{red green blue yellow cyan magenta}
	pack $w.gridcolor.l5grid -side left

	make_labeled_radio $w.gridcolor.l6grid "Level 6 grid color: " $n \
		top $this-level6_grid_color \
		{red green blue yellow cyan magenta}
	pack $w.gridcolor.l6grid -side left


	# set up the stuff for the grid colors
	frame $w.nodecolor
	pack $w.nodecolor -side top -fill x -padx 2 -pady 2

	make_labeled_radio $w.nodecolor.l1node "Level 1 node color: " $n \
		top $this-level1_node_color \
		{red green blue yellow cyan magenta}
	pack $w.nodecolor.l1node -side left

	make_labeled_radio $w.nodecolor.l2node "Level 2 node color: " $n \
		top $this-level2_node_color \
		{red green blue yellow cyan magenta}
	pack $w.nodecolor.l2node -side left

	make_labeled_radio $w.nodecolor.l3node "Level 3 node color: " $n \
		top $this-level3_node_color \
		{red green blue yellow cyan magenta}
	pack $w.nodecolor.l3node -side left

	make_labeled_radio $w.nodecolor.l4node "Level 4 node color: " $n \
		top $this-level4_node_color \
		{red green blue yellow cyan magenta}
	pack $w.nodecolor.l4node -side left

	make_labeled_radio $w.nodecolor.l5node "Level 5 node color: " $n \
		top $this-level5_node_color \
		{red green blue yellow cyan magenta}
	pack $w.nodecolor.l5node -side left

	make_labeled_radio $w.nodecolor.l6node "Level 6 node color: " $n \
		top $this-level6_node_color \
		{red green blue yellow cyan magenta}
	pack $w.nodecolor.l6node -side left

	#other stuff
	frame $w.o
	pack $w.o -side top -fill x -padx 2 -pady 2

	frame $w.o.select
	pack $w.o.select -side left -fill x -padx 2 -pady 2
	
	checkbutton $w.o.select.plane_on -text "Selection plane" -variable \
		$this-plane_on -command $n
	pack $w.o.select.plane_on -side top -pady 2 -ipadx 3

	checkbutton $w.o.select.nselect_on -text "Node Select" -variable \
		$this-node_select_on -command $n
	pack $w.o.select.nselect_on -side top -pady 2 -ipadx 3


	frame $w.o.sphere
	pack $w.o.sphere -side left -fill x -padx 2 -pady 2

	set r [expscale $w.o.sphere.radius -label "Radius:" \
		-orient horizontal -variable $this-radius -command $n ]
	pack $w.o.sphere.radius -side top -fill x

	scale $w.o.sphere.polygons -label "Polygons:" -orient horizontal \
	    -variable $this-polygons -command $n \
	    -from 8 -to 400 -tickinterval 392
	pack $w.o.sphere.polygons -side top -fill x


	frame $w.o.axisbuttons
	pack $w.o.axisbuttons -fill x -side right -padx 2 -pady 2

	button $w.o.axisbuttons.findxy -text "Find XY" -command "$this-c findxy"
	pack $w.o.axisbuttons.findxy -pady 2 -side top -ipadx 3 -anchor e

	button $w.o.axisbuttons.findyz -text "Find YZ" -command "$this-c findyz"
	pack $w.o.axisbuttons.findyz -pady 2 -side top -ipadx 3 -anchor e

	button $w.o.axisbuttons.findxz -text "Find XZ" -command "$this-c findxz"
	pack $w.o.axisbuttons.findxz -pady 2 -side top -ipadx 3 -anchor e

	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
    }
}	
	
    

    

