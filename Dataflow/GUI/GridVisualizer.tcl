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
    inherit Uintah_Visualization_VariablePlotter

    constructor {config} {
	set name GridVisualizer
	Uintah_Visualization_VariablePlotter::set_defaults
	set_defaults
    }

    method set_defaults {} {
	puts "In GridVisualizer"
	
	#the grid colors
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
	global $this-nl
	set $this-nl 0

	#plane selection
	global $this-plane_on
	set $this-plane_on 0
	global $this-node_select_on
	set $this-node_select_on 0
	
	#Node or Cell oriented data
	global $this-var_orientation
	set $this-var_orientation 0

	#sphere stuff
	global $this-radius
	set $this-radius 0.01
	global $this-polygons
	set $this-polygons 8

	#selected node stuff
	global $this-show_selected_node
	set $this-show_selected_node 1
    }
    method make_color_menu {w v n} {
	$w add radiobutton \
		-variable $v \
		-command $n \
		-label red \
		-value red
	$w add radiobutton \
		-variable $v \
		-command $n \
		-label green \
		-value green
	$w add radiobutton \
		-variable $v \
		-command $n \
		-label blue \
		-value blue
	$w add radiobutton \
		-variable $v \
		-command $n \
		-label yellow \
		-value yellow
	$w add radiobutton \
		-variable $v \
		-command $n \
		-label cyan \
		-value cyan
	$w add radiobutton \
		-variable $v \
		-command $n \
		-label magenta \
		-value magenta

    }
    # this is used for seting up the colors used for
    # the levels and grid points
    method setup_color {w text nl n} {
	for {set i 1} { $i <= $nl} {incr i} {
	    set st "$w.l$i"
	    append st "$text"
#	    puts $st
	    menubutton $st -text "Level [expr $i - 1] $text color" \
		    -menu $st.list -relief groove
	    pack $st -side top -anchor w -padx 2 -pady 2
	    
	    menu $st.list
	    set var "$this-level$i"
	    append var "_$text"
	    append var "_color"
	    make_color_menu $st.list $var $n
	}
    }

    method destroyFrames {} {
	set w .ui[modname]
	destroy $w.colormenus
	destroy $w.vars
	set $var_list ""
    }
    method makeFrames {w} {
	$this buildColorMenus $w
	$this buildVarFrame $w
    }
    
    method buildColorMenus {w} {
	set n "$this-c needexecute"
	puts "GridVisualizer.tcl::buildColorMenus:w = $w"
	if {[set $this-nl] > 0} {
	    # color menu stuff
	    frame $w.colormenus -borderwidth 3 -relief ridge
	    pack $w.colormenus -side top -fill x -padx 2 -pady 2
	    
	    # set up the stuff for the grid colors
	    frame $w.colormenus.gridcolor
	    pack $w.colormenus.gridcolor -side left -fill y -padx 2 -pady 2
	    
	    setup_color $w.colormenus.gridcolor "grid" [set $this-nl] $n
	    
	    # set up the stuff for the node colors
	    frame $w.colormenus.nodecolor
	    pack $w.colormenus.nodecolor -side right -fill y -padx 2 -pady 2
	    
	    setup_color $w.colormenus.nodecolor "node" [set $this-nl] $n
	}
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	Uintah_Visualization_VariablePlotter::ui
#	toplevel $w
	wm minsize $w 300 190
	set n "$this-c needexecute "

	#selection stuff
	frame $w.grid
	pack $w.grid -side top -fill x -padx 2 -pady 2

	frame $w.grid.select
	pack $w.grid.select -side left -fill x -padx 2 -pady 2
	
	checkbutton $w.grid.select.plane_on -text "Selection plane" \
		-variable $this-plane_on -command $n
	pack $w.grid.select.plane_on -side top -anchor w -pady 2 -ipadx 3

	checkbutton $w.grid.select.nselect_on -text "Node Select" \
		-variable $this-node_select_on -command $n
	pack $w.grid.select.nselect_on -side top -anchor w -pady 2 -ipadx 3

	checkbutton $w.grid.select.show_select_n -text "Show Selected Node" \
		-variable $this-show_selected_node \
		-command "$this-c update_sn"
	pack $w.grid.select.show_select_n -side top -anchor w -pady 2 -ipadx 3

	button $w.grid.select.findxy -text "Find XY" -command "$this-c findxy"
	pack $w.grid.select.findxy -pady 2 -side top -ipadx 3 -anchor w

	button $w.grid.select.findyz -text "Find YZ" -command "$this-c findyz"
	pack $w.grid.select.findyz -pady 2 -side top -ipadx 3 -anchor w

	button $w.grid.select.findxz -text "Find XZ" -command "$this-c findxz"
	pack $w.grid.select.findxz -pady 2 -side top -ipadx 3 -anchor w

	# right side
	frame $w.grid.r
	pack $w.grid.r -side right -fill x -padx 2 -pady 2

	# sphere stuff
	frame $w.grid.r.sphere
	pack $w.grid.r.sphere -side top -anchor w -fill x -padx 2 -pady 2

	set r [expscale $w.grid.r.sphere.radius -label "Radius:" \
		-orient horizontal -variable $this-radius -command $n ]
	pack $w.grid.r.sphere.radius -side top -fill x

	scale $w.grid.r.sphere.polygons -label "Polygons:" -orient horizontal \
	    -variable $this-polygons -command $n \
	    -from 8 -to 400 -tickinterval 392
	pack $w.grid.r.sphere.polygons -side top -fill x

#	makeFrames $w

#	button $w.ttest -text "Table test" -command "$this table_test"
#	pack $w.ttest -side bottom -expand yes -fill x
#	button $w.gtest -text "Graph test" -command "$this graph_test"
#	pack $w.gtest -side bottom -expand yes -fill x
    }

}	
	
    

    









