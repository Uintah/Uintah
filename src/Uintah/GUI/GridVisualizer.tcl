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

    protected var_list ""
    protected mat_list ""

    constructor {config} {
	set name GridVisualizer
	set_defaults
    }
    
    method set_defaults {} {
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
	$this-c needexecute
	global $this-var_orientation
	set $this-var_orientation 0

	#sphere stuff
	global $this-radius
	set $this-radius 0.01
	global $this-polygons
	set $this-polygons 8

	#var graphing stuff
	global $this-curr_var
	set var_list ""
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
    method setup_color {w text nl n} {
	for {set i 1} { $i <= $nl} {incr i} {
	    set st "$w.l$i"
	    append st "$text"
	    puts $st
	    menubutton $st -text "Level $i $text color" \
		    -menu $st.list -relief groove
	    pack $st -side top -anchor w -padx 2 -pady 2
	    
	    menu $st.list
	    set var "$this-level$i"
	    append var "_$text"
	    append var "_color"
	    make_color_menu $st.list $var $n
	}
    }
    method isVisible {} {
	if {[winfo exists .ui[modname]]} {
	    return 1
	} else {
	    return 0
	}
    }
    method Rebuild {} {
	set w .ui[modname]

	$this destroyFrames
	$this makeFrames $w
    }
    method build {} {
	set w .ui[modname]

	$this makeFrames $w
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
    method graph {name} {
    }
    method make_mat_menu {w mat i} {
	set fname "$w.mat$mat"
	menubutton $fname -text "Material" \
		-menu $fname.list -relief groove
	pack $fname -side right -padx 2 -pady 2
	
	menu $fname.list
	global $this-vmat$i
	set $this-vmat$i 0
	for {set j 0} { $j < $mat} {incr j} {
	    $fname.list add radiobutton \
		    -variable $this-vmat$i \
		    -label "Mat $j" \
		    -value $j
	}
    }
    method graphbutton {name i} {
	$this-c graph $name [set $this-vmat$i] $i
    }
    method addVar {w name mat i} {
	set fname "$w.var$i"
	frame $fname
	pack $fname -side top -fill x -padx 2 -pady 2

	label $fname.label -text "$name"
	pack $fname.label -side left -padx 2 -pady 2

	button $fname.button -text "Graph" -command "$this graphbutton $name $i"
	pack $fname.button -side right -padx 2 -pady 2

	make_mat_menu $fname $mat $i
    }
    method setVar_list { args } {
	set var_list $args
	puts "var_list is now $var_list"
    }
    method setMat_list { args } {
	set mat_list $args
	puts "mat_list is now $mat_list"
    }
    method buildVarFrame {w} {
	if {[llength $var_list] > 0} {
	    frame $w.vars -borderwidth 3 -relief ridge
	    pack $w.vars -side top -fill x -padx 2 -pady 2
	    
	    puts "var_list length [llength $var_list]"
	    for {set i 0} { $i < [llength $var_list] } { incr i } {
		set newvar [lindex $var_list $i]
		set newmat [lindex $mat_list $i]
		addVar $w.vars $newvar $newmat $i
	    }
	}
    }
    method buildColorMenus {w} {
	set n "$this-c needexecute "
	set i 0
	set b 1
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
	toplevel $w
	wm minsize $w 300 190
	set n "$this-c needexecute "

	#selection stuff
	frame $w.o
	pack $w.o -side top -fill x -padx 2 -pady 2

	frame $w.o.select
	pack $w.o.select -side left -fill x -padx 2 -pady 2
	
	checkbutton $w.o.select.plane_on -text "Selection plane" -variable \
		$this-plane_on -command $n
	pack $w.o.select.plane_on -side top -anchor w -pady 2 -ipadx 3

	checkbutton $w.o.select.nselect_on -text "Node Select" -variable \
		$this-node_select_on -command $n
	pack $w.o.select.nselect_on -side top -anchor w -pady 2 -ipadx 3

	radiobutton $w.o.select.node -variable $this-var_orientation \
		-command $n -text "Node Centered" -value 0
	pack $w.o.select.node -side top -anchor w -pady 2 -ipadx 3

	radiobutton $w.o.select.cell -variable $this-var_orientation \
		-command $n -text "Cell Centered" -value 1
	pack $w.o.select.cell -side top -anchor w -pady 2 -ipadx 3
	
	radiobutton $w.o.select.face -variable $this-var_orientation \
		-command $n -text "Face Centered" -value 2
	pack $w.o.select.face -side top -anchor w -pady 2 -ipadx 3
	
	button $w.o.select.findxy -text "Find XY" -command "$this-c findxy"
	pack $w.o.select.findxy -pady 2 -side top -ipadx 3 -anchor w

	button $w.o.select.findyz -text "Find YZ" -command "$this-c findyz"
	pack $w.o.select.findyz -pady 2 -side top -ipadx 3 -anchor w

	button $w.o.select.findxz -text "Find XZ" -command "$this-c findxz"
	pack $w.o.select.findxz -pady 2 -side top -ipadx 3 -anchor w

	# sphere stuff
	frame $w.o.sphere
	pack $w.o.sphere -side left -fill x -padx 2 -pady 2

	set r [expscale $w.o.sphere.radius -label "Radius:" \
		-orient horizontal -variable $this-radius -command $n ]
	pack $w.o.sphere.radius -side top -fill x

	scale $w.o.sphere.polygons -label "Polygons:" -orient horizontal \
	    -variable $this-polygons -command $n \
	    -from 8 -to 400 -tickinterval 392
	pack $w.o.sphere.polygons -side top -fill x


	makeFrames $w

	# close button
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
    }
    method graph_data { id var args } {
	set w .graph[modname]$id
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w
#	wm minsize $w 300 300

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -expand yes -fill x
	
	blt::graph $w.graph -title "Variable Value" -height 250 \
		-plotbackground gray99

	set max 1e-10
	set min 1e+10

	for { set i 1 } { $i < [llength $args] } { set i [expr $i + 2]} {
	    set val [lindex $args $i]
	    if { $max < $val } { set max $val }
	    if { $min > $val } { set min $val }
	}
	if { ($max - $min) > 1000 || ($max - $min) < 1e-3 } {
	    $w.graph yaxis configure -logscale true -title $var
	} else {
	    $w.graph yaxis configure -title $var
	}

	$w.graph xaxis configure -title "Timestep" \
		-loose true
	$w.graph element create "Variable Value" -linewidth 2 -color blue \
	    -pixels 3

	pack $w.graph
	if { $args != "" } {
	    $w.graph element configure "Variable Value" -data "$args"
	}
    }
}	
	
    

    









