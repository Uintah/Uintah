#
#  PatchVisualizer.tcl
#
#  Written by:
#   James Bigler
#   Department of Computer Science
#   University of Utah
#   January 2001
#
#  Copyright (C) 2001 SCI Group
#

itcl_class Uintah_Visualization_PatchVisualizer {
    inherit Module

    protected num_colors 240

    constructor {config} {
	set name PatchVisualizer
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
	
	global $this-nl
	set $this-nl 0

	global $this-patch_seperate
	set $this-patch_seperate 0
    }
    # This creates a menu to select colors for the grid
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
    }
    method makeFrames {w} {
	$this buildColorMenus $w
    }
    method graph {name} {
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
	}
    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v -state disabled
	bind $w.e <Return> $c
	pack $w.e -side right
    }
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

	checkbutton $w.options.seperate -text "Seperate Patches" -variable \
		$this-patch_seperate -command $n
	pack $w.options.seperate -side top -anchor w -pady 2 -ipadx 3

	makeFrames $w

	# close button
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
#	button $w.gtest -text "Graph test" -command "$this graph_test"
#	pack $w.gtest -side bottom -expand yes -fill x
    }
    method get_color { index } {
	set color_scheme {
	    { 255 0 0}  { 255 102 0}
	    { 255 204 0}  { 255 234 0}
	    { 204 255 0}  { 102 255 0}
	    { 0 255 0}    { 0 255 102}
	    { 0 255 204}  { 0 204 255}
	    { 0 102 255}  { 0 0 255}}
	#set color_scheme { {255 0 0} {0 255 0} {0 0 255} }
	set incr {}
	set upper_bounds [expr [llength $color_scheme] -1]
	for {set j 0} { $j < $upper_bounds} {incr j} {
	    set c1 [lindex $color_scheme $j]
	    set c2 [lindex $color_scheme [expr $j + 1]]
	    set incr_a {}
	    lappend incr_a [expr [lindex $c2 0] - [lindex $c1 0]]
	    lappend incr_a [expr [lindex $c2 1] - [lindex $c1 1]]
	    lappend incr_a [expr [lindex $c2 2] - [lindex $c1 2]]
	    lappend incr $incr_a
	}
	lappend incr {0 0 0}
#	puts "incr = $incr"
	set step [expr $num_colors / [llength $color_scheme]]
	set ind [expr $index % $num_colors] 
	set i [expr $ind / $step]
	set im [expr double($ind % $step)/$step]
#	puts "i = $i  im = $im"
	set curr_color [lindex $color_scheme $i]
	set curr_incr [lindex $incr $i]
#	puts "curr_color = $curr_color, curr_incr = $curr_incr"
	set r [expr [lindex $curr_color 0]+round([lindex $curr_incr 0] * $im)] 
	set g [expr [lindex $curr_color 1]+round([lindex $curr_incr 1] * $im)] 
	set b [expr [lindex $curr_color 2]+round([lindex $curr_incr 2] * $im)] 
	set c [format "#%02x%02x%02x" $r $g $b]
#	puts "r=$r, g=$g, b=$b, c=$c"
	return $c
    }
}	
	
    

    









