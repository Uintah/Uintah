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

    constructor {config} {
	set name PatchVisualizer
	set_defaults
    }
    
    method set_defaults {} {
	# There can only be 6 unique representations
	# If there are more than 6 level than the value for level 5 will
	# be used for all subsequent levels

	#the grid colors used for solid coloring
	global $this-level0_grid_color
	set $this-level0_grid_color red
	global $this-level1_grid_color
	set $this-level1_grid_color green
	global $this-level2_grid_color
	set $this-level2_grid_color yellow
	global $this-level3_grid_color
	set $this-level3_grid_color magenta
	global $this-level4_grid_color
	set $this-level4_grid_color cyan
	global $this-level5_grid_color
	set $this-level5_grid_color blue
	
	#the level coloring scheme
	#they are all initialized to solid, because it is faster than the
	#the other coloring methods
	global $this-level0_color_scheme
	set $this-level0_color_scheme solid
	global $this-level1_color_scheme
	set $this-level1_color_scheme solid
	global $this-level2_color_scheme
	set $this-level2_color_scheme solid
	global $this-level3_color_scheme
	set $this-level3_color_scheme solid
	global $this-level4_color_scheme
	set $this-level4_color_scheme solid
	global $this-level5_color_scheme
	set $this-level5_color_scheme solid
	
	# the number of levels
	# this is set by the c-code
	global $this-nl
	set $this-nl 0

	# check button that indicates whether the patch geometry should have
	# some seperation between them.
	# Default is off(0). On is 1.
	global $this-patch_seperate
	set $this-patch_seperate 0
    }
    # This creates a sub menue for a button that will have 6 radial buttons
    # for the selection of the solid color for the level

    # w - the tickle menu to add the radio buttons to
    # v - the variable the radio buttons control
    # n - the command to execute when the buttons are touched
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
    # the levels as well as setting up controls for the type of coloring

    # w - the widget to add all this stuff to
    # nl - the number of levels
    # n - the command to execute when any of the controls are touched
    method setup_color {w nl n} {
	# loop for each level
	for {set i 0} { $i < $nl} {incr i} {
	    # color scheme menu stuff
	    set colorscheme $w.$i
	    frame $colorscheme -borderwidth 3 -relief ridge
	    pack $colorscheme -side left -anchor w -padx 2 -pady 2
	    
	    # construct the variable name by the form
	    #  $this-level(level)_color_scheme
	    set var "$this-level$i"
	    append var "_color_scheme"

	    # add all the radio buttons
	    radiobutton $colorscheme.solid -text "solid" -variable $var \
		    -command $n -value solid
	    pack $colorscheme.solid -side top -anchor w -pady 2 -ipadx 3

	    radiobutton $colorscheme.x -text "x" -variable $var \
		    -command $n -value x
	    pack $colorscheme.x -side top -anchor w -pady 2 -ipadx 3

	    radiobutton $colorscheme.y -text "y" -variable $var \
		    -command $n -value y
	    pack $colorscheme.y -side top -anchor w -pady 2 -ipadx 3

	    radiobutton $colorscheme.z -text "z" -variable $var \
		    -command $n -value z
	    pack $colorscheme.z -side top -anchor w -pady 2 -ipadx 3

	    radiobutton $colorscheme.random -text "random" -variable $var \
		    -command $n -value random
	    pack $colorscheme.random -side top -anchor w -pady 2 -ipadx 3

	    # create the solid color selection menue
	    # st = $w.l(level)grid
	    set st "$w.l$i"
	    append st "grid"

	    # create the button we will create the menu($st.list) later
	    menubutton $st -text "Level [expr $i - 1] grid color" \
		    -menu $st.list -relief groove
	    pack $st -side left -anchor n -padx 2 -pady 2

	    
	    menu $st.list

	    # create the variable, $this-level(level)_grid_color
	    set var "$this-level$i"
	    append var "_grid_color"
	    make_color_menu $st.list $var $n
	}
    }
    
    # returns 1 if the window is visible, 0 otherwise
    method isVisible {} {
	if {[winfo exists .ui[modname]]} {
	    return 1
	} else {
	    return 0
	}
    }

    # destroys and rebuilds the archive specific stuff
    method Rebuild {} {
	set w .ui[modname]

	$this destroyFrames
	$this makeFrames $w
    }

    # destroys all archive specific stuff
    method destroyFrames {} {
	set w .ui[modname]
	destroy $w.colormenus
    }

    # creates all archive specific stuff
    method makeFrames {w} {
	$this buildColorMenus $w
    }

    # creates the color selection stuff
    method buildColorMenus {w} {
	set n "$this-c needexecute "
	if {[set $this-nl] > 0} {
	    # color menu stuff
	    frame $w.colormenus -borderwidth 3 -relief ridge
	    pack $w.colormenus -side top -fill x -padx 2 -pady 2
	    
	    # set up the stuff for the grid colors
	    frame $w.colormenus.gridcolor
	    pack $w.colormenus.gridcolor -side left -fill y -padx 2 -pady 2
	    
	    setup_color $w.colormenus.gridcolor [set $this-nl] $n
	}
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

	# add the Seperate Patches check button
	checkbutton $w.options.seperate -text "Seperate Patches" -variable \
		$this-patch_seperate -command $n
	pack $w.options.seperate -side top -anchor w -pady 2 -ipadx 3

	# add the level specific stuff
	makeFrames $w

	# close button
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
    }
}	
	
    

    









