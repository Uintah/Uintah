#
#  PatchDataVisualizer.tcl
#
#  Written by:
#   James Bigler
#   Department of Computer Science
#   University of Utah
#   January 2001
#
#  Copyright (C) 2001 SCI Group
#

itcl_class Uintah_Visualization_PatchDataVisualizer {
    inherit Module

    constructor {config} {
	set name PatchDataVisualizer
	set_defaults
    }
    
    method set_defaults {} {
	#sphere stuff
	global $this-radius
	set $this-radius 0.1
	global $this-polygons
	set $this-polygons 8

	puts "radius = [set $this-radius]"
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
    }

    # creates all archive specific stuff
    method makeFrames {w} {
    }

    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v
	bind $w.e <Return> $c
	pack $w.e -side right
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

	# sphere stuff
	frame $w.sphere
	pack $w.sphere -side top -anchor w -fill x -padx 2 -pady 2

#	puts "radius = [set $this-radius]"
#	set r [expscale $w.sphere.radius -label "Radius:" \
#		-orient horizontal -variable $this-radius -command $n ]
#	pack $w.sphere.radius -side top -fill x

	make_entry $w.sphere.radius "radius" $this-radius $n
	pack $w.sphere.radius -side top -fill x -padx 2 -pady 2

	scale $w.sphere.polygons -label "Polygons:" -orient horizontal \
	    -variable $this-polygons -command $n \
	    -from 8 -to 400 -tickinterval 392
	pack $w.sphere.polygons -side top -fill x

	# close button
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
    }
}	
	
    

    









