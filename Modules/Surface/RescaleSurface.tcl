#
#  RescaleSurface.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   July 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class RescaleSurface {
    inherit Module
    constructor {config} {
	set name RescaleSurface
	set_defaults
    }
    method set_defaults {} {
	global $this-scale 
	global $this-coreg
	set $this-scale 0
	set $this-coreg 0
    }
    method ui {} {
	set w .ui$this
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 200 50

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2
	global $this-scale
	global $this-coreg
	label $w.f.l -text "Log Scale: "
	scale $w.f.s -variable $this-scale -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	set $this-scale 0
	button $w.f.e -text "Rescale" -command "$this-c needexecute"
	checkbutton $w.f.coreg -variable $this-coreg -text "Coreg Sized" \
		-command "$this-c needexecute"
	pack $w.f.l -side left
	pack $w.f.s -side left -expand 1 -fill x
	pack $w.f.e $w.f.coreg -side left
	pack $w.f
    }	
}
