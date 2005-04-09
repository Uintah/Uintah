#
#  RotateSurface.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   July 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class RotateSurface {
    inherit Module
    constructor {config} {
	set name RotateSurface
	set_defaults
    }
    method set_defaults {} {
	global $this-rx
	global $this-ry
	global $this-rz
	global $this-th
	set $this-rx 0
	set $this-ry 0
	set $this-rz 1
	set $this-th 0
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
	global $this-rx
	global $this-ry
	global $this-rz
	global $this-th
	scale $w.f.x -orient horizontal -variable $this-rx \
		-label "Rotate Axis X:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.y -orient horizontal -variable $this-ry \
		-label "Rotate Axis Y:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.z -orient horizontal -variable $this-rz \
		-label "Rotate Axis Z:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.th -orient horizontal -variable $this-th \
		-label "Rotate Theta:" -showvalue true -from -7.0 -to 7.0 \
		-resolution .01
	pack $w.f.x $w.f.y $w.f.z $w.f.th -fill x -expand 1 -side top
	button $w.f.b -text "Rotate" -command "$this-c needexecute"
	pack $w.f.b -side top
	pack $w.f -fill x -expand 1
    }	
}
