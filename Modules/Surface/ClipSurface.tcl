#
#  ClipSurface.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   July 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class ClipSurface {
    inherit Module
    constructor {config} {
	set name ClipSurface
	set_defaults
    }
    method set_defaults {} {
	global $this-pa
	global $this-pb
	global $this-pc
	global $this-pd
	set $this-pa 0
	set $this-pb 0
	set $this-pc 1
	set $this-pd 0
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

	global $this-pa
	global $this-pb
	global $this-pc
	global $this-pd
	frame $w.f.p -relief groove -borderwidth 5
	scale $w.f.p.a -orient horizontal -variable $this-pa \
		-label "Plane A:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.p.b -orient horizontal -variable $this-pb \
		-label "Plane B:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.p.c -orient horizontal -variable $this-pc \
		-label "Plane C:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	expscale $w.f.p.d -orient horizontal -variable $this-pd \
		-label "Plane D:"
	pack $w.f.p.a $w.f.p.b $w.f.p.c $w.f.p.d -side top -fill x
	pack $w.f.p -side top -fill x -expand 1
	button $w.f.b -text "Clip" -command "$this-c needexecute"
	pack $w.f.b -side top
	pack $w.f -fill x -expand 1 -side top
    }	
}
