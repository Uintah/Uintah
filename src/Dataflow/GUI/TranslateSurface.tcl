#
#  TranslateSurface.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   July 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class PSECommon_Surface_TranslateSurface {
    inherit Module
    constructor {config} {
	set name TranslateSurface
	set_defaults
    }
    method set_defaults {} {
	global $this-tx
	global $this-ty
	global $this-tz
	set $this-tx 0
	set $this-ty 0
	set $this-tz 0
    }
    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 200 50

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2
	global $this-tx
	global $this-ty
	global $this-tz
#	expscale $w.f.x -orient horizontal -variable $this-tx \
#		-label "Translate X:"
#	expscale $w.f.y -orient horizontal -variable $this-ty \
#		-label "Translate Y:"
#	expscale $w.f.z -orient horizontal -variable $this-tz \
#		-label "Translate Z:"
	button $w.f.b -text "Translate" -command "$this-c needexecute"
#	pack $w.f.b $w.f.x $w.f.y $w.f.z -side top
	pack $w.f.b -side top
	pack $w.f
    }	
}
