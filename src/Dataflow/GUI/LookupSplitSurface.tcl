##
 #  LookupSplitSurface.tcl: General surface interpolation module
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jully 1997
 #  Copyright (C) 1997 SCI Group
 #  Log Information:
 ##

itcl_class SCIRun_Surface_LookupSplitSurface {
    inherit Module
    constructor {config} {
        set name LookupSplitSurface
        set_defaults
    }
    method set_defaults {} {
	global $this-splitDirTCL
	global $this-splitValTCL
	set $this-splitDirTCL X
	set $this-splitValTCL 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
	
        toplevel $w
        wm minsize $w 100 30
        frame $w.f
        set n "$this-c needexecute "
	global $this-splitDirTCL
        make_labeled_radio $w.splitdir "Split Direction:" "" left \
		$this-splitDirTCL {X Y Z}
        global $this-splitValTCL
        expscale $w.splitval -orient horizontal -label "Split value:" \
                -variable $this-splitValTCL
	button $w.e -text "Execute" -command $n
	pack $w.splitdir $w.splitval $w.e -side top
	pack $w.f
    }
}
