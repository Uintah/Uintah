##
 #  LookupSurface.tcl: General surface interpolation module
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jully 1997
 #  Copyright (C) 1997 SCI Group
 #  Log Information:
 ##

itcl_class SCIRun_Surface_LookupSurface {
    inherit Module
    constructor {config} {
        set name LookupSurface
        set_defaults
    }
    method set_defaults {} {
	global $this-constElemsTCL
	set $this-constElemsTCL 0
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
	global $this-constElemsTCL
	checkbutton $w.f.b -text "Constant Elements" \
		-variable $this-constElemsTCL
	button $w.f.e -text "Execute" -command $n
	pack $w.f.b $w.f.e -side top
	pack $w.f
    }
}
