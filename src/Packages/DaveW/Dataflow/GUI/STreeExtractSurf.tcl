##
 #  STreeExtractSurf.tcl: Label a specific surface in a surftree
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jully 1997
 #
 #  Copyright (C) 1997 SCI Group
 # 
 #  Log Information:
 #
 #  $Log$
 #  Revision 1.1  1999/08/24 06:22:56  dmw
 #  Added in everything for the DaveW branch
 #
 #  Revision 1.1.1.1  1999/04/24 23:12:17  dav
 #  Import sources
 #
 #  Revision 1.1  1999/01/04 05:31:56  dmw
 #  See Dave for details...
 #
 #  Revision 1.1  1997/08/23 06:27:19  dweinste
 #  Some trivial modules that I needed...
 #
 #
 ##

itcl_class STreeExtractSurf {
    inherit Module
    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }
    constructor {config} {
        set name STreeExtractSurf
        set_defaults
    }
    method set_defaults {} {
	global $this-surfid
	set $this-surfid ""
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
	global $this-surfid
	frame $w.f.surf
	label $w.f.surf.l -text "SurfId: "
	entry $w.f.surf.e -relief sunken -width 10 -textvariable $this-surfid
	pack $w.f.surf.l $w.f.surf.e -side left
	pack $w.f.surf -side left -fill x
        pack $w.f -side top -expand yes
    }
}
