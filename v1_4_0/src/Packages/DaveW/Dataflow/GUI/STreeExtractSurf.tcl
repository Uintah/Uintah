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
 #  Revision 1.8  2000/12/29 12:43:12  dmw
 #  accidentally blew it away... now its back again
 #
 #  Revision 1.4  1999/11/17 00:31:02  dmw
 #  fixed a typo -- all of these modules said Davew instead of DaveW
 #
 #  Revision 1.3  1999/09/02 21:30:45  moulding
 #  took out the modname method; it's in the base itcl clase module (module.tcl)
 #
 #  Revision 1.2  1999/08/29 01:02:20  dmw
 #  updated module names to be consistent with new loading mechanism
 #
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
catch {rename DaveW_EEG_STreeExtractSurf ""}

itcl_class DaveW_EEG_STreeExtractSurf {
    inherit Module
    constructor {config} {
        set name STreeExtractSurf
        set_defaults
    }
    method set_defaults {} {
	global $this-surfid
	set $this-surfid ""
	global $this-remapTCL
	set $this-remapTCL 1
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
	global $this-remapTCL
	checkbutton $w.b -text "Renumber points" -variable $this-remapTCL
        pack $w.f $w.b -side top -expand yes
    }
}
