##
 #  STreeExtractSurf.tcl: Label a specific surface in a surftree
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jully 1997
 #  Copyright (C) 1997 SCI Group
 #  Log Information:
 ##
catch {rename Packages/DaveW_EEG_STreeExtractSurf ""}

itcl_class Packages/DaveW_EEG_STreeExtractSurf {
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
