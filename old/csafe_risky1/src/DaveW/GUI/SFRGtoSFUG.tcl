##
 #  SFRGtoSFUG.tcl: The SFRGtoSFUG UI
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #
 #  Copyright (C) 2000 SCI Group
 #
 ##

catch {rename DaveW_EEG_SFRGtoSFUG ""}

itcl_class DaveW_EEG_SFRGtoSFUG {
    inherit Module
    constructor {config} {
        set name SFRGtoSFUG
        set_defaults
    }
    method set_defaults {} {
        global $this-npts
        set $this-scalarAsCondTCL 1
	set $this-removeAirTCL 1
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 300 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-scalarAsCondTCL
	checkbutton $w.f.s -text "Scalar As Conductivity Index" -variable $this-scalarAsCondTCL
	global $this-removeAirTCL
	checkbutton $w.f.r -text "Remove Air Elements" -variable $this-removeAirTCL
	pack $w.f.s $w.f.r -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
