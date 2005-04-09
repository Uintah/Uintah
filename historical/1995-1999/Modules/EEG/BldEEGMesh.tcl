##
 #  Coregister.tcl: The coregistration UI
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996
 #
 #  Copyright (C) 1996 SCI Group
 #
 ##

itcl_class BldEEGMesh {
    inherit Module
    constructor {config} {
        set name BldEEGMesh
        set_defaults
    }
    method set_defaults {} {
        global $this-npts
	global $this-NOBRAIN
	global $this-ADDBC
        set $this-npts 500
	set $this-NOBRAIN 1
	set $this-ADDBC 1
    }
    method ui {} {
        set w .ui$this
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 300 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        set n "$this-c needexecute "
        global $this-npts
	scale $w.f.np -orient horizontal -label "Number of nodes (approx): " \
		-variable $this-npts -showvalue true -from 100 -to 100000 \
		-resolution 100
	checkbutton $w.f.nob -text "Eliminate Material Inside Cortex" -variable $this-NOBRAIN
	checkbutton $w.f.add -text "Add Dirichlet Boundary Conditions" -variable $this-ADDBC
	pack $w.f.np $w.f.nob $w.f.add -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
