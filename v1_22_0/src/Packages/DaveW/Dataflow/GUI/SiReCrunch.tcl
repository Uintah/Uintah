##
 #  SiReCrunch.tcl: Crunch the data -- k-space to image space
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1998
 #  Copyright (C) 1998 SCI Group
 ##

catch {rename DaveW_SiRe_SiReCrunch ""}

itcl_class DaveW_SiRe_SiReCrunch {
    inherit Module
    constructor {config} {
        set name SiReCrunch
        set_defaults
    }
    method set_defaults {} {
        global $this-npts
        set $this-npts 500
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
        set n "$this-c needexecute "
        global $this-npts
	scale $w.f.np -orient horizontal -label "Number of nodes (approx): " \
		-variable $this-npts -showvalue true -from 100 -to 100000 \
		-resolution 100
	pack $w.f.np -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
