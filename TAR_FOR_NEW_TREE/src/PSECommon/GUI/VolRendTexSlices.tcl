##
 #  VolRendTexSlices.tcl: The volume rendering by slices UI
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1997
 #
 #  Copyright (C) 1997 SCI Group
 #
 ##

itcl_class PSECommon_Visualization_VolRendTexSlices {
    inherit Module
    constructor {config} {
        set name VolRendTexSlices
        set_defaults
    }
    method set_defaults {} {
        global $this-accum
        global $this-bright
        set $this-accum 0.1
        set $this-bright 0.6
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
	global $this-accum
	global $this-bright
	scale $w.f.a -orient horizontal -label "Accumulation: " \
		-variable $this-accum -showvalue true -from 0.00 -to 1.00 \
		-resolution 0.01
	set $this-accum 0.1
	scale $w.f.b -orient horizontal -label "Brightness: " \
		-variable $this-bright -showvalue true -from 0.00 -to 1.00 \
		-resolution 0.01
	set $this-bright 0.6
	pack $w.f.a $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
