##
 #  SelectSurfNodes.tcl: The UI
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   July 1998
 #
 #  Copyright (C) 1998 SCI Group
 #
 ##

itcl_class SelectSurfNodes {
    inherit Module
    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }
    constructor {config} {
        set name SelectSurfNodes
        set_defaults
    }
    method set_defaults {} {
        global $this-method
        global $this-sphereSize
	set $this-method addnode
	set $this-sphereSize 1.0
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
        global $this-method
        global $this-sphereSize
	make_labeled_radio $w.f.m "Method: " "" \
		left $this-method \
		{{"Add Node" addnode} \
		{"Delete Node" delnode}}
	scale $w.f.s -orient horizontal -label "Sphere Radius: " \
		-variable $this-sphereSize -showvalue true \
		-from 0.01 -to 100.00 -resolution 0.01
	button $w.f.b -text "Clear" -command "$this-c clear"
	pack $w.f.m $w.f.s $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
