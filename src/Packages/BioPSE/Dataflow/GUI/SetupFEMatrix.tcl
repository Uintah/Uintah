##
 #  SetupFEMatrix.tcl
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996, March 2001
 #  Copyright (C) 1996 SCI Group
 ##

catch {rename BioPSE_Forward_SetupFEMatrix ""}

itcl_class BioPSE_Forward_SetupFEMatrix {
    inherit Module
    constructor {config} {
        set name SetupFEMatrix
        set_defaults
    }
    method set_defaults {} {
        global $this-BCFlag
	global $this-UseCondTCL
        set $this-BCFlag "none"
	set $this-UseCondTCL 1
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        global $this-BCFlag
        make_labeled_radio $w.f.r "Boundary Conditions:" "" \
                left $this-BCFlag \
                {{"Apply Dirichlet" DirSub} \
                {"No Dirichlet (ping to zero)" PinZero}}
	
	global $this-UseCondTCL
	checkbutton $w.f.b -text "Use Conductivities" -variable $this-UseCondTCL -onvalue 1 -offvalue 0
	pack $w.f.r $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
