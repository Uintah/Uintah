##
 #  BuildFEMatrix.tcl
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

itcl_class PSECommon_FEM_BuildFEMatrix {
    inherit Module
    constructor {config} {
        set name BuildFEMatrix
        set_defaults
    }
    method set_defaults {} {
        global $this-BCFlag
	global $this-UseCondTCL
        set $this-BCFlag "none"
	set $this-UseCondTCL 1
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
                {{"None" none} \
                {"Apply Dirichlet" DirSub} \
                {"Pin Node Zero" PinZero}}
	global $this-UseCondTCL
	checkbutton $w.f.b -text "Use Conductivities" -variable $this-UseCondTCL
	pack $w.f.r $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
