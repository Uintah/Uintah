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
        global $this-DirSubFlag
        set $this-DirSubFlag 1
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
        global $this-DirSubFlag
	checkbutton $w.f.b -variable $this-DirSubFlag -text "Dirichlet Substitution"
	pack $w.f.b -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
