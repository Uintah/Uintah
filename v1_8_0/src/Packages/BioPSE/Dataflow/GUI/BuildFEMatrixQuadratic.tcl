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

itcl_class BioPSE_Forward_BuildFEMatrixQuadratic {
    inherit Module
    constructor {config} {
        set name BuildFEMatrixQuadratic
        set_defaults
    }
    method set_defaults {} {
        global $this-BCFlag
	global $this-UseCondTCL
	global $this-refnodeTCL
        set $this-BCFlag "none"
	set $this-UseCondTCL 1
	set $this-refnodeTCL 0
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
                {{"None" none} \
                {"Apply Dirichlet" DirSub} \
                {"Ground Average DC" AverageGround} \
                {"Use Reference Node" PinZero}}
	global $this-refnodeTCL
	make_entry $w.f.pinned "Reference node:" $this-refnodeTCL {}
	global $this-UseCondTCL
	checkbutton $w.f.b -text "Use Conductivities" -variable $this-UseCondTCL
	pack $w.f.r $w.f.pinned $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
