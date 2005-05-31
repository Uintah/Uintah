
itcl_class Uintah_Operators_Schlieren {

    inherit Module

    constructor {config} {
	set name Schlieren

        # Set up the GUI/C++ interaction variables.
        global $this-dx
        global $this-dy
        global $this-dz

	set $this-dx 1.0
	set $this-dy 1.0
	set $this-dz 1.0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.f_dx
	frame $w.f_dy
	frame $w.f_dz

        label $w.f_dx.dx_l -text "DX:"
        entry $w.f_dx.dx_e -textvariable $this-dx

        label $w.f_dy.dy_l -text "DY:"
        entry $w.f_dy.dy_e -textvariable $this-dy

        label $w.f_dz.dz_l -text "DZ:"
        entry $w.f_dz.dz_e -textvariable $this-dz

        pack $w.f_dx $w.f_dy $w.f_dz

        pack $w.f_dx.dx_l $w.f_dx.dx_e
        pack $w.f_dy.dy_l $w.f_dy.dy_e
        pack $w.f_dz.dz_l $w.f_dz.dz_e

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }

}
