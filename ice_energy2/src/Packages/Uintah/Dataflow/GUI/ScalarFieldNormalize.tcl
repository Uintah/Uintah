
itcl_class Uintah_Operators_ScalarFieldNormalize {
    inherit Module

    constructor {config} {
	set name ScalarFieldNormalize
       # Set up the GUI/C++ interaction variables.
       global $this-xIndex
       global $this-yIndex
       global $this-zIndex

	set $this-xIndex 0
	set $this-yIndex 0
	set $this-zIndex 0
    }
    method ui {} {
      set w .ui[modname]
      if {[winfo exists $w]} {
          return
      }
      toplevel $w

      frame $w.f_x
      frame $w.f_y
      frame $w.f_z

      label $w.f_x.dx_l -text "x cell index:"
      entry $w.f_x.dx_e -textvariable $this-xIndex

      label $w.f_y.dy_l -text "y cell index:"
      entry $w.f_y.dy_e -textvariable $this-yIndex

      label $w.f_z.dz_l -text "z cell index:"
      entry $w.f_z.dz_e -textvariable $this-zIndex

      pack $w.f_x $w.f_y $w.f_z

      pack $w.f_x.dx_l $w.f_x.dx_e
      pack $w.f_y.dy_l $w.f_y.dy_e
      pack $w.f_z.dz_l $w.f_z.dz_e

      makeSciButtonPanel $w $w $this
      moveToCursor $w
    }
}
