itcl_class CardioWave_DiscreteMultiDomain_DMDConductionVelocity {
    inherit Module
    constructor {config} {
        set name DMDConductionVelocity
        set_defaults
    }

    method set_defaults {} {
      global $this-distance
      global $this-threshold
      
      set $this-threshold 0
      set $this-distance 1
    }

    method ui {} {
      set w .ui[modname]
      if {[winfo exists $w]} {
          return
      }
      toplevel $w


      iwidgets::labeledframe $w.m -labeltext "COMPUTE CONDUCTION VELOCITY"
      set cv [$w.m childsite]
      pack $w.m -fill both -expand yes

      label $cv.lab1 -text "Distance"
      entry $cv.en1 -textvariable $this-distance
      label $cv.lab2 -text "Activation Threshold"
      entry $cv.en2 -textvariable $this-threshold

	    grid $cv.lab1 -row 0 -column 0
      grid $cv.en1 -row 0 -column 1
	    grid $cv.lab2 -row 1 -column 0
      grid $cv.en2 -row 1 -column 1

      makeSciButtonPanel $w $w $this
      moveToCursor $w
    }
}


