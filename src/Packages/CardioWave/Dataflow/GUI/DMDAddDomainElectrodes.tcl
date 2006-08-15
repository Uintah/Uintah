itcl_class CardioWave_DiscreteMultiDomain_DMDAddDomainElectrodes {
    inherit Module
    constructor {config} {
        set name DMDAddDomainElectrodes
        set_defaults
    }

    method set_defaults {} {
      global $this-electrodedomain
      set $this-electrodedomain 0.0      
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
     toplevel $w 
      wm minsize $w 100 50

      iwidgets::labeledframe $w.m -labeltext "DOMAIN ELECTRODE"
      set ref [$w.m childsite]
      pack $w.m -fill both -expand yes

      label $ref.lab1 -text "Electrode Domain"
      entry $ref.en1 -textvariable $this-electrodedomain

	    grid $ref.lab1 -row 0 -column 0
      grid $ref.en1 -row 0 -column 1

      makeSciButtonPanel $w $w $this
      moveToCursor $w

    }
}


