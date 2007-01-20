itcl_class CardioWaveInterface_DiscreteMultiDomain_DMDAddReference {
    inherit Module
    constructor {config} {
        set name DMDAddReference
        set_defaults
    }

    method set_defaults {} {
      global $this-referencevalue
      global $this-referencedomain
      global $this-usereferencevalue
      global $this-useelements

      set $this-referencedomain 0.0      
      set $this-referencevalue 0.0
      set $this-usefieldvalue 1
      set $this-uselements 1  
    }

    method ui {} {
      
      set w .ui[modname]
      if {[winfo exists $w]} {
          return
      }

      toplevel $w 
      wm minsize $w 100 150

      iwidgets::labeledframe $w.m -labeltext "REFERENCE ELECTRODE"
      set ref [$w.m childsite]
      pack $w.m -fill both -expand yes

      label $ref.lab1 -text "Reference Value(mV)"
      entry $ref.en1 -textvariable $this-referencevalue
      label $ref.lab2 -text "Reference Domain"
      entry $ref.en2 -textvariable $this-referencedomain

      checkbutton $ref.cb1 -text "Use data in field as reference potential" -variable $this-usefieldvalue
      checkbutton $ref.cb2 -text "Define reference only for elements contained with geometry" -variable $this-useelements

	    grid $ref.lab1 -row 0 -column 0
      grid $ref.en1 -row 0 -column 1
	    grid $ref.lab2 -row 1 -column 0
      grid $ref.en2 -row 1 -column 1
      grid $ref.cb1 -row 2 -column 0 -columnspan 2      
      grid $ref.cb2 -row 3 -column 0 -columnspan 2

      makeSciButtonPanel $w $w $this
      moveToCursor $w
    }

}


