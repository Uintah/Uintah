itcl_class CardioWave_DiscreteMultiDomain_DMDAddReference {
    inherit Module
    constructor {config} {
        set name DMDAddReference
        set_defaults
    }

    method set_defaults {} {
      global $this-referencevalue
      global $this-usereferencevalue
      
      set $this-referencevalue 0.0
      set $this-usefieldvalue 1  
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
      checkbutton $ref.cb1 -text "Use data in field as reference potential" -variable $this-usefieldvalue

	    grid $ref.lab1 -row 0 -column 0
      grid $ref.en1 -row 0 -column 1
      grid $ref.cb1 -row 1 -column 0 -columnspan 2      

      makeSciButtonPanel $w $w $this
      moveToCursor $w
    }

}


