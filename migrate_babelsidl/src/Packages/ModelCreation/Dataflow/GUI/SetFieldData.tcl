itcl_class ModelCreation_FieldsData_SetFieldData {
    inherit Module
    constructor {config} {
        set name SetFieldData
        set_defaults
    }

    method set_defaults {} {
      global $this-keepscalartype
      set $this-keepscalartype 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        checkbutton $w.kst -text "Keep scalar field input type" \
          -variable $this-keepscalartype
        pack $w.kst
        
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


