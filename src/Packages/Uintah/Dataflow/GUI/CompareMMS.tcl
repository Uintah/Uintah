
itcl_class Uintah_Operators_CompareMMS {
    inherit Module

    constructor {config} {
	set name CompareMMS
	set_defaults
    }

    method set_defaults {} {
        global $this-field_name
        set    $this-field_name "---"
    }

    method ui {} {
	
	set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.f1
        pack  $w.f1

        label_pair $w.f1.field_name "Field Name" $this-field_name

        pack $w.f1.field_name -padx 10 -pady 10

	makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}

