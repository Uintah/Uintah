
itcl_class Uintah_Operators_CompareMMS {
    inherit Module

    constructor {config} {
	set name CompareMMS
	set_defaults
    }

    method set_defaults {} {
    }

    method ui {} {
	
	set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}

