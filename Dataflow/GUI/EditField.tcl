itcl_class SCIRun_Fields_EditField {
    inherit Module
    constructor {config} {
        set name EditField
        set_defaults
    }

    method set_defaults {} {
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

    }
}


