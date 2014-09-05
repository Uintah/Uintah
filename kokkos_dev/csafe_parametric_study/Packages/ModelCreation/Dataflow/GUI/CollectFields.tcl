itcl_class ModelCreation_FieldsCreate_CollectFields {
    inherit Module
    constructor {config} {
        set name CollectFields
        set_defaults
    }

    method set_defaults {} {
        global $this-buffersize
        set $this-buffersize 20
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        iwidgets::entryfield $w.bs \
          -labeltext "Buffer Size" \
          -textvariable $this-buffersize
        pack $w.bs -side top -expand yes -fill x

        button $w.reset -text "Reset Buffer" -command "$this-c reset; $this-c needexecute"       
        pack $w.reset -side top 

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


