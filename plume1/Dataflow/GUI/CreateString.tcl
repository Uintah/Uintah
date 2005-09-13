
itcl_class SCIRun_String_CreateString {
    inherit Module
    constructor {config} {
        set name CreateString
        set_defaults
    }

    method set_defaults {} {
        global $this-inputstring
        global $this-get-inputstring
        set $this-inputstring ""
        set $this-get-inputstring "$this update_text"
    }

    method ui {} {

        global $this-inputstring
        global $this-get-inputstring
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
        frame $w.f     
        pack $w.f -expand yes -fill both

        option add *textBackground white	
        iwidgets::scrolledtext $w.f.str -vscrollmode dynamic \
            -labeltext "String Contents" 
        $w.f.str insert end [set $this-inputstring]
        pack $w.f.str -fill both -expand yes

        set $this-inputstring "$this update_text"

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }

    method update_text {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            global $this-inputstring
            set $this-inputstring [$w.f.str get 1.0 end]
        }
        }
}


