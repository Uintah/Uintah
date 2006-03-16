
itcl_class SCIRun_String_CreateString {
    inherit Module

    constructor {config} {
        set name CreateString
    }

    method ui {} {

        global $this-inputstring
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


