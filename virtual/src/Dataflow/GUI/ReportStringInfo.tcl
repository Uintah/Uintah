itcl_class SCIRun_String_ReportStringInfo {
    inherit Module

    constructor {config} {
        set name ReportStringInfo
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

        iwidgets::scrolledtext $w.f.str -vscrollmode dynamic \
            -labeltext "String Contents"  
        $w.f.str insert end [set $this-inputstring]
        pack $w.f.str -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }

    method update {} {
    
        global $this-inputstring
        set w .ui[modname]
        if {[winfo exists $w]} {
            $w.f.str clear
            $w.f.str insert end [set $this-inputstring]
         }
    }

}


