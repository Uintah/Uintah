itcl_class Uintah_Operators_UdaScale {
    inherit Module
    constructor {config} {
        set name UdaScale
        set_defaults
    }

    method set_defaults {} {
#        global $this-cell-scale
#        set $this-cell-scale 1.0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        wm geometry $w ""

        set n "$this-c needexecute"
        
        expscale $w.scale -orient horizontal \
		-label "Cell Scale:" \
		-variable $this-cell-scale -command $n
        pack $w.scale -side top -expand yes -fill x

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


