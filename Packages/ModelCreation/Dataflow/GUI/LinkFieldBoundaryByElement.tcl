itcl_class ModelCreation_FiniteElements_LinkFieldBoundaryByElement {
    inherit Module
    constructor {config} {
        set name LinkFieldBoundaryByElement
        set_defaults
    }

    method set_defaults {} {
        global $this-tol 
        set $this-tol 1e-6
        
        global $this-linkx
        set $this-linkx 1
        global $this-linky
        set $this-linky 1
        global $this-linkz
        set $this-linkz 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        checkbutton $w.linkx -variable $this-linkx \
          -text "Link in the X-direction"
        grid $w.linkx -column 0 -row 0 -sticky news
        checkbutton $w.linky -variable $this-linky \
          -text "Link in the Y-direction"
        grid $w.linky -column 0 -row 1 -sticky news
        checkbutton $w.linkz -variable $this-linkz \
          -text "Link in the Z-direction"
        grid $w.linkz -column 0 -row 2 -sticky news
        
        iwidgets::entryfield $w.tol \
          -labeltext "Tolerance for matching node positions" \
          -textvariable $this-tol
        grid $w.tol -column 0 -row 3 -sticky news

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


