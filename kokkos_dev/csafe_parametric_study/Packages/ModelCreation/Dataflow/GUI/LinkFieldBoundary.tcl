itcl_class ModelCreation_FieldsGeometry_LinkFieldBoundary {
    inherit Module
    constructor {config} {
        set name LinkFieldBoundary
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

        iwidgets::Labeledframe $w.f -labelpos nw \
                    -labeltext "Link Field over opposing boundaries" 
        pack $w.f -fill x
        set f [$w.f childsite]

        checkbutton $f.linkx -variable $this-linkx \
          -text "Link in the X-direction"
        grid $f.linkx -column 0 -row 0 -sticky news
        checkbutton $f.linky -variable $this-linky \
          -text "Link in the Y-direction"        
        grid $f.linky -column 0 -row 1 -sticky news
        checkbutton $f.linkz -variable $this-linkz \
          -text "Link in the Z-direction"
        grid $f.linkz -column 0 -row 2 -sticky news
        
        iwidgets::entryfield $f.tol \
          -labeltext "Tolerance for matching node positions" \
          -textvariable $this-tol
        grid $f.tol -column 0 -row 3 -sticky news

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


