itcl_class ModelCreation_FieldsExample_SphericalSurface {
    inherit Module
    constructor {config} {
        set name SphericalSurface
        set_defaults
    }

    method set_defaults {} {
        global $this-radius
        global $this-discretization    

        set $this-radius          1
        set $this-discretization  10  
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.f
        label $w.f.label1 -text "Discretization elements/(pi*radius)"
        entry $w.f.entry1 -textvariable $this-discretization 

        label $w.f.label2 -text "Sphere radius"
        entry $w.f.entry2 -textvariable $this-radius 

        grid $w.f.label1 -row 0 -column 0 -sticky news
        grid $w.f.entry1 -row 0 -column 1 -sticky news
        grid $w.f.label2 -row 1 -column 0 -sticky news
        grid $w.f.entry2 -row 1 -column 1 -sticky news

        pack $w.f -fill x
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


