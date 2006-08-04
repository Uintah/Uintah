itcl_class ModelCreation_FieldsGeometry_ScaleField {
    inherit Module
    constructor {config} {
        set name ScaleField
        set_defaults
    }

    method set_defaults {} {
        global $this-datascale
        global $this-geomscale
        global $this-usegeomcenter
    
        set $this-datascale 1
        set $this-geomscale 1
        set $this-usegeomcenter 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        iwidgets::labeledframe $w.frame -labeltext "SCALE FACTORS"
        set d [$w.frame childsite]
        pack $w.frame -fill both -expand yes
        
        label $d.lab1 -text "Data scaling factor"
        entry $d.e1 -textvariable $this-datascale       
        label $d.lab2 -text "Geometry scaling factor"
        entry $d.e2 -textvariable $this-geomscale       
        checkbutton $d.center -variable $this-usegeomcenter -text "Use center of geometry for scaling"
        
        grid $d.lab1 -row 0 -column 0  -sticky news
        grid $d.e1 -row 0 -column 1  -sticky news
        grid $d.lab2 -row 1 -column 0  -sticky news
        grid $d.e2 -row 1 -column 1  -sticky news        
        grid $d.center -row 2 -column 0 -columnspan 2 -sticky w

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


