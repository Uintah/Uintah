itcl_class ModelCreation_FieldsCreate_DomainBoundary {
    inherit Module
    constructor {config} {
        set name DomainBoundary
        set_defaults
    }

    method set_defaults {} {
        global $this-userange
        global $this-usevalue
        global $this-minrange
        global $this-maxrange
        global $this-value
        global $this-includeouterboundary
        global $this-innerboundaryonly
        global $this-noinnerboundary
        global $this-disconnect
  
        set $this-userange 0
        set $this-usevalue 0
        set $this-minrange 0.0
        set $this-maxrange 255.0
        set $this-value 1.0
        set $this-includeouterboundary 1
        set $this-innerboundaryonly 0
        set $this-noinnerboundary 0
        set $this-disconnect 1
        
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.f
        checkbutton $w.f.userange -text "Only include compartments in the range between:" -variable $this-userange -command "if { [set $this-usevalue] == 1 } { set $this-usevalue 0} ; $w.f.usevalue deselect"
        checkbutton $w.f.usevalue -text "Only include compartment:" -variable $this-usevalue -command "if { [set $this-userange] == 1 } { set $this-userange 0} ; $w.f.userange deselect"

        label $w.f.valuelabel -text "value:"
        entry $w.f.value  -textvariable  $this-value

        label $w.f.minrangelabel -text "min:"
        entry $w.f.minrange  -textvariable  $this-minrange
        label $w.f.maxrangelabel -text "max:"
        entry $w.f.maxrange  -textvariable  $this-maxrange
        checkbutton $w.f.includeouterboundary -text "Include outer boundary" -variable $this-includeouterboundary
        checkbutton $w.f.innerboundaryonly -text "Include inner boundary only" -variable $this-innerboundaryonly
        checkbutton $w.f.noinnerboundary -text "Exclude inner boundary" -variable $this-noinnerboundary
        checkbutton $w.f.disconnect -text "Disconnect boundaries between different element types" -variable $this-disconnect

        grid $w.f.userange -column 0 -row 0 -columnspan 4 -sticky w
        grid $w.f.minrangelabel -column 0 -row 1 -sticky news
        grid $w.f.minrange -column 1 -row 1 -sticky news
        grid $w.f.maxrangelabel -column 2 -row 1 -sticky news
        grid $w.f.maxrange -column 3 -row 1 -sticky news
        grid $w.f.usevalue -column 0 -row 2 -columnspan 4 -sticky w
        grid $w.f.valuelabel -column 0 -row 3 -sticky news
        grid $w.f.value -column 1 -row 3 -sticky news
        grid $w.f.includeouterboundary -column 0 -row 4 -columnspan 4 -sticky w
        grid $w.f.innerboundaryonly -column 0 -row 5 -columnspan 4 -sticky w
        grid $w.f.noinnerboundary -column 0 -row 6 -columnspan 4 -sticky w
        grid $w.f.disconnect -column 0 -row 7 -columnspan 4 -sticky w

        pack $w.f -fill x
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


