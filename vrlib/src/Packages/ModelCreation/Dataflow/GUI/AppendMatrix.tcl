itcl_class ModelCreation_Math_AppendMatrix {
    inherit Module
    constructor {config} {
        set name AppendMatrix
        set_defaults
    }

    method set_defaults {} {
      global $this-row-or-column
      set $this-row-or-column "row"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
    
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        
        radiobutton $w.f.b1 -text "Append rows" -variable $this-row-or-column -value "row"
        radiobutton $w.f.b2 -text "Append columns" -variable $this-row-or-column -value "column"
        
        grid $w.f.b1 -column 0 -row 0 -sticky w
        grid $w.f.b2 -column 0 -row 1 -sticky w
        
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


