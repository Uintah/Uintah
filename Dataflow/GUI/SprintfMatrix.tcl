itcl_class SCIRun_String_SprintfMatrix {
    inherit Module
    constructor {config} {
        set name SprintfMatrix
        set_defaults
    }

    method set_defaults {} {
        global $this-formatstring
        set    $this-formatstring "time: %5.4f ms"    
    
    }

    method ui {} {
    
    		global $this-formatstring
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.frame
        pack $w.frame -side top -fill x -expand yes -padx 5 -pady 5
        frame $w.frame2
        pack $w.frame2 -side top -fill x -expand yes -padx 5 -pady 5
        
        label $w.frame.label -text "Format string :"
        entry $w.frame.string -textvariable $this-formatstring
        pack $w.frame.label -side left 
        pack $w.frame.string -side right -fill x -expand yes

        label $w.frame2.label -text "Available format strings :"
        label $w.frame2.string -text "%a %d %e %f %g %i %x %E %F %G %A"
        pack $w.frame2.label -side top -anchor w
        pack $w.frame2.string -side top -anchor w


        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


