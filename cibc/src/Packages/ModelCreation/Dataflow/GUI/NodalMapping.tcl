itcl_class ModelCreation_FieldsData_NodalMapping {
    inherit Module
    constructor {config} {
        set name NodalMapping
        set_defaults
    }

    method set_defaults {} {
      global $this-mappingmethod
      global $this-def-value
      set $this-mappingmethod "InterpolatedData"
      set $this-def-value 0.0
    
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        
        frame $w.f
        
        pack $w.f
        label $w.f.lab1 -text "Interpolation/Extrapolation of source data"
        label $w.f.lab4 -text "Default value for unassigned data"
        grid $w.f.lab1 -row 0 -column 0 -sticky e
        grid $w.f.lab4 -row 1 -column 0 -sticky e
        
        myselectionbutton $w.f.sel1 0 1 { ClosestNodalData ClosestInterpolatedData InterpolatedData } $this-mappingmethod
        entry $w.f.ent -textvariable $this-def-value
        grid $w.f.ent -row 1 -column 1 -sticky news
        
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    


   method myselectionbutton { win x y arglist var} {

        frame $win 
        grid $win  -row $x -column $y -sticky news
        iwidgets::optionmenu $win.c -foreground darkred -command " $this comboget $win.c $var "

        set i 0
        set found 0
        set length [llength $arglist]
        for {set elem [lindex $arglist $i]} {$i<$length} {incr i 1; set elem [lindex $arglist $i]} {
          if {"$elem"=="[set $var]"} {
            set found 1
          }
          $win.c insert end $elem
        }

        if {!$found} {
          $win.c insert end [set $var]
        }

        $win.c select [set $var]
  
        pack $win.c	-fill x
    }



    method comboget { win var } {
        if {![winfo exists $win]} {
          return
        }
        if { "$var"!="[$win get]" } {
          set $var [$win get]
        }
    }
        
}


