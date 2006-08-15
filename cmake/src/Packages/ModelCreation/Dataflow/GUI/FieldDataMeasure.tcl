itcl_class ModelCreation_FieldsData_FieldDataMeasure {
    inherit Module
    constructor {config} {
        set name FieldDataMeasure
        set_defaults
    }

    method set_defaults {} {
      global $this-measure
      set $this-measure "sum"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        frame $w.f
        
        pack $w.f
        label $w.f.lab1 -text "Data Measure"
        grid $w.f.lab1 -row 0 -column 0 -sticky e
        myselectionbutton $w.f.sel1 0 1 { sum average minimum maximum } $this-measure
       
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


