itcl_class ModelCreation_FieldsData_IsInsideField {
    inherit Module
    constructor {config} {
        set name IsInsideField
        set_defaults
    }
    
    method set_defaults {} {
      global $this-outputbasis
      global $this-outputtype

      global $this-outval
      global $this-inval
      global $this-partial-inside
      
      set $this-outputbasis "same as input"
      set $this-outputtype "double"
      set $this-outval 0
      set $this-inval  1
      set $this-partial-inside 0
      
    }
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        
        toplevel $w
        
        frame $w.f
        pack $w.f
        label $w.f.lab1 -text "Data location"
        grid $w.f.lab1 -row 0 -column 0 -sticky e
        label $w.f.lab2 -text "Data type"
        grid $w.f.lab2 -row 1 -column 0 -sticky e
        label $w.f.lab3 -text "Outside value"
        grid $w.f.lab3 -row 2 -column 0 -sticky e
        label $w.f.lab4 -text "Inside value"
        grid $w.f.lab4 -row 3 -column 0 -sticky e
        
        
        
        myselectionbutton $w.f.sel1 0 1 { "same as input" "linear" "constant" } $this-outputbasis
        myselectionbutton $w.f.sel2 1 1 { "same as input" "char" "short" "unsigned short" "unsigned int" "int" "float" "double" } $this-outputtype
        entry $w.f.e1 -textvariable $this-outval
        entry $w.f.e2 -textvariable $this-inval
        checkbutton $w.f.e3 -text "Count partial inside elements" -variable $this-partial-inside
        
        grid $w.f.e1 -row 2 -column 1 -sticky news
        grid $w.f.e2 -row 3 -column 1 -sticky news
        grid $w.f.e3 -row 4 -column 0 -columnspan 2
        
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

    