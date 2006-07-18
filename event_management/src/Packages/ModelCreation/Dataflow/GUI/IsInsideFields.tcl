itcl_class ModelCreation_FieldsData_IsInsideFields {
    inherit Module
    constructor {config} {
        set name IsInsideFields
        set_defaults
    }
    
    method set_defaults {} {
      global $this-outputbasis
      global $this-outputtype
      set $this-outputbasis "same as input"
      set $this-outputtype "double"
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
        
        myselectionbutton $w.f.sel1 0 1 { "same as input" "linear" "constant" } $this-outputbasis
        myselectionbutton $w.f.sel2 1 1 { "same as input" "char" "short" "unsigned short" "unsigned int" "int" "float" "double" } $this-outputtype
        
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

        bind $win.c <Map> "$win.c select {[set $var]}"
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

    