itcl_class ModelCreation_FieldsData_ModalMapping {
    inherit Module
    constructor {config} {
        set name ModalMapping
        set_defaults
    }

    method set_defaults {} {
      global $this-mappingmethod
      global $this-integrationmethod
      global $this-integrationfilter
      global $this-def-value
      set $this-mappingmethod "InterpolatedData"
      set $this-integrationmethod "Gaussian2"
      set $this-integrationfilter "Average"
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
        label $w.f.lab2 -text "Integration Sampling Nodes"
        label $w.f.lab3 -text "Integration Sampling Method"
        label $w.f.lab4 -text "Default value for unassigned data"
        grid $w.f.lab1 -row 0 -column 0 -sticky e
        grid $w.f.lab2 -row 1 -column 0 -sticky e
        grid $w.f.lab3 -row 2 -column 0 -sticky e
        grid $w.f.lab4 -row 3 -column 0 -sticky e
        
        myselectionbutton $w.f.sel1 0 1 { ClosestNodalData ClosestInterpolatedData InterpolatedData } $this-mappingmethod
        myselectionbutton $w.f.sel2 1 1 { Gaussian1 Gaussian2 Gaussian3 Regular1 Regular2 Regular3 Regular4 Regular5 } $this-integrationmethod
        myselectionbutton $w.f.sel3 2 1 { Integrate Average WeightedAverage Sum Median Minimum Maximum MostCommon} $this-integrationfilter
        entry $w.f.ent -textvariable $this-def-value
        grid $w.f.ent -row 3 -column 1 -sticky news
        
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
