itcl_class CardioWave_ContinuousBiDomain_CBDSetupSimulation {
    inherit Module
    constructor {config} {
        set name CBDSetupSimulation
        set_defaults
    }

    method set_defaults {} {
    
      global $this-solver-names
      global $this-solver-name
      global $this-solver-param
      global $this-solver-desc
      
      set $this-solver-names {}
      set $this-solver-param ""
      set $this-solver-desc ""
      set $this-solver-name ""

      global $this-tstep-names
      global $this-tstep-name
      global $this-tstep-param
      global $this-tstep-desc
      
      set $this-tstep-names {}
      set $this-tstep-param ""
      set $this-tstep-desc ""
      set $this-tstep-name ""

      global $this-output-names
      global $this-output-name
      global $this-output-param
      global $this-output-desc
      
      set $this-output-names {}
      set $this-output-param ""
      set $this-output-desc ""
      set $this-output-name ""

      global $this-cwave-param
      global $this-cwave-desc
      set $this-cwave-param ""
      set $this-cwave-desc ""
    
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        puts "test"
        $this-c c_set_defaults

        toplevel $w
        wm minsize $w 100 150
        
        iwidgets::labeledframe $w.mainframe -labeltext "CARDIOWAVE SIMULATOR PARAMETERS"
        set mf [$w.mainframe childsite]
        pack $w.mainframe -fill both -expand yes

        iwidgets::labeledframe $mf.solver -labeltext "SOLVER"
        set solver [$mf.solver childsite]
        grid $mf.solver -row 0 -column 0 -sticky news

        set names [set $this-solver-names]      
        labelcombo $solver.select "Solver Method" $names $this-solver-name "$this update_solver"
        grid $solver.select -row 0 -column 0 -sticky w
        option add *textBackground white	
        iwidgets::scrolledtext $solver.param -vscrollmode dynamic \
              -labeltext "Parameters" -height 150 -width 400 
        $solver.param insert end [set $this-solver-param]
        grid $solver.param -row 1 -column 0 -sticky news
        iwidgets::scrolledtext $solver.desc -vscrollmode dynamic \
              -labeltext "Description" -height 100 -width 400
        grid $solver.desc -row 2 -column 0 -sticky news
        $solver.desc insert end [set $this-solver-desc]


        iwidgets::labeledframe $mf.tstep -labeltext "TIME STEPPER"
        set tstep [$mf.tstep childsite]
        grid $mf.tstep -row 0 -column 1 -sticky news

        set names [set $this-tstep-names]      
        labelcombo $tstep.select "Timestepper Method" $names $this-tstep-name "$this update_tstep"
        grid $tstep.select -row 0 -column 0 -sticky w
        option add *textBackground white	
        iwidgets::scrolledtext $tstep.param -vscrollmode dynamic \
              -labeltext "Parameters" -height 150 -width 400 
        $tstep.param insert end [set $this-tstep-param]
        grid $tstep.param -row 1 -column 0 -sticky news
        iwidgets::scrolledtext $tstep.desc -vscrollmode dynamic \
              -labeltext "Description" -height 100 -width 400
        grid $tstep.desc -row 2 -column 0 -sticky news
        $tstep.desc insert end [set $this-tstep-desc]


        iwidgets::labeledframe $mf.output -labeltext "OUTPUT"
        set output [$mf.output childsite]
        grid $mf.output -row 1 -column 0 -sticky news

        set names [set $this-output-names]      
        labelcombo $output.select "Output Method" $names $this-output-name "$this update_output"
        grid $output.select -row 0 -column 0 -sticky w
        option add *textBackground white	
        iwidgets::scrolledtext $output.param -vscrollmode dynamic \
              -labeltext "Parameters" -height 150 -width 400 
        $output.param insert end [set $this-output-param]
        grid $output.param -row 1 -column 0 -sticky news
        iwidgets::scrolledtext $output.desc -vscrollmode dynamic \
              -labeltext "Description" -height 100 -width 400
        grid $output.desc -row 2 -column 0 -sticky news
        $output.desc insert end [set $this-output-desc]


        iwidgets::labeledframe $mf.cwave -labeltext "GENERAL PARAMETERS"
        set cwave [$mf.cwave childsite]
        grid $mf.cwave -row 1 -column 1 -sticky news

        option add *textBackground white	
        iwidgets::scrolledtext $cwave.param -vscrollmode dynamic \
              -labeltext "Parameters" -height 150 -width 400 
        $cwave.param insert end [set $this-cwave-param]
        pack $cwave.param -fill both -expand yes

        iwidgets::scrolledtext $cwave.desc -vscrollmode dynamic \
              -labeltext "Description" -height 100 -width 400
        $cwave.desc insert end [set $this-cwave-desc]
        pack $cwave.desc -fill both -expand yes
 
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
 
  method get_param {} {

      set w .ui[modname]
      if {[winfo exists $w]} {
        set mf [$w.mainframe childsite]
        set solver [$mf.solver childsite]  
        set $this-solver-param [$solver.param get 1.0 end]
        set tstep [$mf.tstep childsite]  
        set $this-tstep-param [$tstep.param get 1.0 end]
        set output [$mf.output childsite]  
        set $this-output-param [$output.param get 1.0 end]
        set cwave [$mf.cwave childsite]  
        set $this-cwave-param [$cwave.param get 1.0 end]
      }    

  }

  method update_cwave {} {

      $this-c set_cwave ""

      set w .ui[modname]
      if {[winfo exists $w]} {
 
        set mf [$w.mainframe childsite]
        set cwave [$mf.cwave childsite]   
        $cwave.param clear
        $cwave.param insert end [set $this-cwave-param]
      }    
    }


  method update_solver {} {

      $this-c set_solver [set $this-solver-name]

      set w .ui[modname]
      if {[winfo exists $w]} {

        set mf [$w.mainframe childsite]
        set solver [$mf.solver childsite]   
        $solver.param clear
        $solver.param insert end [set $this-solver-param]

        $solver.desc clear
        $solver.desc insert end [set $this-solver-desc] 
      }    
    }

  method update_tstep {} {

      $this-c set_tstep [set $this-tstep-name]

      set w .ui[modname]
      if {[winfo exists $w]} {

        set mf [$w.mainframe childsite]
        set tstep [$mf.tstep childsite]   
        $tstep.param clear
        $tstep.param insert end [set $this-tstep-param]

        $tstep.desc clear
        $tstep.desc insert end [set $this-tstep-desc] 
      }    
    }

  method update_output {} {

      $this-c set_output [set $this-output-name]

      set w .ui[modname]
      if {[winfo exists $w]} {

        set mf [$w.mainframe childsite]
        set output [$mf.output childsite]   
        $output.param clear
        $output.param insert end [set $this-output-param]

        $output.desc clear
        $output.desc insert end [set $this-output-desc] 
      }    
    }



  method labelcombo { win text1 arglist var cmd} {
      frame $win 
      pack $win -side top -padx 5
      label $win.l1 -text $text1 -anchor w -just left
      label $win.colon  -text ":" -width 2 -anchor w -just left
      iwidgets::optionmenu $win.c -foreground darkred \
        -command " $this comboget $win.c $var; $cmd "

      set i 0
      set found 0
      set length [llength $arglist]
      for {set elem [lindex $arglist $i]} {$i<$length} \
        {incr i 1; set elem [lindex $arglist $i]} {
        if {"$elem"=="[set $var]"} {
          set found 1
        }
        $win.c insert end $elem
      }

      if {!$found} {
        $win.c insert end [set $var]
      }

      $win.c select [set $var]
  
      label $win.l2 -text "" -width 20 -anchor w -just left

      # hack to associate optionmenus with a textvariable
      # bind $win.c <Map> "$win.c select {[set $var]}"

      pack $win.l1 $win.colon -side left
      pack $win.c $win.l2 -side left	
    }

  method comboget { win var } {
      if {![winfo exists $win]} {
          return
      }
      if { "$var"!="[$win get]" } {
          set $var [$win get]
      }
    }

  method set_combobox { win var name1 name2 op } {
      set w .ui[modname]
      set menu $w.$win
      if {[winfo exists $menu]} {
          $menu select $var
      }
    }
    
    
    
    
}


