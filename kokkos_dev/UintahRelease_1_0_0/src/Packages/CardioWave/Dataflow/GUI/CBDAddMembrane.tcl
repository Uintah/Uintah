itcl_class CardioWave_ContinuousBiDomain_CBDAddMembrane {
    inherit Module
    constructor {config} {
        set name CBDAddMembrane
        set_defaults
    }

    method set_defaults {} {
      global $this-mem-names
      global $this-mem-name
      global $this-mem-param
      global $this-mem-desc
      
      set $this-mem-names {}
      set $this-mem-param ""
      set $this-mem-desc ""
    
    }

    method ui {} {
      
      set w .ui[modname]
      if {[winfo exists $w]} {
          return
      }

      $this-c get_membrane_names
      $this-c set_membrane [set $this-mem-name]

      toplevel $w 
      wm minsize $w 100 150

      iwidgets::labeledframe $w.m -labeltext "MEMBRANE MODEL"
      set mem [$w.m childsite]
      pack $w.m -fill both -expand yes

      set names [set $this-mem-names]      
      labelcombo $mem.select "Membrane model" $names $this-mem-name "$this update_membrane"

      grid $mem.select -row 0 -column 0 -sticky w

      option add *textBackground white	
      iwidgets::scrolledtext $mem.param -vscrollmode dynamic \
            -labeltext "Membrane parameters (default values)" -height 150 
      $mem.param insert end [set $this-mem-param]
      grid $mem.param -row 1 -column 0 -sticky news
      
      iwidgets::scrolledtext $mem.desc -vscrollmode dynamic \
            -labeltext "Description of Membrane parameters" -height 100
            
      grid $mem.desc -row 2 -column 0 -sticky news
      $mem.desc insert end [set $this-mem-desc]
      
      makeSciButtonPanel $w $w $this
      moveToCursor $w
    }


  method get_param {} {

      set w .ui[modname]
      if {[winfo exists $w]} {
        set mem [$w.m childsite]
        set $this-mem-param [$mem.param get 1.0 end]  
      }    

  }

  method update_membrane {} {

      $this-c set_membrane [set $this-mem-name]

      set w .ui[modname]
      if {[winfo exists $w]} {
 
        set mem [$w.m childsite]   
        $mem.param clear
        $mem.param insert end [set $this-mem-param]

        $mem.desc clear
        $mem.desc insert end [set $this-mem-desc] 
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


