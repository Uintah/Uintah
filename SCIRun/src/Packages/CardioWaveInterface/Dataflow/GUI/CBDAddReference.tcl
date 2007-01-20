itcl_class CardioWaveInterface_ContinuousBiDomain_CBDAddReference {
    inherit Module
    constructor {config} {
        set name CBDAddReference
        set_defaults
    }

    method set_defaults {} {
      global $this-referencevalue
      global $this-referencedomain
      global $this-usereferencevalue
      global $this-useelements

      set $this-referencedomain "ExtraCellular"      
      set $this-referencevalue 0.0
      set $this-usefieldvalue 1
      set $this-uselements 1  
    }

    method ui {} {
      
      set w .ui[modname]
      if {[winfo exists $w]} {
          return
      }

      toplevel $w 
      wm minsize $w 100 150

      iwidgets::labeledframe $w.m -labeltext "REFERENCE ELECTRODE"
      set ref [$w.m childsite]
      pack $w.m -fill both -expand yes

      label $ref.lab1 -text "Reference Value(mV)"
      entry $ref.en1 -textvariable $this-referencevalue
      label $ref.lab2 -text "Reference Domain"
      labelcombo $ref.en2 {"ExtraCellular" "IntraCellular"} $this-referencedomain

      checkbutton $ref.cb1 -text "Use data in field as reference potential" -variable $this-usefieldvalue
      checkbutton $ref.cb2 -text "Define reference only for elements contained with geometry"

	    grid $ref.lab1 -row 0 -column 0
      grid $ref.en1 -row 0 -column 1
	    grid $ref.lab2 -row 1 -column 0
      grid $ref.en2 -row 1 -column 1
      grid $ref.cb1 -row 2 -column 0 -columnspan 2      
      grid $ref.cb2 -row 3 -column 0 -columnspan 2

      makeSciButtonPanel $w $w $this
      moveToCursor $w
    }


  method labelcombo { win arglist var} {
      frame $win 
      pack $win -side top -padx 5
      iwidgets::optionmenu $win.c -foreground darkred \
        -command " $this comboget $win.c $var;"

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
  
      pack $win.c -side left	
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


