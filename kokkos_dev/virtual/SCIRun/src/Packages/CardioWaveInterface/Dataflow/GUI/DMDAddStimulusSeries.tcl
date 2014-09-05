itcl_class CardioWaveInterface_DiscreteMultiDomain_DMDAddStimulusSeries {
    inherit Module
    constructor {config} {
        set name DMDAddStimulusSeries
        set_defaults
    }

    method set_defaults {} {
      global $this-stim-domain
      global $this-stim-current
      global $this-stim-start
      global $this-stim-end
      global $this-stim-duration
      global $this-stim-is-current-density
      global $this-stim-uselements
      
      set $this-stim-domain 0
      set $this-stim-current 1.0
      set $this-stim-start 0.400
      set $this-stim-end 0.800
      set $this-stim-duration 450
      set $this-is-current-density 0
      set $this-stim-useelements 1
  
    }

    method ui {} {
      
      set w .ui[modname]
      if {[winfo exists $w]} {
          return
      }

      toplevel $w 
      wm minsize $w 100 150

      iwidgets::labeledframe $w.m -labeltext "BLOCK PULSE STIMULUS"
      set stim [$w.m childsite]
      pack $w.m -fill both -expand yes

      label $stim.lab1 -text "Stimulus Domain (Element type)"
      label $stim.lab2 -text "Stimulus Current (mA)"
      label $stim.lab3 -text "Stimulus Start (ms)"
      label $stim.lab4 -text "Stimulus End (ms)"
      label $stim.lab5 -text "Stimulus Repeat (ms)"

      entry $stim.en1 -textvariable $this-stim-domain
      entry $stim.en2 -textvariable $this-stim-current
      entry $stim.en3 -textvariable $this-stim-start
      entry $stim.en4 -textvariable $this-stim-end
      entry $stim.en5 -textvariable $this-stim-duration
	
      checkbutton $stim.cb1 -text "Current is Current Density" -variable $this-stim-is-current-density
      checkbutton $stim.cb2 -text "Define stimulus only for elements contained within geometry" -variable $this-stim-useelements
      grid $stim.lab1 -row 0 -column 0
      grid $stim.lab2 -row 1 -column 0
      grid $stim.lab3 -row 2 -column 0
      grid $stim.lab4 -row 3 -column 0
      grid $stim.lab5 -row 4 -column 0
      grid $stim.en1 -row 0 -column 1
      grid $stim.en2 -row 1 -column 1
      grid $stim.en3 -row 2 -column 1
      grid $stim.en4 -row 3 -column 1
      grid $stim.en5 -row 4 -column 1
      grid $stim.cb1 -row 5 -column 0 -columnspan 2      
      grid $stim.cb2 -row 6 -column 0 -columnspan 2      


      makeSciButtonPanel $w $w $this
      moveToCursor $w
    }

}


