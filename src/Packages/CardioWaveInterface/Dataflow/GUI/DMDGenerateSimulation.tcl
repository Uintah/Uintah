itcl_class CardioWave_DiscreteMultiDomain_DMDGenerateSimulation {
    inherit Module
    constructor {config} {
        set name DMDGenerateSimulation
        set_defaults
    }

    method set_defaults {} {
      global $this-filename
      global $this-filename-set
      global $this-filename-entry
      global $this-usedebug
      global $this-buildvisbundle
      
      set $this-filename ""
      set $this-filename-set ""
      set $this-filename-entry ""
      set $this-usedebug 1
      set $this-buildvisbundle 1
      set $this-optimize 0
    }

    method ui {} {

        global $this-filename
        global $this-filename-set
        global $this-filename-entry

        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        
        iwidgets::labeledframe $w.fn -labeltext "SIMULATION FILENAME"
        set fileframe [$w.fn childsite]
        pack $w.fn -fill both -expand yes
        
        label $fileframe.label -text "Simulation filename"
        entry $fileframe.file -textvariable $this-filename
        button $fileframe.browse -text "Browse" -command "$this ChooseSimulationFile"
        label $fileframe.note -text "A range files will be create using the base name of the simulation bundle"
        set $this-filename-entry $fileframe.file  
        pack $fileframe.label -side left
        pack $fileframe.file  -side left -fill x -expand yes
        pack $fileframe.browse -side left
#        pack $fileframe.note -side bottom
  
        iwidgets::labeledframe $w.opt -labeltext "SIMULATION OPTIONS"
        set opt [$w.opt childsite]
        pack $w.opt -fill both -expand yes
  
        checkbutton $opt.enabledebug -text "Enable CardioWave Debug mode" -variable $this-usedebug
        checkbutton $opt.buildvisualization -text "Build Visualization bundle" -variable $this-buildvisbundle
        checkbutton $opt.optimize -text "Optimize linear system for parallel solver" -variable $this-optimize
  
        grid $opt.enabledebug -row 0 -column 0 -sticky w
        grid $opt.buildvisualization -row 1 -column 0 -sticky w
        grid $opt.optimize -row 2 -column 0 -sticky w
        
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
	
  method ChooseSimulationFile { } {

		global env
		global $this-filename
		global $this-filename-set

		# Create a unique name for the file selection window
		set w [format "%s-filebox" .ui[modname]]

		# if the file selector is open, bring it to the front
		# in case it is iconified, deiconify
		if { [winfo exists $w] } {
      if { [winfo ismapped $w] == 1} {
				raise $w
      } else {
				wm deiconify $w
      }
      return
		}

		toplevel $w -class TkFDialog

		set initdir ""
	
		# place to put preferred data directory
		# it's used if $this-filename is empty
	
		if {[info exists env(SCIRUN_DATA)]} {
	    		set initdir $env(SCIRUN_DATA)
		} elseif {[info exists env(SCI_DATA)]} {
	    		set initdir $env(SCI_DATA)
		} elseif {[info exists env(PSE_DATA)]} {
	    		set initdir $env(PSE_DATA)
		}
	
 		set $this-formatvar ""
    set $this-filename-set [set $this-filename]
  
		makeSaveFilebox \
			-parent $w \
			-filevar $this-filename-set \
			-command "wm withdraw $w;  $this OpenNewSimulationfile" \
 			-commandname "Set" \
			-cancel "wm withdraw $w" \
			-title "Enter the filename of simulation" \
			-filetypes {{ "Simulation Bundle" "*.sim.bdl" } }\
			-initialdir $initdir \
			-defaultextension "*.*" \
			-selectedfiletype 0 \
			-formatvar $this-formatvar \
      -formats {None} 
          
		wm deiconify $w	
	}
	
	method OpenSimulationfile {} {

		global $this-filename
		global $this-filename-entry
		
		set $this-filename [[set $this-filename-entry] get] 
		
	}

	method OpenNewSimulationfile {} {

		global $this-filename
		global $this-filename-set
		global $this-filename-entry
		
		set $this-filename [set $this-filename-set] 
#    $this-filename-entry insert 0 $this-filename
		
	}    
    
    
    
}


