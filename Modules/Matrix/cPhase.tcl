itcl_class cPhase {
    inherit Module
    constructor {config} {
	set name cPhase
	set_defaults
    }

    method set_defaults {} {	
        global $this-phase

        set $this-phase 0
   }


    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	
	global $this-phase
      
        toplevel $w
	wm minsize $w 100 50

	set n "$this-c needexecute "
        
	scale $w.phase -command $n -orient horizontal -variable $this-phase \
		-from 0 -to 3.15 -resolution .0000001
	pack $w.phase -fill x -expand 1
    }
}

