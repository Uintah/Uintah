itcl_class RescaleParticleColorMap { 

    inherit Module 


    constructor {config} { 
        set name RescaleParticleColorMap 
        set_defaults 
    } 
  


    method set_defaults {} { 
        global $this-tcl_status 
	global $this-minVal
	global $this-maxVal
	global $this-scaleMode
	set exposed 0
    } 
  

    method ui {} { 
	global $this-minVal
	global $this-maxVal
	global $this-scaleMode

        set w .ui$this 
        if {[winfo exists $w]} { 
	    wm deiconify $w
            raise $w 
            return; 
        } 

	set type ""
  
        toplevel $w 
        wm minsize $w 200 50 

	frame $w.f1 -relief flat -borderwidth 2 
	pack $w.f1 -side top -expand yes

	button $w.b -text "Close" -command "destroy $w" -borderwidth 2
	pack $w.b -side bottom -expand yes -fill x

	label $w.f1.l -text "Auto scale mode"
	pack $w.f1.l -side top -expand yes -fill x
	button $w.f1.b -text "Switch to fixed scale mode" \
	    -command "$this fixedScale"
	pack $w.f1.b -side bottom -expand yes -fill both

	$this [set $this-scaleMode]Scale
	

    }	

    method fixedScale {} {
	global $this-minVal
	global $this-maxVal
	global $this-scaleMode
	set w .ui$this

	$w.f1.l configure -text "Fixed scale mode"
	$w.f1.b configure -text "Switch to auto scale mode" \
	    -command "$this autoScale"

	frame $w.f1.f -relief groove -borderwidth 2
	label $w.f1.f.l1 -text "min:" 
	label $w.f1.f.l2 -text "max:"
	entry $w.f1.f.e1 -textvariable $this-minVal
	entry $w.f1.f.e2 -textvariable $this-maxVal
	pack $w.f1.f -side top -expand yes -fill both
	pack  $w.f1.f.l1 -side left -anchor e -expand yes -fill x
	pack  $w.f1.f.e1 -side left -anchor w -expand yes -fill x
	pack  $w.f1.f.l2 -side left -anchor e -expand yes -fill x
	pack  $w.f1.f.e2 -side left -anchor w -expand yes -fill x
	
	set $this-scaleMode "fixed"
	$this-c needexecute
    }

    method autoScale {} {
	global $this-scaleMode
	set w .ui$this

	if { [winfo exists $w.f1.f ] } {
	    destroy $w.f1.f
	}
	$w.f1.l configure -text "Auto scale mode"
	$w.f1.b configure -text "Switch to fixed scale mode" \
	    -command "$this fixedScale"
	
	set $this-scaleMode "auto"
	$this-c needexecute
    }
}
