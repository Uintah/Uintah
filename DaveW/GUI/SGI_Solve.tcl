catch {rename DaveW_ISL_SGI_Solve ""}

itcl_class DaveW_ISL_SGI_Solve {
    inherit Module

    constructor {config} {
	set name SGI_Solve
	set_defaults
    }


    method set_defaults {} {	
        global $this-tcl_status
	
    }


    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	
	
        toplevel $w
	wm minsize $w 100 50
	
	set n "$this-c needexecute "
        
	frame $w.f1 -relief groove -borderwidth 1
	label $w.f1.label -text "Network:"
	
        frame $w.f
	label $w.f.label -text "Status:"
	entry $w.f.t -width 15  -relief sunken -bd 2 -textvariable $this-tcl_status
	pack $w.f.label $w.f.t  -side left -pady 2m         
	
        button $w.go -text "Execute" -relief raised -command $n 
	
	pack $w.f1 $w.f $w.go  -fill x 
	
    }
	
}

