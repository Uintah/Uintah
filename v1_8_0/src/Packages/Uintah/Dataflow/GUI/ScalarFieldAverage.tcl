
itcl_class Uintah_Operators_ScalarFieldAverage {
    inherit Module

    constructor {config} {
	set name ScalarFieldAverage
	set_defaults
    }
    
    method set_defaults {} {
	#field average

	#field time info
	global $this-t0_
	set $this-t0_ 0
	global $this-t1_
	set $this-t1_ 0
	global $this-tsteps
	set $this-tsteps_ 0

	
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
 	toplevel $w

	button $w.b -text Close -command "destroy $w"
	pack $w.b -side bottom -expand yes -fill x -padx 2 -pady 2

	frame $w.t -relief raised -bd 2
	label $w.t.0 -text "Begin Time: "
	entry $w.t.e0 -textvariable $this-t0_ -state disabled \
	    -width 12
	label $w.t.1 -text "End Time: "
	entry $w.t.e1 -textvariable $this-t1_ -state disabled \
	    -width 12
	label $w.t.2 -text "# Time Steps: "
	entry $w.t.e2 -textvariable $this-tsteps_ -state disabled \
	    -width 5
	pack $w.t.0  $w.t.e0 $w.t.1 $w.t.e1 $w.t.2 $w.t.e2 -side left

	pack $w.t -side top -expand yes -fill x
    }
}
