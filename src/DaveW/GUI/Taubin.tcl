catch {rename DaveW_EEG_Taubin ""}

itcl_class DaveW_EEG_Taubin {
    inherit Module
    constructor {config} {
	set name Taubin
	set_defaults
    }
    method set_defaults {} {
	global $this-N
	global $this-pb
	global $this-gamma
	global $this-constrainedTCL
	set $this-pb .4
	set $this-gamma .8
	set $this-N 10
	set $this-constrainedTCL 0
	set $this-constraintTCL 1
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	set n "$this-c needexecute"
	
	global $this-N
	global $this-pb
	global $this-gamma
	global $this-constrainedTCL
	global $this-constraintTCL
	scale $w.f.n -orient horizontal -label "Iterations: " \
		-variable $this-N -showvalue true -from 1 -to 100
	scale $w.f.pb -orient horizontal -label "Pass Band (k) : " \
		-variable $this-pb -resolution .01 -showvalue true \
		-from 0 -to 1
	scale $w.f.gamma -orient horizontal -label "Gamma: " \
		-variable $this-gamma -resolution .01 -showvalue true \
		-from 0 -to 1
	scale $w.f.constr -orient horizontal -label "Constraint: " \
		-variable $this-constraintTCL -resolution .1 -showvalue true \
		-from 1 -to 3
	frame $w.f.b
	button $w.f.b.r -text "Reset" -command "$this-c reset"
	button $w.f.b.e -text "Execute" -command "$this-c tcl_exec"
	button $w.f.b.p -text "Print" -command "$this-c print"
	checkbutton $w.f.b.c -text "Constrained" -variable $this-constrainedTCL
	pack $w.f.b.r $w.f.b.e $w.f.b.p $w.f.b.c -side left -padx 4 -expand 1
	pack $w.f.n $w.f.pb $w.f.gamma $w.f.constr $w.f.b -side top -fill x -expand 1
	$this set_defaults
    }
}
