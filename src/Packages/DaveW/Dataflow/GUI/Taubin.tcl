itcl_class Taubin {
    inherit Module
    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }
    constructor {config} {
	set name Taubin
	set_defaults
    }
    method set_defaults {} {
	global $this-N
	global $this-pb
	global $this-gamma
	set $this-pb .4
	set $this-gamma .8
	set $this-N 10
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
	scale $w.f.n -orient horizontal -label "Iterations: " \
		-variable $this-N -showvalue true -from 1 -to 100
	scale $w.f.pb -orient horizontal -label "Pass Band (k) : " \
		-variable $this-pb -resolution .01 -showvalue true \
		-from 0 -to 1
	scale $w.f.gamma -orient horizontal -label "Gamma: " \
		-variable $this-gamma -resolution .01 -showvalue true \
		-from 0 -to 1
	frame $w.f.b
	button $w.f.b.r -text "Reset" -command "$this-c reset"
	button $w.f.b.e -text "Execute" -command "$this-c tcl_exec"
	button $w.f.b.p -text "Print" -command "$this-c print"
	pack $w.f.b.r $w.f.b.e $w.f.b.p -side left -padx 4 -expand 1
	pack $w.f.n $w.f.pb $w.f.gamma $w.f.b -side top -fill x -expand 1
	$this set_defaults
    }
}
