itcl_class PadFldPow2 {
    inherit Module
    constructor {config} {
	set name PadFldPow2
	set_defaults
    }
    method set_defaults {} {
	global $this-padvalTCL
	set $this-padvalTCL 0
    }
    method ui {} {
	global $this-padfldTCL

	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.p -bd 2 -relief sunken
	label $w.p.l -text "Pad value:"
	entry $w.p.e -width 7 -relief sunken -bd 2 \
		-textvariable $this-padvalTCL
	pack $w.p.l $w.p.e -side left \
		-padx 4 -fill x -expand 1
	button $w.ex -text "Execute" -command "$this-c needexecute"
	pack $w.p $w.ex -side top -fill x -expand 1
    }
}
