catch {rename DaveW_FEM_DipoleSourceRHS ""}

itcl_class DaveW_FEM_DipoleSourceRHS {
    inherit Module
    constructor {config} {
	set name DipoleSourceRHS
	set_defaults
    }
    method set_defaults {} {
	global $this-sourceNodeTCL
	global $this-sinkNodeTCL
	global $this-modeTCL
	set $this-sourceNodeTCL 0
	set $this-sinNodeTCL 1
	set $this-modeTCL dipole
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
	global $this-sourceNodeTCL
	global $this-sinkNodeTCL
	global $this-modeTCL

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	
	make_labeled_radio $w.mode "Source model:" "" left $this-modeTCL \
		{dipole electrodes}
	make_entry $w.source "Source electrode:" $this-sourceNodeTCL {}
	make_entry $w.sink "Sink electrode:" $this-sinkNodeTCL {}
	bind $w.source <Return> "$this-c needexecute"
	bind $w.sink <Return> "$this-c needexecute"
	pack $w.mode $w.source $w.sink -side top -fill x
    }
}
