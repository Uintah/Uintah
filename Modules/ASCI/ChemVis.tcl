
catch {rename ChemVis ""}

itcl_class ChemVis {
    inherit Module
    constructor {config} {
	set name ChemVis
	set_defaults
    }
    method set_defaults {} {
	global $this-current_time
	set $this-current_time 0

    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 20

	scale $w.time -label "Time:" -orient horizontal \
		-variable $this-current_time -command "$this-c needexecute"
	pack $w.time -side top -fill x
	scale $w.sphere -label "Sphere resolution:" -orient horizontal \
		-variable $this-sphere_res -command "$this-c needexecute"
	pack $w.sphere -side top -fill x
    }
}

