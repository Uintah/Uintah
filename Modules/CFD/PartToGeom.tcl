
catch {rename PartToGeom ""}

itcl_class PartToGeom {
    inherit Module
    constructor {config} {
	set name PartToGeom
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

	expscale $w.time -label "Time:" -orient horizontal \
		-variable $this-current_time -command "$this-c needexecute"
	pack $w.time -side top -fill x
    }
}

