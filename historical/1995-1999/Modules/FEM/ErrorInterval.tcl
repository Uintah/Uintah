
catch {rename ErrorInterval ""}

itcl_class ErrorInterval {
    inherit Module
    constructor {config} {
	set name ErrorInterval
	set_defaults
    }
    method set_defaults {} {
	global $this-low
	set $this-nnodes 0

	global $this-high
	set $this-high 1
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 20

	#range $w.low -var_min $this-low -var_max $this-high -from 0 -to 100 \
	    -showvalue true -orient horizontal -command "$this-c needexecute" \
	    -label "Error bounds" -resolution 0.01
	#pack $w.low -side top
	button $w.exec -text "Execute" -command "$this-c needexecute"
	expscale $w.low -label "Low:" -orient horizontal -variable $this-low
	expscale $w.high -label "High:" -orient horizontal \
	    -variable $this-high
	pack $w.exec $w.low $w.high -side top -fill x
    }
}

