itcl_class GenAxes {
    inherit Module
    constructor {config} {
	set name GenAxes
	set_defaults
    }
    method set_defaults {} {
	global $this-size
	set $this-size 1
	$this-c needexecute
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
	
	global $this-size
	scale $w.f.slide -label Size -from 0.01 -to 9.99 \
		-showvalue true \
		-orient horizontal -resolution 0.01 \
		-digits 3 -variable $this-size \
		-command "$this-c size_changed"
	button $w.f.b -text "Execute" -command $n
	pack $w.f.slide $w.f.b -side top
    }
}
