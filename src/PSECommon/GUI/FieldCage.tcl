
itcl_class PSECommon_Visualizations_FieldCage {
    inherit Module
    constructor {config} {
	set name FieldCage
	set_defaults
    }
    method set_defaults {} {
	global $this-numx
	global $this-numy
	global $this-numz
	set $this-numx 3
	set $this-numy 3
	set $this-numz 3
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
	pack $w.f -padx 2 -pady 2 -fill x -expand yes
	set n "$this-c needexecute"
	
	global $this-numx
	global $this-numy
	global $this-numz
	scale $w.f.nx -label Size -from 0 -to 20 \
		-showvalue true \
		-orient horizontal -resolution 1 \
		-variable $this-numx \
		-command $n
	scale $w.f.ny -label Size -from 0 -to 20 \
		-showvalue true \
		-orient horizontal -resolution 1 \
		-variable $this-numy \
		-command $n
	scale $w.f.nz -label Size -from 0 -to 20 \
		-showvalue true \
		-orient horizontal -resolution 1 \
		-variable $this-numz \
		-command $n
	pack $w.f.nx $w.f.ny $w.f.nz -side top -fill x
    }
}
