
catch {rename AddWells ""}

itcl_class PSECommon_Visualization_AddWells {
    inherit Module
    constructor {config} {
	set name AddWells
	set_defaults
    }
    method set_defaults {} {
	global $this-radius
	set $this-radius 1
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	set n "$this-c needexecute "

	entry $w.f -textvariable $this-filename -width 40 \
		-borderwidth 2 -relief sunken
	pack $w.f -side bottom
	bind $w.f <Return> "$this-c needexecute "
#	expscale $w.r -label "Radius:" \
#		-orient horizontal -variable $this-radius -command $n
#	pack $w.r -side top -fill x -expand yes
    }
}
