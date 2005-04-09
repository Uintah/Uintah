
catch {rename AddWells2 ""}

itcl_class AddWells2 {
    inherit Module
    constructor {config} {
	set name AddWells2
	set_defaults
    }
    method set_defaults {} {
	global $this-radius
	set $this-radius 2
	global $this-filename
	set $this-filename "/home/ari/scratch1/egi/stratton/info/TABLE2.TXT"
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	set n "$this-c needexecute "

	entry $w.f -textvariable $this-filename -width 50 \
		-borderwidth 2 -relief sunken
	pack $w.f -side bottom
	bind $w.f <Return> "$this-c needexecute "
	expscale $w.r -label "Radius:" \
		-orient horizontal -variable $this-radius -command $n
	pack $w.r -side top -fill x -expand yes
    }
}
