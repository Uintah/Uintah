itcl_class SurfGen {
    inherit Module
    constructor {config} {
        set name SurfGen
        set_defaults
    }

    method set_defaults {} {
	global $this-nx $this-ny $this-zscale $this-period
	set $this-nx 20
	set $this-ny 20
	set $this-zscale 5
	set $this-period 2
    }

    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2
	scale $w.f.s -label "Z Scale" -variable $this-zscale \
		-orient horizontal \
                -from 1 -to 100 -showvalue true

	scale $w.f.f -label "Frequency" -variable $this-period \
		-orient horizontal \
                -from 1 -to 100 -showvalue true

	scale $w.f.nx -label "X Res" -variable $this-nx \
		-orient horizontal \
                -from 1 -to 250 -showvalue true

	scale $w.f.ny -label "Y Res" -variable $this-ny -orient horizontal \
                -from 1 -to 250 -showvalue true

	button $w.f.e -text "Execute" -command "$this-c needexecute"

	pack $w.f.f $w.f.s $w.f.nx $w.f.ny $w.f.e -fill x
    }
}
