itcl_class ClipField {
    inherit Module
    constructor {config} {
	set name ClipField
	set_defaults
    }
    method set_defaults {} {
	global $this-x_min
	global $this-y_min
	global $this-z_min
	global $this-x_max
	global $this-y_max
	global $this-z_max
	set $this-x_min 0
	set $this-x_max 255
	set $this-y_min 0
	set $this-y_max 255
	set $this-z_min 0
	set $this-z_max 80
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	global $this-x_min
	global $this-x_max
	global $this-y_min
	global $this-y_max
	global $this-z_min
	global $this-z_max
	toplevel $w
	wm minsize $w 300 100
	frame $w.f -width 400 -height 500
	pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
	set n "$this-c needexecute "

	frame $w.f.r
	range $w.f.r.x -from 1 -to 256 -label "X" -showvalue true \
		-var_min $this-x_min -var_max $this-x_max -orient horizontal \
		-length 300 -nonzero 1
	$w.f.r.x setMinMax 1 256
	range $w.f.r.y -from 1 -to 256 -label "Y" -showvalue true \
		-var_min $this-y_min -var_max $this-y_max -orient horizontal \
		-length 300 -nonzero 1
	$w.f.r.y setMinMax 1 256
	range $w.f.r.z -from 1 -to 81 -label "Z" -showvalue true \
		-var_min $this-z_min -var_max $this-z_max -orient horizontal \
		-length 300 -nonzero 1
	$w.f.r.z setMinMax 5 5
	pack $w.f.r.x $w.f.r.y $w.f.r.z -side top
	button $w.f.go -text "Execute" -relief raised -command $n
	checkbutton $w.f.same -variable $this-sameInput -text "Same Inputs?"
	$w.f.same select
	pack $w.f.r $w.f.same $w.f.go -side top
    }
}
