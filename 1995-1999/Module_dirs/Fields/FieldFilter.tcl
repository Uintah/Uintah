itcl_class FieldFilter {
    inherit Module
    constructor {config} {
	set name FieldFilter
	set_defaults
    }
    method set_defaults {} {
	global $this-nx
	global $this-ny
	global $this-nz
	global $this-ox
	global $this-oy
	global $this-oz
	set $this-nx 16
	set $this-ny 16
	set $this-nz 16
	set $this-ox 0
	set $this-oy 0
	set $this-oz 0
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	global $this-nx
	global $this-ny
	global $this-nz
	global $this-ox
	global $this-oy
	global $this-oz
	toplevel $w
	wm minsize $w 300 100
	frame $w.f -width 400 -height 500
	pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
	set n "$this-c needexecute "

	frame $w.f.old -relief groove -borderwidth 2
	label $w.f.old.l -text "Old Dimensions"
	frame $w.f.old.fx
	label $w.f.old.fx.l -text "X: "
	label $w.f.old.fx.v -textvariable $this-ox
	pack $w.f.old.fx.l $w.f.old.fx.v -side left
	frame $w.f.old.fy
	label $w.f.old.fy.l -text "Y: "
	label $w.f.old.fy.v -textvariable $this-oy
	pack $w.f.old.fy.l $w.f.old.fy.v -side left
	frame $w.f.old.fz
	label $w.f.old.fz.l -text "Z: "
	label $w.f.old.fz.v -textvariable $this-oz
	pack $w.f.old.fz.l $w.f.old.fz.v -side left
	pack $w.f.old.l $w.f.old.fx $w.f.old.fy $w.f.old.fz -side top
	
	frame $w.f.new -relief groove -borderwidth 2
	label $w.f.new.l -text "New Dimensions"
	frame $w.f.new.r
	scale $w.f.new.r.x -from 1 -to 64 -label "X" -showvalue true \
		-tickinterval 63 -variable $this-nx -orient vertical \
		-length 100
	$w.f.new.r.x set 16
	scale $w.f.new.r.y -from 1 -to 64 -label "Y" -showvalue true \
		-tickinterval 63 -variable $this-ny -orient vertical \
		-length 100
	$w.f.new.r.y set 16
	scale $w.f.new.r.z -from 1 -to 64 -label "Z" -showvalue true \
		-tickinterval 63 -variable $this-nz -orient vertical \
		-length 100
	$w.f.new.r.z set 16
	pack $w.f.new.r.x $w.f.new.r.y $w.f.new.r.z -side left
	pack $w.f.new.l $w.f.new.r -side top

	pack $w.f.old $w.f.new -side left -fill both -expand 1
	frame $w.b
	pack $w.b -side bottom
	button $w.b.print_values -text "Print Values" -relief raised \
		-command "$this print_values"
	button $w.b.go -text "Execute" -relief raised -command $n
	pack $w.b.print_values $w.b.go -side top
    }
    method print_values {} {
	global $this-nx
	global $this-ny
	global $this-nz
	global $this-ox
	global $this-oy
	global $this-oz
	puts "OLD VALUES"
	puts -nonewline " X: "
	puts [set $this-ox]
	puts -nonewline " Y: "
	puts [set $this-oy]
	puts -nonewline " Z: "
	puts [set $this-oz]
	puts "NEW VALUES"
	puts -nonewline " X: "
	puts [set $this-nx]
	puts -nonewline " Y: "
	puts [set $this-ny]
	puts -nonewline " Z: "
	puts [set $this-nz]
	puts " "
    }
}