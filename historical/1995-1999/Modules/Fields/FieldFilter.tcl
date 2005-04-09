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
	global $this-filterType
	set $this-filterType Triangle    
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
	global $this-nMaxZ
	trace variable $this-nMaxZ w "$this changed_max"
	toplevel $w
	wm minsize $w 300 100
	frame $w.f -width 400 -height 500
	pack $w.f -padx 2 -pady 2 -side top -fill both -expand yes
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
		-tickinterval 20 -variable $this-nx -orient vertical \
		-length 100
	$w.f.new.r.x set 16
	scale $w.f.new.r.y -from 1 -to 64 -label "Y" -showvalue true \
		-tickinterval 20 -variable $this-ny -orient vertical \
		-length 100
	$w.f.new.r.y set 16
	scale $w.f.new.r.z -from 1 -to 64 -label "Z" -showvalue true \
		-tickinterval 20 -variable $this-nz -orient vertical \
		-length 100
	$w.f.new.r.z set 16
	pack $w.f.new.r.x $w.f.new.r.y $w.f.new.r.z -side left -expand 1 -fill y
	pack $w.f.new.l -side top
	pack $w.f.new.r -side top -expand 1 -fill y


	frame $w.f.range -relief groove -borderwidth 2
	label $w.f.range.l -text "Filter Range"
	frame $w.f.range.r
	range $w.f.range.r.x -from 1 -to 16 -label "X" -showvalue true \
		-tickinterval 5 -var_min $this-range_min_x -var_max \
		$this-range_max_x -orient vertical -length 100 -nonzero 1
	$w.f.range.r.x setMinMax 1 12
	range $w.f.range.r.y -from 1 -to 16 -label "Y" -showvalue true \
		-tickinterval 5 -var_min $this-range_min_y -var_max \
		$this-range_max_y -orient vertical -length 100 -nonzero 1
	$w.f.range.r.y setMinMax 1 12
	range $w.f.range.r.z -from 1 -to 16 -label "Z" -showvalue true \
		-tickinterval 5 -var_min $this-range_min_z -var_max \
		$this-range_max_z -orient vertical -length 100 -nonzero 1
	$w.f.range.r.z setMinMax 1 12
	pack $w.f.range.r.x $w.f.range.r.y $w.f.range.r.z -side left -expand 1 -fill y
	pack $w.f.range.l -side top
	pack $w.f.range.r -side top -expand 1 -fill y

	frame $w.f.filt -relief groove -borderwidth 2
        make_labeled_radio $w.f.filt.b "Filter:" "" top $this-filterType \
                {Box Triangle Mitchell}
	pack $w.f.filt.b

	pack $w.f.old $w.f.range $w.f.new $w.f.filt -side left -fill both -expand 1
	frame $w.b
	pack $w.b -side bottom
#	button $w.b.print_values -text "Print Values" -relief raised \
#		-command "$this print_values"
#	checkbutton $w.b.same -text "Same Inputs?" -variable $this-sameInput
#	$w.b.same select
	button $w.b.go -text "Execute" -relief raised -command $n
	pack $w.b.go -side top
#	pack $w.b.same $w.b.print_values $w.b.go -side top
    }
    method print_values {} {
	global $this-nx
	global $this-ny
	global $this-nz
	global $this-ox
	global $this-oy
	global $this-oz
	global $this-nMaxX
	global $this-nMaxY
	global $this-nMaxZ
	global $this-filterType
	puts "OLD VALUES"
	puts -nonewline " X: "
	puts [set $this-ox]
	puts -nonewline " Y: "
	puts [set $this-oy]
	puts -nonewline " Z: "
	puts [set $this-oz]
	puts "MAXES"
	puts -nonewline " X: "
	puts [set $this-nMaxX]
	puts -nonewline " Y: "
	puts [set $this-nMaxY]
	puts -nonewline " Z: "
	puts [set $this-nMaxZ]
	puts "NEW VALUES"
	puts -nonewline " X: "
	puts [set $this-nx]
	puts -nonewline " Y: "
	puts [set $this-ny]
	puts -nonewline " Z: "
	puts [set $this-nz]
	puts -nonewline "FilterType: "
	puts [set $this-filterType]
	puts " "
    }
    method changed_max {v vtmp op} {
	set w .ui$this
	global $w.f.new.r.x
	global $w.f.new.r.y
	global $w.f.new.r.z	
	global $w.f.range.r.x
	global $w.f.range.r.y
	global $w.f.range.r.z	
	global $this-nMaxX
	global $this-nMaxY
	global $this-nMaxZ
	global $this-ox
	global $this-oy
	global $this-oz
	$w.f.new.r.x configure -to [set $this-nMaxX]
	$w.f.new.r.y configure -to [set $this-nMaxY]
	$w.f.new.r.z configure -to [set $this-nMaxZ]
	$w.f.range.r.x configure -to [set $this-ox]
	$w.f.range.r.y configure -to [set $this-oy]
	$w.f.range.r.z configure -to [set $this-oz]
    }
}
