
itcl_class IsoSurface {
    inherit Module
    constructor {config} {
	set name IsoSurface
	set_defaults
    }
    method set_defaults {} {
	global $this-have_seedpoint
	set $this-have_seedpoint 1
	global $this-do_3dwidget
	set $this-do_3dwidget 1
	global $this-isoval
	global $this-emit_surface
	set $this-emit_surface 0
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2
	set n "$this-c needexecute "

	frame $w.f.seedpoint
	pack $w.f.seedpoint
	label $w.f.seedpoint.label -text "Algorithm: "
	radiobutton $w.f.seedpoint.value -text Value -relief flat \
		-variable $this-have_seedpoint -value 0 \
		-command "$this do_value"
	radiobutton $w.f.seedpoint.seedpoint -text Seedpoint -relief flat \
		-variable $this-have_seedpoint -value 1 \
		-command "$this do_seed"

	checkbutton $w.f.seedpoint.w3d -text "3D widget" -relief flat \
		-variable $this-do_3dwidget -command $n
	pack $w.f.seedpoint.label $w.f.seedpoint.value \
		$w.f.seedpoint.seedpoint $w.f.seedpoint.w3d -side left

	scale $w.f.isoval -variable $this-isoval -digits 4 \
		-from 0.0 -to 1.0 -label "IsoValue:" \
		-resolution 0 -showvalue true \
		-orient horizontal \
		-command $n -state disabled
	pack $w.f.isoval -side top -fill x

	makePoint $w.f.seed "Seed Point" $this-seed_point $n
	pack $w.f.seed -fill x
	
	checkbutton $w.f.emit_surface -text "Emitting Surface" -relief flat \
	    -variable $this-emit_surface -command $n
	pack $w.f.emit_surface
    }
    method set_minmax {min max} {
	set w .ui$this
	set min -200
	set max 200
	$w.f.isoval configure -from $min -to $max
    }
    method set_bounds {xmin ymin zmin xmax ymax zmax} {
	set w .ui$this
	$w.f.seed.x configure -from $xmin -to $xmax
	$w.f.seed.y configure -from $ymin -to $ymax
	$w.f.seed.z configure -from $zmin -to $zmax
    }
    method do_value {} {
	set w .ui$this
	$w.f.seedpoint.w3d configure -state disabled
	$w.f.isoval configure -state normal
	$w.f.seed.x configure -state disabled
	$w.f.seed.y configure -state disabled
	$w.f.seed.z configure -state disabled
	$this-c needexecute
    }
    method do_seed {} {
	set w .ui$this
	$w.f.seedpoint.w3d configure -state normal
	$w.f.isoval configure -state disabled
	$w.f.seed.x configure -state normal
	$w.f.seed.y configure -state normal
	$w.f.seed.z configure -state normal
	$this-c needexecute
    }
}
