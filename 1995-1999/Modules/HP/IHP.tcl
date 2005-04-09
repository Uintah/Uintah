
catch {rename IHP ""}

itcl_class IHP {
    inherit Module
    constructor {config} {
	set name IHP
	set_defaults
    }
    method set_defaults {} {
	global $this-have_seedpoint
	set $this-have_seedpoint 0
	global $this-do_3dwidget
	set $this-do_3dwidget 1
	global $this-isoval
	global $this-emit_surface
	set $this-emit_surface 0
	global $this-min $this-max
	set $this-min 0
	set $this-max 255
	puts "set_defaults"
	global $this-xmin $this-xmax $this-ymin $this-ymax $this-zmin $this-zmax
	set $this-xmin 0
	set $this-ymin 0
	set $this-zmin 0
	set $this-xmax 1
	set $this-ymax 1
	set $this-zmax 1
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
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

	global $this-min $this-max
	scale $w.f.isoval -variable $this-isoval \
		-from [set $this-min] -to [set $this-max] -label "IsoValue:" \
		-resolution 0.000001 -showvalue true \
		-orient horizontal \
		-command $n
	pack $w.f.isoval -side top -fill x

	#makePoint $w.f.seed "Seed Point" $this-seed_point $n
	#pack $w.f.seed -fill x
	global $this-xmin $this-ymin $this-zmin $this-xmax $this-ymax $this-zmax
	set_bounds [set $this-xmin] [set $this-ymin] [set $this-zmin] \
		[set $this-xmax] [set $this-ymax] [set $this-zmax]
	
	checkbutton $w.f.emit_surface -text "Emitting Surface" -relief flat \
	    -variable $this-emit_surface -command $n
	pack $w.f.emit_surface
    }
    method set_minmax {min max} {
	set w .ui$this
	global $this-min $this-max
	set $this-min $min
	set $this-max $max
	$w.f.isoval configure -from $min -to $max
    }
    method set_bounds {xmin ymin zmin xmax ymax zmax} {
	return
	set w .ui$this
	$w.f.seed.x configure -from $xmin -to $xmax
	$w.f.seed.y configure -from $ymin -to $ymax
	$w.f.seed.z configure -from $zmin -to $zmax

	global $this-xmin $this-xmax
	set $this-xmin $xmin
	set $this-xmax $xmax
	global $this-ymin $this-ymax
	set $this-ymin $ymin
	set $this-ymax $ymax
	global $this-zmin $this-zmax
	set $this-zmin $zmin
	set $this-zmax $zmax
    }
    method do_value {} {
	set w .ui$this
	$w.f.seedpoint.w3d configure -state disabled
	$w.f.isoval configure -state normal
	#$w.f.seed.x configure -state disabled
	#$w.f.seed.y configure -state disabled
	#$w.f.seed.z configure -state disabled
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
