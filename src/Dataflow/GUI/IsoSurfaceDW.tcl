
catch {rename IsoSurfaceDW ""}

itcl_class PSECommon_Visualization_IsoSurfaceDW {
    inherit Module
    constructor {config} {
	set name IsoSurfaceDW
	set_defaults
    }
    method set_defaults {} {
	global $this-have_seedpoint
	set $this-have_seedpoint 0
	global $this-do_3dwidget
	set $this-do_3dwidget 0
	global $this-isoval
	global $this-emit_surface
	set $this-emit_surface 0
	global $this-single
	set $this-single 0
	global $this-method
	set $this-method MC
	global $this-min $this-max
	set $this-min 0
	set $this-max 200
	global $this-tclBlockSize
	set $this-tclBlockSize 4
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	set $this-clr-r 0.5
	set $this-clr-g 0.7
	set $this-clr-b 0.3

	global $this-xmin $this-xmax $this-ymin $this-ymax $this-zmin $this-zmax
	set $this-xmin 0
	set $this-ymin 0
	set $this-zmin 0
	set $this-xmax 1
	set $this-ymax 1
	set $this-zmax 1
    }
    method raiseColor { col } {
	set w .ui[modname]
	if {[winfo exists $w.color]} {
	    raise $w.color
	    return;
	} else {
	    toplevel $w.color
	    global $this-clr
	    makeColorPicker $w.color $this-clr "$this setColor $col" \
		    "destroy $w.color"
	}
    }
    method setColor { col } {
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	set ir [expr int([set $this-clr-r] * 65535)]
	set ig [expr int([set $this-clr-g] * 65535)]
	set ib [expr int([set $this-clr-b] * 65535)]

	.ui[modname].f.f.col config -background [format #%04x%04x%04x $ir $ig $ib]
    }
    method ui {} {
	set w .ui[modname]
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
		-variable $this-have_seedpoint -value 0
	radiobutton $w.f.seedpoint.seedpoint -text Seedpoint -relief flat \
		-variable $this-have_seedpoint -value 1

	checkbutton $w.f.seedpoint.w3d -text "3D widget" -relief flat \
		-variable $this-do_3dwidget -state disabled
	pack $w.f.seedpoint.label $w.f.seedpoint.value \
		$w.f.seedpoint.seedpoint $w.f.seedpoint.w3d -side left

	global $this-min $this-max
	scale $w.f.isoval -variable $this-isoval \
		-from [set $this-min] -to [set $this-max] -label "IsoValue:" \
		-resolution 0.000001 -showvalue true \
		-orient horizontal \
		-state normal
	pack $w.f.isoval -side top -fill x
	scale $w.f.blocksize -variable $this-tclBlockSize \
		-from 2 -to 64 -label "BlockSize:" \
		-showvalue true -orient horizontal
	pack $w.f.blocksize -side top -fill x

	#makePoint $w.f.seed "Seed Point" $this-seed_point $n
	#pack $w.f.seed -fill x
	global $this-xmin $this-ymin $this-zmin $this-xmax $this-ymax $this-zmax
	set_bounds [set $this-xmin] [set $this-ymin] [set $this-zmin] [set $this-xmax] [set $this-ymax] [set $this-zmax]
	
	frame $w.f.b
	frame $w.f.b.l
	checkbutton $w.f.b.l.emit_surface -text "Emit Surface" -relief flat \
		-variable $this-emit_surface
	checkbutton $w.f.b.l.single -text "Single Processor" -relief flat \
		-variable $this-single
	pack $w.f.b.l.emit_surface $w.f.b.l.single -side top -expand 1
	make_labeled_radio $w.f.b.r "Method: " "" \
		top $this-method \
		{{Hash "Hash"} {Rings "Rings"} {MC "MC"} {None "None"}}
	pack $w.f.b.l $w.f.b.r -side left -expand 1
	pack $w.f.b -side top -fill x -expand 1
	button $w.f.ex -text "Execute" -command $n
	frame $w.f.f
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	set ir [expr int([set $this-clr-r] * 65535)]
	set ig [expr int([set $this-clr-g] * 65535)]
	set ib [expr int([set $this-clr-b] * 65535)]
	frame $w.f.f.col -relief ridge -borderwidth 4 -height 0.7c -width 0.7c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
	button $w.f.f.b -text "Set Color" -command "$this raiseColor $w.f.f.col"
	pack $w.f.f.b $w.f.f.col -side left -fill x -padx 5 -expand 1
	pack $w.f.ex $w.f.f -side top -fill x -expand 1
    }
    method set_minmax {min max} {
	set w .ui[modname]
	global $this-min $this-max
	set $this-min $min
	set $this-max $max
	$w.f.isoval configure -from $min -to $max
    }
    method set_bounds {xmin ymin zmin xmax ymax zmax} {
	return
	set w .ui[modname]
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
}
