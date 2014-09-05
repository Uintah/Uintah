
itcl_class PSECommon_Fields_GenVectorField {

    inherit Module

    constructor {config} {
	set name GenVectorField
	set_defaults
    }

    method set_defaults {} {

	global $this-resx
	global $this-resy
	global $this-resz

	global $this-eqX
	global $this-eqY
	global $this-eqZ

 	global $this-xmin
	global $this-ymin
	global $this-zmin
	global $this-xmax
	global $this-ymax
	global $this-zmax

	set $this-resx	32
	set $this-resy	32
	set $this-resz	32

	set $this-eqX	"0"
	set $this-eqY	"0"
	set $this-eqZ	"0"

	set $this-xmin  0.0
	set $this-xmax  1.0
	set $this-ymin  0.0
	set $this-ymax  1.0
	set $this-zmin  0.0
	set $this-zmax  1.0

#	$this-c needexecute
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	set n "$this-c needexecute"
	
	global $this-resx
	global $this-resy
	global $this-resz

 	global $this-xmin
	global $this-ymin
	global $this-zmin
	global $this-xmax
	global $this-ymax
	global $this-zmax

	global $this-eqX
	global $this-eqY
	global $this-eqZ

	scale $w.f.slide1 -label "x-Resolution" -from 1 -to 256 \
		-showvalue true \
		-orient horizontal -resolution 1 \
		-digits 2 -variable $this-resx
#		-command "$this-c resx"

	scale $w.f.slide2 -label "y-Resolution" -from 1 -to 256 \
		-showvalue true \
		-orient horizontal -resolution 1 \
		-digits 2 -variable $this-resy
#		-command "$this-c resy"

	scale $w.f.slide3 -label "z-Resolution" -from 1 -to 256 \
		-showvalue true \
		-orient horizontal -resolution 1 \
		-digits 2 -variable $this-resz
#		-command "$this-c res"

 	frame $w.f.bx
	label $w.f.bx.l1 -text "min x ="
	entry $w.f.bx.e1 -width 10 -relief sunken -textvariable $this-xmin
	label $w.f.bx.l2 -text "max x ="
	entry $w.f.bx.e2 -width 10 -relief sunken -textvariable $this-xmax
	pack  $w.f.bx.l1 $w.f.bx.e1 $w.f.bx.l2 $w.f.bx.e2 -side left

 	frame $w.f.by
	label $w.f.by.l1 -text "min y ="
	entry $w.f.by.e1 -width 10 -relief sunken -textvariable $this-ymin
	label $w.f.by.l2 -text "max y ="
	entry $w.f.by.e2 -width 10 -relief sunken -textvariable $this-ymax
	pack  $w.f.by.l1 $w.f.by.e1 $w.f.by.l2 $w.f.by.e2 -side left

 	frame $w.f.bz
	label $w.f.bz.l1 -text "min z ="
	entry $w.f.bz.e1 -width 10 -relief sunken -textvariable $this-zmin
	label $w.f.bz.l2 -text "max z ="
	entry $w.f.bz.e2 -width 10 -relief sunken -textvariable $this-zmax
	pack  $w.f.bz.l1 $w.f.bz.e1 $w.f.bz.l2 $w.f.bz.e2 -side left

	frame $w.f.x
	label $w.f.x.eql -text "fx(x,y,z) := "
	entry $w.f.x.eqe -width 40 -relief sunken -textvariable $this-eqX
	pack $w.f.x.eql $w.f.x.eqe -side left

	frame $w.f.y
	label $w.f.y.eql -text "fy(x,y,z) := "
	entry $w.f.y.eqe -width 40 -relief sunken -textvariable $this-eqY
	pack $w.f.y.eql $w.f.y.eqe -side left

	frame $w.f.z
	label $w.f.z.eql -text "fz(x,y,z) := "
	entry $w.f.z.eqe -width 40 -relief sunken -textvariable $this-eqZ
	pack $w.f.z.eql $w.f.z.eqe -side left

	button $w.f.b -text "Execute" -command $n

	pack \
		$w.f.slide1 $w.f.slide2 $w.f.slide3 \
		$w.f.bx $w.f.by $w.f.bz \
		$w.f.x $w.f.y $w.f.z $w.f.b -side top
    }
}
