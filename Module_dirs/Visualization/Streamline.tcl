

itcl_class Streamline {
    inherit Module
    constructor {config} {
	set name Streamline
	set_defaults
    }
    method set_defaults {} {
	global $this-source
	set $this-source "Line"

	global $this-markertype
	set $this-markertype "Ribbon"

	global $this-lineradius
	set $this-trscale .20

	global $this-algorithm
	set $this-algorithm "Euler"

	global $this-stepsizes
	set $this-stepsizes 0.02

	global $this-maxstepss
	set $this-maxstepss 50

	global $this-widget_scale
	set $this-widget_scale 1

	global $this-need_find
	set $this-need_find 1
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	#
	# Setup toplevel window
	#
	toplevel $w
	wm minsize $w 100 100
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x -expand yes
	set n "$this-c needexecute "

	#
	# Selector for source type - point, line, square, ring
	#
	make_labeled_radio $w.f.widgets "Source:" $n left $this-source \
		{Point Line Square Ring}
	pack $w.f.widgets -side top -fill x
	
	#
	# Selector for marker type - line, ribbon, surface
	#
	make_labeled_radio $w.f.marker "Marker:" $n left $this-markertype \
		{Line Tube Ribbon Surface}
	pack $w.f.marker

	#
	# Line radius and ribbon scale
	#
	scale $w.f.trscale -variable $this-trscale -digits 3 \
		-from 0.0 -to 5.0 -label "Tube/Ribbon Scale:" \
		-resolution 0 -showvalue true -tickinterval .2 \
		-orient horizontal
	pack $w.f.trscale -fill x -pady 2

	#
	# Selector for algorithm - Euler, RK4, PC, Stream function
	#
	make_labeled_radio $w.f.alg "Algorithm:" $n left $this-algorithm \
		{Euler RK4}
	pack $w.f.alg -side top -fill x

	#
	# Parameters
	scale $w.f.stepsize -variable $this-stepsizes -digits 3 \
		-from -2.0 -to 2.0 -label "Step size:" \
		-resolution .01 -showvalue true -tickinterval 2 \
		-orient horizontal
	pack $w.f.stepsize -fill x -pady 2
	
	scale $w.f.maxsteps -variable $this-maxstepss \
		-from 0 -to 1000 -label "Maximum steps:" \
		-showvalue true -tickinterval 200 \
		-orient horizontal
	pack $w.f.maxsteps -fill x -pady 2
    }
}
