itcl_class SurfToGeom {
    inherit Module
    constructor {config} {
	set name SurfToGeom
	set_defaults
    }
    method set_defaults {} {
	global $this-range_min
	global $this-range_max
	global $this-best
	global $this-invert
	global $this-ntype
	global $this-rad
	set $this-range_min -1
	set $this-range_max 1
	set $this-best 1
	set $this-invert 0
	set $this-nodes 0
	set $this-named 0
	set $this-ntype points
	set $this-rad 1.0
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	set n "$this-c needexecute"
	global $this-range_min
	global $this-range_max
	global $this-nodes
	global $this-ntype
	global $this-rad
	global $this-named
	toplevel $w
	frame $w.d
	checkbutton $w.d.n -text "Render Nodes Only" -variable $this-nodes
	make_labeled_radio $w.d.nodes "Render Nodes As: " "" \
		left $this-ntype \
		{{"Spheres" spheres} \
		{"Points" points}}
	checkbutton $w.d.named -text "Only Render Named (SurfTree)" \
		-variable $this-named
	
	frame $w.d.sph -relief groove -borderwidth 2
	expscale $w.d.sph.radius -orient horizontal -variable $this-rad \
		-label "Radius:" \
		-command "$this-c needexecute"
	scale $w.d.sph.res -orient horizontal -variable $this-resol \
		-label "Resolution:" -from 4 -to 20 \
		-command "$this-c needexecute"
	pack $w.d.sph.radius $w.d.sph.res -side left -fill x -expand 1
	pack $w.d.n $w.d.nodes $w.d.named $w.d.sph -side top -fill x -expand 1
	frame $w.f
	expscale $w.f.min -orient horizontal -variable $this-range_min \
		-label "range min:"
	expscale $w.f.max -orient horizontal -variable $this-range_max \
		-label "range max:"
	pack $w.f.min $w.f.max -side left -expand 1
	frame $w.b
	global $this-best
	global $this-invert
	checkbutton $w.b.b -text Best -variable $this-best
	checkbutton $w.b.i -text Invert -variable $this-invert
	pack $w.b.b $w.b.i -side left -expand 1 -fill x
	pack $w.d $w.f $w.b -side top -fill x
    }
}
