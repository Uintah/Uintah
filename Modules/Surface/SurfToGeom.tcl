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
	global $this-nodes
	global $this-named
	global $this-ntype
	global $this-rad
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	global $this-normals
	set $this-range_min -1
	set $this-range_max 1
	set $this-best 1
	set $this-invert 0
	set $this-nodes 0
	set $this-named 0
	set $this-ntype points
	set $this-rad 1.0
	set $this-clr-r 0.5
	set $this-clr-g 0.7
	set $this-clr-b 0.3
	set $this-normals 0
    }
    method raiseColor { col } {
	set w .ui$this
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

	.ui$this.p.f.col config -background [format #%04x%04x%04x $ir $ig $ib]
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
	global $this-clr
	global $this-clr-r
	global $this-clr-g
	global $this-clr-b
	toplevel $w

	frame $w.p -relief groove -borderwidth 2
	checkbutton $w.p.n -text "Average Normals" -variable $this-normals
	frame $w.p.f
	set ir [expr int([set $this-clr-r] * 65535)]
	set ig [expr int([set $this-clr-g] * 65535)]
	set ib [expr int([set $this-clr-b] * 65535)]
	frame $w.p.f.col -relief ridge -borderwidth 4 -height 0.7c -width 0.7c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
	button $w.p.f.b -text "Set Color" -command "$this raiseColor $w.f.b.col"
	pack $w.p -side top -fill x
	pack $w.p.n $w.p.f -side top
	pack $w.p.f.b $w.p.f.col -side left -padx 5
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
