
catch {rename Bundles ""}

itcl_class DaveW_Tensor_Bundles {
    inherit Module
    constructor {config} {
	set name Bundles
	set_defaults
    }
    method set_defaults {} {
	global $this-nsteps
	global $this-nfibers
	global $this-niters
	global $this-stepsize
	global $this-bundleradx
	global $this-bundlescy
	global $this-bundlescz
	global $this-whichdir
	global $this-puncture
	global $this-demarcelle
	global $this-startx
	global $this-starty
	global $this-startz
	global $this-endx
	global $this-endy
	global $this-endz
	global $this-uniform
	global $this-seed
	global $this-bundlers
	set $this-nsteps 100
	set $this-nfibers 100
	set $this-niters 10
	set $this-stepsize 1
	set $this-bundleradx 1
	set $this-bundlescy 1
	set $this-bundlescz 1
	set $this-whichdir 2
	set $this-puncture 0.5
	set $this-demarcelle 0
	set $this-startx 4
	set $this-starty 4
	set $this-startz 4
	set $this-endx 4
	set $this-endy 4
	set $this-endz 7
	set $this-uniform 0
	set $this-seed 0
	set $this-bundlers 0
    }

    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	# Setup toplevel window
	toplevel $w
	wm minsize $w 100 100
	set n "$this-c needexecute "
	
	scale $w.nsteps -from 1 -to 1000 -variable $this-nsteps \
		-orient horizontal -label "Steps"
	scale $w.nfibers -from 1 -to 100 -variable $this-nfibers \
		-orient horizontal -label "Number of Fibers"
	scale $w.niters -from 1 -to 100 -variable $this-niters \
		-orient horizontal -label "Number of Iterations"
	scale $w.puncture -from 0 -to 1 -variable $this-puncture \
		-orient horizontal -label "Puncture" \
		-resolution 0.01
	scale $w.demarcelle -from 0 -to 1 -variable $this-demarcelle \
		-orient horizontal -label "DW/GK vs Demarcelle" \
		-resolution 0.01
	expscale $w.stepsize -label "Stepsize" -orient horizontal \
		-variable $this-stepsize
	expscale $w.bundleradx -label "XRadius" -orient horizontal \
		-variable $this-bundleradx
	expscale $w.bundlescy -label "YRadScale" -orient horizontal \
		-variable $this-bundlescy
	expscale $w.bundlescz -label "ZRadScale" -orient horizontal \
		-variable $this-bundlescz
	make_labeled_radio $w.whichdir "Advection Direction:" "" \
		left $this-whichdir \
		{{"forward" 0} {"backward" 1} {"both" 2}}
	frame $w.f1
	checkbutton $w.f1.un -text "Uniform Distribution" -variable $this-uniform
	label $w.f1.l -text "Rand Seed"
	entry $w.f1.e -width 5 -relief sunken -bd 2 -textvariable $this-seed
	checkbutton $w.f1.b -text "Bundlers" -variable $this-bundlers
	pack $w.f1.un $w.f1.l $w.f1.e $w.f1.b -side left -fill x -expand 1
	frame $w.f
	button $w.f.ex -text "Execute" -command $n
	button $w.f.is -text "IncrStep" -command "$this incrstep"
	button $w.f.sv -text "ValuesFromC" -command "$this-c set_points"
	button $w.f.gv -text "ValuesToC" -command "$this-c get_points"
	pack $w.f.ex $w.f.is $w.f.sv $w.f.gv -side left -fill x -expand 1
	frame $w.pts
	frame $w.pts.s
	frame $w.pts.s.x
	label $w.pts.s.x.l -text "StartX: "
	entry $w.pts.s.x.e -width 7 -relief sunken -bd 2 -textvariable $this-startx
	pack $w.pts.s.x.l $w.pts.s.x.e -side left
	frame $w.pts.s.y
	label $w.pts.s.y.l -text "StartY: "
	entry $w.pts.s.y.e -width 7 -relief sunken -bd 2 -textvariable $this-starty
	pack $w.pts.s.y.l $w.pts.s.y.e -side left
	frame $w.pts.s.z
	label $w.pts.s.z.l -text "StartZ: "
	entry $w.pts.s.z.e -width 7 -relief sunken -bd 2 -textvariable $this-startz
	pack $w.pts.s.z.l $w.pts.s.z.e -side left
	pack $w.pts.s.x $w.pts.s.y $w.pts.s.z -side top
	frame $w.pts.e
	frame $w.pts.e.x
	label $w.pts.e.x.l -text "EndX: "
	entry $w.pts.e.x.e -width 7 -relief sunken -bd 2 -textvariable $this-endx
	pack $w.pts.e.x.l $w.pts.e.x.e -side left
	frame $w.pts.e.y
	label $w.pts.e.y.l -text "EndY: "
	entry $w.pts.e.y.e -width 7 -relief sunken -bd 2 -textvariable $this-endy
	pack $w.pts.e.y.l $w.pts.e.y.e -side left
	frame $w.pts.e.z
	label $w.pts.e.z.l -text "EndZ: "
	entry $w.pts.e.z.e -width 7 -relief sunken -bd 2 -textvariable $this-endz
	pack $w.pts.e.z.l $w.pts.e.z.e -side left
	pack $w.pts.e.x $w.pts.e.y $w.pts.e.z -side top
	pack $w.pts.s $w.pts.e -side left -fill x -expand 1
	pack $w.nsteps $w.nfibers $w.niters $w.puncture $w.demarcelle \
		$w.stepsize $w.bundleradx $w.bundlescy $w.bundlescz \
		$w.whichdir $w.f1 $w.pts $w.f -side top -fill x -expand 1
    }
    method incrstep {} {
	global $this-nsteps
	incr $this-nsteps
	$this-c needexecute
    }
}
