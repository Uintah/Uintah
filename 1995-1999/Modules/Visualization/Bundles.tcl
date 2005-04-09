
catch {rename Bundles ""}

itcl_class Bundles {
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
	global $this-whichdir
	set $this-nsteps 100
	set $this-nfibers 100
	set $this-niters 10
	set $this-stepsize 1
	set $this-whichdir 2
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
	set n "$this-c needexecute "
	
	scale $w.nsteps -from 1 -to 1000 -variable $this-nsteps \
		-orient horizontal -label "Steps"
	scale $w.nfibers -from 1 -to 100 -variable $this-nfibers \
		-orient horizontal -label "Number of Fibers"
	scale $w.niters -from 1 -to 100 -variable $this-niters \
		-orient horizontal -label "Number of Iterations"
	expscale $w.stepsize -label "Stepsize" -orient horizontal \
		-variable $this-stepsize
	make_labeled_radio $w.whichdir "Advection Direction:" "" \
		left $this-whichdir \
		{{"forward" 0} {"backward" 1} {"both" 2}}
	pack $w.nsteps $w.nfibers $w.niters $w.stepsize $w.whichdir \
		-side top -fill x -expand 1
    }
}
