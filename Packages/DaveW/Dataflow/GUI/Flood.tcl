
catch {rename Flood ""}

itcl_class DaveW_Tensor_Flood {
    inherit Module
    constructor {config} {
	set name Flood
	set_defaults
    }
    method set_defaults {} {
	global $this-nsteps
	global $this-stepsize
	global $this-seed_x
	global $this-seed_y
	global $this-seed_z
	set $this-nsteps 50
	set $this-stepsize 1
	set $this-seed_x 4
	set $this-seed_y 4
	set $this-seed_z 4
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	# Setup toplevel window
	toplevel $w
	wm minsize $w 100 100
	set n "$this-c needexecute "
	
	scale $w.nsteps -from 1 -to 100 -variable $this-nsteps \
		-orient horizontal -label "Steps"
	expscale $w.stepsize -label "Stepsize" -orient horizontal \
		-variable $this-stepsize
	frame $w.f
	button $w.f.ex -text "Execute" -command $n
	button $w.f.r -text "Reset" -command "$this-c reset"
	button $w.f.b -text "Break" -command "$this-c break"
	button $w.f.sv -text "ValuesFromC" -command "$this-c set_seed"
	button $w.f.gv -text "ValuesToC" -command "$this-c get_seed"
	pack $w.f.ex $w.f.r $w.f.b $w.f.sv $w.f.gv -side left -fill x -expand 1
	frame $w.seed
	frame $w.seed.s
	frame $w.seed.s.x
	label $w.seed.s.x.l -text "Seed X: "
	entry $w.seed.s.x.e -width 7 -relief sunken -bd 2 -textvariable $this-seed_x
	pack $w.seed.s.x.l $w.seed.s.x.e -side left
	frame $w.seed.s.y
	label $w.seed.s.y.l -text "Seed Y: "
	entry $w.seed.s.y.e -width 7 -relief sunken -bd 2 -textvariable $this-seed_y
	pack $w.seed.s.y.l $w.seed.s.y.e -side left
	frame $w.seed.s.z
	label $w.seed.s.z.l -text "Seed Z: "
	entry $w.seed.s.z.e -width 7 -relief sunken -bd 2 -textvariable $this-seed_z
	pack $w.seed.s.z.l $w.seed.s.z.e -side left
	pack $w.seed.s.x $w.seed.s.y $w.seed.s.z -side top
	pack $w.seed.s -side left -fill x -expand 1
	pack $w.nsteps $w.stepsize $w.seed $w.f -side top -fill x -expand 1
    }
}
