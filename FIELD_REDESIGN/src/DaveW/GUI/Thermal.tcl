catch {rename DaveW_EEG_Thermal ""}

itcl_class DaveW_EEG_Thermal {
    inherit Module
    constructor {config} {
	set name Thermal
	set_defaults
    }
    method set_defaults {} {
	global $this-iters
	global $this-scale
	global $this-maxFlag
	global $this-maxVal
	global $this-stickFlag
	global $this-stickVal
	global $this-cond0
	global $this-cond1
	global $this-cond2
	global $this-cond3
	global $this-cond4
	global $this-cond5
	global $this-cond6
	set $this-iters 1.0
	set $this-iters 10

# maxFlag gets 1 if we never want to increase colder voxels past a certain temp
	set $this-maxFlag 0
	set $this-maxVal 10

# stickFlag gets 1 if we never want to CHANGE the temp of voxels below a value
	set $this-stickFlag 0
	set $this-stickVal -30

# these are the conductivities for slurpy
	set $this-cond0 0.00075
	set $this-cond1 0.3
	set $this-cond2 0.55
	set $this-cond3 2.5
	set $this-cond4 0.5
	set $this-cond5 0.15
	set $this-cond6 0.5

# these are the conductivities for gas fumes
#	set $this-cond0 0.5
#	set $this-cond1 0.3
#	set $this-cond2 0.3
#	set $this-cond3 0.3
#	set $this-cond4 0.3
#	set $this-cond5 0.3
#	set $this-cond6 0.5
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	global $this-z_max
	toplevel $w
	wm minsize $w 300 100
	frame $w.f -width 400 -height 500
	pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
	set n "$this-c needexecute "

	scale $w.f.i -orient horizontal -label "Iterations: " \
		-variable $this-iters -showvalue true -from 1 -to 100
	scale $w.f.s -orient horizontal -label "Scale: " \
		-variable $this-scale -resolution .1 -showvalue true \
		-from 0.1 -to 10.0
	frame $w.f.b
	button $w.f.b.r -text "Reset" -command "$this-c reset"
	button $w.f.b.e -text "Execute" -command "$this-c tcl_exec"
	button $w.f.b.h -text "Heat On" -command "$this-c heat_on"
	button $w.f.b.o -text "Heat Off" -command "$this-c heat_off"
	pack $w.f.b.r $w.f.b.e $w.f.b.h $w.f.b.o -side left -padx 4
	pack $w.f.i $w.f.s -side top -fill x -expand 1
	pack $w.f.b -side top
    }
}
