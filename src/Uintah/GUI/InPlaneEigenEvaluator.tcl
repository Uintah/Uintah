
itcl_class Uintah_Visualization_InPlaneEigenEvaluator {
    inherit Module

    constructor {config} {
	set name InPlaneEigenEvaluator
	set_defaults
    }

    method set_defaults {} {
	global $this-planeSelect
	global $this-calcType
	global $this-delta
	set $this-planeSelect 2
	set $this-calcType 0
	set $this-delta 1
    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v
	bind $w.e <Return> $c
	pack $w.e -side right
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	set n "$this-c needexecute"

	frame $w.m 
	pack $w.m -padx 2 -pady 2 -fill x -expand yes

	make_labeled_radio $w.m.rp "Plane" $n left \
		$this-planeSelect {{"xy" 2} {"xz" 1} {"yz" 0}}

	frame $w.m.calc
	label $w.m.calc.l -text "Calculation:"
	radiobutton $w.m.calc.r1 -text "|e2 - e1|" \
		-variable $this-calcType -value 0 \
		-command "$this absDiffCalc"
	radiobutton $w.m.calc.r2 -text "sin(|e2 - e1| / Delta)" \
		-variable $this-calcType -value 1 \
		-command "$this sinDiffCalc"
	pack $w.m.calc.l -anchor w
	pack $w.m.calc.r1 $w.m.calc.r2 -side left

	make_entry $w.m.delta "Delta" $this-delta $n

	pack $w.m.rp $w.m.calc $w.m.delta -anchor w -pady 5

	button $w.b -text Close -command "destroy $w"
	pack $w.b -side bottom -padx 2 -pady 2
	
	absDiffCalc
    }

    method absDiffCalc {} {
	set w .ui[modname]
	set color "#505050"

	$w.m.delta.e configure -state disabled
	$w.m.delta.e configure -foreground $color
	$w.m.delta.l configure -foreground $color
	$this-c needexecute
    }

    method sinDiffCalc {} {
	set w .ui[modname]
	$w.m.delta.e configure -state normal
	$w.m.delta.e configure -foreground black
	$w.m.delta.l configure -foreground black
	$this-c needexecute
    }
}

