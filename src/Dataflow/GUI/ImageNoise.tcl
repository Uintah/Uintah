#
#

itcl_class SCIRun_Image_Noise {
    inherit Module
    constructor {config} {
	set name Noise
	set_defaults
    }
    method set_defaults {} {
	global $this-freq
	set $this-freq 0
	global $this-mag
	set $this-mag 1
	global $this-funcname
	set $this-funcname Gaussian
	global $this-min
	global $this-max
	global $this-mag2
	set $this-min 0
	set $this-max 255
	set $this-mag2 1

	$this-c needexecute
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 30 
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	label $w.f.lab -text "Frequency: "
        entry $w.f.n1 -relief sunken -width 7 -textvariable $this-freq
	label $w.f.lab2 -text " Max Magnitude:"
	entry $w.f.n2 -relief sunken -width 7 -textvariable $this-mag
	pack $w.f.lab $w.f.n1 $w.f.lab2 $w.f.n2 -side left

	label $w.f.lab3 -text "Min cutoff: "
        entry $w.f.n3 -relief sunken -width 7 -textvariable $this-min
	label $w.f.lab4 -text " Max cutoff: "
	entry $w.f.n4 -relief sunken -width 7 -textvariable $this-max
	pack $w.f.lab3 $w.f.n3 $w.f.lab4 $w.f.n4 -side left

	make_labeled_radio $w.funcname "Type: " "" left $this-funcname \
		{Salt&Pepper Gaussian}
	pack $w.funcname -side top
	entry $w.f2 -textvariable $this-funcname -width 40 \
		-borderwidth 2 -relief sunken
	#pack $w.f2 -side bottom

	button $w.f.doit -text " Execute " -command "$this rflush"
	pack $w.f.doit
    }

    method rflush {} {
	global $this-freq
	global $this-mag
	global $this-min
	global $this-max
	
	$this-c initmatrix [set $this-freq] [set $this-mag] [set $this-min] \
		[set $this-max]
	$this-c needexecute
    }
}





