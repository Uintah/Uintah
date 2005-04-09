#
#

itcl_class ImageConvolve {
    inherit Module
    constructor {config} {
	set name ImageConvolve
	set_defaults
    }
    method set_defaults {} {
	global $this-n1 $this-n2 $this-n3 
	global $this-n4 $this-n5 $this-n6 
	global $this-n7 $this-n8 $this-n9 
	global $this-norm	
	global $this-denom

	set $this-n1 1.0
	set $this-n2 1.0
	set $this-n3 1.0

	set $this-n4 1.0
	set $this-n5 1.0
	set $this-n6 1.0

	set $this-n7 1.0
	set $this-n8 1.0
	set $this-n9 1.0

	set $this-norm 1.0
	set $this-denom 1.0

	$this-c needexecute
    }
    method ui {} {
	set w .ui$this
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 30 
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	frame $w.f.r1
	frame $w.f.r2
	frame $w.f.r3
	frame $w.f.r4

	frame $w.f.v
	pack $w.f.v -side top -fill x
	label $w.f.v.lab -text "Kernel for Convolution:"
	pack $w.f.v.lab -side left

	pack $w.f.r1 $w.f.r2 $w.f.r3 $w.f.r4 -anchor nw

	entry $w.f.r1.n1 -relief sunken -width 7 -textvariable $this-n1
	entry $w.f.r1.n2 -relief sunken -width 7 -textvariable $this-n2
	entry $w.f.r1.n3 -relief sunken -width 7 -textvariable $this-n3

	entry $w.f.r2.n1 -relief sunken -width 7 -textvariable $this-n4
	entry $w.f.r2.n2 -relief sunken -width 7 -textvariable $this-n5
	entry $w.f.r2.n3 -relief sunken -width 7 -textvariable $this-n6

	entry $w.f.r3.n1 -relief sunken -width 7 -textvariable $this-n7
	entry $w.f.r3.n2 -relief sunken -width 7 -textvariable $this-n8
	entry $w.f.r3.n3 -relief sunken -width 7 -textvariable $this-n9

	label $w.f.r4.lab -text "Scale: "
	entry $w.f.r4.n1 -relief sunken -width 7 -textvariable $this-norm
	label $w.f.r4.lab2 -text " / "
	entry $w.f.r4.n2 -relief sunken -width 7 -textvariable $this-denom
	

	pack $w.f.r1.n1 $w.f.r1.n2 $w.f.r1.n3 -side left
	pack $w.f.r2.n1 $w.f.r2.n2 $w.f.r2.n3 -side left
	pack $w.f.r3.n1 $w.f.r3.n2 $w.f.r3.n3 -side left
	pack $w.f.r4.lab $w.f.r4.n1 $w.f.r4.lab2 $w.f.r4.n2 -side left

	button $w.f.doit -text " Execute " -command "$this rflush"
	pack $w.f.doit
    }

    method rflush {} {
	global $this-n1 $this-n2 $this-n3 
	global $this-n4 $this-n5 $this-n6 
	global $this-n7 $this-n8 $this-n9 
	global $this-norm
	global $this-denom
	
	
	$this-c initmatrix [set $this-n1] [set $this-n2] [set $this-n3] [set $this-n4] [set $this-n5] [set $this-n6] [set $this-n7] [set $this-n8] [set $this-n9] [set $this-norm] [set $this-denom]
	$this-c needexecute
    }
}





