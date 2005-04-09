#
#

itcl_class Subsample {
    inherit Module
    constructor {config} {
	set name Subsample
	set_defaults
    }
    method set_defaults {} {
	global $this-funcname
	global $this-width	
	global $this-height
	global $this-xscale
	global $this-yscale

	set $this-funcname Mitchell
	set $this-width 256
	set $this-height 256
	set $this-xscale 1.0
	set $this-yscale 1.0

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

	frame $w.f.r4
	pack $w.f.r4
	
	label $w.f.r4.lab -text "Width: "
	entry $w.f.r4.n1 -relief sunken -width 7 -textvariable $this-width
	label $w.f.r4.lab2 -text " Height: "
	entry $w.f.r4.n2 -relief sunken -width 7 -textvariable $this-height

	button $w.f.r4.doit -text " Execute " -command "$this rflush"
	
	pack $w.f.r4.lab $w.f.r4.n1 $w.f.r4.lab2 $w.f.r4.n2 $w.f.r4.doit -side left

	frame $w.f.r5
	pack $w.f.r5 -side bottom

	label $w.f.r5.lab -text "Xscale: "
	entry $w.f.r5.n1 -relief sunken -width 7 -textvariable $this-xscale
	label $w.f.r5.lab2 -text " Yscale: "
	entry $w.f.r5.n2 -relief sunken -width 7 -textvariable $this-yscale
	
	button $w.f.r5.doit2 -text " Execute " -command "$this rflush2"

	pack $w.f.r5.lab $w.f.r5.n1 $w.f.r5.lab2 $w.f.r5.n2 $w.f.r5.doit2 -side left

	make_labeled_radio $w.funcname "Function:" "" left $this-funcname \
		{Mitchell Fast}
	pack $w.funcname -side bottom
	entry $w.f.ted -textvariable $this-funcname -width 40 \
		-borderwidth 2 -relief sunken

    }

    method rflush {} {
	global $this-width
	global $this-height
	
	$this-c initsize [set $this-width] [set $this-height] 
	$this-c needexecute
    }
    method rflush2 {} {
	global $this-xscale
	global $this-yscale

	$this-c initscale [set $this-xscale] [set $this-yscale]
	$this-c needexecute
    }
}





