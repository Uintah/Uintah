#
#

itcl_class Edge {
    inherit Module
    constructor {config} {
	set name Edge
	set_defaults
    }
    method set_defaults {} {
	global $this-funcname
	set $this-funcname Roberts
	global $this-norm
	set $this-norm 1.0
	global $this-denom
	set $this-denom 1.0
	global $this-t1
	set $this-t1 0

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
#	frame $w.f 
#	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	make_labeled_radio $w.funcname "Function:" "" left $this-funcname \
		{Roberts Sobel Prewitt Canny Laplacian}
	pack $w.funcname -side top
	entry $w.f -textvariable $this-funcname -width 40 \
		-borderwidth 2 -relief sunken
	pack $w.f -side bottom
	
#	frame $w.f.r4
#	pack $w.f.r4 -anchor sw

	label $w.f.lab -text "Scale: "
	entry $w.f.n1 -relief sunken -width 7 -textvariable $this-norm
	label $w.f.lab2 -text " / "
	entry $w.f.n2 -relief sunken -width 7 -textvariable $this-denom
	pack $w.f.lab $w.f.n1 $w.f.lab2 $w.f.n2 -side left


	button $w.f.doit -text " Execute " -command "$this rflush"
	pack $w.f.doit

	expscale $w.thresh -orient horizontal -label "Canny Threshold:" -variable $this-t1 -command $n
	$w.thresh-win- configure
	pack $w.thresh -fill x -pady 2

    }

    method rflush {} {
	global $this-funcname
	global $this-norm
	global $this-denom
	
	$this-c initmatrix [set $this-norm] [set $this-denom]
	$this-c needexecute
    }
}





