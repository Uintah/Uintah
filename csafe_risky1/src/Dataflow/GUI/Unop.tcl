#
#

itcl_class SCIRun_Image_Unop {
    inherit Module
    constructor {config} {
	set name Unop
	set_defaults
    }
    method set_defaults {} {
	global $this-funcname

	set $this-funcname Max/Min

	$this-c needexecute
    }
    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 30 
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	frame $w.f.r4
	pack $w.f.r4
	
	make_labeled_radio $w.funcname "Operation:" "" top $this-funcname \
		{Abs Negative Invert Max/Min Grayscale A^2 Sqrt(A) arctan nonzero resize-to-power-of-2}
	pack $w.funcname -side top
	entry $w.f.ted -textvariable $this-funcname -width 40 \
		-borderwidth 2 -relief sunken

	button $w.f.doit -text " Execute " -command $n
	pack $w.f.doit -side top

    }

}





