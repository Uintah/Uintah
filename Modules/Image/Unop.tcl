#
#

itcl_class Unop {
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
	
	make_labeled_radio $w.funcname "Operation:" "" left $this-funcname \
		{Abs Negative Invert Max/Min}
	pack $w.funcname -side top
	entry $w.f.ted -textvariable $this-funcname -width 40 \
		-borderwidth 2 -relief sunken

	button $w.f.doit -text " Execute " -command $n
	pack $w.f.doit -side bottom

    }

}





