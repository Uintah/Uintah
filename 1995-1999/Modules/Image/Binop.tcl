#
#

itcl_class Binop {
    inherit Module
    constructor {config} {
	set name Binop
	set_defaults
    }
    method set_defaults {} {
	global $this-funcname

	set $this-funcname A+B

	$this-c needexecute
    }
    method ui {} {
	set w .ui$this
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
		{A+B A-B A*B A/B AorB AandB AxorB max(A,B) min(A,B)}
	pack $w.funcname -side top
	entry $w.f.ted -textvariable $this-funcname -width 40 \
		-borderwidth 2 -relief sunken

	button $w.f.doit -text " Execute " -command $n
	pack $w.f.doit -side top

    }

}





