#
#

itcl_class Sharpen {
    inherit Module
    constructor {config} {
	set name Sharpen
	set_defaults
    }
    method set_defaults {} {
	global $this-fact
	set $this-fact 1.0

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

	label $w.f.lab -text "Sharpen Factor ? "
        entry $w.f.n1 -relief sunken -width 7 -textvariable $this-fact
	pack $w.f.lab $w.f.n1 -side left

	button $w.f.doit -text " Execute " -command "$this rflush"
	pack $w.f.doit
    }

    method rflush {} {
	global $this-fact
	
	$this-c initmatrix [set $this-fact]
	$this-c needexecute
    }
}





