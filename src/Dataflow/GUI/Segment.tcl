#
#

itcl_class Segment {
    inherit Module
    constructor {config} {
	set name Segment
	set_defaults
    }
    method set_defaults {} {
	global $this-conn
	set $this-conn Eight

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
#	frame $w.f 
#	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	make_labeled_radio $w.funcname "Connectivity: " "" left $this-conn \
		{Four Eight}
	pack $w.funcname -side top
	entry $w.f -textvariable $this-conn -width 40 \
		-borderwidth 2 -relief sunken
	pack $w.f -side bottom

	button $w.f.doit -text " Execute " -command "$this rflush"
	pack $w.f.doit
	
	
    }

    method rflush {} {
	$this-c needexecute
    }

}





