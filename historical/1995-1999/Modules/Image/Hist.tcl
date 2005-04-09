#
#

itcl_class Hist {
    inherit Module
    constructor {config} {
	set name Hist
	set_defaults
    }
    method set_defaults {} {
       global $this-include
       set $this-include 0
       global $this-numbins
       set $this-numbins 500

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
	checkbutton $w.f.v1 -text "Include Zero" -relief flat \
		-variable $this-include
	pack $w.f.v1 -side right
	
	label $w.f.lab -text "Number of Bins: "
        entry $w.f.n1 -relief sunken -width 7 -textvariable $this-numbins
	pack $w.f.lab $w.f.n1 -side left

	button $w.f.doit -text " Execute " -command "$this exec"
	pack $w.f.doit -side bottom

    }
    method exec {} {
	$this-c needexecute
    }

}





