#
#

itcl_class SCIRun_Image_Radon {
    inherit Module
    constructor {config} {
	set name Radon
	set_defaults
    }
    method set_defaults {} {
	global $this-lowval
	set $this-lowval 1
	global $this-higval
	set $this-higval 180
	global $this-num
	set $this-num 180

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

	button $w.f.doit -text " Execute " -command "$this rflush"
	pack $w.f.doit -side bottom
	
	frame $w.f.r1
	pack $w.f.r1 -anchor nw
	
	entry $w.f.r1.n1 -relief sunken -width 11 -textvariable $this-lowval
	entry $w.f.r1.n2 -relief sunken -width 11 -textvariable $this-higval
	entry $w.f.r1.n3 -relief sunken -width 11 -textvariable $this-num

	frame $w.f.r1.lab
	pack $w.f.r1.lab -side top -fill x
	label $w.f.r1.lab.hv -text "Theta Range   "
	label $w.f.r1.lab.mv -text "High       "
	label $w.f.r1.lab.lv -text "Number of Thetas"
	pack $w.f.r1.lab.hv $w.f.r1.lab.mv $w.f.r1.lab.lv -side left

	pack $w.f.r1.n1 $w.f.r1.n2 $w.f.r1.n3 -side left
    }

    method rflush {} {
	$this-c needexecute
    }

}





