#
#

itcl_class Threshold {
    inherit Module
    constructor {config} {
	set name Threshold
	set_defaults
    }
    method set_defaults {} {
	global $this-low
	set $this-low 0
	global $this-high
	set $this-high 255
	global $this-lowval
	set $this-lowval 0
	global $this-medval
	set $this-medval -1
	global $this-higval
	set $this-higval 0
	$this-c needexecute
    }
    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 400 200    
	frame $w.f -width 400
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

#	expscale $w.low -orient horizontal -label "Low Threshold:" -variable $this-low -command $n
#	$w.low-win- configure
#	pack $w.low -fill x -pady 2
#	expscale $w.high -orient horizontal -label "High Threshold:" -variable $this-high -command $n
#	pack $w.high -fill x -pady 2

	frame $w.f.r1
	pack $w.f.r1 -anchor nw
	
	entry $w.f.r1.n1 -relief sunken -width 9 -textvariable $this-lowval
	entry $w.f.r1.n2 -relief sunken -width 9 -textvariable $this-medval
	entry $w.f.r1.n3 -relief sunken -width 9 -textvariable $this-higval

	frame $w.f.r1.v
	pack $w.f.r1.v -side top -fill x
	label $w.f.r1.v.lab -text "           Values : (-1 = same)           "
	pack $w.f.r1.v.lab -side left

	frame $w.f.r1.lab
	pack $w.f.r1.lab -side top -fill x
	label $w.f.r1.lab.hv -text "Low           "
	label $w.f.r1.lab.mv -text "Middle       "
	label $w.f.r1.lab.lv -text "High"
	pack $w.f.r1.lab.hv $w.f.r1.lab.mv $w.f.r1.lab.lv -side left

	pack $w.f.r1.n1 $w.f.r1.n2 $w.f.r1.n3 -side left




    }
}

