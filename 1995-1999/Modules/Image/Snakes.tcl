#
#

itcl_class Snakes {
    inherit Module
    constructor {config} {
	set name Snakes
	set_defaults
    }
    method set_defaults {} {
	global $this-a
	set $this-a 1.0
	global $this-fixed
	set $this-fixed 0
	global $this-b
	set $this-b 1.0
	global $this-maxx
	set $this-maxx 2
	global $this-maxy
	set $this-maxy 2
	global $this-resx
	set $this-resx 1
	global $this-resy
	set $this-resy 1
	global $this-iter
	set $this-iter 1
	$this-c needexecute
    }
    method ui {} {
	set w .ui$this
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 400 200    
	frame $w.f -width 400
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	expscale $w.low -orient horizontal -label "Alpha:" -variable $this-a -command $n
	$w.low-win- configure
	pack $w.low -fill x -pady 2
	expscale $w.high -orient horizontal -label "Beta:" -variable $this-b -command $n
	pack $w.high -fill x -pady 2

	frame $w.f.r1
	pack $w.f.r1 -anchor nw

	checkbutton $w.f.v1 -text "Fixed End Points" -relief flat \
		-variable $this-fixed
	pack $w.f.v1 -side left
	
	entry $w.f.r1.n1 -relief sunken -width 9 -textvariable $this-maxx
	entry $w.f.r1.n2 -relief sunken -width 9 -textvariable $this-maxy
	entry $w.f.r1.n3 -relief sunken -width 9 -textvariable $this-resx
	entry $w.f.r1.n4 -relief sunken -width 9 -textvariable $this-resy
	entry $w.f.r1.n5 -relief sunken -width 9 -textvariable $this-iter

	frame $w.f.r1.v
	pack $w.f.r1.v -side top -fill x
	label $w.f.r1.v.lab -text "           Snake Parameters :          "
	pack $w.f.r1.v.lab -side left

	frame $w.f.r1.lab
	pack $w.f.r1.lab -side top -fill x
	label $w.f.r1.lab.hv -text "Max DeltaX"
	label $w.f.r1.lab.mv -text "Max DeltaY"
	label $w.f.r1.lab.lv -text "X Resolution"
	label $w.f.r1.lab.tv -text "Y Resolution"
	label $w.f.r1.lab.vv -text "# of Iterations"
	pack $w.f.r1.lab.hv $w.f.r1.lab.mv $w.f.r1.lab.lv $w.f.r1.lab.tv $w.f.r1.lab.vv -side left

	pack $w.f.r1.n1 $w.f.r1.n2 $w.f.r1.n3 $w.f.r1.n4 $w.f.r1.n5 -side left




    }
}

