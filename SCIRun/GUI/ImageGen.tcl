#
#

itcl_class ImageGen {
    inherit Module
    constructor {config} {
	set name ImageGen
	set_defaults
    }
    method set_defaults {} {
	global $this-width
	set $this-width 512
	global $this-height
	set $this-height 512
	global $this-period
	set $this-period 6.3
	global $this-amp
	set $this-amp 255
	global $this-horizontal
	set $this-horizontal 1
	global $this-vertical
	set $this-vertical 0

    }
    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 200 100    
	frame $w.f -width 200
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "
	pack $w.f

#	expscale $w.period -orient horizontal -label "Period:" -variable $this-period 
#	$w.period-win- configure
#	pack $w.period -fill x -pady 2
#	expscale $w.amp -orient horizontal -label "Amplitude:" -variable $this-amp
#	pack $w.amp -fill x -pady 2

	frame $w.f.r1
	pack $w.f.r1 -anchor nw

	entry $w.f.n1 -relief sunken -width 7 -textvariable $this-width


	entry $w.f.n2 -relief sunken -width 7 -textvariable $this-height

	label $w.f.lab -text " BY "

	frame $w.f.v
	pack $w.f.v -side top	
	label $w.f.v.lab -text "     Size of Image:      "
	pack $w.f.v.lab -side left

	pack $w.f.n1 $w.f.lab $w.f.n2 -side left

	checkbutton $w.f.v1 -text "Horizontal" -relief flat \
		-variable $this-horizontal
	checkbutton $w.f.v2 -text "Vertical  " -relief flat \
		-variable $this-vertical
	pack $w.f.v1 $w.f.v2 -side right


	button $w.f.doit -text " Execute " -command "$this rflush"
	pack $w.f.doit
    }

    method rflush {} {
	global $this-width
	global $this-height
	
	$this-c setsize [set $this-width] [set $this-height]
	$this-c needexecute
    }


   
}

