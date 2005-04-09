#
#

itcl_class HistEq {
    inherit Module
    constructor {config} {
	set name HistEq
	set_defaults
    }
    method set_defaults {} {
	global $this-clip
	set $this-clip 1.0
	global $this-bins
	set $this-bins 128
	global $this-conx
	set $this-conx 2
	global $this-cony
	set $this-cony 2
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

	expscale $w.low -orient horizontal -label "ClipLimit:" -variable $this-clip -command $n
	$w.low-win- configure
	pack $w.low -fill x -pady 2

	frame $w.f.r1
	pack $w.f.r1 -anchor nw

	entry $w.f.r1.n1 -relief sunken -width 9 -textvariable $this-bins
	entry $w.f.r1.n2 -relief sunken -width 9 -textvariable $this-conx
	entry $w.f.r1.n3 -relief sunken -width 9 -textvariable $this-cony

	frame $w.f.r1.v
	pack $w.f.r1.v -side top -fill x
	label $w.f.r1.v.lab -text "           HistEq Parameters :          "
	pack $w.f.r1.v.lab -side left

	frame $w.f.r1.lab
	pack $w.f.r1.lab -side top -fill x
	label $w.f.r1.lab.hv -text "# of Bins"
	label $w.f.r1.lab.mv -text "X Contextial Regions"
	label $w.f.r1.lab.lv -text "Y Contextial Regions"
	pack $w.f.r1.lab.hv $w.f.r1.lab.mv $w.f.r1.lab.lv -side left

	pack $w.f.r1.n1 $w.f.r1.n2 $w.f.r1.n3 -side left




    }
}

