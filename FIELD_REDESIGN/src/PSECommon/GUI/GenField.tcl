itcl_class PSECommon_Fields_GenField {
    inherit Module
    constructor {config} {
	set name GenField
	set_defaults
    }
    method set_defaults {} {
	global $this-nx
	global $this-ny	
	global $this-nz
	global $this-fval

	set $this-nx 5
	set $this-ny 5
	set $this-nz 5
	set $this-fval 0
    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v
	bind $w.e <Return> $c
	pack $w.e -side right
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	global $this-nx
	global $this-ny
	global $this-nz
	global $this-fval

	toplevel $w
	wm minsize $w 200 150
	frame $w.f -width 400 -height 500
	pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
	set n "$this-c needexecute "
	frame $w.f.r
	make_entry $w.f.r.nx "x:" $this-nx "$this-c needexecute"
	make_entry $w.f.r.ny "y:" $this-ny "$this-c needexecute "
	make_entry $w.f.r.nz "z:" $this-nz "$this-c needexecute "
	make_entry $w.f.r.fval "Value:" $this-fval "$this-c needexecute"
	pack $w.f.r.nx $w.f.r.ny $w.f.r.nz -fill x
	pack $w.f.r.fval -pady 5 -fill x
	button $w.f.go -text "Execute" -relief raised -command $n
	pack $w.f.r $w.f.go -fill y
	set_defaults
    }
}
