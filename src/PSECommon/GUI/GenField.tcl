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

	global $this-sx
	global $this-sy	
	global $this-sz

	global $this-ex
	global $this-ey	
	global $this-ez

	global $this-fval

	set $this-nx 10
	set $this-ny 10
	set $this-nz 10

	set $this-sx -1.0
	set $this-sy -1.0
	set $this-sz -1.0

	set $this-ex 1.0
	set $this-ey 1.0
	set $this-ez 1.0

	set $this-fval "x^2 + y^2 + z^2"
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

	global $this-sx
	global $this-sy
	global $this-sz

	global $this-ex
	global $this-ey
	global $this-ez

	global $this-fval

	toplevel $w
	wm minsize $w 300 250
	frame $w.f -width 400 -height 500
	pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
	set n "$this-c needexecute "
	frame $w.f.r

	 frame $w.f.r.geom
	 radiobutton $w.f.r.geom.structured -text Structured -variable $this-geomtype -value 1
	 radiobutton $w.f.r.geom.unstructured -text Unstructured -variable $this-geomtype -value 2
	 pack $w.f.r.geom.structured -side left
	 pack $w.f.r.geom.unstructured -side right
	 pack $w.f.r.geom -side top -fill x

	frame $w.f.r.attrib

	radiobutton $w.f.r.attrib.flat -text Flat -variable $this-attribtype -value 1
	pack $w.f.r.attrib.flat -side left

	radiobutton $w.f.r.attrib.accel -text Accel -variable $this-attribtype -value 2
	pack $w.f.r.attrib.accel -side left

	radiobutton $w.f.r.attrib.brick -text Brick -variable $this-attribtype -value 3
	pack $w.f.r.attrib.brick -side left

	radiobutton $w.f.r.attrib.constant -text Constant -variable $this-attribtype -value 4
	pack $w.f.r.attrib.constant -side left

	pack $w.f.r.attrib -side top -fill x

	make_entry $w.f.r.nx "x:" $this-nx "$this-c needexecute"
	make_entry $w.f.r.ny "y:" $this-ny "$this-c needexecute "
	make_entry $w.f.r.nz "z:" $this-nz "$this-c needexecute "
	pack $w.f.r.nx $w.f.r.ny $w.f.r.nz -fill x

	make_entry $w.f.r.sx "sx:" $this-sx "$this-c needexecute"
	make_entry $w.f.r.sy "sy:" $this-sy "$this-c needexecute "
	make_entry $w.f.r.sz "sz:" $this-sz "$this-c needexecute "
	pack $w.f.r.sx $w.f.r.sy $w.f.r.sz -fill x

	make_entry $w.f.r.ex "ex:" $this-ex "$this-c needexecute"
	make_entry $w.f.r.ey "ey:" $this-ey "$this-c needexecute "
	make_entry $w.f.r.ez "ez:" $this-ez "$this-c needexecute "
	pack $w.f.r.ex $w.f.r.ey $w.f.r.ez -fill x

	make_entry $w.f.r.fval "Value:" $this-fval "$this-c needexecute"
	pack $w.f.r.fval -pady 5 -fill x

	button $w.f.go -text "Execute" -relief raised -command $n
	pack $w.f.r $w.f.go -fill y

	set_defaults
    }
}
