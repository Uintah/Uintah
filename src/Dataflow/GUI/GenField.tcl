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

	global $this-fval1
	global $this-fval2
	global $this-fval3
	global $this-fval4
	global $this-fval5
	global $this-fval6
	global $this-fval7
	global $this-fval8
	global $this-fval9

	set $this-nx 10
	set $this-ny 10
	set $this-nz 10

	set $this-sx -1.0
	set $this-sy -1.0
	set $this-sz -1.0

	set $this-ex 1.0
	set $this-ey 1.0
	set $this-ez 1.0

	set $this-fval1 "x^2 + y^2 + z^2"
	set $this-fval2 "2 * y"
	set $this-fval3 "2 * z"
	set $this-fval4 "y * x"
	set $this-fval5 "y * y"
	set $this-fval6 "y * z"
	set $this-fval7 "z * x"
	set $this-fval8 "z * y"
	set $this-fval9 "z * z"
    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v
	bind $w.e <Return> $c
	pack $w.e -side right
    }
    method make_range_entry {w text text1 text2 v1 v2 c} {
	frame $w
	label $w.l -text "$text"
	label $w.t1 -text "$text1"
	entry $w.e1 -textvariable $v1 -width 8
	bind $w.e1 <Return> $c
	label $w.t2 -text "$text2"
	entry $w.e2 -textvariable $v2 -width 8
	bind $w.e2 <Return> $c
	pack $w.l -side left
	pack $w.e2 $w.t2 $w.e1 $w.t1 -side right
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

	global $this-fval1
	global $this-fval2
	global $this-fval3
	global $this-fval4
	global $this-fval5
	global $this-fval6
	global $this-fval7
	global $this-fval8
	global $this-fval9

	global $this-indexed

	toplevel $w
	wm minsize $w 300 250
	frame $w.f -width 400 -height 500
	pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
	set n "$this-c needexecute"
	frame $w.f.r


	iwidgets::tabnotebook $w.f.r.functions -width 3i -height 1i

	set page1 [$w.f.r.functions add -label "Scalar"]
	make_entry $page1.fval1 "Function" $this-fval1 "$this-c needexecute"
	pack $page1.fval1

	set page2 [$w.f.r.functions add -label "Vector"]
	make_entry $page2.fval1 "Function X" $this-fval1 "$this-c needexecute"
	make_entry $page2.fval2 "Function Y" $this-fval2 "$this-c needexecute"
	make_entry $page2.fval3 "Function Z" $this-fval3 "$this-c needexecute"
	pack $page2.fval1 $page2.fval2 $page2.fval3

	# set page3 [$w.f.r.functions add -label "Tensor"]
	# make_entry $page3.fval1 "Function XX" $this-fval1 "$this-c needexecute"
	# make_entry $page3.fval2 "Function XY" $this-fval2 "$this-c needexecute"
	# make_entry $page3.fval3 "Function XZ" $this-fval3 "$this-c needexecute"
	# make_entry $page3.fval4 "Function YX" $this-fval4 "$this-c needexecute"
	# make_entry $page3.fval5 "Function YY" $this-fval5 "$this-c needexecute"
	# make_entry $page3.fval6 "Function YZ" $this-fval6 "$this-c needexecute"
	# make_entry $page3.fval7 "Function ZX" $this-fval7 "$this-c needexecute"
	# make_entry $page3.fval8 "Function ZY" $this-fval8 "$this-c needexecute"
	# make_entry $page3.fval9 "Function ZZ" $this-fval9 "$this-c needexecute"
	# pack $page3.fval1 $page3.fval2 $page3.fval3
	# pack $page3.fval4 $page3.fval5 $page3.fval6
	# pack $page3.fval7 $page3.fval8 $page3.fval9

	$w.f.r.functions view "Scalar"
	$w.f.r.functions configure -tabpos n

	pack $w.f.r.functions


	# Attribute Type
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

	# Structured or Unstructured geometry?
	frame $w.f.r.geom
	radiobutton $w.f.r.geom.structured -text Structured -variable $this-geomtype -value 1
	radiobutton $w.f.r.geom.unstructured -text Unstructured -variable $this-geomtype -value 2
	pack $w.f.r.geom.structured -side left
	pack $w.f.r.geom.unstructured -side right
	pack $w.f.r.geom -side top -fill x

	make_entry $w.f.r.nx "X Sample Size" $this-nx "$this-c needexecute"
	make_entry $w.f.r.ny "Y Sample Size" $this-ny "$this-c needexecute"
	make_entry $w.f.r.nz "Z Sample Size" $this-nz "$this-c needexecute"
	pack $w.f.r.nx $w.f.r.ny $w.f.r.nz -fill x

	make_range_entry $w.f.r.rangex "X Location" "From" "To" $this-sx $this-ex "$this-c needexecute"
	make_range_entry $w.f.r.rangey "Y Location" "From" "To" $this-sy $this-ey "$this-c needexecute"
	make_range_entry $w.f.r.rangez "Z Location" "From" "To" $this-sz $this-ez "$this-c needexecute"
	pack $w.f.r.rangex $w.f.r.rangey $w.f.r.rangez -fill x


	checkbutton $w.f.r.indexed -text "Force Indexed" -variable $this-indexed
	pack $w.f.r.indexed -fill x

	button $w.f.r.go -text "Execute" -relief raised -command $n
	pack $w.f.r $w.f.r.go -fill y

	set_defaults
    }
}
