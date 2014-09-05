
itcl_class Uintah_Operators_VectorOperator {
    inherit Module

    method set_defaults {} {
	global $this-operation
	set $this-operation 0
	
	# element extractor
	global $this-row
	global $this-column
	global $this-elem
	set $this-row 2
	set $this-column 2
	set $this-elem 5

	# 2D eigen evaluator
	global $this-planeSelect
	global $this-delta
	global $this-eigen2D-calc-type
	set $this-planeSelect 2
	set $this-eigen2D-calc-type 0
	set $this-delta 1

    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left -anchor w
	entry $w.e -textvariable $v -width 8
	bind $w.e <Return> $c
	pack $w.e -side right -anchor e
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
 	toplevel $w

	button $w.b -text Close -command "destroy $w"
	pack $w.b -side bottom -expand yes -fill x -padx 2 -pady 2

	frame $w.calc -relief raised -bd 1
	label $w.calc.l -text "Select Operation"
	radiobutton $w.calc.u -text "U" \
		-variable $this-operation -value 0 \
		-command "$this-c needexecute"
	radiobutton $w.calc.v -text "V" \
		-variable $this-operation -value 1 \
		-command "$this-c needexecute"
	radiobutton $w.calc.w -text "W" \
		-variable $this-operation -value 2 \
		-command "$this-c needexecute"
	radiobutton $w.calc.length -text "Length" \
		-variable $this-operation -value 3 \
		-command "$this-c needexecute"
	radiobutton $w.calc.vort -text "Vorticity" \
		-variable $this-operation -value 4 \
		-command "$this-c needexecute"
	pack $w.calc.l $w.calc.u $w.calc.v \
	    $w.calc.w $w.calc.length $w.calc.vort -anchor w
	pack $w.calc -side left -padx 2 -pady 2 -fill y

    }

}
