
itcl_class Uintah_Operators_ScalarOperator {
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


	label $w.calc.l \
	    -text "Select Operation \n All output fields are LatVolFields"
	radiobutton $w.calc.log -text "Natural Log" \
		-variable $this-operation -value 0 \
		-command "$this-c needexecute"
	radiobutton $w.calc.e -text "Exponential" \
		-variable $this-operation -value 1 \
		-command "$this-c needexecute"
	radiobutton $w.calc.e -text "No Op" \
		-variable $this-operation -value 20 \
		-command "$this-c needexecute"
	pack $w.calc.l $w.calc.log $w.calc.e -anchor w 
	pack $w.calc -side left -padx 2 -pady 2 -fill y

    }

}
