
itcl_class Uintah_Operators_ScalarFieldBinaryOperator {
    inherit Module

    constructor {config} {
	set name ScalarFieldBinaryOperator
	set_defaults
    }

    method set_defaults {} {
	global $this-operation
	set $this-operation 0
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
	radiobutton $w.calc.add -text "Add" \
		-variable $this-operation -value 0 \
		-command "$this-c needexecute"
	radiobutton $w.calc.sub -text "Subtract" \
		-variable $this-operation -value 1 \
		-command "$this-c needexecute"
	radiobutton $w.calc.mul -text "Multiply" \
		-variable $this-operation -value 2 \
		-command "$this-c needexecute"
	radiobutton $w.calc.div -text "Divide" \
		-variable $this-operation -value 3 \
		-command "$this-c needexecute"
	radiobutton $w.calc.ave -text "Average" \
		-variable $this-operation -value 4 \
		-command "$this-c needexecute"

	pack $w.calc.l $w.calc.add $w.calc.sub $w.calc.mul $w.calc.div \
	    $w.calc.ave -anchor w 
	pack $w.calc -side left -padx 2 -pady 2 -fill y

    }

}
