itcl_class Uintah_Operators_TensorToTensorConvertor {
    inherit Module

    protected update_type

    constructor {config} {
        set name TensorToTensorConvertor
        set_defaults
    }

    method set_defaults {} {
        global $this-operation
        set $this-operation 0

	set update_type "Auto"
    }

    # If you call this function instead of $this-c needexecute
    # directly you will be able to selectively auto execute or not.
    method execute_maybe {} {
	# Check to see if we should update or not
	if { $update_type == "Auto" } {
	    eval "$this-c needexecute"
	}
    }

    method set_update_type { w } {
	# I don't believe this needs to be global, but try it if things break.
#	global $w
	set update_type [$w get]
    }
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	set c "$this execute_maybe"

	# Which operation to perform
        frame $w.frame 
        label $w.frame.label -text "Convert Into .."

        radiobutton $w.frame.none -text "None" \
                -variable $this-operation -value 0 -command $c

        radiobutton $w.frame.green -text "Green-Lagrange Strain Tensor" \
                -variable $this-operation -value 1 -command $c

        radiobutton $w.frame.cauchy -text "Cauchy-Green Deformation Tensor" \
                -variable $this-operation -value 2 -command $c 

        radiobutton $w.frame.finger -text "Finger Deformation Tensor" \
                -variable $this-operation -value 3 -command $c 

        pack $w.frame.label $w.frame.none $w.frame.green $w.frame.cauchy \
             $w.frame.finger

        pack $w.frame -side top -padx 2 -pady 2 -fill y -fill x

	#  This is a seperator
	frame $w.separator0 -height 2 -relief sunken -borderwidth 2
	pack  $w.separator0 -fill x -pady 5
	
	# The auto update stuff
	frame $w.update
	
	iwidgets::optionmenu $w.update.menu -labeltext "Update:" \
	    -labelpos w -command "$this set_update_type $w.update.menu"
	$w.update.menu insert end "Manual" "Auto"
	$w.update.menu select [set update_type]

	pack $w.update.menu -side left

	pack $w.update -side top -padx 2 -pady 2 -fill x

	# This is the SCI standard button
	makeSciButtonPanel $w $w $this
    }
}


