itcl_class Insight_DataIO_Switch {
    inherit Module
    constructor {config} {
        set name Switch

	global $this-which_port

        set_defaults
    }

    method set_defaults {} {
	set $this-which_port 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }
        toplevel $w

	label $w.options -text "Select Input"
	pack $w.options 

	radiobutton $w.a -text "Port 1" \
	    -variable $this-which_port \
	    -value 1
	pack $w.a 

	radiobutton $w.b -text "Port 2" \
	    -variable $this-which_port \
	    -value 2
	pack $w.b 

	radiobutton $w.c -text "Port 3" \
	    -variable $this-which_port \
	    -value 3
	pack $w.c 

	radiobutton $w.d -text "Port 4" \
	    -variable $this-which_port \
	    -value 4
	pack $w.d 



	button $w.execute -text "Execute" -command "$this-c needexecute"
	button $w.close -text "Close" -command "destroy $w"
	pack $w.execute $w.close 

    }
}


