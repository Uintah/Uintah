#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

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


