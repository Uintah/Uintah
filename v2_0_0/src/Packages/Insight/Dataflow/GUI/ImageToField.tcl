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

itcl_class Insight_Converters_ImageToField {
    inherit Module
    constructor {config} {
        set name ImageToField

	global $this-copy

        set_defaults
    }

    method set_defaults {} {
	set $this-copy 0
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

	radiobutton $w.a -text "Reference ITK Image Data" \
	    -variable $this-copy \
	    -value 0
	pack $w.a 

	radiobutton $w.b -text "Copy Data" \
	    -variable $this-copy \
	    -value 1
	pack $w.b 


	frame $w.buttons
        button $w.buttons.execute -text "Execute" -command "$this-c needexecute"
        button $w.buttons.close -text "Close" -command "destroy $w"
        pack $w.buttons.execute $w.buttons.close -side left
	pack $w.buttons -side top -padx 5 -pady 5
    }
}


