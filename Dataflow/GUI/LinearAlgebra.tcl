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

itcl_class SCIRun_Math_LinearAlgebra {
    inherit Module
    constructor {config} {
        set name LinearAlgebra
	
	global $this-function

        set_defaults
    }

    method set_defaults {} {
	set $this-function "A"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.row1	
	entry $w.row1.entry -textvariable $this-function -width 60
	pack $w.row1.entry -side left -fill x -padx 5

	frame $w.row2
	button $w.row2.execute -text "Execute" -command "$this-c needexecute"
	pack $w.row2.execute -side left -e y -f both -padx 5 -pady 5
	pack $w.row1 $w.row2 -side top -e y -f both -padx 5 -pady 5
    }
}


